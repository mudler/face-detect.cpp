# face-detect.cpp container image.
#
# Multi-stage build: a fat build stage compiles facedetect-cli (and the ggml
# backends it links against), then a slim runtime stage carries only the binary
# plus the ggml shared libraries.
#
# The same Dockerfile produces the CPU and CUDA variants. Select with build args:
#
#   CPU (default):
#     docker build -t face-detect.cpp:cpu .
#
#   CUDA (GGML_CUDA_NO_VMM=ON drops the libcuda driver-lib link dependency, which
#   a GPU-less build container does not have):
#     docker build -t face-detect.cpp:cuda \
#       --build-arg BUILD_BASE=nvidia/cuda:13.0.1-devel-ubuntu24.04 \
#       --build-arg RUNTIME_BASE=nvidia/cuda:13.0.1-runtime-ubuntu24.04 \
#       --build-arg "CMAKE_EXTRA_ARGS=-DFACEDETECT_GGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON" .
#
# The build context must be a checkout with the ggml submodule populated
# (git clone --recursive, or actions/checkout with submodules: recursive).
# Models are not bundled: mount a pre-converted .gguf at runtime.

ARG BUILD_BASE=ubuntu:24.04
ARG RUNTIME_BASE=ubuntu:24.04

# ---------------------------------------------------------------------------
# build: configure + compile facedetect-cli and the ggml backends.
# ---------------------------------------------------------------------------
FROM ${BUILD_BASE} AS build

ARG CMAKE_EXTRA_ARGS=""
# CUDA architectures (e.g. "90;121-real"). Empty = ggml's default broad list.
ARG CUDA_ARCHS=""

ENV DEBIAN_FRONTEND=noninteractive
# git + ca-certificates are also required at build time: CMake fetches and builds
# libjpeg-turbo (static, for cv2.imread-parity JPEG decode) via ExternalProject.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# CMake auto-applies the in-tree ggml patches during configure via
# scripts/apply_ggml_patches.sh, which uses `git apply` and therefore needs
# third_party/ggml to be a git repo. Re-init it as a throwaway repo.
RUN rm -rf third_party/ggml/.git && git -C third_party/ggml init -q

# GGML_NATIVE=OFF keeps the binary portable across the CPUs that pull the image.
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DFACEDETECT_BUILD_CLI=ON \
        -DFACEDETECT_BUILD_TESTS=OFF \
        ${CMAKE_EXTRA_ARGS} \
        ${CUDA_ARCHS:+"-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"} \
    && cmake --build build -j"$(nproc)"

# Stage the binary and every backend shared library into a clean prefix.
RUN mkdir -p /install/bin /install/lib \
    && cp build/examples/cli/facedetect-cli /install/bin/ \
    && find build -name '*.so*' -exec cp -av {} /install/lib/ \;

# ---------------------------------------------------------------------------
# runtime: slim image with just the binary and its shared libraries.
# ---------------------------------------------------------------------------
FROM ${RUNTIME_BASE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /install/bin/ /usr/local/bin/
COPY --from=build /install/lib/ /usr/local/lib/
RUN ldconfig

WORKDIR /work
ENTRYPOINT ["facedetect-cli"]
CMD ["--help"]
