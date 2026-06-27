cmake_minimum_required(VERSION 3.18)
project(face_detect_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(FACEDETECT_BUILD_TESTS "Build ctest targets" OFF)
option(FACEDETECT_BUILD_CLI   "Build facedetect-cli" ON)
option(FACEDETECT_SHARED      "Build libfacedetect as shared" OFF)
option(FACEDETECT_GGML_CUDA   "Forward GGML_CUDA" OFF)
option(FACEDETECT_GGML_CUDNN  "Use cuDNN for conv2d on CUDA (implicit-GEMM, no im2col spill)" OFF)
option(FACEDETECT_GGML_METAL  "Forward GGML_METAL" OFF)
option(FACEDETECT_GGML_VULKAN "Forward GGML_VULKAN" OFF)
option(FACEDETECT_GGML_HIP    "Forward GGML_HIP (ROCm)" OFF)

set(GGML_CUDA   ${FACEDETECT_GGML_CUDA}   CACHE BOOL "" FORCE)
set(GGML_METAL  ${FACEDETECT_GGML_METAL}  CACHE BOOL "" FORCE)
set(GGML_VULKAN ${FACEDETECT_GGML_VULKAN} CACHE BOOL "" FORCE)
set(GGML_HIP    ${FACEDETECT_GGML_HIP}    CACHE BOOL "" FORCE)

# CUDA graphs: ggml defaults this OFF, but on CUDA builds it captures/replays the
# compute graph for a small free speedup (never negative). Turn it on whenever we
# forward CUDA. The runtime kill-switch GGML_CUDA_DISABLE_GRAPHS=1 still works.
if(FACEDETECT_GGML_CUDA)
  set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
endif()

# cuDNN conv path. When FACEDETECT_GGML_CUDNN is on we (a) forward
# GGML_CUDA_USE_CUDNN so ggml-cuda compiles + links the cuDNN implicit-GEMM
# conv (src/ggml-cuda/cudnn-conv.cu, applied as an in-tree ggml patch) and (b)
# define FACEDETECT_GGML_CUDNN for libfacedetect so fd::conv2d emits
# GGML_OP_CONV_2D (ggml_conv_2d_direct) on CUDA instead of im2col+SGEMM, which
# ggml-cuda then routes to cuDNN. Requires CUDA. Runtime A/B: FACEDETECT_CONV=im2col
# (never emit the op) / GGML_CUDA_USE_CUDNN=0 (force the native fallback inside
# ggml-cuda).
if(FACEDETECT_GGML_CUDNN)
  if(NOT FACEDETECT_GGML_CUDA)
    message(FATAL_ERROR "FACEDETECT_GGML_CUDNN requires FACEDETECT_GGML_CUDA=ON")
  endif()
  set(GGML_CUDA_USE_CUDNN ON CACHE BOOL "" FORCE)
endif()

# Performance: -march=native (GGML_NATIVE) and tinyBLAS SGEMM (GGML_LLAMAFILE)
# give meaningful free speedups. Force them on unless the caller overrides.
if(NOT DEFINED GGML_NATIVE)
  set(GGML_NATIVE ON CACHE BOOL "ggml: optimize the build for the current system" FORCE)
endif()
if(NOT DEFINED GGML_LLAMAFILE)
  set(GGML_LLAMAFILE ON CACHE BOOL "ggml: use LLAMAFILE (tinyBLAS SGEMM)" FORCE)
endif()

# Apply any in-tree ggml patches before configuring the submodule. Idempotent.
find_program(BASH_EXECUTABLE bash)
if(BASH_EXECUTABLE AND EXISTS "${CMAKE_SOURCE_DIR}/scripts/apply_ggml_patches.sh")
  execute_process(
    COMMAND "${BASH_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/apply_ggml_patches.sh"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE _patch_result
    OUTPUT_VARIABLE _patch_output
    ERROR_VARIABLE  _patch_errors
  )
  if(_patch_output)
    message(STATUS "${_patch_output}")
  endif()
  if(_patch_result)
    message(WARNING "apply_ggml_patches.sh returned ${_patch_result}: ${_patch_errors}")
  endif()
endif()

add_subdirectory(third_party/ggml)

# ---------------------------------------------------------------------------
# libjpeg-turbo: the JPEG decoder. The fixtures (and any real-world input) are
# JPEG, and the parity gates require the C++ decode to match cv2.imread
# (insightface's loader) pixel-for-pixel. OpenCV decodes JPEG via libjpeg-turbo
# with JDCT_ISLOW IDCT + fancy chroma upsampling; matching the *same* library +
# settings removes the 1-3 LSB decode drift that stb_image otherwise introduced
# (which amplified to ~1.8e-3 max-abs in the 512-d embedding, breaking the strict
# 1e-3 gate). See src/image_io.cpp.
#
# Vendored via ExternalProject_Add and STATIC-linked: this keeps the eventual
# LocalAI `.so` (Phase 9) self-contained with no external libjpeg runtime dep.
# Built C-only (WITH_SIMD OFF) to avoid a NASM build dependency; libjpeg-turbo's
# scalar and SIMD decode paths are bit-identical (the project verifies both
# against the same reference checksums), so the C path still matches cv2's SIMD
# build bit-for-bit.
option(FACEDETECT_VENDOR_LIBJPEG "Vendor + static-link libjpeg-turbo (ExternalProject)" ON)
if(FACEDETECT_VENDOR_LIBJPEG)
  # libjpeg-turbo refuses add_subdirectory()/FetchContent_MakeAvailable on
  # purpose, so build it as an ExternalProject and import the resulting static
  # archive. PIC is forced so the archive folds cleanly into the eventual
  # LocalAI shared `.so` (Phase 9). C-only (WITH_SIMD OFF) to avoid a NASM build
  # dep; libjpeg-turbo's scalar and SIMD decode paths are bit-identical.
  include(ExternalProject)
  set(_jpeg_prefix  "${CMAKE_BINARY_DIR}/libjpeg-turbo")
  set(_jpeg_install "${_jpeg_prefix}/install")
  set(_jpeg_lib     "${_jpeg_install}/lib/libjpeg.a")
  set(_jpeg_include "${_jpeg_install}/include")
  ExternalProject_Add(libjpeg_turbo_ext
    GIT_REPOSITORY    https://github.com/libjpeg-turbo/libjpeg-turbo.git
    # Pin to an immutable commit SHA (tags are mutable) for reproducible builds.
    GIT_TAG           f29eda648547b36aa594c4116c7764a6c8a079b9  # libjpeg-turbo 3.0.4
    # GIT_SHALLOW requires a branch/tag ref; a bare commit SHA cannot be cloned
    # shallowly against GitHub, so a full clone is needed to reach the pin.
    GIT_SHALLOW       FALSE
    PREFIX            "${_jpeg_prefix}"
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${_jpeg_install}
      -DCMAKE_BUILD_TYPE=Release
      -DENABLE_SHARED=OFF
      -DENABLE_STATIC=ON
      -DWITH_TURBOJPEG=OFF
      -DWITH_SIMD=OFF
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    BUILD_BYPRODUCTS  "${_jpeg_lib}"
    UPDATE_DISCONNECTED TRUE)
  # INTERFACE_INCLUDE_DIRECTORIES must exist at configure time.
  file(MAKE_DIRECTORY "${_jpeg_include}")
  add_library(facedetect_jpeg STATIC IMPORTED)
  set_target_properties(facedetect_jpeg PROPERTIES
    IMPORTED_LOCATION "${_jpeg_lib}"
    INTERFACE_INCLUDE_DIRECTORIES "${_jpeg_include}")
  set(FACEDETECT_JPEG_TARGET facedetect_jpeg)
  set(FACEDETECT_JPEG_DEP libjpeg_turbo_ext)
else()
  find_package(JPEG REQUIRED)
  set(FACEDETECT_JPEG_TARGET JPEG::JPEG)
  set(FACEDETECT_JPEG_DEP "")
endif()

set(FACEDETECT_SRC
    src/facedetect.cpp
    src/facedetect_capi.cpp
    src/common.cpp
    src/model.cpp
    src/model_loader.cpp
    src/backend.cpp
    src/image_io.cpp
    src/align.cpp
    src/preprocess.cpp
    src/graph_ops.cpp
    src/winograd.cpp
    src/directconv.cpp
    src/scrfd_graph.cpp
    src/yunet_graph.cpp
    src/arcface_graph.cpp
    src/sface_graph.cpp
    src/genderage_graph.cpp
    src/antispoof_graph.cpp
    src/landmark_graph.cpp
    src/detect.cpp)

if(FACEDETECT_SHARED)
  add_library(facedetect SHARED ${FACEDETECT_SRC})
  # The C-API header marks symbols with plain extern "C", no __declspec. On MSVC
  # auto-export every symbol so the DLL is usable for dlopen/ctypes consumers.
  set_target_properties(facedetect PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  add_library(facedetect STATIC ${FACEDETECT_SRC})
endif()
target_include_directories(facedetect PUBLIC include PRIVATE src ${CMAKE_SOURCE_DIR}/third_party)
target_compile_definitions(facedetect PUBLIC $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>)
target_compile_definitions(facedetect PRIVATE FACEDETECT_HAVE_LIBJPEG=1)
target_link_libraries(facedetect PUBLIC ggml PRIVATE ${FACEDETECT_JPEG_TARGET})
if(FACEDETECT_GGML_CUDNN)
  target_compile_definitions(facedetect PRIVATE FACEDETECT_GGML_CUDNN)
endif()
if(FACEDETECT_JPEG_DEP)
  add_dependencies(facedetect ${FACEDETECT_JPEG_DEP})
endif()

# Winograd's hot GEMM uses AVX2 intrinsics. The production build sets
# GGML_NATIVE=OFF, so the library target is NOT compiled with -march=native;
# enable AVX2+FMA for just winograd.cpp when the host compiler supports them
# (the file falls back to a scalar path otherwise). Mirrors depth-anything.cpp's
# per-file flag.
#
# AVX-512: winograd.cpp also carries a zmm register-blocked GEMM microkernel
# (gemm_block_avx512_t8). It is RUNTIME-CPUID-DISPATCHED, not gated on a global
# flag: the zmm code lives in functions tagged
# __attribute__((target("avx512f,avx512bw,avx512vl"))) (GCC/Clang function
# multiversioning) and runs ONLY when __builtin_cpu_supports reports AVX-512 at
# run time; AVX2-only CPUs take the AVX2 microkernel. We deliberately DO NOT pass
# -mavx512f to the whole translation unit -- that would let the compiler hoist
# zmm into the always-run code and SIGILL on an AVX2-only host. So the default
# portable build is one binary that uses AVX-512 where present and AVX2 elsewhere,
# and ggml's CPU backend stays AVX2 (GGML_NATIVE=OFF). FACEDETECT_WINO_AVX512 is
# set only when the compiler can emit such a per-function AVX-512 path without a
# global flag (probed below); otherwise the file compiles AVX2-only.
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
check_cxx_compiler_flag("-mavx2" FACEDETECT_HAS_AVX2)
if(FACEDETECT_HAS_AVX2)
  set_source_files_properties(src/winograd.cpp PROPERTIES
      COMPILE_OPTIONS "-mavx2;-mfma")
  # directconv.cpp uses the identical AVX2-base + runtime-CPUID-dispatched AVX-512
  # function-multiversioning scheme as winograd.cpp (its zmm microkernel runs only
  # when the host advertises AVX-512). Same per-file -mavx2 -mfma base, never a
  # global -mavx512f; FACEDETECT_DCONV_AVX512 enables the per-function path when
  # the multiversioning probe (below) succeeds.
  set_source_files_properties(src/directconv.cpp PROPERTIES
      COMPILE_OPTIONS "-mavx2;-mfma")
  # Probe function-multiversioning: can the compiler build an AVX-512 function via
  # a per-function target attribute (using the masked load/store the microkernel
  # needs) WITHOUT -mavx512f on the command line? GCC/Clang headers force-enable
  # the ISA for such functions, so the intrinsics are available even though the TU
  # is AVX2-only. Only winograd.cpp gets the AVX2 base flags here, never -mavx512f.
  set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma")
  check_cxx_source_compiles("
    #include <immintrin.h>
    __attribute__((target(\"avx512f,avx512bw,avx512vl\")))
    static void k(const float* u, const float* v, float* m, int oc) {
        __mmask16 mask = (__mmask16)((1u << (oc & 15)) - 1u);
        __m512 a = _mm512_maskz_loadu_ps(mask, u);
        a = _mm512_fmadd_ps(a, _mm512_set1_ps(v[0]), a);
        _mm512_mask_storeu_ps(m, mask, a);
        _mm512_storeu_ps(m, _mm512_loadu_ps(u));
    }
    int main() { float u[16]={0}, v[1]={0}, m[16]; k(u, v, m, 7); return (int)m[0]; }
  " FACEDETECT_WINO_AVX512_DISPATCH)
  unset(CMAKE_REQUIRED_FLAGS)
  if(FACEDETECT_WINO_AVX512_DISPATCH)
    set_source_files_properties(src/winograd.cpp PROPERTIES
        COMPILE_DEFINITIONS "FACEDETECT_WINO_AVX512=1")
    set_source_files_properties(src/directconv.cpp PROPERTIES
        COMPILE_DEFINITIONS "FACEDETECT_DCONV_AVX512=1")
  endif()
endif()

# Symbol hygiene for the shared build: the static libjpeg-turbo archive is folded
# into libfacedetect.so. With default visibility its jpeg_* symbols would be
# re-exported and could interpose against (or be interposed by) another libjpeg
# already loaded in the LocalAI host process (Phase 9). --exclude-libs,ALL
# localizes every symbol pulled from static archives (jpeg_*, ggml internals)
# while leaving this target's own facedetect_capi_* symbols exported. The C-API
# symbols are plain extern "C" with no visibility attribute, so -fvisibility=hidden
# would hide them too; the linker approach is the correct one here. GNU ld / lld
# only (guarded by a feature check; a no-op on Apple ld and MSVC).
if(FACEDETECT_SHARED AND NOT MSVC)
  include(CheckLinkerFlag)
  check_linker_flag(CXX "-Wl,--exclude-libs,ALL" FACEDETECT_LD_HAS_EXCLUDE_LIBS)
  if(FACEDETECT_LD_HAS_EXCLUDE_LIBS)
    target_link_options(facedetect PRIVATE "-Wl,--exclude-libs,ALL")
  endif()
endif()

if(FACEDETECT_BUILD_CLI)
  add_subdirectory(examples/cli)
endif()
if(FACEDETECT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
