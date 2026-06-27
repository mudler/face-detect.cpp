#!/usr/bin/env bash
#
# gpu_verify.sh - build face-detect.cpp with the ggml CUDA backend, confirm the
# numerical parity gates still hold when the engine runs on the GPU, and append
# GPU latency rows to the comparative benchmark table.
#
# IMPORTANT: this script MUST run on a CUDA host (e.g. a DGX). It is deliberately
# NOT wired into CI here, because the CI/dev box that produced the CPU numbers in
# benchmarks/RESULTS.md has no NVIDIA GPU (no nvidia-smi / nvcc). Run it by hand
# on a machine with the CUDA toolkit + driver installed.
#
# What it does:
#   1. build/  -> build-cuda/  with -DFACEDETECT_GGML_CUDA=ON.
#   2. Runs the existing parity runner scripts (run_embedding_parity.sh,
#      run_detector_parity.sh, run_sface_parity.sh, run_yunet_parity.sh) pointed
#      at build-cuda/, with FACEDETECT_DEVICE forcing the ggml CUDA device. Each
#      script regenerates the CPU reference goldens (insightface / cv2 /
#      onnxruntime - device-independent) and diffs the GPU engine output against
#      them: the SAME strict gates as on CPU (recognizer cosine >= 0.9999,
#      max|d| <= 1e-3; detector boxes+landmarks <= 1 px; aligned crop <= ~1 LSB).
#   3. Runs scripts/bench_compare.py --device <dev> --append-gpu to append GPU
#      ggml ms/image rows to benchmarks/RESULTS.md.
#
# Device selection: the engine's runtime backend pick is driven by
# FACEDETECT_DEVICE (see src/backend.cpp): "cpu" forces CPU, otherwise it matches
# a ggml backend-registry device by name (case-insensitive), e.g. "CUDA0". This
# script exports FACEDETECT_DEVICE=$FD_GPU_DEVICE (default CUDA0) for every gate
# and bench run. Override with: FD_GPU_DEVICE=CUDA1 ./scripts/gpu_verify.sh
#
# Usage:
#   ./scripts/gpu_verify.sh                 # full build + parity + GPU bench
#   FD_GPU_DEVICE=CUDA0 N=20 ./scripts/gpu_verify.sh
#   SKIP_BUILD=1 ./scripts/gpu_verify.sh    # reuse an existing build-cuda/
#   PACKS="buffalo_l" ./scripts/gpu_verify.sh   # limit the buffalo packs gated
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEV="${FD_GPU_DEVICE:-CUDA0}"
BUILD_DIR="${BUILD_DIR:-build-cuda}"
N="${N:-20}"
IMAGE="${IMAGE:-tests/fixtures/face_a.jpg}"
PY="${PY:-$ROOT/.venv/bin/python}"
CLI="$ROOT/$BUILD_DIR/examples/cli/facedetect-cli"
PACKS="${PACKS:-buffalo_l buffalo_m buffalo_s}"

# --- 0. CUDA host guard -----------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. gpu_verify.sh MUST run on a CUDA host." >&2
    echo "       (This is intentionally NOT run in the CPU-only CI/dev box.)" >&2
    exit 2
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "WARN: nvcc not on PATH; relying on CMake to locate the CUDA toolkit." >&2
fi
echo "== GPU host =="; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# --- 1. Build with the ggml CUDA backend ------------------------------------
if [ "${SKIP_BUILD:-0}" != "1" ]; then
    echo "== Building $BUILD_DIR with -DFACEDETECT_GGML_CUDA=ON =="
    cmake -B "$BUILD_DIR" \
        -DFACEDETECT_BUILD_TESTS=ON \
        -DFACEDETECT_GGML_CUDA=ON \
        -DGGML_NATIVE=ON
    cmake --build "$BUILD_DIR" -j
fi
[ -x "$CLI" ] || { echo "ERROR: $CLI not built" >&2; exit 1; }
[ -x "$PY" ]  || { echo "ERROR: venv python $PY missing (need it for reference goldens)" >&2; exit 1; }

# Sanity: confirm the CUDA device is actually visible to the ggml registry and
# that the engine selects it (the CLI logs "fd::Backend using device: ...").
echo "== Confirming ggml sees device '$DEV' =="
FACEDETECT_DEVICE="$DEV" "$CLI" info "$ROOT/models/buffalo_l-f16.gguf" >/dev/null 2>&1 || true

# --- 2. Parity gates with the engine forced onto the GPU --------------------
# Re-use the CPU parity runner scripts unchanged: they regenerate the (device-
# independent) reference goldens and run the test binaries against the build we
# point them at, honoring FACEDETECT_DEVICE through the engine's backend pick.
FAIL=0
export FACEDETECT_BUILD="$ROOT/$BUILD_DIR"
export FACEDETECT_DEVICE="$DEV"

run_gate() {  # run_gate <label> <script> [extra env assignments...]
    local label="$1"; shift
    local script="$1"; shift
    echo "== Parity [$label] on device '$DEV' =="
    if env "$@" bash "$script"; then
        echo "   PASS [$label]"
    else
        echo "   FAIL [$label]" >&2; FAIL=1
    fi
}

# SCRFD detector + ArcFace/MobileFaceNet recognizer (the buffalo packs).
for pack in $PACKS; do
    run_gate "detect/$pack"  scripts/run_detector_parity.sh  FACEDETECT_PACK="$pack"
    run_gate "embed/$pack"   scripts/run_embedding_parity.sh FACEDETECT_PACK="$pack"
done
# YuNet detector + SFace recognizer (the Apache pipeline).
run_gate "yunet" scripts/run_yunet_parity.sh
run_gate "sface" scripts/run_sface_parity.sh

# --- 3. Append GPU latency rows to the results table ------------------------
echo "== GPU bench (appending rows to benchmarks/RESULTS.md) =="
"$PY" scripts/bench_compare.py \
    --image "$IMAGE" --n "$N" --threads 1 --cli "$CLI" \
    --device "$DEV" --append-gpu --out benchmarks/RESULTS.md

if [ "$FAIL" -ne 0 ]; then
    echo "GPU PARITY FAILED - see failures above" >&2
    exit 1
fi
echo "GPU verify + bench complete. Parity holds on '$DEV'; GPU rows appended to benchmarks/RESULTS.md"
