#!/usr/bin/env bash
#
# run_detector_parity.sh
#
# Drive the SCRFD detector parity gates end to end for a pack:
#
#   1. activate the project venv,
#   2. convert the pack GGUF if absent (embeds the detector ONNX topology as the
#      facedetect.detector.graph KV for det_2.5g / det_500m; det_10g stays
#      hand-mapped),
#   3. dump the reference baselines for face_a/b/c + multi (per-stride raw heads
#      scrfd_out_0..8, the exact letterbox/blob, and the post-NMS det set),
#   4. gate per-stride raw heads (tests/test_scrfd_graph, max|d| ~1e-3 vs the ONNX
#      reference) on the single-face and multi images, then
#   5. gate end-to-end detection (tests/test_detect: boxes + 5 landmarks <= 1 px
#      from a real JPEG decode, all faces, after the same NMS).
#
# This is the detector analogue of run_embedding_parity.sh. det_2.5g (buffalo_m)
# and det_500m (buffalo_s) run through the metadata-driven interpreter; det_10g
# (buffalo_l) runs the hand-mapped path. The crowded multi fixture can differ by
# one face on the weakest detector (det_500m) when a face sits ON the 0.5 score
# threshold and the production letterbox's <=1 LSB drift flips it; test_detect
# tolerates that single boundary face while still requiring every supra-threshold
# face to match within 1 px (see its header).
#
# Usage:
#   bash scripts/run_detector_parity.sh                  # buffalo_l
#   FACEDETECT_PACK=buffalo_m bash scripts/run_detector_parity.sh
#   FACEDETECT_PACK=buffalo_s bash scripts/run_detector_parity.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PACK="${FACEDETECT_PACK:-buffalo_l}"
VENV="${FACEDETECT_VENV:-${PROJECT_ROOT}/.venv}"
DTYPE="${FACEDETECT_DTYPE:-f16}"
MODEL="${FACEDETECT_MODEL:-${PROJECT_ROOT}/models/${PACK}-${DTYPE}.gguf}"
BUILD="${FACEDETECT_BUILD:-${PROJECT_ROOT}/build}"
OUTDIR="${PROJECT_ROOT}/out/${PACK}"
PY="${VENV}/bin/python"

if [[ ! -x "${PY}" ]]; then
    echo "error: venv python not found at ${PY}" >&2
    exit 1
fi
if [[ ! -d "${BUILD}" ]]; then
    echo "error: build dir ${BUILD} not found; configure + build first:" >&2
    echo "       cmake -B build -DFACEDETECT_BUILD_TESTS=ON && cmake --build build -j" >&2
    exit 1
fi

# 1. Convert the model GGUF if needed.
if [[ -f "${MODEL}" ]]; then
    echo "model: ${MODEL} (already present)"
else
    echo "model: converting ${PACK} -> ${MODEL} (dtype ${DTYPE})"
    mkdir -p "$(dirname "${MODEL}")"
    "${PY}" scripts/convert_facedetect_to_gguf.py \
        --model "${PACK}" --output "${MODEL}" --dtype "${DTYPE}"
fi

# 2. Dump the per-fixture baselines (single-face + multi).
mkdir -p "${OUTDIR}"
for tag in a b c multi; do
    img="tests/fixtures/$([[ "${tag}" == multi ]] && echo multi || echo "face_${tag}").jpg"
    out="${OUTDIR}/baseline_${tag}.gguf"
    if [[ ! -f "${img}" ]]; then echo "error: fixture ${img} missing" >&2; exit 1; fi
    echo "baseline: ${img} -> ${out}"
    "${PY}" scripts/gen_baseline.py --model "${PACK}" --image "${img}" --output "${out}"
done

# 3. Per-stride raw head gate (face_a + multi) and end-to-end detection gate (all).
fail=0
for tag in a multi; do
    img="tests/fixtures/$([[ "${tag}" == multi ]] && echo multi || echo "face_${tag}").jpg"
    echo "=== per-stride raw heads: ${PACK} ${tag} ==="
    FACEDETECT_TEST_GGUF="${MODEL}" \
    FACEDETECT_TEST_BASELINE="${OUTDIR}/baseline_${tag}.gguf" \
    FACEDETECT_TEST_IMAGE="${img}" \
        "${BUILD}/tests/test_scrfd_graph" || fail=1
done
for tag in a b c multi; do
    img="tests/fixtures/$([[ "${tag}" == multi ]] && echo multi || echo "face_${tag}").jpg"
    echo "=== end-to-end detection: ${PACK} ${tag} ==="
    FACEDETECT_TEST_GGUF="${MODEL}" \
    FACEDETECT_TEST_BASELINE="${OUTDIR}/baseline_${tag}.gguf" \
    FACEDETECT_TEST_IMAGE="${img}" \
        "${BUILD}/tests/test_detect" || fail=1
done

if [[ "${fail}" -ne 0 ]]; then
    echo "DETECTOR PARITY FAILED for ${PACK}" >&2
    exit 1
fi
echo "DETECTOR PARITY OK for ${PACK}"
