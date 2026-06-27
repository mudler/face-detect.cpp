#!/usr/bin/env bash
#
# run_yunet_parity.sh
#
# Drive the YuNet detector (OpenCV-Zoo, Apache-2.0) parity gate end to end:
#
#   1. download the YuNet ONNX from OpenCV Zoo if absent,
#   2. convert it to GGUF (embeds the detector ONNX topology as the
#      facedetect.detector.graph KV; the C++ replays it metadata-driven through
#      the SAME interpreter as the SCRFD det_2.5g detector - no new ggml op),
#   3. dump the cv2.FaceDetectorYN reference baselines for face_a/b/c + multi
#      (raw heads yunet_out_0..11, the exact 640 resized pixels + input blob, and
#      the decoded boxes/scores/landmarks in the 640 net space + yunet_scale),
#   4. gate raw heads + the anchor-free decode + NMS + the full end-to-end path
#      (tests/test_yunet): raw heads <=1e-3, decoded boxes/landmarks <=1px (decode
#      from the reference's exact pixels is bit-exact vs cv2.FaceDetectorYN).
#
# This is the Apache-2.0 analogue of run_detector_parity.sh (insightface SCRFD).
#
# Usage:
#   bash scripts/run_yunet_parity.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

VENV="${FACEDETECT_VENV:-${PROJECT_ROOT}/.venv}"
DTYPE="${FACEDETECT_DTYPE:-f32}"
ONNX="${PROJECT_ROOT}/models/yunet/face_detection_yunet_2023mar.onnx"
MODEL="${FACEDETECT_MODEL:-${PROJECT_ROOT}/models/yunet-${DTYPE}.gguf}"
BUILD="${FACEDETECT_BUILD:-${PROJECT_ROOT}/build}"
OUTDIR="${PROJECT_ROOT}/out/yunet"
PY="${VENV}/bin/python"
URL="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

if [[ ! -x "${PY}" ]]; then echo "error: venv python not found at ${PY}" >&2; exit 1; fi
if [[ ! -d "${BUILD}" ]]; then
    echo "error: build dir ${BUILD} not found; configure + build first:" >&2
    echo "       cmake -B build -DFACEDETECT_BUILD_TESTS=ON && cmake --build build -j" >&2
    exit 1
fi

# 1. Download the YuNet ONNX if needed.
if [[ ! -f "${ONNX}" ]]; then
    echo "model: downloading YuNet ONNX -> ${ONNX}"
    mkdir -p "$(dirname "${ONNX}")"
    curl -sL "${URL}" -o "${ONNX}"
fi

# 2. Convert the GGUF if needed.
if [[ -f "${MODEL}" ]]; then
    echo "model: ${MODEL} (already present)"
else
    echo "model: converting YuNet -> ${MODEL} (dtype ${DTYPE})"
    "${PY}" scripts/convert_facedetect_to_gguf.py --model "${ONNX}" --output "${MODEL}" --dtype "${DTYPE}"
fi

# 3. Dump the per-fixture baselines.
mkdir -p "${OUTDIR}"
for tag in a b c multi; do
    img="tests/fixtures/$([[ "${tag}" == multi ]] && echo multi || echo "face_${tag}").jpg"
    out="${OUTDIR}/baseline_${tag}.gguf"
    if [[ ! -f "${img}" ]]; then echo "error: fixture ${img} missing" >&2; exit 1; fi
    echo "baseline: ${img} -> ${out}"
    "${PY}" scripts/gen_baseline.py --model "${ONNX}" --image "${img}" --output "${out}"
done

# 4. Gate raw heads + decode + end-to-end (all fixtures).
fail=0
for tag in a b c multi; do
    img="tests/fixtures/$([[ "${tag}" == multi ]] && echo multi || echo "face_${tag}").jpg"
    echo "=== yunet parity: ${tag} ==="
    FACEDETECT_TEST_GGUF="${MODEL}" \
    FACEDETECT_TEST_BASELINE="${OUTDIR}/baseline_${tag}.gguf" \
    FACEDETECT_TEST_IMAGE="${img}" \
        "${BUILD}/tests/test_yunet" || fail=1
done

if [[ "${fail}" -ne 0 ]]; then echo "YUNET PARITY FAILED" >&2; exit 1; fi
echo "YUNET PARITY OK"
