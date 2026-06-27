#!/usr/bin/env bash
#
# run_sface_parity.sh
#
# Drive the SFace recognizer (OpenCV-Zoo, Apache-2.0, 128-d) parity gate end to
# end - the commercial-friendly recognizer paired with the YuNet detector:
#
#   1. download the YuNet detector + SFace recognizer ONNX from OpenCV Zoo if absent,
#   2. convert the yunet+sface pack to GGUF (embeds the SFace ONNX topology as the
#      facedetect.recognizer.graph KV; the C++ replays it metadata-driven through
#      the SAME interpreter as the MobileFaceNet recognizer - Task 8.3 adds only the
#      Sub/Dropout ops + spatial BatchNorm + per-node epsilon, no new ggml op),
#   3. dump the cv2.FaceRecognizerSF reference baselines for face_a/b/c (the SFace
#      alignCrop 112x112 crop + the raw 128-d feature, plus the source pixels +
#      primary landmarks for the alignment gate),
#   4. gate the aligned crop (norm_crop vs alignCrop, <= a few LSB) + the 128-d
#      embedding (tests/test_sface_embedding): raw feature max|d| <= 1e-3 AND the
#      L2-normalized cosine >= 0.9999, fed the reference's EXACT aligned crop.
#
# This is the Apache-2.0 recognizer analogue of run_embedding_parity.sh (insightface
# ArcFace) and the recognizer half of the YuNet+SFace pack (run_yunet_parity.sh).
#
# Usage:
#   bash scripts/run_sface_parity.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

VENV="${FACEDETECT_VENV:-${PROJECT_ROOT}/.venv}"
DTYPE="${FACEDETECT_DTYPE:-f32}"
DET_ONNX="${PROJECT_ROOT}/models/yunet/face_detection_yunet_2023mar.onnx"
REC_ONNX="${PROJECT_ROOT}/models/sface/face_recognition_sface_2021dec.onnx"
MODEL="${FACEDETECT_MODEL:-${PROJECT_ROOT}/models/yunet-sface-${DTYPE}.gguf}"
BUILD="${FACEDETECT_BUILD:-${PROJECT_ROOT}/build}"
OUTDIR="${PROJECT_ROOT}/out/sface"
PY="${VENV}/bin/python"
DET_URL="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
REC_URL="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

if [[ ! -x "${PY}" ]]; then echo "error: venv python not found at ${PY}" >&2; exit 1; fi
if [[ ! -d "${BUILD}" ]]; then
    echo "error: build dir ${BUILD} not found; configure + build first:" >&2
    echo "       cmake -B build -DFACEDETECT_BUILD_TESTS=ON && cmake --build build -j" >&2
    exit 1
fi

# 1. Download the YuNet + SFace ONNX if needed.
if [[ ! -f "${DET_ONNX}" ]]; then
    echo "model: downloading YuNet ONNX -> ${DET_ONNX}"
    mkdir -p "$(dirname "${DET_ONNX}")"
    curl -sL "${DET_URL}" -o "${DET_ONNX}"
fi
if [[ ! -f "${REC_ONNX}" ]]; then
    echo "model: downloading SFace ONNX -> ${REC_ONNX}"
    mkdir -p "$(dirname "${REC_ONNX}")"
    curl -sL "${REC_URL}" -o "${REC_ONNX}"
fi

# 2. Convert the yunet+sface GGUF if needed (resolve_pack pairs the sibling SFace).
if [[ -f "${MODEL}" ]]; then
    echo "model: ${MODEL} (already present)"
else
    echo "model: converting yunet+sface -> ${MODEL} (dtype ${DTYPE})"
    "${PY}" scripts/convert_facedetect_to_gguf.py --model "${DET_ONNX}" --output "${MODEL}" --dtype "${DTYPE}"
fi

# 3. Dump the per-fixture SFace baselines (cv2 FaceRecognizerSF golden).
mkdir -p "${OUTDIR}"
for tag in a b c; do
    img="tests/fixtures/face_${tag}.jpg"
    out="${OUTDIR}/baseline_${tag}.gguf"
    if [[ ! -f "${img}" ]]; then echo "error: fixture ${img} missing" >&2; exit 1; fi
    echo "baseline: ${img} -> ${out}"
    FACEDETECT_DET_ONNX="${DET_ONNX}" \
        "${PY}" scripts/gen_baseline.py --model "${REC_ONNX}" --image "${img}" --output "${out}"
done

# 4. Gate the aligned crop + the 128-d embedding (all fixtures).
fail=0
for tag in a b c; do
    echo "=== sface parity: ${tag} ==="
    FACEDETECT_TEST_GGUF="${MODEL}" \
    FACEDETECT_TEST_BASELINE="${OUTDIR}/baseline_${tag}.gguf" \
        "${BUILD}/tests/test_sface_embedding" || fail=1
done

if [[ "${fail}" -ne 0 ]]; then echo "SFACE PARITY FAILED" >&2; exit 1; fi
echo "SFACE PARITY OK"
