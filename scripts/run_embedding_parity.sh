#!/usr/bin/env bash
#
# run_embedding_parity.sh
#
# Drive the strict all-three-fixture embedding parity gate end to end:
#
#   1. activate the project venv,
#   2. convert the buffalo_l model GGUF if it is not already present,
#   3. dump the reference baselines for face_a/b/c into the layout the test
#      discovers via siblings_of (out/baseline_{a,b,c}.gguf +
#      tests/fixtures/face_{a,b,c}.jpg),
#   4. run tests/test_embedding.cpp path C, which gates the strict
#      cosine >= 0.9999 AND max|d| <= 1e-3 bound on ALL THREE fixtures.
#
# Exits non-zero if any step (conversion, baseline dump, or the gate) fails.
#
# Usage:
#   bash scripts/run_embedding_parity.sh                 # buffalo_l
#   FACEDETECT_PACK=buffalo_m bash scripts/run_embedding_parity.sh
#   FACEDETECT_PACK=buffalo_s bash scripts/run_embedding_parity.sh
#
# Env overrides:
#   FACEDETECT_PACK   insightface pack    (default: buffalo_l)
#   FACEDETECT_VENV   venv dir            (default: <root>/.venv)
#   FACEDETECT_MODEL  model GGUF path     (default: <root>/models/<pack>-<dtype>.gguf)
#   FACEDETECT_DTYPE  converter dtype     (default: f16)
#   FACEDETECT_BUILD  cmake build dir     (default: <root>/build)
#
# buffalo_m (det_2.5g + w600k_r50) and buffalo_s (det_500m + w600k_mbf,
# MobileFaceNet) now have their SCRFD detectors ported (Task 8.1b: the det_2.5g /
# det_500m topology is replayed through the metadata-driven interpreter), so ALL
# packs gate the full pipeline (detect -> align -> recognize) here. The strict
# recognizer bound (cosine >= 0.9999 AND max|d| <= 1e-3) is still enforced on the
# golden-landmark (C) and golden-crop (B) paths; path A is the end-to-end cosine.

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
    echo "       create it with: python3 -m venv .venv && \\" >&2
    echo "         .venv/bin/pip install -r scripts/requirements.txt" >&2
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

# 2. Dump the three per-fixture baselines into the standard layout.
mkdir -p "${OUTDIR}"
for tag in a b c; do
    img="tests/fixtures/face_${tag}.jpg"
    out="${OUTDIR}/baseline_${tag}.gguf"
    if [[ ! -f "${img}" ]]; then
        echo "error: fixture ${img} missing" >&2
        exit 1
    fi
    echo "baseline: ${img} -> ${out}"
    "${PY}" scripts/gen_baseline.py --model "${PACK}" \
        --image "${img}" --output "${out}"
done

# 3. Run the strict all-three gate. The test derives face_b/c + baseline_b/c
#    from the face_a/baseline_a primary and FAILS (not skips) if a sibling is
#    missing, so a partial run cannot masquerade as a pass.
if [[ ! -d "${BUILD}" ]]; then
    echo "error: build dir ${BUILD} not found; configure + build first:" >&2
    echo "       cmake -B build -DFACEDETECT_BUILD_TESTS=ON && cmake --build build -j" >&2
    exit 1
fi

echo "gate: running test_embedding all-three path C for ${PACK}"
# Every pack now has a working detector, so the full production path (A) runs for
# all of them; FACEDETECT_TEST_NO_DETECTOR is left unset.
FACEDETECT_TEST_GGUF="${MODEL}" \
FACEDETECT_TEST_IMAGE="${PROJECT_ROOT}/tests/fixtures/face_a.jpg" \
FACEDETECT_TEST_BASELINE="${OUTDIR}/baseline_a.gguf" \
    ctest --test-dir "${BUILD}" -R test_embedding --output-on-failure
