#!/usr/bin/env bash
#
# apply_ggml_patches.sh
#
# Apply the in-tree ggml patches to third_party/ggml. Idempotent: re-running
# is a no-op once everything is applied.
#
# Patches live in third_party/ggml-patches/ and are applied in filename order
# (the numeric prefix from `git format-patch` gives us the right ordering).
#
# Usage:
#   bash scripts/apply_ggml_patches.sh
#
# Exits 0 on success, non-zero on any failure. Designed to be called by CMake
# during configure but also runnable standalone for debugging.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GGML_DIR="${PROJECT_ROOT}/third_party/ggml"
PATCH_DIR="${PROJECT_ROOT}/third_party/ggml-patches"

if [[ ! -d "${GGML_DIR}" ]]; then
    echo "error: ggml submodule not found at ${GGML_DIR}" >&2
    echo "       did you forget 'git submodule update --init --recursive'?" >&2
    exit 1
fi

if [[ ! -d "${GGML_DIR}/.git" && ! -f "${GGML_DIR}/.git" ]]; then
    echo "error: ${GGML_DIR} is not a git repository" >&2
    exit 1
fi

# No patches directory (or an empty one) is fine: nothing to do.
if [[ ! -d "${PATCH_DIR}" ]]; then
    echo "ggml patches: no patch directory at ${PATCH_DIR} (nothing to do)"
    exit 0
fi

shopt -s nullglob
PATCHES=("${PATCH_DIR}"/*.patch)
shopt -u nullglob

if [[ ${#PATCHES[@]} -eq 0 ]]; then
    echo "ggml patches: no patches found in ${PATCH_DIR} (nothing to do)"
    exit 0
fi

# Sort by filename so the numeric prefix (0001-, 0002-, ...) determines order.
IFS=$'\n' PATCHES=($(printf '%s\n' "${PATCHES[@]}" | sort))
unset IFS

applied=0
skipped=0

cd "${GGML_DIR}"

# Serialise concurrent invocations against the shared submodule tree (parallel
# CMake configures, e.g. a downstream per-CPU-variant build matrix, race here).
if [[ -z "${FACEDETECT_PATCH_FLOCK_HELD:-}" ]] && command -v flock >/dev/null 2>&1; then
    LOCK_FILE="${PROJECT_ROOT}/third_party/.ggml-patch.lock"
    : > "${LOCK_FILE}" 2>/dev/null || true
    if [[ -e "${LOCK_FILE}" ]]; then
        export FACEDETECT_PATCH_FLOCK_HELD=1
        SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
        exec flock "${LOCK_FILE}" bash "${SCRIPT_PATH}" "$@"
    fi
fi

for patch in "${PATCHES[@]}"; do
    name="$(basename "${patch}")"

    if git apply --check --reverse "${patch}" >/dev/null 2>&1; then
        echo "ggml patches: skipping ${name} (already applied)"
        skipped=$((skipped + 1))
        continue
    fi

    if git apply --check "${patch}" >/dev/null 2>&1; then
        if ! git apply "${patch}"; then
            echo "error: failed to apply ${name} after --check succeeded" >&2
            exit 1
        fi
        echo "ggml patches: applied ${name}"
        applied=$((applied + 1))
        continue
    fi

    echo "error: cannot apply ${name}" >&2
    git apply --check "${patch}" 2>&1 | sed 's/^/         /' >&2 || true
    echo "       submodule HEAD: $(git rev-parse HEAD)" >&2
    exit 1
done

echo "ggml patches: applied ${applied}, skipped ${skipped}"
