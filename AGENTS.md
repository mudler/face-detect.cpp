# Contributor & agent guide

This is the working guide for humans and AI coding assistants on face-detect.cpp.

## AI-assisted contributions

face-detect.cpp follows the Linux kernel project's guidelines for AI coding
assistants. Before submitting AI-assisted code:

- **No `Signed-off-by` from an AI.** Only the human submitter signs off on the
  Developer Certificate of Origin (DCO). Every commit must carry the human's
  `Signed-off-by:` trailer.
- **No `Co-Authored-By: <AI>` trailers.** The human contributor owns the change.
- **Use an `Assisted-by:` trailer** to attribute AI involvement. Format:
  `Assisted-by: AGENT_NAME:MODEL_VERSION [TOOL]`.
- **The human submitter is responsible** for reviewing, testing, and
  understanding every line of generated code.

## What this project is

A standalone C++17/ggml face-recognition engine with GGUF weights, consumed by
LocalAI through a flat C ABI (dlopen/purego). It replaces LocalAI's Python
`insightface` backend. Pipeline: decode image -> SCRFD detect -> 5-landmark
align (insightface `norm_crop`) -> ArcFace embed -> L2-normalized 512-d
embedding, plus verify / analyze / detect and an optional MiniFASNet anti-spoof
veto.

## Repository layout

```
include/           public headers: facedetect.h, facedetect_capi.h (the frozen ABI)
src/               flat .cpp/.hpp stages:
  facedetect.cpp     version + fd:: convenience layer
  facedetect_capi.cpp flat C-ABI implementation
  model.{cpp,hpp}    load-once orchestration (detect/embed/analyze)
  model_loader.*     GGUF KV + tensor reader (facedetect.* namespace)
  backend.*          persistent ggml_backend_t + gallocr wrapper (+ global_backend)
  image_io.*         image decode to RGB (libjpeg-turbo for JPEG = cv2.imread
                     parity; vendored stb_image for PNG/BMP + fallback)
  align.*            norm_crop 5-landmark similarity transform (TODO)
  detect.*           SCRFD anchor decode + NMS, host-side (NMS done; decode TODO)
examples/cli/      facedetect-cli (info/embed/detect/verify/analyze/bench)
tests/             parity.hpp harness + ctest targets + python/check_convert.py
scripts/           convert_facedetect_to_gguf.py, gen_baseline.py, requirements.txt,
                   apply_ggml_patches.sh
docs/              conversion.md, parity.md, quantization.md
third_party/       ggml submodule + vendored stb_image.h + ggml-patches/
models/            MANIFEST.md (the GGUFs themselves are git-ignored)
```

## Build

```bash
cmake -B build -DFACEDETECT_BUILD_TESTS=ON -DFACEDETECT_BUILD_CLI=ON -DGGML_NATIVE=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure -LE model
```

### CMake options

`FACEDETECT_BUILD_CLI`, `FACEDETECT_BUILD_TESTS`, `FACEDETECT_SHARED`,
`FACEDETECT_GGML_{CUDA,METAL,VULKAN,HIP}` forward to the matching `GGML_*`.
`GGML_NATIVE` / `GGML_LLAMAFILE` are forced ON unless overridden.

## Running tests

- **Model-independent** (run anywhere, no checkpoint): `ctest -LE model`. These
  are the gate in CI (`test_smoke`, `test_nms`).
- **Model/baseline-dependent**: carry the `model` label and skip with exit code
  77 when the reference baseline is absent (`test_align`, `check_convert`).
  Generate baselines with `scripts/gen_baseline.py`.

## Converting a model

See `docs/conversion.md`. insightface packs are ONNX, so the converter reads ONNX
initializers verbatim and writes a metadata-driven GGUF (`facedetect.*` KV).

## C-API and LocalAI integration

`include/facedetect_capi.h` defines the flat C-API. Build `libfacedetect.so` with
`-DFACEDETECT_SHARED=ON`. Verify exports with
`nm -D build/libfacedetect.so | grep facedetect_capi`.

The LocalAI backend lives in the LocalAI repo and dlopens `libfacedetect.so`.
These symbols are FROZEN: do not remove or change any signature without a
coordinated bump on the LocalAI side AND an `facedetect_capi_abi_version` bump.

```
facedetect_capi_abi_version
facedetect_capi_load
facedetect_capi_free
facedetect_capi_last_error
facedetect_capi_free_string
facedetect_capi_free_vec
facedetect_capi_embed_path        # primary face -> L2-normalized embedding (file)
facedetect_capi_embed_rgb         # primary face -> embedding (raw RGB bytes)
facedetect_capi_detect_path_json  # all faces: boxes + score + 5 landmarks as JSON
facedetect_capi_verify_paths      # two images -> cosine distance + verdict (+ anti-spoof)
facedetect_capi_analyze_path_json # all faces: age + gender as JSON
```

### ABI version bump rule

`facedetect_capi_abi_version()` returns an integer LocalAI checks for
compatibility. **Bump it on any breaking change** to the signatures or semantics
of the frozen symbols above, and add a dated entry to the changelog comment block
at the top of `facedetect_capi.h`. **Additive changes (brand-new functions) do
NOT require a bump** - LocalAI feature-detects them by `dlsym`. Keep the changelog
in `facedetect_capi.h`, the `#define FACEDETECT_CAPI_ABI_VERSION` in
`src/facedetect_capi.cpp`, and this list in sync.

Ownership contract: returned `char*` (JSON) free with `facedetect_capi_free_string`;
returned `float*` (embeddings) free with `facedetect_capi_free_vec`; the
`facedetect_ctx` last-error string is owned by the context. No C++ exception ever
crosses the boundary.

## Dumping reference baselines

`scripts/gen_baseline.py` runs the insightface reference pipeline and dumps the
detector outputs / aligned crop / embedding into a baseline GGUF for the parity
tests (`docs/parity.md`). Requires the venv (`scripts/requirements.txt`).

## ggml submodule

`third_party/ggml` is pinned. Any local ggml patches go in
`third_party/ggml-patches/*.patch` and are applied at configure time by
`scripts/apply_ggml_patches.sh` (idempotent).

## Common maintenance tasks

- **Add a new pack**: add a branch in `resolve_pack()` + the tensor map in the
  converter, document it in `docs/conversion.md` + `README.md`, add a row to
  `models/MANIFEST.md`, and a parity baseline.
- **Update ggml**: bump the submodule, re-check the patches still apply.
