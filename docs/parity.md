# Parity vs insightface

face-detect.cpp is validated stage-by-stage against the reference insightface
pipeline. `scripts/gen_baseline.py` dumps the reference intermediates to a
baseline GGUF; the ctest suite (`tests/test_*.cpp`, via `tests/parity.hpp`) diffs
each C++ stage against them. Tests skip with exit code 77 when a baseline asset
is absent, so CI without the reference venv stays green.

## Gates

**Embedding (the primary gate).** For the same input image, the C++ ArcFace
embedding must match the reference:

- cosine similarity `>= 0.9999`, and
- max-abs-diff `<= 1e-3` element-wise,

both computed against the L2-normalized 512-d reference embedding
(`embedding` in the baseline). `tests/parity.hpp::cosine` / `::compare`.

**Verify verdicts.** For a labelled set of image pairs (same-identity and
different-identity), the C++ verdict (`distance <= threshold`) must be identical
to insightface's at the buffalo default threshold (cosine distance, 0.35). A
single flipped verdict fails the gate.

**Detection (boxes + landmarks).** A small golden detection set (a handful of
images with one or more faces) pins the SCRFD output. For each image the decoded
boxes must match within `<= 1 px` (max corner error) after the same NMS, and the
5 landmarks per face within `<= 1 px`, for EVERY face (not just the primary).
This catches anchor-decode / NMS / stride mistakes independently of the
recognizer. `tests/test_detect.cpp`; drive all three packs over face_a/b/c + multi
with `scripts/run_detector_parity.sh`.

The three buffalo packs use three SCRFD detector variants. buffalo_l's det_10g
runs a hand-mapped ggml graph (`src/scrfd_graph.cpp`). buffalo_m's det_2.5g and
buffalo_s's det_500m have different backbones (det_500m is a lighter,
depthwise-separable stem with no pooling) and different ONNX node numbering, so
hand-mapping them would not scale; instead the converter embeds each detector's
ONNX topology as the `facedetect.detector.graph` KV and the C++ replays it through
the shared metadata-driven interpreter (`run_onnx_graph_multi`, the same
interpreter that runs the MiniFASNet / MobileFaceNet graphs, extended with
MaxPool / AveragePool / Resize / Transpose / Reshape). The interpreter emits the 9
per-stride heads (scores / bbox / kps for strides 8/16/32) already flattened to
the host decode's `(H*W*num_anchors, C)` layout, feeding the SAME parity-proven
anchor decode + NMS. The per-stride raw heads are gated against the ONNX reference
at `max|d| ~1e-5` (`tests/test_scrfd_graph.cpp`), and end-to-end detection at
`<= 1 px` for both packs (buffalo_m/buffalo_s).

> **Detection boundary note (det_500m on crowded scenes).** The raw heads are
> bit-faithful to the ONNX reference even on the multi-face fixture
> (`max|d| ~4e-7` on scores). But the PRODUCTION letterbox resize differs from
> cv2 by up to 1 LSB on a minority of pixels (the same `cv_resize_linear_u8`
> caveat the graph gate sidesteps by feeding reference pixels). On the weakest
> detector (det_500m / buffalo_s) the crowded `multi` fixture has one face at
> golden score `0.50303` - 0.003 above the 0.5 detection threshold - and that
> sub-LSB letterbox drift flips it below threshold, yielding 10 of 11 faces. Every
> face comfortably above threshold still matches within 1 px; `test_detect`
> tolerates the single threshold-boundary face but fails on any dropped
> supra-threshold or spurious face. Closing it would need a cv2-bit-exact
> letterbox resize (the detector analogue of the Task 4.4 decoder reconciliation),
> out of scope for the detector topology port. det_2.5g (buffalo_m) and det_10g
> (buffalo_l) detect all 11 multi faces exactly.

**YuNet detector (Apache-2.0).** The OpenCV-Zoo YuNet detector
(`tests/test_yunet.cpp`, `scripts/run_yunet_parity.sh`) runs its WHOLE forward
through the shared `run_onnx_graph_multi` interpreter - no new ggml op - and is
gated three ways vs `cv2.FaceDetectorYN`: (1) the 12 raw per-stride heads
(cls/obj/bbox/kps x 8/16/32) at `max|d| ~1e-6`; (2) the anchor-free decode
(`score=sqrt(cls*obj)`, `cx=(c+dx)*s`, `w=exp(dw)*s`, kps `(k+c)*s`) + OpenCV NMS
(standard IoU, no Pascal-VOC `+1`) **bit-exact** at `<= 1e-4 px` on the
reference's exact 640 pixels; and (3) the full production path (libjpeg decode +
plain resize to the static 640 net + scale-back) within 2 px. YuNet plain-RESIZES
(no letterbox - it distorts to the square net, like `FaceDetectorYN`), so on a
small face in a heavily downscaled crowd the `cv_resize_linear_u8` <=1 LSB drift
maps to ~1.5 net px; the strict <=1px proof is gate (2).

**Decode (cv2.imread parity).** insightface loads images with `cv2.imread`
(libjpeg-turbo, `JDCT_ISLOW` IDCT + fancy chroma upsampling). `fd::load_image_rgb`
decodes JPEG through the same vendored libjpeg-turbo with those settings, so the
decoded RGB matches the golden `src_image` (cv2 pixels) **bit-exact** (`max|d| ==
0`). `tests/test_decode.cpp`. This was added in Task 4.4: the previous stb_image
decode drifted 1-3 LSB/pixel, which amplified to ~1.8e-3 max-abs through the
50-layer ArcFace trunk and broke the strict 1e-3 embedding bound. PNG/BMP still
decode via the vendored stb_image (and stb is the fallback if a libjpeg decode
fails).

> **Production residual (known concern).** With the decoder reconciled, the
> strict 1e-3 embedding bound is met end-to-end from the REAL libjpeg decode when
> fed the reference landmarks (`tests/test_embedding.cpp` path C, gated on ALL
> THREE single-face fixtures: cosine 1.000000, max-abs <= 4e-5). The full
> production path (this port's own SCRFD landmarks) still measures ~1.5-2.6e-3
> max-abs: the residual is sub-pixel landmark drift (this port's detected
> landmarks differ from insightface by ~0.02-0.1 px, within the <=1 px detection
> gate), perturbing the umeyama crop. Production cosine: face_a 0.999940 (passes
> the >= 0.9999 gate), face_b 0.999892, face_c 0.999840.
>
> Task 4.5 ruled OUT detector weight dtype as the cause: the SCRFD detector
> tensors (`det.*`) are already stored F32 in BOTH the f16 and f32 model GGUFs
> (the converter keeps conv kernels F32 because their innermost ggml `ne` is 1 or
> 3, below the 32-wide quantization floor; only one ArcFace recognizer FC weight
> is F16 in the f16 build). An all-FP32 model therefore leaves the detected
> landmarks - and the production cosine - bit-identical to the f16 build on all
> three fixtures; it only tightens the recognizer-graph max-abs (path C drops from
> ~1e-5 to ~1e-7). The localized root cause is ggml-vs-onnxruntime conv
> accumulation in the SCRFD trunk / regression head, amplified through the crop;
> closing it needs exact landmark parity, outside the decoder reconciliation scope.

**Aligned crop.** The 112x112 `norm_crop` output is compared against the
reference aligned crop (`aligned_crop`) with a max-abs-diff `<= 1` on 8-bit pixel
values (bilinear sampling differences are sub-LSB). Alignment must reproduce
insightface `norm_crop` / `estimate_norm` (umeyama similarity fit) exactly,
because the embedding gate is downstream of it.

## Running

### Single fixture (ad-hoc)

Dump one baseline and point the suite at it. Any path works; the embedding
all-three gate (below) degrades gracefully to a single fixture when the path does
not match the `face_<tag>` / `baseline_<tag>` layout:

```bash
.venv/bin/python scripts/gen_baseline.py --model buffalo_l \
    --image tests/fixtures/face_a.jpg --output /tmp/fd_baseline.gguf
FACEDETECT_TEST_BASELINE=/tmp/fd_baseline.gguf \
    ctest --test-dir build --output-on-failure
```

### All-three strict embedding gate (face_a/b/c)

`tests/test_embedding.cpp` path C runs the strict real-libjpeg-decode +
golden-landmark gate (`cosine >= 0.9999` AND `max|d| <= 1e-3`) on **all three**
single-face fixtures from a single invocation. It discovers the siblings by
substituting the tag character in the primary paths (`siblings_of`), so the
baselines and fixtures MUST follow this exact layout:

| fixture | image                       | baseline             |
|---------|-----------------------------|----------------------|
| face_a  | `tests/fixtures/face_a.jpg` | `out/baseline_a.gguf` |
| face_b  | `tests/fixtures/face_b.jpg` | `out/baseline_b.gguf` |
| face_c  | `tests/fixtures/face_c.jpg` | `out/baseline_c.gguf` |

Generate the three per-fixture baselines (one `gen_baseline.py` call each):

```bash
for tag in a b c; do
    .venv/bin/python scripts/gen_baseline.py --model buffalo_l \
        --image "tests/fixtures/face_${tag}.jpg" \
        --output "out/baseline_${tag}.gguf"
done
```

Then drive the gate. `FACEDETECT_TEST_IMAGE` / `FACEDETECT_TEST_BASELINE` point at
the `face_a` / `baseline_a` primary; the test derives face_b/c and baseline_b/c
from them and gates all three (it FAILS, not skips, if a sibling is missing):

```bash
FACEDETECT_TEST_GGUF=models/buffalo_l-f16.gguf \
FACEDETECT_TEST_IMAGE=tests/fixtures/face_a.jpg \
FACEDETECT_TEST_BASELINE=out/baseline_a.gguf \
    ctest --test-dir build -R test_embedding --output-on-failure
```

`scripts/run_embedding_parity.sh` wraps the whole flow (venv activation, model
conversion if needed, the three `gen_baseline.py` dumps, and the all-three gate),
exiting non-zero on failure.

## Dense-landmark heads (`test_landmarks`, engine-only)

`gen_baseline.py` dumps, per head (`landmark_{2d,3d}_{crop,raw,M,points}`), the
exact 192x192 face crop, the raw `fc1` output, the forward crop affine `M`, and the
final image-space points the insightface `app.get` produced. `test_landmarks` feeds
the reference crop into the C++ head (ISOLATED, like the genderage gate), gates the
raw `fc1` (`<= 1.5e-2`), then decodes + maps to image space with the reference's OWN
`M` and gates the points within `1px`. f32 is bit-exact (2d/3d raw `~2e-7`, points
`~5e-5 px`); f16 stays well under (points `~7e-3 px`). These heads are engine-level
only - no LocalAI RPC consumes them.

```bash
FACEDETECT_LANDMARKS_GGUF=models/landmarks-2d106-1k3d68-f16.gguf \
FACEDETECT_TEST_BASELINE=out/baseline_a.gguf \
    ctest --test-dir build -R test_landmarks --output-on-failure
```
