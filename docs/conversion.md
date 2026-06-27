# Conversion: insightface pack -> GGUF

`scripts/convert_facedetect_to_gguf.py` converts an insightface model pack into a
single metadata-driven GGUF that the C++/ggml engine loads. insightface ships its
buffalo packs as **ONNX** graphs, so the converter reads ONNX *initializers*
(weights) rather than a PyTorch `state_dict`.

## Principles

- **Metadata-driven.** All config lives in GGUF KV under the `facedetect.*`
  namespace (see the schema below). The C++ side reads it via `ModelLoader` and
  never hardcodes geometry.
- **Verbatim tensor names.** Tensor names are kept exactly as the source ONNX
  initializers name them, so the C++ graph is a 1:1 mapping with no rename table.
- **Selective quantization.** Only the large conv/linear weights consumed via
  `ggml_mul_mat` / `ggml_conv` are quantized (`--dtype f16|q8_0`); everything the
  host reads as raw F32 (BN running stats, biases, small heads, the 5-point
  reference template) stays F32. See `docs/quantization.md` and `should_quantize`.

## KV schema (`facedetect.*`)

| Key | Type | Meaning |
|-----|------|---------|
| `general.architecture` | str | always `facedetect` |
| `facedetect.arch` | str | `scrfd+arcface` / `yunet+sface` |
| `facedetect.detector.kind` | str | `scrfd` / `yunet` |
| `facedetect.detector.input_size` | u32 | square detector input (letterboxed) |
| `facedetect.detector.strides` | i32[] | FPN strides, e.g. `[8,16,32]` |
| `facedetect.detector.num_anchors` | u32 | anchors per location (SCRFD: 2) |
| `facedetect.detector.score_thresh` | f32 | detection score threshold |
| `facedetect.detector.nms_thresh` | f32 | NMS IoU threshold |
| `facedetect.recognizer.kind` | str | `arcface` (IResNet) / `arcface_mbf` (MobileFaceNet) / `sface` |
| `facedetect.recognizer.input_size` | u32 | aligned crop size (ArcFace: 112) |
| `facedetect.recognizer.embed_dim` | u32 | embedding dim (ArcFace 512, SFace 128) |
| `facedetect.recognizer.verify_threshold` | f32 | default cosine-distance match threshold |
| `facedetect.recognizer.graph` | str[] | MobileFaceNet (`w600k_mbf`) / SFace ONNX node topology, replayed by the shared interpreter; absent for IResNet50 packs |
| `facedetect.recognizer.input` | str | recognizer graph input tensor name (e.g. `input.1`) |
| `facedetect.recognizer.output` | str | recognizer graph output tensor name |
| `facedetect.genderage.present` | bool | genderage head bundled |
| `facedetect.antispoof.present` | bool | MiniFASNet ensemble bundled |
| `facedetect.antispoof.input_size` | u32 | MiniFASNet input (80) |
| `facedetect.antispoof.scales` | f32[] | per-model crop scales, e.g. `[2.7, 4.0]` |

## Tensor namespacing

insightface's buffalo packs reuse auto-numbered ONNX initializer names (e.g.
`685`, `689`, `691`, `695`, `697`, `701`) across the SCRFD and ArcFace graphs for
*different* real weights, and the MiniFASNet anti-spoof ensemble repeats every
name across its two members. Because the C++ loader keys every tensor by name in
a single ggml context, each sub-model's initializers are written under a stable
namespace **prefix**, with the ONNX name otherwise kept verbatim:

| sub-model | source file | GGUF prefix |
|---|---|---|
| SCRFD detector | `det_*.onnx` | `det.` |
| ArcFace recognizer | `w600k_*.onnx` | `rec.` |
| GenderAge head | `genderage.onnx` | `ga.` |
| MiniFASNet anti-spoof (i-th) | `MiniFASNet*.onnx` | `as<i>.` |

The prefix is a single deterministic tag per sub-model, not a per-tensor rename
table, so the C++ graph mapping stays 1:1. The full per-tensor contract (every
name, shape, ggml `ne`, and quant decision) lives in
[`scripts/tensor_manifest_buffalo_l.md`](../scripts/tensor_manifest_buffalo_l.md),
produced by `scripts/trace_onnx_manifest.py`.

## buffalo_l (implemented): SCRFD + ArcFace ResNet50, 512-d

Primary target, **non-commercial** license (see README). `resolve_pack` locates
the pack under `~/.insightface/models/buffalo_l/` (or a directory passed via
`--model`) and reads `det_10g.onnx`, `w600k_r50.onnx`, and `genderage.onnx`.
buffalo_l ships **no** MiniFASNet ONNX, so `facedetect.antispoof.*` is omitted;
the `1k3d68.onnx` / `2d106det.onnx` landmark models are not consumed (SCRFD emits
its own 5-point landmarks for `norm_crop`).

KV written for buffalo_l:

| Key | Value |
|-----|-------|
| `facedetect.arch` | `scrfd+arcface` |
| `facedetect.detector.kind` | `scrfd` |
| `facedetect.detector.input_size` | `640` |
| `facedetect.detector.strides` | `[8, 16, 32]` |
| `facedetect.detector.num_anchors` | `2` |
| `facedetect.detector.score_thresh` | `0.5` |
| `facedetect.detector.nms_thresh` | `0.4` |
| `facedetect.recognizer.kind` | `arcface` |
| `facedetect.recognizer.input_size` | `112` |
| `facedetect.recognizer.embed_dim` | `512` |
| `facedetect.recognizer.verify_threshold` | `0.35` |
| `facedetect.genderage.present` | `true` |
| `facedetect.genderage.input_size` | `96` |

Tensor counts: 125 (det) + 237 (rec) + 161 (genderage) = **523** initializers.
Under `should_quantize`, a weight is quantized only when both of its two innermost
ggml `ne` dims are >= 32, so 3x3 / 1x1 conv kernels (innermost `ne` = 3 or 1)
stay F32 and only the large 2-D ArcFace embedding `fc` Gemm is stored F16/Q8_0.
Measured GGUF footprints: f32 ~184 MB, f16 ~160 MB, q8_0 ~148 MB.

## buffalo_m / buffalo_s: smaller SCRFD + ArcFace packs

Same `scrfd+arcface` arch family, smaller backbones. `resolve_pack` matches them by
the same `det_*.onnx` / `w600k_*.onnx` globs; per-tensor contracts are in
[`tensor_manifest_buffalo_m.md`](../scripts/tensor_manifest_buffalo_m.md) and
[`tensor_manifest_buffalo_s.md`](../scripts/tensor_manifest_buffalo_s.md).

| pack | detector | recognizer | recognizer status |
|---|---|---|---|
| buffalo_m | `det_2.5g` | `w600k_r50` (**byte-identical** to buffalo_l) | reuses the hand-mapped IResNet50 graph |
| buffalo_s | `det_500m` | `w600k_mbf` (MobileFaceNet) | new metadata-driven graph (see below) |

**Recognizer parity (the embedding gate): GREEN for both** at cosine = 1.000000,
max\|d\| ~1e-5 on all three single-face fixtures (`scripts/run_embedding_parity.sh`,
`FACEDETECT_PACK=buffalo_m` / `buffalo_s`). buffalo_m's recognizer is the same
`w600k_r50` IResNet50 as buffalo_l, so it runs the existing hand-mapped graph
unchanged. buffalo_s ships **MobileFaceNet** (`w600k_mbf`): a different backbone
(depthwise-separable + one genuinely grouped conv, a `Gemm` head) that does **not**
share the IResNet50 node names. Rather than hand-map a second trunk, the converter
embeds its ONNX forward topology (`facedetect.recognizer.graph`) and the C++ runs it
through the **shared metadata-driven graph interpreter** (the same path as the
MiniFASNet anti-spoof ensemble), then L2-normalizes the raw 512-d output. The
interpreter gained `Gemm` (transB) and general grouped-conv (channel split + concat)
support for this.

**Detector status (det_2.5g / det_500m): not yet ported.** The SCRFD detector graph
in `src/scrfd_graph.cpp` is hand-mapped to `det_10g`'s exact topology and
auto-numbered node names; `det_2.5g` is a different-depth ResNet variant and
`det_500m` is a different backbone entirely (no pooling). Each needs a new detector
graph or a detector-side generalization of the interpreter (which would add
`Resize` / `MaxPool` / `AveragePool` plus the dynamic head post-process). Until then
the embedding gate runs with `FACEDETECT_TEST_NO_DETECTOR=1`, which strictly gates
the recognizer on golden landmarks/crops and skips the detector-dependent production
path; detection parity for these two packs is deferred.

## buffalo_sc / antelopev2: the remaining insightface packs

These two complete parity with the Python insightface backend's pack list. Both
ride the existing metadata-driven paths with **no new interpreter ops**.

| pack | detector | recognizer | how it is routed |
|---|---|---|---|
| buffalo_sc | `det_500m` | `w600k_mbf` (MobileFaceNet) | identical sub-models to buffalo_s; rides the same SCRFD interpreter + MobileFaceNet recognizer graph (no genderage / landmark heads) |
| antelopev2 | `scrfd_10g_bnkps` | `glintr100` (ArcFace **glint360k R100**, IResNet100) | SCRFD-10G via the interpreter; R100 via `recognizer.graph` |

**buffalo_sc** is the "small compact" detection+recognition-only pack. Its two ONNX
files are exactly buffalo_s's (`det_500m` + `w600k_mbf`), so `resolve_pack` simply
adds it to the buffalo name gate and everything else is reused.

**antelopev2** is the higher-accuracy pack. Two routing notes:

- Its detector `scrfd_10g_bnkps.onnx` has a **byte-identical topology and
  graph-output names** to buffalo_l's `det_10g` (same SCRFD family the
  `det_2.5g`/`det_500m` interpreter already replays: FPN 2x-nearest `Resize` + GFL
  per-stride heads). Because it is **not** named `det_10g`, it takes the
  metadata-driven `facedetect.detector.graph` path rather than the hand-mapped C++
  graph, and reproduces it bit-faithfully (detection <= 0.73 px boxes / 0.10 px
  landmarks vs the reference, all fixtures).
- Its recognizer `glintr100` is a **100-layer IResNet** (`Conv` / `PRelu` /
  `BatchNormalization` / `Add` / `Flatten` / `Gemm` - every op already covered by
  the interpreter), a deeper trunk than the hand-mapped r50 block table. Rather than
  extend that table, the `rec_use_graph` flag forces the converter to embed the R100
  ONNX forward topology as `facedetect.recognizer.graph`; the C++ replays it through
  the shared interpreter (the MobileFaceNet path) and L2-normalizes the raw 512-d
  output. Recognizer parity is GREEN: cosine = 1.000000 / max\|d\| ~1e-5 on all three
  golden-landmark fixtures, end-to-end production cosine 0.999946.

Reproduce both with `FACEDETECT_PACK=buffalo_sc` / `antelopev2` in
`scripts/run_detector_parity.sh` and `scripts/run_embedding_parity.sh`.

## YuNet detector (OpenCV-Zoo, Apache-2.0)

The commercial-friendly alternative to insightface's non-commercial SCRFD. Pass
the YuNet ONNX path (or `--model yunet`, which globs `models/yunet/`):

```
python scripts/convert_facedetect_to_gguf.py \
    --model models/yunet/face_detection_yunet_2023mar.onnx \
    --output models/yunet-f32.gguf --dtype f32
```

`resolve_pack`'s `yunet` branch emits `arch=yunet`, `detector.kind=yunet`,
`detector.engine=onnx_direct`, `det_input_size=640`, `strides=[8,16,32]`,
`num_anchors=1`, and embeds the forward topology as `facedetect.detector.graph`
(the same KV the SCRFD det_2.5g path uses). The C++ engine replays it
metadata-driven through the shared `run_onnx_graph_multi` interpreter - YuNet adds
**no new ggml op** - then runs its anchor-free decode (`score=sqrt(cls*obj)`,
`exp`-bbox, kps) + OpenCV-convention NMS in `src/yunet_graph.cpp`. See
`scripts/tensor_manifest_yunet_sface.md` and `scripts/run_yunet_parity.sh`.

The paired **SFace** recognizer (`face_recognition_sface_2021dec.onnx`, Apache-2.0,
128-d) is wired in Task 8.3: `resolve_pack`'s `yunet` branch resolves a sibling
`models/sface/*sface*.onnx`, sets `recognizer.kind=sface`, `embed_dim=128`,
`verify_threshold=0.363`, and embeds the SFace ONNX topology as
`facedetect.recognizer.graph` (`rec.` prefix). The C++ (`src/sface_graph.cpp`)
replays it through the shared `run_onnx_graph` interpreter - the MobileFaceNet path,
extended with the `Sub` / `Dropout` ops, a spatial (4-D) BatchNorm, and per-node BN
epsilon (`e=`), **no new ggml op**. SFace normalizes `(x-127.5)/128` in-graph and
shares insightface's `arcface_dst` alignment template. Gated vs `cv2.FaceRecognizerSF`
(cosine 1.000000, max-abs 2.6e-6 at f32); see `scripts/run_sface_parity.sh`.

## Dense-landmark heads (`--model landmarks`)

The two insightface dense-landmark regressors buffalo_l ships (`2d106det` 2D
106-point + `1k3d68` 3D 68-point) convert to their OWN GGUF
(`--model landmarks --output landmarks-2d106-1k3d68.gguf`). They are
**engine-level capability only**: no LocalAI proto RPC / API endpoint consumes
dense landmarks yet (the Detect RPC returns only the 5-point SCRFD kps), so they
are NOT a gallery pack - reachable from the C++ engine, `facedetect-cli landmarks`,
and the parity gate only. Exposing them through LocalAI needs a future
dense-landmark RPC + endpoint + capability registry.

Both heads embed their ONNX topology (`facedetect.landmark.{2d,3d}.graph`, `l2d.` /
`l3d.` prefixes) and run through the SAME `run_onnx_graph` interpreter as the
MobileFaceNet recognizer - the only op added is `Identity` (1k3d68's `data -> id`
passthrough), plus `kernel_shape` (`k=`) now emitted so 1k3d68's 3x3 `MaxPool`
replays exactly. Both are fed raw [0,255] R,G,B (`input_mean 0` / `input_std 1`):
2d106det normalizes in-graph (Sub/Mul scalar leaves), 1k3d68 via its leading
`bn_data` BatchNorm. f32 is bit-exact vs onnxruntime; f16 is near-lossless. See
`docs/parity.md` and `models/MANIFEST.md`.

## Other packs (not yet implemented)

- **MiniFASNet anti-spoof ensemble** - V2@scale2.7 + V1SE@scale4.0, 80x80;
  wired through `as<i>.` prefixes when present.

When `resolve_pack` returns `None` the converter prints
`FACEDETECT_MODEL_UNAVAILABLE` and exits 2 so CI skips cleanly.
