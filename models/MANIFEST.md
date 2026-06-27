# face-detect.cpp - Model Publishing Manifest

This file lists the expected set of published GGUF model packs for
face-detect.cpp. Each row is one source pack × one quantization variant.

The GGUFs themselves are **not committed** (`models/` is git-ignored); only this
manifest is tracked. Run `scripts/convert_facedetect_to_gguf.py` to produce the
GGUFs locally.

Validation is against the insightface reference pipeline (see `docs/parity.md`):
the embedding gate is cosine `>= 0.9999` / max-abs-diff `<= 1e-3`, plus identical
verify verdicts and a golden detection set (boxes + landmarks).

---

## Expected published set

### `buffalo_l` - SCRFD + ArcFace ResNet50, 512-d (primary, non-commercial)

Measured against `out/baseline_a.gguf` (insightface golden), CPU, Task 7.1. The
embedding parity column is the decode-isolated gate (golden aligned crop ->
arcface_embed vs the reference 512-d embedding); `verdict` is the same/different
identity pair (face_a vs face_b same, face_a vs face_c different) at the buffalo
0.35 cosine-distance threshold.

| Variant | Size (measured) | Embedding cosine | Embedding max\|d\| | Gate (cos>=0.9999, max\|d\|<=1e-3) | Verdicts | Publish |
|---|---:|---:|---:|---|---|---|
| F32  | 187.0 MiB | 1.000000 | 9.7e-08 | PASS | 2/2 | baseline |
| Q8_0 | 151.0 MiB | 0.999997 | 3.8e-04 | PASS | 2/2 | yes |
| Q4_0 | 144.9 MiB | 0.999628 | 3.6e-03 | FAIL (below 0.9999) | 2/2 | verdict-only |

> Selective quant only stores the large 2-D ArcFace embedding `rec.fc.weight`
> Gemm ([25088,512]) as Q8_0/Q4_0; the SCRFD detector and every conv kernel stay
> F32 (their innermost ggml `ne` is 3 or 1, below the 32-wide block floor), so
> the footprint is dominated by the F32 backbones. Q8_0 is near-lossless and
> publishable; Q4_0 preserves the verify verdict (distance margin to 0.35 stays
> wide) but does NOT meet the strict near-lossless embedding bound, so it is not
> advertised as embedding-parity. Q4_K is in the allowlist but the bundled
> pure-python `gguf` has no Q4_K packer (the converter exits cleanly).

### `buffalo_m` - SCRFD + ArcFace ResNet50, 512-d (non-commercial)

| Variant | Approx size | Embedding parity | Validated |
|---|---:|---|---|
| F16  | ~85 MB | not yet measured | - |
| Q8_0 | ~50 MB | not yet measured | - |

### `buffalo_s` - SCRFD + ArcFace ResNet50, 512-d (non-commercial)

| Variant | Approx size | Embedding parity | Validated |
|---|---:|---|---|
| F16  | ~30 MB | not yet measured | - |
| Q8_0 | ~18 MB | not yet measured | - |

### `buffalo_sc` - SCRFD det_500m + MobileFaceNet, 512-d (non-commercial)

The "small compact" detection+recognition-only pack: the same two sub-models as
buffalo_s (`det_500m` + `w600k_mbf`), no genderage / landmark heads. Detector runs
the metadata-driven SCRFD interpreter; recognizer runs the MobileFaceNet graph.

| Variant | Size (measured) | Embed cosine (golden) | Embed cosine (e2e) | Detection (boxes/lmk) | Publish |
|---|---:|---:|---:|---|---|
| F16 | 12.9 MB | 1.000000 | 0.999936 | <=0.16 px / <=0.28 px | yes |

### `antelopev2` - SCRFD-10G + ArcFace glint360k R100, 512-d (non-commercial)

The higher-accuracy pack: `scrfd_10g_bnkps` (byte-identical topology to buffalo_l's
det_10g, run through the interpreter) + `glintr100`, a 100-layer IResNet routed
through the shared `recognizer.graph` interpreter (`rec_use_graph`). genderage head
present.

| Variant | Size (measured) | Embed cosine (golden) | Embed cosine (e2e) | Detection (boxes/lmk) | Publish |
|---|---:|---:|---:|---|---|
| F16 | 253.2 MB | 1.000000 | 0.999946 | <=0.73 px / <=0.10 px | yes |

> Both gate the strict golden-landmark recognizer bound (cosine 1.000000,
> max\|d\| ~1e-5) and end-to-end detection within 1 px of the insightface reference
> on all fixtures. As with the other packs, f16 only stores the large 2-D ArcFace
> Gemm head as F16; the SCRFD detector and every conv kernel stay F32. Reproduce
> with `FACEDETECT_PACK=buffalo_sc` / `antelopev2` in the parity scripts.

### `yunet+sface` - YuNet + SFace, 128-d (OpenCV-zoo, Apache-2.0)

Measured against `out/sface/baseline_{a,b,c}.gguf` (cv2 `FaceDetectorYN` +
`FaceRecognizerSF` golden), CPU, Task 8.3. The embedding column is the
decode-isolated gate (cv2 `alignCrop` -> `sface_feature` vs the reference raw 128-d
`FaceRecognizerSF.feature`); SFace returns an UN-normalized feature, so the gate
also L2-normalizes both sides for the cosine. `sface_embed` returns the normalized
embedding. The aligned-crop gate (`norm_crop` vs cv2 `alignCrop`, same arcface_dst
template) holds to <= 2 LSB.

| Variant | Size (measured) | Embedding cosine | Embedding max\|d\| | Gate (cos>=0.9999, max\|d\|<=1e-3) | Publish |
|---|---:|---:|---:|---|---|
| F32  | 37.1 MiB | 1.000000 | 2.6e-06 | PASS | baseline |
| F16  | 24.9 MiB | 1.000000 | 1.0e-04 | PASS | yes |

> Only the large 2-D SFace head `rec.pre_fc1_weight` Gemm ([50176,128]) is
> quantizable (innermost ggml `ne` 50176 >= 32); every conv kernel and the YuNet
> detector stay F32 (innermost `ne` 1 or 3), so the F16 footprint is dominated by
> the F32 backbones. SFace normalizes its input `(x-127.5)/128` IN-graph and tags
> each BatchNormalization with its own epsilon (1e-3 conv BNs vs 2e-5 bn1/fc1); the
> C++ runs it through the shared `run_onnx_graph` interpreter (the MobileFaceNet
> path), which Task 8.3 extends with `Sub`/`Dropout` + spatial (4-D) BatchNorm +
> per-node epsilon. Reproduce with `scripts/run_sface_parity.sh`.

---

### `landmarks-2d106-1k3d68` - dense-landmark heads (engine-only, non-commercial)

The two insightface dense-landmark regressors buffalo_l ships alongside the
detector/recognizer: `2d106det` (2D 106-point, MobileNet) + `1k3d68` (3D 68-point,
IResNet). **ENGINE-LEVEL capability only** - no LocalAI proto RPC / API endpoint
consumes dense landmarks yet (the Detect RPC returns only the 5-point SCRFD kps),
so these ship as their OWN GGUF, NOT a gallery pack. Both heads are 192x192,
fed raw [0,255] R,G,B (2d106det normalizes in-graph via Sub/Mul; 1k3d68 via its
leading `bn_data`), decoded per insightface `Landmark.get` (crop = box×1.5
scale-about-center; `(pred+1)×96`; image-space via the inverse crop affine; 1k3d68
takes the last 68 of 1103 output rows). Built `--model landmarks --dtype f16`,
gated by `tests/test_landmarks.cpp` (isolated: golden 192 crop -> head -> raw `fc1`
+ decoded points via the reference's own affine).

| Variant | Size (measured) | 2d raw max\|d\| | 2d points max\|d\| (px) | 3d raw max\|d\| | 3d points max\|d\| (px) | Gate (raw<=1.5e-2, points<=1px) | Publish |
|---|---:|---:|---:|---:|---:|---|---|
| F32  | 141.8 MiB | 1.8e-07 | 3.1e-05 | 3.0e-07 | 5.3e-05 | PASS | baseline |
| F16  | 127.0 MiB | 3.1e-05 | 5.1e-03 | 4.3e-05 | 7.1e-03 | PASS | yes |

> Published f16 as `landmarks-2d106-1k3d68.gguf`
> (https://huggingface.co/mudler/face-detect-gguf/resolve/main/landmarks-2d106-1k3d68.gguf,
> sha256 b0595a5cb20e1e77ede97a7b8fe2b391640033e99a1bb9c4e8583528da59b3ce). Only the
> two landmark-regression Gemm heads (`fc1_weight`) are quantizable; every conv / BN /
> bias stays F32, so the f16 footprint is dominated by the F32 backbones. **Exposing
> these through LocalAI needs a future dense-landmark RPC + endpoint + capability
> registry** - this is engine + CLI (`facedetect-cli landmarks`) + parity only.

---

## Notes

- The buffalo packs are released under a **non-commercial** license; YuNet +
  SFace are **Apache-2.0** and the commercial-friendly alternative.
- The genderage head and the MiniFASNet anti-spoof ensemble (V2@2.7 + V1SE@4.0,
  80x80) are bundled into the buffalo packs when present (`facedetect.genderage.*`
  / `facedetect.antispoof.*` KV).
- Sizes are estimates from the source ONNX footprints; exact sizes will be
  populated after conversion is implemented.
