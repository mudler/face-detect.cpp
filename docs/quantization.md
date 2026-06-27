# Quantization policy

`scripts/convert_facedetect_to_gguf.py` takes `--dtype f32|f16|q8_0|q4_0|q4_k`.
The dtype is applied **selectively**: only the large weights the C++ engine hands
to ggml's matmul/conv kernels are quantized, and only when the innermost ggml dim
is block-aligned (32 for q8_0/q4_0, 256 for q4_k); ggml dequantizes them on the
fly inside the compute graph. Everything the host code reads as raw F32 stays F32.

For the buffalo packs the only weight that qualifies is the ArcFace recognizer FC
(`rec.fc.weight`, ggml `ne=[25088,512]`); the SCRFD detector and every conv kernel
keep their tiny innermost `ne` (1 or 3) and stay F32. Keeping the detector F32 is
deliberate: Task 4.5 showed detector precision drives the landmark and hence the
embedding parity, so it is never quantized.

> `q4_k` is accepted by the allowlist for forward compatibility, but the bundled
> pure-python `gguf` does not implement a Q4_K packer; the converter exits cleanly
> (`FACEDETECT_QUANT_UNSUPPORTED`) rather than emitting a partial file.

## What gets quantized

- Only the ArcFace recognizer FC weight `rec.fc.weight` (`ne=[25088,512]`). It is
  the one weight that is 2D, runs directly through `ggml_mul_mat`, and has a
  block-aligned innermost ggml dim (`>= 32` and divisible by 32), so q8_0/q4_0
  on-the-fly dequant applies. It dominates the file size.

Nothing else is quantized:

- ALL convolution weights stay F32 - the SCRFD detector backbone + FPN heads
  (`det.`) and the ArcFace ResNet50 trunk convs (`rec.`). Their innermost ggml
  `ne` is 1 or 3, which is not block-aligned, so `should_quantize()` returns F32.
- The SCRFD detector stays F32 on purpose: Task 4.5 showed detector precision
  drives the landmark and hence the embedding parity.
- The genderage (`ga.*`) and MiniFASNet anti-spoof (`as*.`) heads are explicitly
  `skip_q=True` and never quantized.

## What stays F32

- Batch-norm running stats (mean/var) and all biases.
- The small genderage / MiniFASNet heads.
- The 5-point reference template (`arcface_dst`) and any host-side constants.

The allowlist lives in `should_quantize()` in the converter.

## Accuracy

The parity gates in `docs/parity.md` (cosine `>= 0.9999`, max-abs-diff `<= 1e-3`
on the embedding; identical verify verdicts) are evaluated per dtype and validated
before a variant is published. Measured on buffalo_l (decode-isolated embedding
gate vs the insightface golden, CPU; verify verdicts = face_a/face_b same +
face_a/face_c different at threshold 0.35):

| dtype | size | embedding cosine | embedding max\|d\| | embedding gate | verdicts |
|---|---:|---:|---:|---|---|
| f32  | 187.0 MiB | 1.000000 | 9.7e-08 | PASS | 2/2 |
| q8_0 | 151.0 MiB | 0.999997 | 3.8e-04 | PASS | 2/2 |
| q4_0 | 144.9 MiB | 0.999628 | 3.6e-03 | FAIL | 2/2 |

Degradation is monotonic (f32 < q8_0 < q4_0 in error). **q8_0** is near-lossless
and is the recommended published quant. **q4_0** keeps the verify verdicts
(distance margin to the 0.35 threshold stays wide) but exceeds the strict 1e-3
embedding bound, so it is published as a verdict-only variant, not advertised as
embedding-parity. The genderage and MiniFASNet anti-spoof heads are stored F32
(`skip_q=True`) and are unaffected by `--dtype`.
