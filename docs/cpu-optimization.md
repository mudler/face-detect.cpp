# CPU optimization and the path to MLAS parity

Durable record of the face-detect.cpp CPU optimization campaign: the final standing
of each face model against onnxruntime / MLAS, and a cross-reference to the shared
parity-path narrative documented in full on the sibling voice-detect.cpp engine.

Discipline throughout: `optimizing-ggml-inference-speed` plus the depth-anything.cpp
recipe. Parity-first - profile host-side work before tuning kernels, change one thing
at a time, re-gate parity after every change (SCRFD boxes/landmarks <= 1 px vs ONNX;
ArcFace stem/block + golden-landmark + isolated embed cosine = 1.000000), keep only
end-to-end wins, record dead-ends. Numbers below were measured on an AMD Ryzen 9
9950X3D 16-core (Zen5, 2 CCDs) on `tests/fixtures/face_a.jpg`; absolute ms are
machine-specific, the ratios are the signal.

## Final CPU standing

Ratio = reference_ms / ggml_ms at 8 threads (matched thread budget on both sides).
`> 1` means ggml wins. Reference: CPU onnxruntime / insightface (MLAS conv).

| Model | dominant op | ggml conv path | ratio @ 8t (ref/ggml) | verdict |
| ----- | ----------- | -------------- | --------------------- | ------- |
| SCRFD detect (640x640) | 3x3 stride-1 conv (large spatial) | custom AVX2 Winograd F(2x2,3x3) | **~0.69-0.92** | closest on Winograd |
| ArcFace recognizer (112x112) | 3x3 stride-1 conv (deep channel) | blocked direct-conv, shape-gated | **~0.60 (Winograd) / +25-27% with directconv @ 8t** | directconv-gated body |
| SFace recognizer (112x112) | conv + matmul | im2col + sgemm | behind | behind |
| genderage (analyze) | conv | im2col + sgemm | behind | behind |

Read this honestly: **no face model reaches MLAS parity on CPU.** SCRFD detect on the
custom AVX2 Winograd kernel is the closest (~0.69-0.92 depending on input shape and
thermal load). The ArcFace recognizer body gains a further +25-27% at 8 threads when
the blocked direct-conv microkernel is routed to it (shape-gated, see below) - the
recognizer is the closest face model to parity once that kernel is in play. The
campaign narrowed every gap substantially but did not beat MLAS on a convolutional
face net on CPU.

## Techniques applied (face-specific)

- **Default thread count 1 -> min(hardware_concurrency, 8).** The process-global
  backend defaulted to 1 thread, below ggml's own default; every consumer that did
  not pass `--threads` paid a ~3.3x latency tax. Capped at 8 (the bandwidth-bound
  detector plateaus there). Out-of-the-box absolute win, parity unchanged
  (thread-count does not change ggml f32 reduction order).
- **conv_2d_direct routing, thread-gated.** Route K>1 convs through
  `ggml_conv_2d_direct` on CPU at `n_threads >= 8`, else im2col; 1x1 convs always stay
  im2col (pure GEMM); GPU always im2col. The two kernels cross over by memory-bandwidth
  pressure, which is set by thread count (im2col's 9x patch buffer is a large stream;
  direct has far less traffic but a less throughput-dense kernel).
- **Custom AVX2 Winograd F(2x2,3x3).** Beat tinyBLAS sgemm on the large 3x3 stride-1
  pad-1 convs that dominate the SCRFD backbone/neck/head stems (min feature-map dim
  >= 80) plus the ArcFace 112x112 stem. Transformed filter cached once per shape;
  tiles split across ggml threads. This is the default detector conv kernel.
- **ArcFace BatchNorm fold.** Fold inference-mode BN affine to host constants once per
  loaded model, collapsing a ~5-op live chain repeated ~25x per embed to one mul + add
  per BN (algebraic, parity preserved). PReLU could NOT be fused without a CPU-only
  custom op (this ggml exposes neither per-channel PReLU nor a binary max), so it was
  left numerically identical.
- **MLAS-class blocked direct-conv microkernel, shape-gated to the ArcFace body.** The
  nChw16c blocked, AVX-512 register-tiled direct-conv microkernel (shared with
  voice-detect.cpp, ~263 GFLOP/s, within 6-7% of MLAS per-op efficiency) is routed to
  the ArcFace recognizer body only (its deep-channel 3x3 stride-1 convs are exactly the
  shape the kernel targets). **SCRFD stays on Winograd by design**: the detector's
  large-spatial feature maps make the Winograd FLOP-reduction the right win, not blocked
  direct conv.

## Dead-ends (do not re-chase)

- **`-DGGML_NATIVE=ON` (global AVX-512 on Zen5).** Net negative for the pipeline: helps
  the small recognizer but regresses the dominant bandwidth-bound SCRFD detector
  (classic AVX-512 down-clock on a memory-bound im2col conv). Build stays AVX2/FMA +
  tinyBLAS; the AVX-512 win has to be a targeted per-op microkernel, not a global flag.
- **Blanket `ggml_conv_2d_direct` for all thread counts.** Regresses 1-thread detect
  +29% and 2-thread +17%; the direct win only appears once threads saturate bandwidth
  (>= 8 here). Hence the thread gate, not a blanket switch.

## Path to true MLAS parity (cross-reference)

The full, durable parity-path plan (Phases A/B/C, economics, decision gate, per-model
applicability) is documented once on the sibling engine:
[`voice-detect.cpp/docs/cpu-optimization.md`](../../voice-detect.cpp/docs/cpu-optimization.md).
Both engines share the same blocked direct-conv microkernel and the same plan.
Summary of the three phases:

- **Phase A - whole-backbone blocked layout.** The current blocked direct-conv kernel
  is the per-conv-reorder form (re-pays a ~15% nChw16c reorder tax per conv). True
  parity carries nChw16c across the whole backbone (reorder once after the stem, run
  all stages blocked, reorder once before pooling), needing ~4-5 new blocked-aware
  custom ops (bias-add, ReLU, residual-add, strided-1x1 downsample) plus a parallel
  blocked forward path.
- **Phase B - per-ISA microkernel family.** AVX-512 (tuned) + AVX2 + NEON, each
  separately hand-tuned (tile width / OC-block / prefetch).
- **Phase C - perpetual parity maintenance.** The direct-conv FMA order is
  within-fp32-tolerance, not bit-identical; every variant and retune re-passes
  cosine >= 0.9999.

Economics: ~600-1000 lines of intrinsics, roughly doubling the hand-SIMD surface, for
a realistic ceiling of ~1.1-1.3x MLAS. **Decision gate: pursue only if CPU latency is
a measured product bottleneck; do NOT build it for the GPU/CUDA path.**

Per-model applicability on the face side: directconv is **shape-gated to the ArcFace
recognizer body** (deep-channel 3x3 stride-1); **SCRFD stays on Winograd** (large
spatial, FLOP-reduction wins); SFace / genderage gain little from a blocked 3x3 kernel.

Per-pass measured tables and the GPU comparisons live in `benchmarks/RESULTS.md` and
`docs/benchmarks.md`.
