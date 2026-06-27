# Benchmarks

This documents the comparative latency harness for face-detect.cpp: the ggml
engine vs the native Python (insightface / OpenCV / onnxruntime) reference each
model came from. The measured CPU results live in
[../benchmarks/RESULTS.md](../benchmarks/RESULTS.md); GPU rows are appended there
by the CUDA-host script.

## What is measured

End-to-end per-image latency (ms/image, mean over N timed passes, one warmup pass
excluded) for the face pipelines:

| Pipeline | ggml side | reference side |
| -------- | --------- | -------------- |
| buffalo_l pipeline (detect+align+embed) | `facedetect-cli bench --mode pipeline` | insightface `FaceAnalysis(name=buffalo_l).get` |
| ArcFace recognizer (embed only) | `bench --mode recognizer` | onnxruntime on `w600k_r50.onnx` (aligned crop) |
| YuNet+SFace pipeline (detect+align+embed) | `bench --mode pipeline` | cv2 `FaceDetectorYN` + `FaceRecognizerSF` |
| SFace recognizer (embed only) | `bench --mode recognizer` | onnxruntime on the SFace ONNX (aligned crop) |
| buffalo_l genderage (detect+analyze) | `bench --mode analyze` | insightface `FaceAnalysis` det+genderage |
| buffalo_l detect (SCRFD only) | `bench --mode detect` | insightface SCRFD detector `.detect` |

The ggml side calls the real C++ engine: image decode + SCRFD/YuNet detect +
5-landmark `norm_crop` align + ArcFace/SFace embed (`pipeline`), or the recognizer
graph alone on an aligned crop (`recognizer`), or detect-only (`detect`), or
detect+genderage (`analyze`). The bench subcommand reports a machine-parseable
`<mode>: <ms> ms/image over <N> passes` line that the harness scrapes.

Anti-spoof (the MiniFASNet ensemble) is not benched as a standalone row: it is a
verify-time veto, not a primary inference path, and the engine exposes no
single-stage entry point for it. Its parity is gated by `tests/test_antispoof`.

## Running the CPU comparison

```bash
# build the CLI (CPU) first. Use AVX2/FMA + tinyBLAS (GGML_NATIVE=OFF): on Zen5,
# -march=native turns on AVX512, which HELPS the small recognizer but REGRESSES
# the dominant bandwidth-bound SCRFD detector (AVX512 down-clock on a memory-bound
# im2col conv). See the dead-end note in benchmarks/RESULTS.md.
cmake -B build -DFACEDETECT_BUILD_TESTS=ON -DGGML_NATIVE=OFF && cmake --build build -j

# reference venv (see docs/conversion.md / scripts/requirements.txt). Pass the
# SAME thread count to both sides for an honest comparison; --ref-threads defaults
# to --threads. The engine's own default is now min(hardware_concurrency, 8).
.venv/bin/python scripts/bench_compare.py \
    --image tests/fixtures/face_a.jpg --n 20 --threads 8 \
    --out benchmarks/RESULTS.md
```

The GGUF packs are resolved under `models/` (`buffalo_l-f32.gguf`,
`yunet-sface-f32.gguf`); the insightface reference ONNX (`w600k_r50.onnx`,
`det_10g.onnx`, `genderage.onnx`) is resolved from `~/.insightface/models/<pack>/`
(override with `INSIGHTFACE_HOME`), and the YuNet/SFace ONNX from `models/yunet/`
and `models/sface/`. A pipeline whose reference deps or weights are missing is
recorded as `n/a` in the `note` column, never fatal.

## Methodology (honest)

- **Matched threads, both sides.** The ggml CLI runs `--threads N`; the reference
  is held to the SAME count (`--ref-threads`, default = `--threads`): onnxruntime
  `intra-op = N` / `inter-op = 1` (injected into every insightface session, since
  `FaceAnalysis` otherwise builds its own multi-threaded sessions) and
  `cv2.setNumThreads(N)`. This is a core-for-core comparison at the chosen
  parallelism. `RESULTS.md` reports both the matched 1-thread and matched 8-thread
  runs; comparing at 1 thread isolates per-core kernel efficiency, while 8 threads
  reflects the engine's new default (`min(hardware_concurrency, 8)`).
- **Warmup excluded.** One untimed warmup pass, then N timed passes (default 20),
  wall-clock mean.
- **`FaceAnalysis` does more work.** insightface `FaceAnalysis.get` also runs the
  genderage and 2d106/1k3d68 landmark heads, so the full-pipeline reference does
  strictly more work than the ggml detect->align->embed pipeline. The
  `detect`-only and `analyze` rows isolate the comparable stages.
- **Recognizer-alone rows.** Both sides feed a plain-resized image as a stand-in
  aligned crop (latency is content-independent), isolating the embed graph from
  detection and alignment.
- **Image I/O.** The reference loads the image once; the ggml `pipeline` / `detect`
  / `analyze` modes re-decode the (small) JPEG each call inside the C++ entry
  point, a sub-millisecond overhead.
- **`speedup = reference / ggml`** (>1 means ggml is faster). Absolute numbers
  are machine-specific; see the machine/version block at the top of
  `benchmarks/RESULTS.md`.

### Reading the current CPU numbers

Two honest facts, both in `benchmarks/RESULTS.md`:

1. **The thread default was the big free win.** The process-global backend used to
   default to 1 thread; it is now `min(hardware_concurrency, 8)`
   (`FACEDETECT_THREADS` overrides). These forwards are compute-bound and scale
   ~linearly to ~8 threads, so the default change cut absolute latency 3.2-3.9x
   (e.g. SCRFD detect 515 -> 154 ms) for every consumer that took the default,
   with parity unchanged (matmul reduction order is thread-count-independent).
2. **It does not close the gap to onnxruntime.** At matched threads the ggml engine
   is still ~2-6x slower than onnxruntime/OpenCV: the reference multithreads at
   least as well, and ggml's conv (im2col + `mul_mat`) is intrinsically behind a
   dedicated cache-blocked conv kernel on CPU. The realistic CPU ceiling for this
   port is "fast enough, deployable, parity-exact," not "beats onnxruntime."

The value of the port is no-Python deployment, a small `libfacedetect.so`, GGUF
quantization, and the same code path on any ggml GPU backend. The
persistent-`ggml_gallocr` / zero-copy-weights levers are in place; `-march=native`
(AVX512) is a measured dead-end for the detector-dominated CPU path (see
`RESULTS.md`); quantized packs and a CUDA build (below) are the obvious next wins.

**Round 2 update (stacked levers, vs prior baseline `523aee1`).** A second
optimization round narrowed the gap with two parity-safe levers (numbers are
contention-robust back-to-back A/B, median-of-N, on a loaded box; speedups are the
signal, absolute ms are load-specific). See
[../benchmarks/RESULTS.md](../benchmarks/RESULTS.md) for the full tables. The
headline is a **custom AVX2 Winograd F(2x2,3x3)** for the large 3x3 stride-1 convs
that **BEAT tinyBLAS sgemm** (refuting the prior pass's doubt that it could):
SCRFD detect 1.37x at 1 thread and ~1.7-2.05x at 8 threads (the full buffalo_l
pipeline inherits 1.28x / 1.58-1.77x). The second lever folds ArcFace's
inference-mode BatchNorm affine to host constants + reduces graph overhead
(recognizer 8t ~1.06-1.13x); PReLU could not be fused without breaking the GPU
path and was left numerically identical. Parity held (SCRFD <= 1 px; embedding
cosine = 1.0 isolated/golden). These deltas are vs our OWN prior ggml baseline,
not a fresh head-to-head vs onnxruntime; they narrow the gap (SCRFD detect
~1.7-2x at 8t) but a residual conv-kernel gap remains intrinsic on these
conv-bound detectors/recognizers, while GEMM-bound transformer heads are the most
likely to reach near-parity. No model regressed.

## GPU verification + benchmark (CUDA host only)

`scripts/gpu_verify.sh` MUST run on a machine with an NVIDIA GPU (CUDA toolkit +
driver). It is intentionally **not** part of CI here, because the CI/dev box that
produced the CPU table has no GPU (`nvidia-smi`/`nvcc` absent). On a CUDA host it:

1. builds `build-cuda/` with `-DFACEDETECT_GGML_CUDA=ON`,
2. runs the existing parity runner scripts (detector / embedding / yunet / sface)
   pointed at `build-cuda/` with the engine forced onto the GPU
   (`FACEDETECT_DEVICE=CUDA0`), confirming the same strict gates (recognizer
   cosine `>= 0.9999` / max|d| `<= 1e-3`, detector boxes+landmarks `<= 1 px`)
   still hold on GPU against the CPU reference goldens,
3. runs the bench on the GPU and appends GPU rows to `benchmarks/RESULTS.md`.

```bash
# on a DGX / CUDA host:
./scripts/gpu_verify.sh
FD_GPU_DEVICE=CUDA0 N=50 ./scripts/gpu_verify.sh   # pick device / passes
SKIP_BUILD=1 ./scripts/gpu_verify.sh               # reuse build-cuda/
PACKS="buffalo_l" ./scripts/gpu_verify.sh          # limit the buffalo packs gated
```

Device selection follows the engine's runtime backend pick in `src/backend.cpp`:
`FACEDETECT_DEVICE=cpu` forces CPU, otherwise the value matches a ggml
backend-registry device by name (case-insensitive), e.g. `CUDA0`. The script
exports it for every parity gate and bench run.
