#!/usr/bin/env python3
"""Comparative latency benchmark: face-detect.cpp (ggml) vs the reference stack.

For each face pipeline this measures end-to-end per-image latency two ways and
prints a comparison table (ggml ms/image | reference ms/image | speedup):

* OUR ENGINE  - the ``facedetect-cli bench`` subcommand, which times the real
  C++ entry point (image decode + SCRFD/YuNet detect + 5-landmark norm_crop
  align + ArcFace/SFace embed, or the recognizer graph alone on an aligned crop,
  or detect-only, or detect+genderage), warmup excluded.
* REFERENCE   - the SAME model via its native Python path on CPU:
    - buffalo_l full pipeline -> insightface ``FaceAnalysis(name=buffalo_l).get``
    - ArcFace recognizer alone -> onnxruntime on ``w600k_r50.onnx`` (aligned crop)
    - YuNet+SFace pipeline     -> cv2 ``FaceDetectorYN`` + ``FaceRecognizerSF``
    - SFace recognizer alone   -> onnxruntime on the SFace ONNX (aligned crop)
    - genderage (analyze)      -> insightface ``FaceAnalysis`` det+genderage
    - detect                   -> insightface SCRFD detector ``.detect``
  warmup excluded, same image, same N.

Methodology (honest): both sides run SINGLE-THREADED by default (``--threads 1``;
onnxruntime intra/inter-op = 1 - injected into EVERY insightface session via a
monkeypatch so the reference is genuinely single-thread; cv2.setNumThreads(REF_THREADS)).
This is a core-for-core comparison, not throughput-at-max-cores. The reference
loads the image once and times detect/align/forward per iteration; the ggml side
re-decodes the (small) JPEG every ``pipeline``/``detect``/``analyze`` call inside
the C++ entry point, a sub-millisecond overhead. The recognizer-alone rows feed a
plain-resized image as a stand-in aligned crop on BOTH sides (latency is content
independent). insightface ``FaceAnalysis.get`` ALSO runs the genderage +
2d106/1k3d68 landmark heads, so the full-pipeline reference does strictly more
work than the ggml detect->align->embed pipeline; that is noted in the table.
Numbers are wall-clock means over N timed passes after a warmup pass; absolute
values are machine-specific (see the machine block in the emitted report).

Usage:
    .venv/bin/python scripts/bench_compare.py \
        --image tests/fixtures/face_a.jpg --n 20 --threads 1 \
        --out benchmarks/RESULTS.md
    # add --device CUDA0 (and a CUDA-enabled CLI build) to time the ggml side on GPU.

Exit codes: 0 = report written (per-model failures are recorded as "n/a", not
fatal). Models whose reference deps/weights are unavailable are clearly marked.
"""
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone


def _first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def _insightface_pack_file(pack, fname):
    home = os.environ.get("INSIGHTFACE_HOME") or os.path.expanduser("~/.insightface")
    return os.path.join(home, "models", pack, fname)


def discover_models(root):
    """Return the pipeline registry, resolving ggml GGUF + reference paths."""
    m = lambda *p: os.path.join(root, *p)
    return [
        dict(name="buffalo_l pipeline (detect+align+embed)", mode="pipeline",
             ggml=m("models", "buffalo_l-f32.gguf"),
             ref="insightface_full", pack="buffalo_l",
             note_extra="ref FaceAnalysis also runs genderage+landmark heads"),
        dict(name="ArcFace recognizer (embed only)", mode="recognizer",
             ggml=m("models", "buffalo_l-f32.gguf"),
             ref="onnx_recognizer", onnx=_insightface_pack_file("buffalo_l", "w600k_r50.onnx"),
             rec_mean=127.5, rec_std=127.5, size=112),
        dict(name="YuNet+SFace pipeline (detect+align+embed)", mode="pipeline",
             ggml=m("models", "yunet-sface-f32.gguf"),
             ref="cv2_yunet_sface",
             det_onnx=m("models", "yunet", "face_detection_yunet_2023mar.onnx"),
             rec_onnx=m("models", "sface", "face_recognition_sface_2021dec.onnx")),
        dict(name="SFace recognizer (embed only)", mode="recognizer",
             ggml=m("models", "yunet-sface-f32.gguf"),
             ref="onnx_recognizer",
             onnx=m("models", "sface", "face_recognition_sface_2021dec.onnx"),
             rec_mean=0.0, rec_std=1.0, size=112),
        dict(name="buffalo_l genderage (detect+analyze)", mode="analyze",
             ggml=m("models", "buffalo_l-f32.gguf"),
             ref="insightface_analyze", pack="buffalo_l"),
        dict(name="buffalo_l detect (SCRFD only)", mode="detect",
             ggml=m("models", "buffalo_l-f32.gguf"),
             ref="insightface_detect", pack="buffalo_l"),
    ]


# ---------------------------------------------------------------------------
# ggml side: drive the CLI bench subcommand.
# ---------------------------------------------------------------------------
def bench_ggml(cli, model, image, mode, n, threads, device):
    if not model or not os.path.isfile(model):
        return None, "gguf missing"
    env = dict(os.environ)
    if device:
        env["FACEDETECT_DEVICE"] = device  # "cpu" or a ggml device name e.g. "CUDA0"
    cmd = [cli, "bench", "--model", model, "--input", image,
           "--mode", mode, "--n", str(n), "--threads", str(threads)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    except Exception as e:  # noqa: BLE001
        return None, f"cli error: {e}"
    if out.returncode != 0:
        tail = (out.stderr or out.stdout).strip().splitlines()[-1:] or [""]
        return None, f"cli rc={out.returncode}: {tail[0]}"
    mobj = re.search(r"([0-9.]+)\s*ms/image", out.stdout)
    if not mobj:
        return None, "no timing line"
    return float(mobj.group(1)), None


# ---------------------------------------------------------------------------
# Reference side: native Python forward, CPU, single-thread, warmup excluded.
# ---------------------------------------------------------------------------
# Reference-side thread count, set from --ref-threads in main() (defaults to
# --threads so ggml and the reference are matched core-for-core). The reference
# is genuinely single-thread at 1, and onnxruntime intra-op-parallel at >1.
REF_THREADS = 1


def _ref_session_options():
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = REF_THREADS
    so.inter_op_num_threads = 1
    return so


def _set_cv2_threads():
    import cv2
    cv2.setNumThreads(REF_THREADS)


def _force_single_thread_insightface():
    """Inject matched-thread SessionOptions into insightface's session wrapper so
    every model FaceAnalysis loads runs at REF_THREADS (a core-for-core compare).

    insightface defines ``PickableInferenceSession(onnxruntime.InferenceSession)``
    at import time, so patching ``onnxruntime.InferenceSession`` itself is both too
    late (the subclass is already bound) and breaks the subclass; instead we wrap
    that wrapper's ``__init__`` to add ``sess_options`` when FaceAnalysis omits it.
    Must run AFTER ``import insightface`` and BEFORE the FaceAnalysis sessions are
    built (i.e. before constructing FaceAnalysis)."""
    import onnxruntime as ort
    from insightface.model_zoo import model_zoo as mz
    if getattr(mz, "_fd_single_thread", False):
        return
    orig_init = mz.PickableInferenceSession.__init__

    def init(self, model_path, **kwargs):
        if kwargs.get("sess_options") is None:
            kwargs["sess_options"] = _ref_session_options()
        orig_init(self, model_path, **kwargs)

    mz.PickableInferenceSession.__init__ = init
    mz._fd_single_thread = True


def _timeit(fwd, n):
    fwd()  # warmup (excluded)
    t0 = time.perf_counter()
    for _ in range(n):
        fwd()
    return (time.perf_counter() - t0) * 1000.0 / n


def bench_ref_insightface_full(spec, image_path, n):
    import cv2
    cv2.setNumThreads(REF_THREADS)
    from insightface.app import FaceAnalysis
    _force_single_thread_insightface()
    app = FaceAnalysis(name=spec["pack"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    def fwd():
        app.get(img)
    return _timeit(fwd, n)


def bench_ref_insightface_analyze(spec, image_path, n):
    import cv2
    cv2.setNumThreads(REF_THREADS)
    from insightface.app import FaceAnalysis
    _force_single_thread_insightface()
    app = FaceAnalysis(name=spec["pack"],
                       allowed_modules=["detection", "genderage"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    def fwd():
        app.get(img)
    return _timeit(fwd, n)


def bench_ref_insightface_detect(spec, image_path, n):
    import cv2
    cv2.setNumThreads(REF_THREADS)
    from insightface.app import FaceAnalysis
    _force_single_thread_insightface()
    app = FaceAnalysis(name=spec["pack"], allowed_modules=["detection"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)
    det = app.models["detection"]
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    def fwd():
        det.detect(img, max_num=0, metric="default")
    return _timeit(fwd, n)


def bench_ref_onnx_recognizer(spec, image_path, n):
    import cv2
    import numpy as np
    import onnxruntime as ort
    cv2.setNumThreads(REF_THREADS)
    if not spec.get("onnx") or not os.path.isfile(spec["onnx"]):
        raise FileNotFoundError(spec.get("onnx") or "<recognizer onnx>")
    so = _ref_session_options()
    sess = ort.InferenceSession(spec["onnx"], sess_options=so,
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    sz = int(spec.get("size", 112))
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    # Plain-resize the source to the recognizer input (stand-in aligned crop), the
    # SAME thing the ggml `recognizer` mode does; latency is content-independent.
    crop = cv2.resize(img, (sz, sz))
    mean = float(spec.get("rec_mean", 127.5))
    std = float(spec.get("rec_std", 127.5)) or 1.0
    blob = cv2.dnn.blobFromImage(crop, 1.0 / std, (sz, sz),
                                 (mean, mean, mean), swapRB=True)

    def fwd():
        sess.run(None, {in_name: blob})
    return _timeit(fwd, n)


def bench_ref_cv2_yunet_sface(spec, image_path, n):
    import cv2
    cv2.setNumThreads(REF_THREADS)
    if not os.path.isfile(spec["det_onnx"]):
        raise FileNotFoundError(spec["det_onnx"])
    if not os.path.isfile(spec["rec_onnx"]):
        raise FileNotFoundError(spec["rec_onnx"])
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    oh, ow = img.shape[:2]
    det = cv2.FaceDetectorYN.create(spec["det_onnx"], "", (ow, oh),
                                    score_threshold=0.5, nms_threshold=0.3, top_k=5000)
    rec = cv2.FaceRecognizerSF.create(spec["rec_onnx"], "")

    def fwd():
        _n, faces = det.detect(img)
        if faces is None or len(faces) == 0:
            return
        import numpy as np
        primary = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]
        aligned = rec.alignCrop(img, primary)
        rec.feature(aligned)
    return _timeit(fwd, n)


REF_FN = {
    "insightface_full": bench_ref_insightface_full,
    "insightface_analyze": bench_ref_insightface_analyze,
    "insightface_detect": bench_ref_insightface_detect,
    "onnx_recognizer": bench_ref_onnx_recognizer,
    "cv2_yunet_sface": bench_ref_cv2_yunet_sface,
}


def bench_ref(spec, image_path, n):
    try:
        return REF_FN[spec["ref"]](spec, image_path, n), None
    except SystemExit as e:
        return None, f"unavailable ({e})"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
def machine_info():
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    vers = {}
    for mod, attr in (("numpy", "numpy"), ("onnxruntime", "onnxruntime"),
                      ("insightface", "insightface"), ("cv2", "cv2")):
        try:
            vers[mod] = __import__(attr).__version__
        except Exception:  # noqa: BLE001
            vers[mod] = "n/a"
    return cpu, vers


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="tests/fixtures/face_a.jpg")
    ap.add_argument("--n", type=int, default=20, help="timed passes (warmup excluded)")
    ap.add_argument("--threads", type=int, default=1, help="ggml CLI threads")
    ap.add_argument("--ref-threads", type=int, default=None,
                    help="reference CPU threads (onnxruntime intra-op + cv2); "
                         "defaults to --threads for a matched core-for-core compare")
    ap.add_argument("--cli", default=None, help="path to facedetect-cli")
    ap.add_argument("--device", default="cpu",
                    help="ggml device for FACEDETECT_DEVICE (cpu, or e.g. CUDA0)")
    ap.add_argument("--out", default="benchmarks/RESULTS.md")
    ap.add_argument("--append-gpu", action="store_true",
                    help="append GPU rows to --out instead of rewriting it (for gpu_verify.sh)")
    args = ap.parse_args()

    global REF_THREADS
    REF_THREADS = args.ref_threads if args.ref_threads is not None else args.threads

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cli = args.cli or _first_existing([
        os.path.join(root, "build", "examples", "cli", "facedetect-cli"),
        os.path.join(root, "build-cuda", "examples", "cli", "facedetect-cli"),
        shutil.which("facedetect-cli") or "",
    ])
    if not cli:
        print("ERROR: facedetect-cli not found; build it or pass --cli", file=sys.stderr)
        return 2
    image = args.image if os.path.isfile(args.image) else os.path.join(root, args.image)

    models = discover_models(root)
    rows = []
    for spec in models:
        sys.stderr.write(f"[bench] {spec['name']} (ggml {args.device}) ... ")
        sys.stderr.flush()
        ggml_ms, gerr = bench_ggml(cli, spec["ggml"], image, spec["mode"],
                                   args.n, args.threads, args.device)
        sys.stderr.write(f"{ggml_ms if ggml_ms else gerr}\n")
        ref_ms = ref_err = None
        if not args.append_gpu:  # reference is CPU-only; skip on GPU append pass
            sys.stderr.write(f"[bench] {spec['name']} (reference) ... ")
            sys.stderr.flush()
            ref_ms, ref_err = bench_ref(spec, image, args.n)
            sys.stderr.write(f"{ref_ms if ref_ms else ref_err}\n")
        rows.append((spec, ggml_ms, gerr, ref_ms, ref_err))

    cpu, vers = machine_info()
    label = (f"GPU ({args.device})" if args.device != "cpu"
             else f"CPU ({args.threads} thread, ref {REF_THREADS} thread)")

    def fmt(v):
        return f"{v:.1f}" if isinstance(v, (int, float)) else "n/a"

    lines = []
    if args.device != "cpu":
        lines.append(f"\n### GPU rows - device `{args.device}` ({datetime.now(timezone.utc):%Y-%m-%d})\n")
        lines.append("| Pipeline | ggml ms/image (GPU) | reference ms/image (CPU) | note |")
        lines.append("| -------- | ------------------- | ------------------------ | ---- |")
    else:
        lines.append("| Pipeline | ggml ms/image | reference ms/image | speedup (ref/ggml) | note |")
        lines.append("| -------- | ------------- | ------------------ | ------------------ | ---- |")
    for spec, ggml_ms, gerr, ref_ms, ref_err in rows:
        speed = (f"{ref_ms / ggml_ms:.2f}x"
                 if isinstance(ggml_ms, (int, float)) and isinstance(ref_ms, (int, float))
                 else "n/a")
        note = "; ".join(x for x in (gerr, ref_err, spec.get("note_extra")) if x) or "ok"
        if args.device != "cpu":
            lines.append(f"| {spec['name']} | {fmt(ggml_ms)} | {fmt(ref_ms)} | {note} |")
        else:
            lines.append(f"| {spec['name']} | {fmt(ggml_ms)} | {fmt(ref_ms)} | {speed} | {note} |")
    table = "\n".join(lines) + "\n"

    header = (
        "# face-detect.cpp comparative benchmark\n\n"
        f"Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} "
        f"by `scripts/bench_compare.py` (N={args.n}, warmup excluded).\n\n"
        f"- Machine: {cpu}\n"
        f"- ggml side: `facedetect-cli bench`, device `{args.device}`, "
        f"`--threads {args.threads}`\n"
        f"- Reference: CPU, onnxruntime intra-op={REF_THREADS} inter-op=1, "
        f"cv2.setNumThreads({REF_THREADS}) (matched to ggml --threads)\n"
        f"- Versions: insightface {vers['insightface']}, onnxruntime "
        f"{vers['onnxruntime']}, opencv {vers['cv2']}, numpy {vers['numpy']}\n"
        f"- Image: `{os.path.relpath(image, root)}`\n\n"
        "speedup = reference / ggml (>1 means ggml is faster). Absolute numbers are "
        "machine-specific. insightface `FaceAnalysis.get` runs the genderage + "
        "landmark heads in addition to detect+embed, so the full-pipeline reference "
        "does strictly more work than the ggml detect->align->embed pipeline.\n\n"
        f"## Results - {label}\n\n"
    )

    out_path = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if args.append_gpu and os.path.isfile(out_path):
        with open(out_path, "a") as f:
            f.write(table)
        sys.stderr.write(f"appended GPU rows to {out_path}\n")
    else:
        with open(out_path, "w") as f:
            f.write(header + table)
        sys.stderr.write(f"wrote {out_path}\n")
    print(header + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
