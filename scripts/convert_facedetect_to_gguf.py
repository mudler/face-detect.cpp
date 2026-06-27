#!/usr/bin/env python3
"""Convert an insightface face-recognition model pack to GGUF (f32 / f16 / q8_0 / q4_0 / q4_k).

The GGUF is fully metadata-driven: all config lives in KV, and tensor names are
kept verbatim from the source ONNX initializers under a stable per-sub-model
namespace prefix (``det.`` / ``rec.`` / ``ga.`` / ``as<i>.``) so the C++ port is a
1:1 mapping. insightface ships its buffalo packs as ONNX graphs (SCRFD detector +
ArcFace recognizer, plus optional genderage + the MiniFASNet anti-spoof
ensemble), so the converter reads ONNX *initializers* (weights), not a PyTorch
state_dict. The prefix is required because the SCRFD and ArcFace graphs reuse
auto-numbered initializer names for different weights (see docs/conversion.md and
scripts/tensor_manifest_buffalo_l.md).

Quantization (``--dtype f16|q8_0|q4_0|q4_k``) is applied **only** to the large
convolution / linear weights the C++ engine consumes directly via ``ggml_mul_mat``
/ ``ggml_conv`` (the SCRFD backbone + heads and the ArcFace ResNet trunk), and only
when the innermost ggml dim is block-aligned (32 for q8_0/q4_0, 256 for q4_k); in
practice the ArcFace recognizer FC (``rec.fc.weight``) is the dominant quantizable
weight. ggml dequantizes those on the fly. Everything read as raw F32 (BN running
stats, biases, the 5-point reference template, small projection heads) stays F32 -
see ``should_quantize`` and ``docs/quantization.md``.

Supported packs:
  * buffalo_l / buffalo_m / buffalo_s : SCRFD + ArcFace ResNet50, 512-d (primary,
                                        implemented; +genderage when present)
  * MiniFASNet ensemble               : V2@2.7 + V1SE@4.0, 80x80 anti-spoof
                                        (wired via as<i>. prefix when present)
  * YuNet + SFace (OpenCV-zoo, Apache): 128-d alt detector + recognizer (Phase 8)

See ``docs/conversion.md`` for the full KV schema.
"""
import argparse
import sys

try:
    import gguf  # noqa: F401
except ImportError as e:  # pragma: no cover - env guard
    print(f"converter: missing dependency 'gguf': {e}", file=sys.stderr)
    print("FACEDETECT_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)

try:
    import onnx  # noqa: F401
    from onnx import numpy_helper
except ImportError as e:  # pragma: no cover - env guard
    print(f"converter: missing dependency 'onnx': {e}", file=sys.stderr)
    print("FACEDETECT_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)

try:
    import numpy as np
except ImportError as e:  # pragma: no cover - env guard
    print(f"converter: missing dependency 'numpy': {e}", file=sys.stderr)
    print("FACEDETECT_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)


# Only weights fed DIRECTLY to ggml_mul_mat / ggml_conv may be quantized; ggml
# dequantizes them on the fly. Large 2-D Gemm/MatMul weights qualify (both
# innermost ggml dims >= 32). 3x3 / 1x1 conv kernels keep their tiny innermost
# `ne` (3 or 1) and therefore stay F32, as do BN stats, biases, PReLU slopes and
# the 5-point template. See docs/quantization.md and tensor_manifest_buffalo_l.md.
def should_quantize(name, arr, dtype):
    if dtype == "f32" or arr.ndim < 2:
        return None
    ggml_ne = list(arr.shape[::-1])          # ggml ne is reversed
    if ggml_ne[0] < 32 or ggml_ne[1] < 32:
        return None
    if dtype == "f16":
        return gguf.GGMLQuantizationType.F16
    if dtype == "q8_0":
        return None if ggml_ne[0] % 32 else gguf.GGMLQuantizationType.Q8_0
    if dtype == "q4_0":
        return None if ggml_ne[0] % 32 else gguf.GGMLQuantizationType.Q4_0
    if dtype == "q4_k":
        return None if ggml_ne[0] % 256 else gguf.GGMLQuantizationType.Q4_K
    return None


def onnx_graph_spec(onnx_path):
    """Serialize an ONNX graph's node topology to a list of compact strings.

    The GGUF carries only the MiniFASNet *initializers* (weights); the C++ port
    rebuilds the forward graph from this topology rather than hard-coding two
    pruned architectures (V2 and V1SE differ in channel counts and V1SE adds SE
    blocks). Each node is ``op;out;in0,in1,...;k=v,k=v`` where the attrs carry the
    Conv stride (``s``), pad (``p``) and group (``g``). Inputs are resolved by the
    interpreter as activations (seen as a prior node output) or as named weights.
    Returns ``(nodes, output_name)``.
    """
    m = onnx.load(onnx_path)
    g = m.graph
    nodes = []
    for n in g.node:
        attrs = {}
        for a in n.attribute:
            if a.name == "strides":
                attrs["s"] = a.ints[0]
            elif a.name == "pads":
                attrs["p"] = a.ints[0]
            elif a.name == "group":
                attrs["g"] = a.i
            elif a.name == "transB":
                attrs["tb"] = a.i      # ONNX Gemm: B is pre-transposed [out,in]
            elif a.name == "epsilon":
                # SFace tags each BatchNormalization with its own epsilon (1e-3 conv
                # BNs vs 2e-5 bn1/fc1); the C++ interpreter parses `e=` as a float.
                attrs["e"] = a.f
            elif a.name == "kernel_shape":
                # MaxPool / AveragePool kernel (the 1k3d68 landmark stem is a 3x3
                # MaxPool, not the 2x2 the interpreter would otherwise default to).
                # Harmless for Conv (the interpreter takes its kernel from the weight
                # tensor shape and ignores `k=`).
                attrs["k"] = a.ints[0]
        astr = ",".join(f"{k}={v}" for k, v in attrs.items())
        nodes.append(";".join([n.op_type, n.output[0], ",".join(n.input), astr]))
    return nodes, g.input[0].name, g.output[0].name


def detector_graph_spec(onnx_path):
    """Serialize an SCRFD detector's ONNX topology to a compact, prunable spec.

    Mirrors ``onnx_graph_spec`` (used for the MiniFASNet / MobileFaceNet graphs)
    but tailored to the SCRFD det_2.5g / det_500m variants so the C++ engine can
    replay them metadata-driven instead of hand-mapping each variant's node names
    (the det_10g graph stays hand-mapped). Two SCRFD-specific concerns are handled
    here so the interpreter need not implement ONNX's dynamic-shape machinery:

      * The FPN ``Resize`` (top-down upsample) is nearest / asymmetric / floor with
        its target size computed by a Shape/Gather/Concat subgraph. On the square,
        letterboxed 640 input every feature map is even, so it is exactly a 2x
        nearest upsample (what the det_10g hand path already does via
        ``ggml_upscale``). We emit ``Resize`` with only its data input and DROP the
        whole Shape/Gather/Unsqueeze/Slice/Concat shape-plumbing (it feeds nothing
        but the Resize sizes).
      * Each per-stride head ends ``Transpose(perm) -> Reshape(->[-1,C]) [-> Sigmoid]``
        where the Reshape shape is a static ``[-1, C]`` initializer. We carry the
        perm verbatim (``pm=...``) and the reshape's last dim (``rc=C``) as attrs so
        the interpreter reshapes without evaluating a shape tensor.

    The input separator is ``|`` (not ``,``): SCRFD head weight names embed commas
    (e.g. ``bbox_head.stride_cls.(8, 8).weight``), which would corrupt a
    comma-joined input list. Returns ``(nodes, input_name, output_names)`` where
    ``output_names`` is the ONNX graph-output order (scores s8/16/32, bbox s8/16/32,
    kps s8/16/32) the host decode + ScrfdRawOut mapping expect.
    """
    m = onnx.load(onnx_path)
    g = m.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    # Pure shape-plumbing ops: present only to compute the Resize target sizes.
    DROP = {"Shape", "Gather", "Unsqueeze", "Slice", "Concat", "Constant"}
    nodes = []
    for n in g.node:
        if n.op_type in DROP:
            continue
        attrs = {}
        ins = list(n.input)
        if n.op_type == "Resize":
            ins = [n.input[0]]            # data only; 2x nearest is implied
        elif n.op_type == "Reshape":
            shp = inits.get(n.input[1])
            if shp is None:
                raise ValueError(f"detector Reshape {n.output[0]} has a non-constant shape")
            attrs["rc"] = int(shp.reshape(-1)[-1])   # reshape to [-1, rc]
            ins = [n.input[0]]
        elif n.op_type == "Transpose":
            for a in n.attribute:
                if a.name == "perm":
                    attrs["pm"] = "/".join(str(v) for v in a.ints)
        else:
            for a in n.attribute:
                if a.name == "strides":
                    attrs["s"] = a.ints[0]
                elif a.name == "pads":
                    attrs["p"] = a.ints[0]
                elif a.name == "group":
                    attrs["g"] = a.i
                elif a.name == "kernel_shape":
                    attrs["k"] = a.ints[0]
        astr = "|".join(f"{k}={v}" for k, v in attrs.items())
        nodes.append(";".join([n.op_type, n.output[0], "|".join(ins), astr]))
    return nodes, g.input[0].name, [o.name for o in g.output]


def add_onnx_initializers(w, onnx_path, dtype, skip_q=False, prefix=""):
    """Write every initializer of an ONNX graph VERBATIM. Returns count.

    Each initializer name is kept exactly as exported, but carries a stable
    per-sub-model ``prefix`` (``det.`` / ``rec.`` / ``ga.`` / ``as<i>.``). The
    insightface buffalo packs reuse auto-numbered ONNX initializer names (e.g.
    ``685``, ``691``) across the SCRFD and ArcFace graphs for *different* real
    weights, and the MiniFASNet anti-spoof ensemble repeats every name across its
    two members; since the C++ loader holds all tensors in one ggml context keyed
    by name, the namespace prefix is required to keep the mapping 1:1 and
    collision-free. The prefix is a single deterministic tag, not a rename table.
    See scripts/tensor_manifest_buffalo_l.md.
    """
    m = onnx.load(onnx_path)
    n = 0
    for init in m.graph.initializer:
        arr = np.ascontiguousarray(numpy_helper.to_array(init), dtype=np.float32)
        if arr.ndim == 0:                    # scalar bookkeeping
            continue
        name = prefix + init.name
        qt = None if skip_q else should_quantize(init.name, arr, dtype)
        if qt is None:
            w.add_tensor(name, arr)
        else:
            try:
                raw = gguf.quantize(arr, qt)
            except NotImplementedError:
                # Some quant types (notably Q4_K) are listed in the allowlist but
                # have no packer in the bundled pure-python `gguf`; fail clearly
                # rather than emit a raw traceback half-way through the file.
                print(f"converter: dtype {dtype!r} ({qt.name}) is not implemented "
                      f"by this gguf build; pick q8_0/q4_0/f16/f32", file=sys.stderr)
                print("FACEDETECT_QUANT_UNSUPPORTED", file=sys.stderr)
                sys.exit(2)
            w.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=qt)
        n += 1
    return n


def write_config_kv(w, pack):
    """Write the facedetect.* config KV block. `pack` carries the per-variant
    detector/recognizer geometry resolved from the ONNX pack."""
    # general.architecture is already written as "facedetect" by the GGUFWriter
    # constructor below; do not re-add it (avoids a duplicate-key warning).
    w.add_string("facedetect.arch", pack["arch"])
    w.add_string("facedetect.detector.kind", pack["detector"])
    if pack.get("engine"):
        # Provenance tag: "onnx_direct" packs carry the detector ONNX topology
        # verbatim as the detector.graph KV (YuNet) rather than a hand-mapped C++
        # graph. The loader does not read this; it documents how the pack was built.
        w.add_string("facedetect.detector.engine", pack["engine"])
    w.add_uint32("facedetect.detector.input_size", pack["det_input_size"])
    w.add_array("facedetect.detector.strides", pack["det_strides"])
    w.add_uint32("facedetect.detector.num_anchors", pack["det_num_anchors"])
    w.add_float32("facedetect.detector.score_thresh", pack["det_score_thresh"])
    w.add_float32("facedetect.detector.nms_thresh", pack["det_nms_thresh"])
    w.add_string("facedetect.recognizer.kind", pack["recognizer"])
    w.add_uint32("facedetect.recognizer.input_size", pack["rec_input_size"])
    w.add_uint32("facedetect.recognizer.embed_dim", pack["embed_dim"])
    w.add_float32("facedetect.recognizer.verify_threshold", pack["verify_threshold"])
    if pack.get("genderage_onnx"):
        w.add_bool("facedetect.genderage.present", True)
        w.add_uint32("facedetect.genderage.input_size", 96)
    if pack.get("antispoof"):
        w.add_bool("facedetect.antispoof.present", True)
        w.add_uint32("facedetect.antispoof.input_size", 80)
        # Scales are paired with each model IN ORDER (as0=V2@2.7, as1=V1SE@4.0);
        # see resolve_pack. The C++ host warp reads these so the per-model crop
        # matches the reference exactly.
        w.add_array("facedetect.antispoof.scales",
                    [scale for _path, scale in pack["antispoof"]])


def resolve_antispoof(base):
    """Find the MiniFASNet ensemble and pair each model with its crop scale.

    The Silent-Face ensemble is order-sensitive: MiniFASNetV2 uses a 2.7 crop
    scale and MiniFASNetV1SE uses 4.0 (upstream test.py). A plain sorted glob
    would put V1SE before V2 and silently swap the scales, so we pair each model
    by filename. Returns an ORDERED list of ``(path, scale)`` -> as0=V2, as1=V1SE.
    """
    import os
    import glob
    hits = glob.glob(os.path.join(base, "**", "MiniFASNet*.onnx"), recursive=True)

    def pick(tag):
        for h in hits:
            if tag in os.path.basename(h):
                return h
        return None

    pairs = []
    v2 = pick("MiniFASNetV2")
    if v2:
        pairs.append((v2, 2.7))
    v1se = pick("MiniFASNetV1SE")
    if v1se:
        pairs.append((v1se, 4.0))
    return pairs


def resolve_landmarks(model):
    """Find the two insightface dense-landmark ONNX heads (2d106det + 1k3d68).

    These are the dense-landmark regressors the buffalo_l pack ships alongside the
    detector / recognizer: ``2d106det.onnx`` (2D 106-point) and ``1k3d68.onnx``
    (3D 68-point). They are ENGINE-LEVEL capability only - no LocalAI proto RPC or
    API endpoint consumes dense landmarks yet (the Detect RPC returns just the
    5-point SCRFD kps), so they ship as their own GGUF rather than as a gallery
    pack. ``model == "landmarks"`` resolves them from the buffalo_l cache; a
    directory path resolves them from there. Returns ``(d2_path, d3_path)`` or None.
    """
    import os
    import glob
    base = model if os.path.isdir(model) else \
        os.path.expanduser("~/.insightface/models/buffalo_l")

    def find(pat):
        hits = glob.glob(os.path.join(base, "**", pat), recursive=True)
        return hits[0] if hits else None

    d2 = find("2d106det.onnx")
    d3 = find("1k3d68.onnx")
    if not d2 or not d3:
        return None
    return d2, d3


def write_landmarks_gguf(d2, d3, output, dtype):
    """Write the engine-only dense-landmark GGUF (2D 106-pt + 3D 68-pt).

    Both heads are metadata-driven through the SAME ONNX-graph interpreter the
    MobileFaceNet recognizer / MiniFASNet ensemble already use (see
    src/antispoof_graph.cpp): the topology is embedded verbatim and the C++ engine
    replays it against the per-head ``l2d.`` / ``l3d.`` weights. The two heads are
    both 192x192 input, but differ in normalization and decode:

      * 2d106det: in-graph (x-127.5)*(1/128) via Sub/Mul leaf scalars, so the host
        feeds raw R,G,B (mean 0 / std 1); output ``fc1`` = 212 = 106*2.
      * 1k3d68 : NO in-graph norm (a bn_data layer instead), so the host applies
        (x-127.5)/128; output ``fc1`` = 3309 = 1103*3, of which the last 68 rows
        are the landmarks (insightface Landmark.get).

    Quantization (``--dtype``) follows the rest of the converter: only the large 2-D
    landmark-regression Gemm heads (``fc1_weight``) are quantizable; every conv / BN /
    bias / PReLU slope stays F32 (innermost ggml dim 3 or 1). f32 is bit-exact vs the
    onnxruntime reference; f16 is near-lossless (within the ~1px parity gate).
    ``arch="landmarks"``; embed_dim defaults to >0 in the loader so a head-only pack
    still passes the load sanity check.
    """
    w = gguf.GGUFWriter(output, "facedetect")
    w.add_string("facedetect.arch", "landmarks")
    # insightface Landmark feeds BOTH heads raw [0,255] R,G,B (input_mean 0 /
    # input_std 1): 2d106det carries (x-127.5)/128 in-graph via Sub/Mul leaf
    # scalars, and 1k3d68 absorbs the input scale into its leading `bn_data`
    # BatchNormalization (calibrated on raw pixels). So neither head normalizes
    # host-side. Output `fc1` = 212 = 106*2 (2D) / 3309 = 1103*3 (3D, last 68 rows
    # are the landmarks).
    heads = [
        # tag, onnx, num_points, dim, input_mean, input_std, prefix
        ("2d", d2, 106, 2, 0.0, 1.0, "l2d."),
        ("3d", d3, 68, 3, 0.0, 1.0, "l3d."),
    ]
    total = 0
    for tag, path, npts, dim, mean, std, prefix in heads:
        nodes, in_name, out_name = onnx_graph_spec(path)
        w.add_bool(f"facedetect.landmark.{tag}.present", True)
        w.add_uint32(f"facedetect.landmark.{tag}.input_size", 192)
        w.add_uint32(f"facedetect.landmark.{tag}.num_points", npts)
        w.add_uint32(f"facedetect.landmark.{tag}.dim", dim)
        w.add_float32(f"facedetect.landmark.{tag}.input_mean", mean)
        w.add_float32(f"facedetect.landmark.{tag}.input_std", std)
        w.add_array(f"facedetect.landmark.{tag}.graph", nodes)
        w.add_string(f"facedetect.landmark.{tag}.input", in_name)
        w.add_string(f"facedetect.landmark.{tag}.output", out_name)
        total += add_onnx_initializers(w, path, dtype, prefix=prefix)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"converter: wrote {total} tensors -> {output}", file=sys.stderr)


def resolve_pack(model):
    import os
    import glob

    # YuNet (OpenCV-Zoo, Apache-2.0): the commercial-friendly alternative to the
    # non-commercial insightface SCRFD detector. A small CNN (backbone + FPN neck +
    # per-stride cls/obj/bbox/kps heads) that the C++ engine runs metadata-driven
    # through the SAME ONNX-graph interpreter as the SCRFD det_2.5g path: the
    # converter embeds the topology as facedetect.detector.graph and the loader's
    # det_graph branch replays it. The static-input file ships [1,3,640,640], so the
    # detector input is 640 (plain-resized, NOT letterboxed - FaceDetectorYN's exact
    # blobFromImage preprocess). num_anchors=1, strides [8,16,32]. The decode (anchor
    # priors, score=sqrt(cls*obj), exp-bbox, kps) + NMS are YuNet-specific (yunet_detect).
    # SFace (the paired Apache recognizer) is Task 8.3; the detector-only GGUF still
    # loads (embed_dim>0).  arch="yunet", engine="onnx_direct".
    if "yunet" in os.path.basename(model.rstrip("/")).lower() or model.lower() == "yunet":
        if os.path.isfile(model) and model.endswith(".onnx"):
            det = model
        else:
            ybase = model if os.path.isdir(model) else "models/yunet"
            hits = glob.glob(os.path.join(ybase, "**", "*yunet*.onnx"), recursive=True)
            det = hits[0] if hits else None
        if not det:
            return None
        # Optional paired SFace recognizer (Task 8.3); detector-only is valid for 8.2.
        # The SFace ONNX may sit beside the detector or in a sibling models/sface dir.
        sdir = os.path.dirname(det)
        search = [sdir, os.path.join(os.path.dirname(sdir), "sface"),
                  os.path.join("models", "sface")]
        rec = None
        for d in search:
            shits = glob.glob(os.path.join(d, "**", "*[sS]face*.onnx"), recursive=True)
            if shits:
                rec = shits[0]
                break
        return dict(arch="yunet", detector="yunet", engine="onnx_direct",
                    det_input_size=640, det_strides=[8, 16, 32], det_num_anchors=1,
                    det_score_thresh=0.5, det_nms_thresh=0.3,
                    recognizer="sface", rec_is_mbf=False, rec_input_size=112,
                    embed_dim=128, verify_threshold=0.363, det_onnx=det, rec_onnx=rec,
                    genderage_onnx=None, antispoof=[])

    base = model if os.path.isdir(model) else \
        os.path.expanduser(f"~/.insightface/models/{model}")

    def find(pat):
        import glob
        hits = glob.glob(os.path.join(base, "**", pat), recursive=True)
        return hits[0] if hits else None

    # antelopev2 (insightface, NON-COMMERCIAL): the higher-accuracy pack - the
    # SCRFD-10G detector (scrfd_10g_bnkps, whose ONNX topology + graph-output names
    # are byte-identical to buffalo_l's det_10g) paired with ArcFace glint360k R100
    # (glintr100), a 100-layer IResNet, 512-d. Two routing choices versus buffalo_l:
    #   * The detector is NOT named "det_10g", so it takes the metadata-driven
    #     interpreter path (facedetect.detector.graph) rather than the hand-mapped
    #     det_10g C++ graph. det_10g and scrfd_10g_bnkps share the exact SCRFD
    #     topology the det_2.5g/det_500m interpreter already replays (FPN 2x-nearest
    #     Resize + GFL per-stride heads), so the graph path reproduces it.
    #   * The R100 recognizer is a DEEPER IResNet than the hand-mapped r50 block
    #     table; rather than extend that table, embed the R100 ONNX forward topology
    #     and let the C++ interpreter replay it (same path as MobileFaceNet). Its op
    #     set (Conv/PRelu/BatchNormalization/Add/Flatten/Gemm) is fully covered.
    #     rec_use_graph forces the recognizer.graph KV regardless of backbone tag.
    if os.path.basename(base.rstrip("/")) == "antelopev2" or model == "antelopev2":
        det = find("scrfd_*.onnx")
        rec = find("glintr100.onnx") or find("glint*.onnx")
        if not det or not rec:
            return None
        return dict(arch="scrfd+arcface", detector="scrfd", det_input_size=640,
                    det_strides=[8, 16, 32], det_num_anchors=2,
                    det_score_thresh=0.5, det_nms_thresh=0.4,
                    recognizer="arcface", rec_is_mbf=False, rec_use_graph=True,
                    rec_input_size=112, embed_dim=512, verify_threshold=0.35,
                    det_onnx=det, rec_onnx=rec,
                    genderage_onnx=find("genderage.onnx"),
                    antispoof=resolve_antispoof(base))

    # buffalo_sc shares buffalo_s's two sub-models exactly (det_500m + w600k_mbf
    # MobileFaceNet) - it is the detection+recognition-only "small compact" pack, no
    # genderage/landmark heads - so it rides the same SCRFD-interpreter + MobileFaceNet
    # graph path with nothing pack-specific beyond the name gate below.
    if os.path.basename(base.rstrip("/")) in ("buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc") \
            or model in ("buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc"):
        det = find("det_*.onnx")
        rec = find("w600k_*.onnx")
        if not det or not rec:
            return None
        # buffalo_s ships the MobileFaceNet recognizer (w600k_mbf): a DIFFERENT
        # backbone (depthwise-separable, Gemm head) from the IResNet50 w600k_r50
        # used by buffalo_l/buffalo_m. The C++ ArcFace IResNet graph is hand-mapped
        # to w600k_r50's verbatim node names; MobileFaceNet does not share them, so
        # the converter embeds its ONNX forward topology and the C++ runs it through
        # the shared metadata-driven graph interpreter (the same path as the
        # MiniFASNet anti-spoof ensemble). The recognizer is still ArcFace-trained
        # (cosine embedding), only the backbone differs - hence "arcface_mbf".
        is_mbf = "mbf" in os.path.basename(rec).lower()
        return dict(arch="scrfd+arcface", detector="scrfd", det_input_size=640,
                    det_strides=[8, 16, 32], det_num_anchors=2,
                    det_score_thresh=0.5, det_nms_thresh=0.4,
                    recognizer=("arcface_mbf" if is_mbf else "arcface"),
                    rec_is_mbf=is_mbf, rec_input_size=112, embed_dim=512,
                    verify_threshold=0.35, det_onnx=det, rec_onnx=rec,
                    genderage_onnx=find("genderage.onnx"),
                    antispoof=resolve_antispoof(base))
    return None


def main():
    ap = argparse.ArgumentParser(description="insightface pack -> GGUF converter")
    ap.add_argument("--model", required=True,
                    help="pack name (buffalo_l/buffalo_m/buffalo_s/yunet+sface) "
                         "or a path to an insightface model directory")
    ap.add_argument("--output", required=True, help="output .gguf path")
    ap.add_argument("--dtype", default="f16",
                    choices=["f32", "f16", "q8_0", "q4_0", "q4_k"],
                    help="quantization for the allowlisted large weights")
    args = ap.parse_args()

    # Engine-only dense-landmark heads (2d106det + 1k3d68). Not a detector+recognizer
    # pack: a standalone GGUF with the two landmark regressors insightface buffalo_l
    # ships. NO LocalAI API consumes dense landmarks yet (engine-level capability).
    if args.model == "landmarks" or \
            (args.model.endswith("landmarks") and resolve_landmarks(args.model)):
        lmk = resolve_landmarks(args.model)
        if lmk is None:
            print(f"converter: landmark heads (2d106det/1k3d68) not found for "
                  f"{args.model!r}", file=sys.stderr)
            print("FACEDETECT_MODEL_UNAVAILABLE", file=sys.stderr)
            sys.exit(2)
        write_landmarks_gguf(lmk[0], lmk[1], args.output, args.dtype)
        sys.exit(0)

    pack = resolve_pack(args.model)
    if pack is None:
        print(f"converter: unknown/unavailable pack {args.model!r}", file=sys.stderr)
        print("FACEDETECT_MODEL_UNAVAILABLE", file=sys.stderr)
        sys.exit(2)

    w = gguf.GGUFWriter(args.output, "facedetect")
    write_config_kv(w, pack)
    total = add_onnx_initializers(w, pack["det_onnx"], args.dtype, prefix="det.")
    # det_2.5g (buffalo_m) / det_500m (buffalo_s) have different backbones + node
    # numbering than the hand-mapped det_10g (buffalo_l) graph, so embed their ONNX
    # topology and let the C++ interpreter replay it metadata-driven (same approach
    # as the MobileFaceNet recognizer). det_10g stays on its hand-mapped path: no
    # graph KV is emitted for it, so the C++ guard `if (!det_graph.empty())` keeps
    # buffalo_l untouched.
    import os as _os
    if "det_10g" not in _os.path.basename(pack["det_onnx"]):
        dnodes, din_name, dout_names = detector_graph_spec(pack["det_onnx"])
        w.add_array("facedetect.detector.graph", dnodes)
        w.add_string("facedetect.detector.input", din_name)
        w.add_array("facedetect.detector.outputs", dout_names)
    if pack.get("rec_onnx"):
        total += add_onnx_initializers(w, pack["rec_onnx"], args.dtype, prefix="rec.")
    if pack.get("rec_is_mbf") or pack.get("rec_use_graph") or pack.get("recognizer") == "sface":
        # Embed the recognizer forward topology so the C++ engine rebuilds the exact
        # graph from metadata (no per-arch C++). Three packs take this path: the
        # MobileFaceNet (w600k_mbf) ArcFace recognizer, the antelopev2 glint360k R100
        # IResNet (rec_use_graph - a deeper trunk than the hand-mapped r50), and the
        # Apache SFace recognizer. The interpreter resolves each node input as a prior activation
        # or a `rec.`-prefixed weight, seeds the graph input under its real ONNX name
        # (SFace: "data"), and reads the output by name (SFace: "fc1"). SFace carries
        # its (x-127.5)/128 normalization in-graph (Sub/Mul) and per-BN epsilons.
        nodes, in_name, out_name = onnx_graph_spec(pack["rec_onnx"])
        w.add_array("facedetect.recognizer.graph", nodes)
        w.add_string("facedetect.recognizer.input", in_name)
        w.add_string("facedetect.recognizer.output", out_name)
    if pack.get("genderage_onnx"):
        total += add_onnx_initializers(w, pack["genderage_onnx"], args.dtype,
                                       skip_q=True, prefix="ga.")
    for i, (asp, _scale) in enumerate(pack.get("antispoof", [])):
        total += add_onnx_initializers(w, asp, args.dtype, skip_q=True,
                                       prefix=f"as{i}.")
        # Embed the node topology so the C++ engine rebuilds the exact forward
        # graph (the GGUF holds only the per-model weights).
        nodes, _in_name, out_name = onnx_graph_spec(asp)
        w.add_array(f"facedetect.antispoof.{i}.graph", nodes)
        w.add_string(f"facedetect.antispoof.{i}.output", out_name)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"converter: wrote {total} tensors -> {args.output}", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
