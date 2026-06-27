#!/usr/bin/env python3
"""Dump insightface reference intermediates to ``baseline.gguf`` for C++ parity.

Runs the reference insightface pipeline on an input image and captures each
stage's output so the C++ port can be diffed stage-by-stage (see ``docs/parity.md``):

  * ``det_scores``    SCRFD per-stride classification scores (post-sigmoid)
  * ``det_boxes``     decoded boxes ``[N,4]`` after distance->box + NMS
  * ``det_landmarks`` decoded 5-point landmarks ``[N,5,2]``
  * ``aligned_crop``  the 112x112 RGB norm_crop of the primary face ``[112,112,3]``
  * ``embedding``     the L2-normalized ArcFace embedding ``[512]``
  * ``age`` / ``gender`` (when the genderage head is present)

The embedding parity gate is cosine >= 0.9999 and max-abs-diff <= 1e-3 against
this dump; the detection gate compares boxes + landmarks against ``det_*``.

This is a skeleton: it documents the captured tensors and guards missing deps,
but the actual capture (forward hooks on the insightface ``FaceAnalysis`` app /
the individual ONNX sessions) is a TODO.
"""
import argparse
import sys

try:
    import gguf  # noqa: F401
    import numpy as np  # noqa: F401
except ImportError as e:  # pragma: no cover - env guard
    print(f"gen_baseline: missing dependency: {e}", file=sys.stderr)
    print("FACEDETECT_BASELINE_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)

def gen_yunet_baseline(args):
    """Dump the OpenCV-Zoo YuNet reference (Apache-2.0 detector) for C++ parity.

    YuNet's static-input ONNX is [1,3,640,640], so the reference runs at a fixed
    640x640 input. We reproduce ``cv2.FaceDetectorYN`` exactly:

      * PLAIN resize the BGR source to 640x640 (cv2.resize, INTER_LINEAR) - NOT a
        letterbox; FaceDetectorYN's ``blobFromImage(img, 1.0, (640,640))`` distorts
        aspect ratio and the decode is in that 640 space.
      * blob = raw BGR planes, NO mean / NO std / NO swapRB.
      * Raw heads via onnxruntime (12 tensors: cls/obj/bbox/kps x strides 8/16/32),
        post-sigmoid for cls/obj. Dumped as ``yunet_out_0..11`` in graph-output order
        for the isolated C++ conv-graph gate (fed the EXACT 640 pixels, like SCRFD).
      * Decoded boxes/landmarks/scores via ``cv2.FaceDetectorYN`` (score=sqrt(cls*obj),
        cx=(c+dx)*s, w=exp(dw)*s, kps=(k+c)*s, then cv2.dnn NMS). Dumped in 640 space;
        ``yunet_scale=[origW/640, origH/640]`` maps them back to source.

    Captured tensors (docs/parity.md, tests/test_yunet.cpp):
      yunet_resized_rgb [640,640,3]  yunet_input_blob [1,3,640,640]
      yunet_out_0..11                yunet_scale [2]
      yunet_boxes [N,4] (x1,y1,x2,y2, 640 space)  yunet_scores [N]
      yunet_landmarks [N,5,2] (640 space)
    """
    import os
    import glob
    import numpy as np, cv2, gguf, onnxruntime as ort

    onnx_path = args.model
    if not (os.path.isfile(onnx_path) and onnx_path.endswith(".onnx")):
        ybase = args.model if os.path.isdir(args.model) else "models/yunet"
        hits = glob.glob(os.path.join(ybase, "**", "*yunet*.onnx"), recursive=True)
        if not hits:
            print(f"gen_baseline: no YuNet onnx under {ybase!r}", file=sys.stderr)
            print("FACEDETECT_MODEL_UNAVAILABLE", file=sys.stderr)
            sys.exit(2)
        onnx_path = hits[0]

    img = cv2.imread(args.image)              # BGR uint8
    if img is None:
        print(f"gen_baseline: cannot read {args.image}", file=sys.stderr); sys.exit(2)
    oh, ow = img.shape[:2]
    S = 640
    resized = cv2.resize(img, (S, S))         # BGR uint8, plain resize (distorts)
    sx, sy = ow / float(S), oh / float(S)

    # Raw heads (post-sigmoid cls/obj) via onnxruntime on the exact blob.
    blob = cv2.dnn.blobFromImage(resized, 1.0, (S, S), (0, 0, 0), swapRB=False)  # BGR raw
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onames = [o.name for o in sess.get_outputs()]
    raw_outs = sess.run(onames, {sess.get_inputs()[0].name: blob})
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Decoded golden via cv2.FaceDetectorYN (faces = [x,y,w,h, 5x(lx,ly), score]).
    det = cv2.FaceDetectorYN.create(onnx_path, "", (S, S),
                                    score_threshold=0.5, nms_threshold=0.3, top_k=5000)
    _n, faces = det.detect(resized)
    if faces is None:
        faces = np.zeros((0, 15), dtype=np.float32)
    faces = faces[np.argsort(-faces[:, 14])]   # descending score
    boxes = np.zeros((len(faces), 4), dtype=np.float32)
    lmks = np.zeros((len(faces), 5, 2), dtype=np.float32)
    scores = np.zeros((len(faces),), dtype=np.float32)
    for i, f in enumerate(faces):
        x, y, w, h = f[0], f[1], f[2], f[3]
        boxes[i] = [x, y, x + w, y + h]        # x1,y1,x2,y2 in 640 space
        lmks[i] = np.array(f[4:14], dtype=np.float32).reshape(5, 2)
        scores[i] = f[14]

    w = gguf.GGUFWriter(args.output, "facedetect-baseline")
    w.add_tensor("yunet_resized_rgb", np.ascontiguousarray(resized_rgb, dtype=np.float32))
    w.add_tensor("yunet_input_blob", np.ascontiguousarray(blob, dtype=np.float32))
    w.add_tensor("yunet_scale", np.array([sx, sy], dtype=np.float32))
    for i, t in enumerate(raw_outs):
        w.add_tensor(f"yunet_out_{i}", np.ascontiguousarray(t, dtype=np.float32))
    w.add_tensor("yunet_boxes", np.ascontiguousarray(boxes, dtype=np.float32))
    w.add_tensor("yunet_scores", np.ascontiguousarray(scores, dtype=np.float32))
    w.add_tensor("yunet_landmarks", np.ascontiguousarray(lmks, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file()
    w.write_tensors_to_file(); w.close()
    print(f"gen_baseline: wrote {args.output} ({len(faces)} YuNet faces)", file=sys.stderr)


def gen_sface_baseline(args):
    """Dump the OpenCV SFace recognizer (Apache-2.0, 128-d) reference for parity.

    SFace pairs with the YuNet detector. We reproduce ``cv2.FaceRecognizerSF``
    exactly: detect with ``cv2.FaceDetectorYN``, pick the primary (largest-area)
    face, ``alignCrop`` it to a 112x112 crop (SFace's own similarity warp - the
    SAME 5-point arcface_dst template insightface uses, verified to match
    ``norm_crop`` within ~2 LSB), and ``feature`` it to the RAW 128-d embedding
    (NOT L2-normalized; the cosine match normalizes). SFace carries its
    ``(x-127.5)/128`` input normalization IN-graph, so ``feature`` feeds the raw
    BGR crop (default ``blobFromImage``: scale 1, mean 0, no swapRB).

    Captured tensors (tests/test_sface_embedding.cpp):
      sface_src_rgb [H,W,3]        the RGB source (so the C++ align gate warps the
                                   SAME pixels, isolating the warp from JPEG decode)
      sface_landmarks [5,2]        the primary face's 5 landmarks (source space)
      sface_aligned_crop [112,112,3] RGB  cv2 FaceRecognizerSF.alignCrop output
      sface_embedding [128]        raw cv2 FaceRecognizerSF.feature output
    """
    import os
    import glob
    import numpy as np, cv2, gguf

    rec_path = args.model
    if not (os.path.isfile(rec_path) and rec_path.endswith(".onnx")):
        sbase = args.model if os.path.isdir(args.model) else "models/sface"
        hits = glob.glob(os.path.join(sbase, "**", "*[sS]face*.onnx"), recursive=True)
        if not hits:
            print(f"gen_baseline: no SFace onnx under {sbase!r}", file=sys.stderr)
            print("FACEDETECT_MODEL_UNAVAILABLE", file=sys.stderr)
            sys.exit(2)
        rec_path = hits[0]

    det_path = os.environ.get("FACEDETECT_DET_ONNX", "")
    if not (det_path and os.path.isfile(det_path)):
        dhits = glob.glob(os.path.join("models", "yunet", "**", "*yunet*.onnx"),
                          recursive=True)
        if not dhits:
            print("gen_baseline: no YuNet detector onnx for SFace baseline",
                  file=sys.stderr)
            print("FACEDETECT_MODEL_UNAVAILABLE", file=sys.stderr)
            sys.exit(2)
        det_path = dhits[0]

    img = cv2.imread(args.image)              # BGR uint8
    if img is None:
        print(f"gen_baseline: cannot read {args.image}", file=sys.stderr); sys.exit(2)
    oh, ow = img.shape[:2]

    det = cv2.FaceDetectorYN.create(det_path, "", (ow, oh),
                                    score_threshold=0.5, nms_threshold=0.3, top_k=5000)
    _n, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        print(f"gen_baseline: no face detected in {args.image}", file=sys.stderr)
        sys.exit(2)
    # Primary = largest-area box (faces[:, 2:4] are w,h). Matches the C++
    # Model::embed primary selection.
    areas = faces[:, 2] * faces[:, 3]
    primary = faces[int(np.argmax(areas))]
    landmarks = np.array(primary[4:14], dtype=np.float32).reshape(5, 2)  # source space

    rec = cv2.FaceRecognizerSF.create(rec_path, "")
    aligned = rec.alignCrop(img, primary)         # 112x112 BGR uint8
    embedding = rec.feature(aligned).reshape(-1).astype(np.float32)  # raw 128-d
    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
    src_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    w = gguf.GGUFWriter(args.output, "facedetect-baseline")
    w.add_tensor("sface_src_rgb", np.ascontiguousarray(src_rgb, dtype=np.float32))
    w.add_tensor("sface_landmarks", np.ascontiguousarray(landmarks, dtype=np.float32))
    w.add_tensor("sface_aligned_crop",
                 np.ascontiguousarray(aligned_rgb, dtype=np.float32))
    w.add_tensor("sface_embedding", np.ascontiguousarray(embedding, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file()
    w.write_tensors_to_file(); w.close()
    print(f"gen_baseline: wrote {args.output} (SFace 128-d, ||emb||={np.linalg.norm(embedding):.4f})",
          file=sys.stderr)


try:
    import insightface  # noqa: F401
except ImportError as e:  # pragma: no cover - env guard
    insightface = None    # YuNet path does not need insightface; buffalo path guards below.


def main():
    import os
    ap = argparse.ArgumentParser(description="dump insightface reference baseline")
    ap.add_argument("--model", default="buffalo_l")
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # SFace (OpenCV-Zoo, Apache) recognizer: self-contained reference (cv2 only),
    # paired with the YuNet detector. Checked BEFORE the yunet branch so an SFace
    # model path is not mis-routed.
    if "sface" in os.path.basename(args.model.rstrip("/")).lower() \
            or args.model.lower() == "sface":
        gen_sface_baseline(args)
        return

    # YuNet (OpenCV-Zoo, Apache) detector: a self-contained reference that needs
    # only onnxruntime + cv2 (no insightface FaceAnalysis pack).
    if "yunet" in os.path.basename(args.model.rstrip("/")).lower() \
            or args.model.lower() == "yunet":
        gen_yunet_baseline(args)
        return
    if insightface is None:
        print("gen_baseline: missing dependency 'insightface'", file=sys.stderr)
        print("FACEDETECT_BASELINE_DEPS_MISSING", file=sys.stderr)
        sys.exit(2)

    import numpy as np, cv2, gguf
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=args.model,
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)
    img = cv2.imread(args.image)              # BGR uint8
    if img is None:
        print(f"gen_baseline: cannot read {args.image}", file=sys.stderr); sys.exit(2)

    det = app.models["detection"]
    # Capture raw per-stride sigmoid scores by re-running forward at det_scale.
    bboxes, kpss = det.detect(img, max_num=0, metric="default")  # [N,5], [N,5,2]

    # --- Raw SCRFD head outputs (Task 3.1 graph gate) ------------------------
    # Reproduce insightface SCRFD.detect's letterbox EXACTLY, then run the raw
    # ONNX session on that blob and dump the 9 per-stride head tensors in the
    # ONNX graph's output order (scrfd_out_0..8 ==
    #   [score s8, score s16, score s32, bbox s8/16/32, kps s8/16/32]).
    # The C++ conv graph is gated component-by-component against these BEFORE the
    # anchor decode (Task 3.2). We ALSO dump the exact letterboxed 640x640 RGB
    # pixels (scrfd_letterbox_rgb) and the normalized input blob
    # (scrfd_input_blob) so the C++ graph gate can feed the SAME pixels the
    # reference consumed, isolating the conv graph from JPEG-decoder / resize
    # differences (the same precedent as the align gate's src_image dump: stb vs
    # libjpeg decode drifts 1-3 LSB, which propagates to ~1e-1 in the raw bbox
    # heads, swamping a 1e-3 graph gate).
    input_size = (int(det.input_size[0]), int(det.input_size[1])) \
        if getattr(det, "input_size", None) else (640, 640)
    im_ratio = float(img.shape[0]) / img.shape[1]
    model_ratio = float(input_size[1]) / input_size[0]
    if im_ratio > model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
    det_scale = float(new_height) / img.shape[0]
    resized_img = cv2.resize(img, (new_width, new_height))           # BGR uint8
    det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized_img                # BGR uint8
    input_mean = float(getattr(det, "input_mean", 127.5))
    input_std = float(getattr(det, "input_std", 128.0))
    blob = cv2.dnn.blobFromImage(
        det_img, 1.0 / input_std, input_size,
        (input_mean, input_mean, input_mean), swapRB=True)            # [1,3,H,W] R,G,B
    raw_outs = det.session.run(det.output_names,
                               {det.input_name: blob})                # 9 tensors
    det_img_rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB).astype(np.float32)

    faces = app.get(img)                       # full pipeline (det+align+rec+genderage)
    faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                   reverse=True)
    primary = faces[0]
    from insightface.utils import face_align
    aligned = face_align.norm_crop(img, primary.kps, image_size=112)  # 112x112 BGR
    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
    emb = primary.normed_embedding.astype(np.float32)  # L2-normalized 512-d

    # --- ArcFace stem intermediate (Task 4.1 graph gate) --------------------
    # Run the recognizer ONNX on the SAME aligned crop, but with the IResNet
    # stem's PReLU output (ONNX value '479') promoted to a graph output, so the
    # C++ stem (first conv + folded BN + PReLU) can be gated component-wise
    # BEFORE the residual trunk. The blob is the recognizer's exact input
    # (blobFromImage on the BGR crop, swapRB=True -> R,G,B planes); we dump it as
    # arcface_input_blob so the C++ to_blob(swap_rb=false) on the RGB aligned_crop
    # can be gated against the identical R,G,B planes.
    import onnxruntime, onnx
    from onnx import helper
    rec = app.models["recognition"]
    rec_mean = float(getattr(rec, "input_mean", 127.5))
    rec_std = float(getattr(rec, "input_std", 127.5))
    rec_size = tuple(int(s) for s in getattr(rec, "input_size", (112, 112)))
    arc_blob = cv2.dnn.blobFromImage(
        aligned, 1.0 / rec_std, rec_size, (rec_mean, rec_mean, rec_mean),
        swapRB=True)                                          # [1,3,112,112] R,G,B
    rec_model = onnx.load(rec.model_file)
    stem_out_name = next(n.output[0] for n in rec_model.graph.node
                         if n.op_type == "PRelu")            # first PReLU == stem '479'
    # First IR residual block (layer1.0) output: the first Add node fuses conv2's
    # bn3-folded output with the downsample shortcut (ONNX value '489'). Promoting
    # it lets the C++ IR block (bn1 -> conv1 -> prelu -> conv2(stride) + downsample)
    # be gated component-wise against the reference BEFORE the rest of the trunk.
    block0_out_name = next(n.output[0] for n in rec_model.graph.node
                           if n.op_type == "Add")            # first residual Add == '489'
    for nm in (stem_out_name, block0_out_name):
        rec_model.graph.output.append(helper.ValueInfoProto(name=nm))
    rec_sess = onnxruntime.InferenceSession(rec_model.SerializeToString(),
                                            providers=["CPUExecutionProvider"])
    arc_stem_out, arc_block0_out = rec_sess.run(
        [stem_out_name, block0_out_name],
        {rec.input_name: arc_blob})                          # [1,64,112,112], [1,64,56,56]

    # The exact RGB pixels the reference fed to norm_crop. Dumping these lets the
    # C++ alignment parity gate (test_align) warp the SAME source pixels as the
    # golden crop, isolating the umeyama transform + bilinear warp from JPEG
    # decoder differences (stb_image vs OpenCV libjpeg differ by 1-3 LSB, which
    # would otherwise swamp the <=1 aligned-crop gate). Stored RGB to match
    # aligned_crop's channel order.
    src_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)  # [H,W,3]

    w = gguf.GGUFWriter(args.output, "facedetect-baseline")
    w.add_tensor("src_image", np.ascontiguousarray(src_rgb, dtype=np.float32))
    # The exact 5 landmarks (insightface order) of the primary face that produced
    # aligned_crop. det_landmarks is in detection-score order; primary is the
    # largest-by-area face, so for multi-face images row 0 of det_landmarks is NOT
    # the primary. The align gate reconstructs norm_crop from THESE landmarks.
    w.add_tensor("primary_landmarks",
                 np.ascontiguousarray(primary.kps, dtype=np.float32))  # [5,2]
    w.add_tensor("det_boxes", np.ascontiguousarray(bboxes[:, :4], dtype=np.float32))
    w.add_tensor("det_scores", np.ascontiguousarray(bboxes[:, 4], dtype=np.float32))
    w.add_tensor("det_landmarks", np.ascontiguousarray(kpss, dtype=np.float32))
    w.add_tensor("aligned_crop", np.ascontiguousarray(aligned_rgb, dtype=np.float32))
    w.add_tensor("embedding", np.ascontiguousarray(emb, dtype=np.float32))
    # ArcFace stem gate (Task 4.1): recognizer input blob + stem PReLU output.
    w.add_tensor("arcface_input_blob", np.ascontiguousarray(arc_blob, dtype=np.float32))
    w.add_tensor("arcface_stem_out", np.ascontiguousarray(arc_stem_out, dtype=np.float32))
    # ArcFace first IR residual block output (Task 4.2 graph gate), ONNX '489'.
    w.add_tensor("arcface_stage1_block0_out",
                 np.ascontiguousarray(arc_block0_out, dtype=np.float32))
    # Raw SCRFD head outputs + the exact letterbox inputs (Task 3.1 graph gate).
    w.add_tensor("scrfd_letterbox_rgb",
                 np.ascontiguousarray(det_img_rgb, dtype=np.float32))   # [640,640,3] RGB
    w.add_tensor("scrfd_input_blob",
                 np.ascontiguousarray(blob, dtype=np.float32))          # [1,3,640,640]
    w.add_tensor("scrfd_det_scale", np.array([det_scale], dtype=np.float32))
    for i, t in enumerate(raw_outs):
        w.add_tensor(f"scrfd_out_{i}", np.ascontiguousarray(t, dtype=np.float32))
    if getattr(primary, "age", None) is not None:
        w.add_tensor("age", np.array([float(primary.age)], dtype=np.float32))
        w.add_tensor("gender", np.array([float(primary.gender)], dtype=np.float32))
        # GenderAge head gate (Task 6.1). Replicate insightface Attribute.get on the
        # primary face's bbox: scale-about-center crop (box expanded 1.5x, fit to
        # 96), then the raw genderage ONNX output `fc1` = [g0, g1, age_raw]
        # (gender=argmax(out[:2]), age=round(out[2]*100)). Dump BOTH the exact 96x96
        # RGB crop (genderage_crop) and the raw 3-float output (genderage_out) so the
        # C++ graph can be gated on the reference's exact crop pixels (isolated from
        # the host-side bbox warp), mirroring the ArcFace aligned_crop isolated gate.
        ga = app.models.get("genderage")
        if ga is not None:
            gbbox = primary.bbox
            gw, gh = (gbbox[2] - gbbox[0]), (gbbox[3] - gbbox[1])
            gcenter = ((gbbox[2] + gbbox[0]) / 2, (gbbox[3] + gbbox[1]) / 2)
            gscale = ga.input_size[0] / (max(gw, gh) * 1.5)
            gaimg, _gM = face_align.transform(img, gcenter, ga.input_size[0], gscale, 0)
            gsize = tuple(gaimg.shape[0:2][::-1])
            gblob = cv2.dnn.blobFromImage(
                gaimg, 1.0 / ga.input_std, gsize,
                (ga.input_mean,) * 3, swapRB=True)            # [1,3,96,96] R,G,B
            gpred = ga.session.run(ga.output_names,
                                   {ga.input_name: gblob})[0][0]   # 3 floats
            gaimg_rgb = cv2.cvtColor(gaimg, cv2.COLOR_BGR2RGB).astype(np.float32)
            w.add_tensor("genderage_crop",
                         np.ascontiguousarray(gaimg_rgb, dtype=np.float32))  # [96,96,3]
            w.add_tensor("genderage_out",
                         np.ascontiguousarray(gpred, dtype=np.float32))      # [3]

    # --- MiniFASNet anti-spoof ensemble gate (Task 6.2) ----------------------
    # Run the Silent-Face MiniFASNet ensemble on the PRIMARY face's detection box
    # exactly as backend/python/insightface/engines.py Antispoofer.predict does:
    # per model, an integer scale-expanded crop of the BGR source resized to 80x80,
    # fed raw (no /255, no mean/std, no swapRB), 3 logits -> softmax; average index
    # 1 ("real") across members. Dump per-model logits + the averaged real prob and
    # the exact 80x80 crops (as RGB) so the C++ graph is gated on identical pixels.
    # as0 = MiniFASNetV2 @ scale 2.7, as1 = MiniFASNetV1SE @ scale 4.0.
    import glob as _glob
    _asdir = os.path.dirname(rec.model_file)
    _members = []
    for _name, _scale in (("MiniFASNetV2", 2.7), ("MiniFASNetV1SE", 4.0)):
        _hits = _glob.glob(os.path.join(_asdir, "**", _name + "*.onnx"), recursive=True)
        if _hits:
            _members.append((_hits[0], _scale))
    if _members:
        def _crop_face(im, bbox, scale):
            sh, sw = im.shape[:2]
            bx1, by1, bx2, by2 = bbox
            bw = max(1.0, bx2 - bx1); bh = max(1.0, by2 - by1)
            scale = min((sh - 1) / bh, (sw - 1) / bw, scale)
            nw = bw * scale; nh = bh * scale
            ccx = bx1 + bw / 2.0; ccy = by1 + bh / 2.0
            kx1 = max(0, int(ccx - nw / 2.0)); ky1 = max(0, int(ccy - nh / 2.0))
            kx2 = min(sw - 1, int(ccx + nw / 2.0)); ky2 = min(sh - 1, int(ccy + nh / 2.0))
            cr = im[ky1:ky2 + 1, kx1:kx2 + 1]
            if cr.size == 0:
                cr = im
            return cv2.resize(cr, (80, 80))

        accum = np.zeros((3,), dtype=np.float64)
        for mi, (mpath, mscale) in enumerate(_members):
            sess = onnxruntime.InferenceSession(mpath, providers=["CPUExecutionProvider"])
            iname = sess.get_inputs()[0].name; oname = sess.get_outputs()[0].name
            crop = _crop_face(img, primary.bbox, mscale).astype(np.float32)  # BGR 80x80
            tensor = np.transpose(crop, (2, 0, 1))[np.newaxis, ...]          # NCHW B,G,R
            logits = sess.run([oname], {iname: tensor})[0][0]                # [3]
            e = np.exp(logits - np.max(logits)); sm = e / e.sum()
            accum += sm
            crop_rgb = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32)
            w.add_tensor(f"antispoof_crop_{mi}",
                         np.ascontiguousarray(crop_rgb, dtype=np.float32))   # [80,80,3] RGB
            w.add_tensor(f"antispoof_logits_{mi}",
                         np.ascontiguousarray(logits, dtype=np.float32))     # [3]
        accum /= float(len(_members))
        w.add_tensor("antispoof_real_prob",
                     np.array([accum[1]], dtype=np.float32))                 # averaged softmax[1]

    # --- Dense-landmark heads gate (2d106det 106-pt 2D / 1k3d68 68-pt 3D) --------
    # ENGINE-LEVEL capability only: no LocalAI RPC consumes dense landmarks yet. We
    # replicate insightface Landmark.get per head: scale-about-center crop (box
    # expanded 1.5x, fit to the head's 192 input), blobFromImage with the head's own
    # input_mean/std. BOTH heads feed RAW [0,255] input (reference input_mean 0 /
    # input_std 1): 2d106det carries (x-127.5)/128 as in-graph Sub/Mul leaves, and
    # 1k3d68 absorbs the input scale into its leading bn_data BatchNormalization
    # (do NOT "align" this to mean 127.5/std 128 -- that reintroduces the 160px bug).
    # We dump raw `fc1` output, and the final image-space
    # points the full app.get produced. We ALSO dump the forward affine M so the C++
    # gate maps crop-space points back with the IDENTICAL transform, isolating the
    # parity check to the network + decode-scaling.
    for _tag, _key, _npts, _dim in (("2d", "landmark_2d_106", 106, 2),
                                    ("3d", "landmark_3d_68", 68, 3)):
        lm = app.models.get(_key)
        if lm is None or _key not in primary:
            continue
        lbbox = primary.bbox
        lw, lh = (lbbox[2] - lbbox[0]), (lbbox[3] - lbbox[1])
        lcenter = ((lbbox[2] + lbbox[0]) / 2, (lbbox[3] + lbbox[1]) / 2)
        lscale = lm.input_size[0] / (max(lw, lh) * 1.5)
        laimg, lM = face_align.transform(img, lcenter, lm.input_size[0], lscale, 0)
        lsize = tuple(laimg.shape[0:2][::-1])
        lblob = cv2.dnn.blobFromImage(
            laimg, 1.0 / lm.input_std, lsize,
            (lm.input_mean,) * 3, swapRB=True)
        lraw = lm.session.run(lm.output_names, {lm.input_name: lblob})[0][0]  # 212 / 3309
        laimg_rgb = cv2.cvtColor(laimg, cv2.COLOR_BGR2RGB).astype(np.float32)
        w.add_tensor(f"landmark_{_tag}_crop",
                     np.ascontiguousarray(laimg_rgb, dtype=np.float32))       # [192,192,3]
        w.add_tensor(f"landmark_{_tag}_raw",
                     np.ascontiguousarray(lraw, dtype=np.float32))            # [212]/[3309]
        w.add_tensor(f"landmark_{_tag}_M",
                     np.ascontiguousarray(lM, dtype=np.float32))              # [2,3] forward affine
        w.add_tensor(f"landmark_{_tag}_points",
                     np.ascontiguousarray(primary[_key], dtype=np.float32))   # image-space [N,dim]
        # Primary face bbox (x1,y1,x2,y2) that drove the crop -> lets the C++ gate
        # exercise fd::landmark_crop end-to-end (bbox -> M / crop), not just the
        # reference's own dumped M.
        w.add_tensor(f"landmark_{_tag}_bbox",
                     np.ascontiguousarray(lbbox, dtype=np.float32))            # [4]

    w.write_header_to_file(); w.write_kv_data_to_file()
    w.write_tensors_to_file(); w.close()
    print(f"gen_baseline: wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
