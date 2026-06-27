# YuNet (+ SFace) tensor-name manifest (verbatim ONNX initializers)

> Converter contract for the OpenCV-Zoo **Apache-2.0** face pack, the
> commercial-friendly alternative to the non-commercial insightface
> SCRFD+ArcFace pack. `scripts/convert_facedetect_to_gguf.py`
> (`resolve_pack` `yunet` branch) writes every initializer below into
> the GGUF under the `det.` namespace prefix, keeping the ONNX name
> otherwise **verbatim**, and embeds the forward topology as the
> `facedetect.detector.graph` KV array. The C++ engine
> (`src/yunet_graph.cpp`) replays that graph through the SHARED
> metadata-driven ONNX interpreter (`run_onnx_graph_multi`, the same
> path as the SCRFD det_2.5g detector) - YuNet adds **no new ggml op**.

## Provenance

- Detector: `face_detection_yunet_2023mar.onnx`, OpenCV Zoo
  (`opencv/opencv_zoo`, `models/face_detection_yunet/`).
  - License: **Apache-2.0** (commercial-friendly).
  - Source URL (Git-LFS):
    `https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`
  - Size: 232589 bytes. SHA256:
    `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
- Recognizer: **SFace** (`face_recognition_sface_2021dec.onnx`, OpenCV Zoo,
  `models/face_recognition_sface/`).
  - License: **Apache-2.0** (commercial-friendly).
  - Source URL (Git-LFS):
    `https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx`
  - Size: 38696353 bytes. SHA256:
    `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`.

## Architecture (recognizer, Task 8.3)

- Static input `data` = `[1,3,112,112]` (NCHW), output `fc1` = `[1,128]` (the
  RAW, **un-normalized** feature; OpenCV's cosine match L2-normalizes).
- MobileFaceNet-style depthwise-separable CNN: 14 `conv_<n>` blocks (a 1x1
  pointwise conv then a 3x3 depthwise conv, each `Conv -> BatchNormalization ->
  PRelu`), a trailing `bn1` spatial BatchNorm, `Dropout` (identity at eval),
  `Flatten`, a `pre_fc1` Gemm (`transB=1`, `[50176,128]`) and a final `fc1`
  BatchNorm over the 128 features.
- Input normalization is **in-graph**: the leading `Sub`(`scalar_op1`=127.5) then
  `Mul`(`scalar_op2`=0.0078125 = 1/128) reproduce `(x-127.5)/128`, so the blob is
  RAW pixels. `cv2.FaceRecognizerSF.feature` uses `blobFromImage(swapRB=true)` on
  its BGR `alignCrop`, so the net sees R,G,B - the C++ feeds the already-RGB crop
  with no channel swap.
- Each `BatchNormalization` carries its **own epsilon** (1e-3 on the conv BNs,
  2e-5 on `bn1`/`fc1`); the converter emits it as `e=` in the graph spec.
- `alignCrop` uses the **same** 5-point `arcface_dst` template as insightface's
  `norm_crop` (verified to agree within ~2 LSB), so no separate aligner is needed.

The C++ engine (`src/sface_graph.cpp`) replays this topology (embedded as the
`facedetect.recognizer.graph` KV, `rec.` namespace prefix) through the SHARED
`run_onnx_graph` interpreter - the same path as the MobileFaceNet (w600k_mbf)
recognizer. Task 8.3 extends that interpreter with the `Sub` / `Dropout` ops, a
spatial (4-D) BatchNorm, and per-node epsilon; it adds **no new ggml op**. Gated
vs `cv2.FaceRecognizerSF` (`alignCrop` + raw `feature`) in
`tests/test_sface_embedding.cpp`: aligned crop <= 2 LSB, raw feature max\|d\|
<= 1e-3, L2-normalized cosine >= 0.9999 (measured: cosine 1.000000, max\|d\| 2.6e-6
at f32). Reproduce with `scripts/run_sface_parity.sh`.

> The 174 SFace initializers (incl. `rec.scalar_op1` / `rec.scalar_op2`) are
> written verbatim under the `rec.` prefix; only the large `rec.pre_fc1_weight`
> Gemm ([50176,128]) is quantizable.

## Architecture (detector)

- Static input `input` = `[1,3,640,640]` (NCHW). The file ships a FIXED
  640 spatial size, so the detector input is **640** (not 320): YuNet is
  plain-RESIZED to the square net (`cv2.FaceDetectorYN` /
  `blobFromImage(img, 1.0, (640,640))`, raw BGR, **no** mean/std,
  **no** letterbox), and boxes are scaled back per-axis
  (`sx,sy = orig_w/640, orig_h/640`).
- Backbone (stride-2 stem conv + ConvDPUnit blocks: a 1x1 pointwise conv
  then a **depthwise** 3x3 conv, `group`=channels) with 4 `MaxPool` 2x2
  downsamples -> feature maps at strides 8 / 16 / 32.
- FPN neck: per-level lateral (1x1 + depthwise 3x3) with 2x nearest
  `Resize` top-down merges (`Add`).
- Heads, per stride, 4 branches (each a 1x1 conv then a depthwise 3x3):
  `cls` / `obj` / `bbox` / `kps`, then `Transpose([0,2,3,1])` ->
  `Reshape([1,-1,C])` (`C` = 1 / 1 / 4 / 10), with `cls` & `obj` ending
  in `Sigmoid` in-graph.
- 12 graph outputs, in order: `cls_8 cls_16 cls_32 obj_8 obj_16 obj_32
  bbox_8 bbox_16 bbox_32 kps_8 kps_16 kps_32`. `num_anchors` = 1;
  rows per stride = (640/stride)^2 = 6400 / 1600 / 400.

## Decode (cv2.FaceDetectorYN, reproduced bit-exact in yunet_decode)

For each stride `s` and grid cell `(c,r)` (row-major, `idx = r*cols+c`):

- `score = sqrt(cls[idx] * obj[idx])` (both post-sigmoid in `[0,1]`).
- `cx = (c + bbox[idx].dx) * s`, `cy = (r + bbox[idx].dy) * s`,
  `w = exp(bbox[idx].dw) * s`, `h = exp(bbox[idx].dh) * s`,
  `x1 = cx - w/2`, `y1 = cy - h/2`.
- landmark `k`: `lx = (kps[idx][2k] + c) * s`,
  `ly = (kps[idx][2k+1] + r) * s` (5 points).
- Filter `score >= 0.5`, then greedy NMS at IoU `0.3` with the OpenCV
  `cv2.dnn.NMSBoxes` convention (standard box area, **NO** Pascal-VOC
  `+1` - that `+1` is the insightface SCRFD convention).

Gated bit-exact (`<=1e-3` raw heads, `<=1e-4 px` decoded boxes/landmarks)
against `cv2.FaceDetectorYN` in `tests/test_yunet.cpp`.

## Detector initializers (`det.` prefix, all F32)

| ONNX name (prefixed) | shape | ggml ne | dtype |
|---|---|---|---|
| `det.420` | (16, 3, 3, 3) | [3, 3, 3, 16] | F32 |
| `det.421` | (16,) | [16] | F32 |
| `det.423` | (16, 1, 3, 3) | [3, 3, 1, 16] | F32 |
| `det.424` | (16,) | [16] | F32 |
| `det.426` | (16, 1, 3, 3) | [3, 3, 1, 16] | F32 |
| `det.427` | (16,) | [16] | F32 |
| `det.429` | (32, 1, 3, 3) | [3, 3, 1, 32] | F32 |
| `det.430` | (32,) | [32] | F32 |
| `det.432` | (32, 1, 3, 3) | [3, 3, 1, 32] | F32 |
| `det.433` | (32,) | [32] | F32 |
| `det.435` | (64, 1, 3, 3) | [3, 3, 1, 64] | F32 |
| `det.436` | (64,) | [64] | F32 |
| `det.438` | (64, 1, 3, 3) | [3, 3, 1, 64] | F32 |
| `det.439` | (64,) | [64] | F32 |
| `det.441` | (64, 1, 3, 3) | [3, 3, 1, 64] | F32 |
| `det.442` | (64,) | [64] | F32 |
| `det.444` | (64, 1, 3, 3) | [3, 3, 1, 64] | F32 |
| `det.445` | (64,) | [64] | F32 |
| `det.447` | (64, 1, 3, 3) | [3, 3, 1, 64] | F32 |
| `det.448` | (64,) | [64] | F32 |
| `det.450` | (64, 1, 3, 3) | [3, 3, 1, 64] | F32 |
| `det.451` | (64,) | [64] | F32 |
| `det.453` | (64, 1, 3, 3) | [3, 3, 1, 64] | F32 |
| `det.454` | (64,) | [64] | F32 |
| `det.456` | (64, 1, 3, 3) | [3, 3, 1, 64] | F32 |
| `det.457` | (64,) | [64] | F32 |
| `det.459` | (64, 1, 3, 3) | [3, 3, 1, 64] | F32 |
| `det.460` | (64,) | [64] | F32 |
| `det.462` | (64, 1, 3, 3) | [3, 3, 1, 64] | F32 |
| `det.463` | (64,) | [64] | F32 |
| `det.464` | (4,) | [4] | F32 |
| `det.465` | (4,) | [4] | F32 |
| `det.backbone.model0.conv2.conv1.bias` | (16,) | [16] | F32 |
| `det.backbone.model0.conv2.conv1.weight` | (16, 16, 1, 1) | [1, 1, 16, 16] | F32 |
| `det.backbone.model1.conv1.conv1.bias` | (16,) | [16] | F32 |
| `det.backbone.model1.conv1.conv1.weight` | (16, 16, 1, 1) | [1, 1, 16, 16] | F32 |
| `det.backbone.model1.conv2.conv1.bias` | (32,) | [32] | F32 |
| `det.backbone.model1.conv2.conv1.weight` | (32, 16, 1, 1) | [1, 1, 16, 32] | F32 |
| `det.backbone.model2.conv1.conv1.bias` | (32,) | [32] | F32 |
| `det.backbone.model2.conv1.conv1.weight` | (32, 32, 1, 1) | [1, 1, 32, 32] | F32 |
| `det.backbone.model2.conv2.conv1.bias` | (64,) | [64] | F32 |
| `det.backbone.model2.conv2.conv1.weight` | (64, 32, 1, 1) | [1, 1, 32, 64] | F32 |
| `det.backbone.model3.conv1.conv1.bias` | (64,) | [64] | F32 |
| `det.backbone.model3.conv1.conv1.weight` | (64, 64, 1, 1) | [1, 1, 64, 64] | F32 |
| `det.backbone.model3.conv2.conv1.bias` | (64,) | [64] | F32 |
| `det.backbone.model3.conv2.conv1.weight` | (64, 64, 1, 1) | [1, 1, 64, 64] | F32 |
| `det.backbone.model4.conv1.conv1.bias` | (64,) | [64] | F32 |
| `det.backbone.model4.conv1.conv1.weight` | (64, 64, 1, 1) | [1, 1, 64, 64] | F32 |
| `det.backbone.model4.conv2.conv1.bias` | (64,) | [64] | F32 |
| `det.backbone.model4.conv2.conv1.weight` | (64, 64, 1, 1) | [1, 1, 64, 64] | F32 |
| `det.backbone.model5.conv1.conv1.bias` | (64,) | [64] | F32 |
| `det.backbone.model5.conv1.conv1.weight` | (64, 64, 1, 1) | [1, 1, 64, 64] | F32 |
| `det.backbone.model5.conv2.conv1.bias` | (64,) | [64] | F32 |
| `det.backbone.model5.conv2.conv1.weight` | (64, 64, 1, 1) | [1, 1, 64, 64] | F32 |
| `det.bbox_head.multi_level_bbox.0.conv1.bias` | (4,) | [4] | F32 |
| `det.bbox_head.multi_level_bbox.0.conv1.weight` | (4, 64, 1, 1) | [1, 1, 64, 4] | F32 |
| `det.bbox_head.multi_level_bbox.0.conv2.bias` | (4,) | [4] | F32 |
| `det.bbox_head.multi_level_bbox.0.conv2.weight` | (4, 1, 3, 3) | [3, 3, 1, 4] | F32 |
| `det.bbox_head.multi_level_bbox.1.conv1.bias` | (4,) | [4] | F32 |
| `det.bbox_head.multi_level_bbox.1.conv1.weight` | (4, 64, 1, 1) | [1, 1, 64, 4] | F32 |
| `det.bbox_head.multi_level_bbox.1.conv2.bias` | (4,) | [4] | F32 |
| `det.bbox_head.multi_level_bbox.1.conv2.weight` | (4, 1, 3, 3) | [3, 3, 1, 4] | F32 |
| `det.bbox_head.multi_level_bbox.2.conv1.bias` | (4,) | [4] | F32 |
| `det.bbox_head.multi_level_bbox.2.conv1.weight` | (4, 64, 1, 1) | [1, 1, 64, 4] | F32 |
| `det.bbox_head.multi_level_bbox.2.conv2.bias` | (4,) | [4] | F32 |
| `det.bbox_head.multi_level_bbox.2.conv2.weight` | (4, 1, 3, 3) | [3, 3, 1, 4] | F32 |
| `det.bbox_head.multi_level_cls.0.conv1.bias` | (1,) | [1] | F32 |
| `det.bbox_head.multi_level_cls.0.conv1.weight` | (1, 64, 1, 1) | [1, 1, 64, 1] | F32 |
| `det.bbox_head.multi_level_cls.0.conv2.bias` | (1,) | [1] | F32 |
| `det.bbox_head.multi_level_cls.0.conv2.weight` | (1, 1, 3, 3) | [3, 3, 1, 1] | F32 |
| `det.bbox_head.multi_level_cls.1.conv1.bias` | (1,) | [1] | F32 |
| `det.bbox_head.multi_level_cls.1.conv1.weight` | (1, 64, 1, 1) | [1, 1, 64, 1] | F32 |
| `det.bbox_head.multi_level_cls.1.conv2.bias` | (1,) | [1] | F32 |
| `det.bbox_head.multi_level_cls.1.conv2.weight` | (1, 1, 3, 3) | [3, 3, 1, 1] | F32 |
| `det.bbox_head.multi_level_cls.2.conv1.bias` | (1,) | [1] | F32 |
| `det.bbox_head.multi_level_cls.2.conv1.weight` | (1, 64, 1, 1) | [1, 1, 64, 1] | F32 |
| `det.bbox_head.multi_level_cls.2.conv2.bias` | (1,) | [1] | F32 |
| `det.bbox_head.multi_level_cls.2.conv2.weight` | (1, 1, 3, 3) | [3, 3, 1, 1] | F32 |
| `det.bbox_head.multi_level_kps.0.conv1.bias` | (10,) | [10] | F32 |
| `det.bbox_head.multi_level_kps.0.conv1.weight` | (10, 64, 1, 1) | [1, 1, 64, 10] | F32 |
| `det.bbox_head.multi_level_kps.0.conv2.bias` | (10,) | [10] | F32 |
| `det.bbox_head.multi_level_kps.0.conv2.weight` | (10, 1, 3, 3) | [3, 3, 1, 10] | F32 |
| `det.bbox_head.multi_level_kps.1.conv1.bias` | (10,) | [10] | F32 |
| `det.bbox_head.multi_level_kps.1.conv1.weight` | (10, 64, 1, 1) | [1, 1, 64, 10] | F32 |
| `det.bbox_head.multi_level_kps.1.conv2.bias` | (10,) | [10] | F32 |
| `det.bbox_head.multi_level_kps.1.conv2.weight` | (10, 1, 3, 3) | [3, 3, 1, 10] | F32 |
| `det.bbox_head.multi_level_kps.2.conv1.bias` | (10,) | [10] | F32 |
| `det.bbox_head.multi_level_kps.2.conv1.weight` | (10, 64, 1, 1) | [1, 1, 64, 10] | F32 |
| `det.bbox_head.multi_level_kps.2.conv2.bias` | (10,) | [10] | F32 |
| `det.bbox_head.multi_level_kps.2.conv2.weight` | (10, 1, 3, 3) | [3, 3, 1, 10] | F32 |
| `det.bbox_head.multi_level_obj.0.conv1.bias` | (1,) | [1] | F32 |
| `det.bbox_head.multi_level_obj.0.conv1.weight` | (1, 64, 1, 1) | [1, 1, 64, 1] | F32 |
| `det.bbox_head.multi_level_obj.0.conv2.bias` | (1,) | [1] | F32 |
| `det.bbox_head.multi_level_obj.0.conv2.weight` | (1, 1, 3, 3) | [3, 3, 1, 1] | F32 |
| `det.bbox_head.multi_level_obj.1.conv1.bias` | (1,) | [1] | F32 |
| `det.bbox_head.multi_level_obj.1.conv1.weight` | (1, 64, 1, 1) | [1, 1, 64, 1] | F32 |
| `det.bbox_head.multi_level_obj.1.conv2.bias` | (1,) | [1] | F32 |
| `det.bbox_head.multi_level_obj.1.conv2.weight` | (1, 1, 3, 3) | [3, 3, 1, 1] | F32 |
| `det.bbox_head.multi_level_obj.2.conv1.bias` | (1,) | [1] | F32 |
| `det.bbox_head.multi_level_obj.2.conv1.weight` | (1, 64, 1, 1) | [1, 1, 64, 1] | F32 |
| `det.bbox_head.multi_level_obj.2.conv2.bias` | (1,) | [1] | F32 |
| `det.bbox_head.multi_level_obj.2.conv2.weight` | (1, 1, 3, 3) | [3, 3, 1, 1] | F32 |
| `det.neck.lateral_convs.0.conv1.bias` | (64,) | [64] | F32 |
| `det.neck.lateral_convs.0.conv1.weight` | (64, 64, 1, 1) | [1, 1, 64, 64] | F32 |
| `det.neck.lateral_convs.1.conv1.bias` | (64,) | [64] | F32 |
| `det.neck.lateral_convs.1.conv1.weight` | (64, 64, 1, 1) | [1, 1, 64, 64] | F32 |
| `det.neck.lateral_convs.2.conv1.bias` | (64,) | [64] | F32 |
| `det.neck.lateral_convs.2.conv1.weight` | (64, 64, 1, 1) | [1, 1, 64, 64] | F32 |
| `det.240` | (0,) | [0] | F32 |
| `det.290` | (3,) | [3] | F32 |
| `det.362` | (3,) | [3] | F32 |
| `det.395` | (3,) | [3] | F32 |

_112 detector initializers total. None quantized: every conv kernel's innermost ggml `ne` is 1 or 3 (< 32), so `should_quantize` keeps them all F32 regardless of `--dtype`._
