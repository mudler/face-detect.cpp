// Task 3.2 gate: the FULL production detection path from a REAL stb JPEG decode
// end to end -- load_image_rgb (stb) -> scrfd_letterbox -> to_blob ->
// scrfd_forward -> anchor decode (distance2bbox / distance2kps) -> NMS -> final
// pixel-space boxes + 5 landmarks -- gated against the insightface golden
// detection set (det_boxes / det_scores / det_landmarks).
//
// Unlike the Task 3.1 graph gate (which fed the reference's EXACT letterbox
// pixels to isolate the conv graph from decoder drift), THIS gate deliberately
// starts from a fresh stb_image JPEG decode, closing the deferred stb-vs-libjpeg
// decode-drift risk: the end-to-end result must still match the libjpeg-decoded
// reference within <= 1 px (max box-corner error) and <= 1 px on each of the 5
// landmarks, for the highest-scoring face.
#include "detect.hpp"
#include "model_loader.hpp"
#include "image_io.hpp"
#include "parity.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

int main() {
    // Default to CPU so the conv graph matches the onnxruntime CPU reference
    // closely, but RESPECT an externally-set FACEDETECT_DEVICE (overwrite=0) so
    // GPU verification can run the same gates on CUDA (the 1px box/landmark
    // bounds are FINAL outputs and hold on both devices).
    setenv("FACEDETECT_DEVICE", "cpu", /*overwrite=*/0);
    fdtest::BackendGuard backend_guard;

    const char* gguf = std::getenv("FACEDETECT_TEST_GGUF");
    const char* base = std::getenv("FACEDETECT_TEST_BASELINE");
    const char* img  = std::getenv("FACEDETECT_TEST_IMAGE");
    if (!gguf || !base || !img) { std::fprintf(stderr, "env unset; skip\n"); return 77; }

    fd::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "load gguf failed\n"); return 1; }

    fd::Image src;
    if (!fd::load_image_rgb(img, src)) { std::fprintf(stderr, "load image failed\n"); return 1; }

    auto dets = fd::scrfd_detect(ml, src);

    std::vector<float> rb, rs, rl; std::vector<int64_t> s1, s2, s3;
    if (!fdtest::load_baseline(base, "det_boxes", rb, s1)) return 77;
    if (!fdtest::load_baseline(base, "det_scores", rs, s2)) return 77;
    if (!fdtest::load_baseline(base, "det_landmarks", rl, s3)) return 77;

    // insightface returns NMS-keep order (descending score); sort ours the same
    // way so the i-th of ours pairs with the i-th golden face. The golden rows
    // (det_boxes/scores/landmarks) are already in that descending-score order.
    std::sort(dets.begin(), dets.end(),
              [](const fd::Detection& a, const fd::Detection& b) { return a.score > b.score; });
    if (dets.empty()) { std::fprintf(stderr, "no detections\n"); return 1; }

    const size_t golden_n = (size_t)s2[0];
    std::fprintf(stderr, "[detect] got %zu faces, golden %zu faces\n", dets.size(), golden_n);

    // A face whose score sits within `kBoundary` of the detection threshold can be
    // kept or dropped depending on a sub-LSB perturbation of the network input: the
    // raw per-stride heads are bit-faithful vs the ONNX reference (test_scrfd_graph,
    // max|d| ~1e-5 incl. the multi image), but the PRODUCTION letterbox resize
    // differs from cv2 by up to 1 LSB on a minority of pixels, which can flip a face
    // sitting essentially ON the threshold (e.g. det_500m on the crowded multi
    // fixture has one face at golden score 0.50303, 0.003 above the 0.5 cutoff). A
    // count difference is tolerated ONLY for such boundary faces; every face
    // comfortably above threshold MUST be detected and matched within 1 px.
    const float thr = ml.config().det_score_thresh;
    const float kBoundary = 0.02f;

    bool ok = true;
    // Every golden face that is NOT a boundary face must have a matched detection.
    size_t golden_solid = 0;
    for (size_t i = 0; i < golden_n; ++i)
        if (rs[i] > thr + kBoundary) ++golden_solid;
    if (dets.size() < golden_solid) {
        std::fprintf(stderr, "dropped a supra-threshold face: got %zu < solid %zu\n",
                     dets.size(), golden_solid);
        ok = false;
    }
    if (dets.size() > golden_n) {
        std::fprintf(stderr, "spurious faces: got %zu > golden %zu\n", dets.size(), golden_n);
        ok = false;
    }

    // Gate box + 5 landmarks <= 1 px for EVERY paired face (the top min(got,golden)
    // by score), not just the primary - so the multi-face fixtures actually verify
    // all faces, not only the highest-scoring one.
    const size_t pairs = std::min(dets.size(), golden_n);
    std::fprintf(stderr, "[detect] top score got=%.5f ref=%.5f\n", dets[0].score, rs[0]);
    float max_box_err = 0.f, max_lmk_err = 0.f;
    for (size_t f = 0; f < pairs; ++f) {
        const fd::Detection& d = dets[f];
        const float got_corner[4] = { d.x1, d.y1, d.x2, d.y2 };
        for (int k = 0; k < 4; ++k) {
            float e = std::fabs(got_corner[k] - rb[f * 4 + k]);
            if (e > max_box_err) max_box_err = e;
            if (e > 1.0f) {
                std::fprintf(stderr, "face %zu box corner %d off: %.4f vs %.4f (|d|=%.4f)\n",
                             f, k, got_corner[k], rb[f * 4 + k], e);
                ok = false;
            }
        }
        for (int p = 0; p < 5; ++p)
            for (int c = 0; c < 2; ++c) {
                float e = std::fabs(d.landmarks[p][c] - rl[(f * 5 + p) * 2 + c]);
                if (e > max_lmk_err) max_lmk_err = e;
                if (e > 1.0f) {
                    std::fprintf(stderr, "face %zu landmark %d.%d off: %.4f vs %.4f (|d|=%.4f)\n",
                                 f, p, c, d.landmarks[p][c], rl[(f * 5 + p) * 2 + c], e);
                    ok = false;
                }
            }
    }

    std::fprintf(stderr, "[detect] max box-corner err=%.4f px, max landmark err=%.4f px\n",
                 max_box_err, max_lmk_err);
    if (!ok) return 1;
    if (dets.size() != golden_n)
        std::fprintf(stderr, "[detect] NOTE: %zu boundary face(s) near thr=%.2f "
                     "(got %zu / golden %zu); all supra-threshold faces matched\n",
                     golden_n - dets.size(), thr, dets.size(), golden_n);
    std::printf("detect OK: %zu/%zu faces matched, box<=%.4fpx lmk<=%.4fpx\n",
                pairs, golden_n, max_box_err, max_lmk_err);
    return 0;
}
