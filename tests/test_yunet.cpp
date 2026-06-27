// Task 8.2 gate: the YuNet detector (OpenCV-Zoo, Apache-2.0) - raw per-stride heads
// + the anchor-free decode + NMS, vs the cv2.FaceDetectorYN reference dumped by
// scripts/gen_baseline.py (yunet_out_0..11 raw heads, yunet_boxes/scores/landmarks
// decoded faces in the 640 net space, yunet_scale to map back to source).
//
// Three gates, mirroring the SCRFD pattern:
//   (a) RAW HEADS  - fed the reference's EXACT resized 640x640 pixels (isolating the
//       conv graph from resize drift), to_blob -> yunet_forward, the 12 head tensors
//       must match the ONNX reference to <=1e-3.
//   (b) DECODE     - from those same reference pixels, yunet_decode in 640 space must
//       reproduce cv2.FaceDetectorYN's boxes + 5 landmarks to <=1 px (and the same
//       score-sqrt(cls*obj) + NMS face set).
//   (c) END TO END - from a fresh libjpeg-turbo decode (bit-exact vs cv2.imread),
//       yunet_detect -> source-space boxes + landmarks must match the reference
//       (scaled by yunet_scale) within a small slack, with a boundary-face count
//       tolerance (the production resize differs from cv2.resize by up to 1 LSB,
//       which can flip a face sitting on the score threshold).
#include "yunet_graph.hpp"
#include "detect.hpp"
#include "model_loader.hpp"
#include "preprocess.hpp"
#include "image_io.hpp"
#include "backend.hpp"
#include "parity.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

int main() {
    // Default to CPU to match the onnxruntime CPU reference, but RESPECT an
    // externally-set FACEDETECT_DEVICE (overwrite=0) so GPU verification can run
    // the same gates on CUDA.
    setenv("FACEDETECT_DEVICE", "cpu", /*overwrite=*/0);
    fdtest::BackendGuard backend_guard;

    const char* gguf = std::getenv("FACEDETECT_TEST_GGUF");
    const char* base = std::getenv("FACEDETECT_TEST_BASELINE");
    const char* img  = std::getenv("FACEDETECT_TEST_IMAGE");
    if (!gguf || !base || !img) { std::fprintf(stderr, "env unset; skip\n"); return 77; }

    fd::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "load gguf failed\n"); return 1; }
    if (ml.config().detector != "yunet") { std::fprintf(stderr, "not a yunet pack; skip\n"); return 77; }
    const int S = (int)ml.config().det_input_size;

    // Reconstruct the reference's resized 640x640 RGB image (the exact pixels the
    // ONNX session + FaceDetectorYN consumed) from the baseline.
    std::vector<float> rzf; std::vector<int64_t> rzsh;
    if (!fdtest::load_baseline(base, "yunet_resized_rgb", rzf, rzsh)) return 77;
    fd::Image resized;
    resized.width = S; resized.height = S;
    resized.rgb.resize(rzf.size());
    for (size_t i = 0; i < rzf.size(); ++i)
        resized.rgb[i] = (uint8_t)(rzf[i] < 0 ? 0 : (rzf[i] > 255 ? 255 : rzf[i] + 0.5f));

    bool ok = true;

    // --- Blob gate: to_blob(reference pixels) must equal the reference input blob.
    std::vector<float> blob_ref; std::vector<int64_t> blob_sh;
    if (!fdtest::load_baseline(base, "yunet_input_blob", blob_ref, blob_sh)) return 77;
    std::vector<float> blob = fd::to_blob(resized, S, 0.0f, 1.0f, /*swap_rb=*/true);
    ok &= fdtest::compare(blob, blob_ref, "yunet_input_blob", 1e-4f, 1e-4f);

    // --- (a) Raw head gate: 12 tensors, ONNX graph-output order.
    auto heads = fd::yunet_forward(ml, resized, fd::global_backend());
    if (heads.size() != 3) { std::fprintf(stderr, "yunet_forward returned %zu strides\n", heads.size()); return 1; }
    const std::vector<float>* got[12] = {
        &heads[0].cls,  &heads[1].cls,  &heads[2].cls,
        &heads[0].obj,  &heads[1].obj,  &heads[2].obj,
        &heads[0].bbox, &heads[1].bbox, &heads[2].bbox,
        &heads[0].kps,  &heads[1].kps,  &heads[2].kps,
    };
    for (int i = 0; i < 12; ++i) {
        std::vector<float> ref; std::vector<int64_t> sh;
        char name[32]; std::snprintf(name, sizeof(name), "yunet_out_%d", i);
        if (!fdtest::load_baseline(base, name, ref, sh)) return 77;
        char label[32]; std::snprintf(label, sizeof(label), "yunet_out_%d", i);
        // INTERMEDIATE raw per-stride head outputs: strict 1e-3 on CPU, looser
        // on GPU (FP reduction-order non-determinism).
        ok &= fdtest::compare(*got[i], ref, label,
                              fdtest::intermediate_atol(1e-3f), 1e-3f);
    }

    // Golden decoded faces (640 net space), descending score.
    std::vector<float> gb, gs, gl; std::vector<int64_t> s1, s2, s3;
    if (!fdtest::load_baseline(base, "yunet_boxes", gb, s1)) return 77;
    if (!fdtest::load_baseline(base, "yunet_scores", gs, s2)) return 77;
    if (!fdtest::load_baseline(base, "yunet_landmarks", gl, s3)) return 77;
    std::vector<float> scale; std::vector<int64_t> ssh;
    if (!fdtest::load_baseline(base, "yunet_scale", scale, ssh) || scale.size() != 2) return 77;
    const float sx = scale[0], sy = scale[1];
    const size_t golden_n = gs.size();
    std::fprintf(stderr, "[yunet] golden %zu faces, scale=(%.4f,%.4f)\n", golden_n, sx, sy);

    auto sort_desc = [](std::vector<fd::Detection>& d) {
        std::sort(d.begin(), d.end(),
                  [](const fd::Detection& a, const fd::Detection& b) { return a.score > b.score; });
    };

    // --- (b) Decode gate in 640 space (from the reference pixels): exact face set.
    {
        auto dets = fd::yunet_decode(ml, heads, 1.0f, 1.0f);
        sort_desc(dets);
        std::fprintf(stderr, "[yunet decode@640] got %zu faces\n", dets.size());
        if (dets.size() != golden_n) {
            std::fprintf(stderr, "decode face count %zu != golden %zu\n", dets.size(), golden_n);
            ok = false;
        }
        const size_t pairs = std::min(dets.size(), golden_n);
        float maxbox = 0.f, maxlmk = 0.f;
        for (size_t f = 0; f < pairs; ++f) {
            const fd::Detection& d = dets[f];
            const float gc[4] = { gb[f*4+0], gb[f*4+1], gb[f*4+2], gb[f*4+3] };
            const float dc[4] = { d.x1, d.y1, d.x2, d.y2 };
            for (int k = 0; k < 4; ++k) {
                float e = std::fabs(dc[k] - gc[k]);
                maxbox = std::max(maxbox, e);
                if (e > 1.0f) { std::fprintf(stderr, "decode face %zu corner %d %.4f vs %.4f\n", f, k, dc[k], gc[k]); ok = false; }
            }
            for (int p = 0; p < 5; ++p)
                for (int c = 0; c < 2; ++c) {
                    float e = std::fabs(d.landmarks[p][c] - gl[(f*5+p)*2+c]);
                    maxlmk = std::max(maxlmk, e);
                    if (e > 1.0f) { std::fprintf(stderr, "decode face %zu lmk %d.%d %.4f vs %.4f\n", f, p, c, d.landmarks[p][c], gl[(f*5+p)*2+c]); ok = false; }
                }
        }
        std::fprintf(stderr, "[yunet decode@640] max box=%.4f px lmk=%.4f px\n", maxbox, maxlmk);
    }

    // --- (c) End-to-end gate from a fresh libjpeg-turbo decode, in source space.
    {
        fd::Image src;
        if (!fd::load_image_rgb(img, src)) { std::fprintf(stderr, "load image failed\n"); return 1; }
        auto dets = fd::yunet_detect(ml, src);
        sort_desc(dets);
        const float thr = ml.config().det_score_thresh;
        const float kBoundary = 0.02f;
        size_t solid = 0;
        for (size_t i = 0; i < golden_n; ++i) if (gs[i] > thr + kBoundary) ++solid;
        std::fprintf(stderr, "[yunet e2e] got %zu faces (golden %zu, solid %zu)\n",
                     dets.size(), golden_n, solid);
        if (dets.size() < solid) { std::fprintf(stderr, "dropped a supra-threshold face\n"); ok = false; }
        if (dets.size() > golden_n) { std::fprintf(stderr, "spurious faces\n"); ok = false; }
        // Pair each golden face to the detection with the nearest box center, NOT by
        // sorted-score index: the multi fixture has several faces with near-tied scores
        // (0.79-0.89), and the production resize's <=1 LSB drift perturbs scores enough
        // to swap their descending order vs the reference. Matching by geometry is the
        // robust comparison (the decode@640 gate above already pins the exact decode +
        // order on identical pixels). Each detection matches at most one golden face.
        std::vector<char> used(dets.size(), 0);
        float maxbox = 0.f, maxlmk = 0.f;
        for (size_t f = 0; f < golden_n; ++f) {
            const float gc[4] = { gb[f*4+0]*sx, gb[f*4+1]*sy, gb[f*4+2]*sx, gb[f*4+3]*sy };
            const float gcx = 0.5f*(gc[0]+gc[2]), gcy = 0.5f*(gc[1]+gc[3]);
            int best = -1; float bestd = 1e30f;
            for (size_t j = 0; j < dets.size(); ++j) {
                if (used[j]) continue;
                const float dcx = 0.5f*(dets[j].x1+dets[j].x2), dcy = 0.5f*(dets[j].y1+dets[j].y2);
                const float dd = (dcx-gcx)*(dcx-gcx) + (dcy-gcy)*(dcy-gcy);
                if (dd < bestd) { bestd = dd; best = (int)j; }
            }
            if (best < 0) continue;          // fewer dets than golden (boundary face)
            used[best] = 1;
            const fd::Detection& d = dets[best];
            const float dc[4] = { d.x1, d.y1, d.x2, d.y2 };
            // The STRICT <=1px decoded-box/landmark parity proof is the decode@640 gate
            // above (bit-exact on the reference's exact pixels). THIS end-to-end gate
            // additionally closes the production resize drift: our 11-bit cv_resize path
            // differs from cv2.resize by up to ~1 LSB (the JPEG decode itself is
            // libjpeg-turbo, bit-exact vs cv2.imread), and on a small face in a heavily
            // downscaled crowd (the multi fixture: 1024x684 -> 640) that drift shifts a
            // regressed landmark by ~1.5 net px. Allow a 2px end-to-end slack (still
            // <0.5% of the 640 net), as test_detect does for resize-drift-sensitive
            // faces; box corners stay well inside it.
            const float kE2E = 2.0f;
            for (int k = 0; k < 4; ++k) {
                float e = std::fabs(dc[k] - gc[k]);
                maxbox = std::max(maxbox, e);
                if (e > kE2E) { std::fprintf(stderr, "e2e golden %zu corner %d %.4f vs %.4f\n", f, k, dc[k], gc[k]); ok = false; }
            }
            for (int p = 0; p < 5; ++p) {
                float ex = std::fabs(d.landmarks[p][0] - gl[(f*5+p)*2+0]*sx);
                float ey = std::fabs(d.landmarks[p][1] - gl[(f*5+p)*2+1]*sy);
                maxlmk = std::max({maxlmk, ex, ey});
                if (ex > kE2E || ey > kE2E) { std::fprintf(stderr, "e2e golden %zu lmk %d off (%.4f,%.4f vs %.4f,%.4f)\n", f, p, d.landmarks[p][0], d.landmarks[p][1], gl[(f*5+p)*2+0]*sx, gl[(f*5+p)*2+1]*sy); ok = false; }
            }
        }
        std::fprintf(stderr, "[yunet e2e] max box=%.4f px lmk=%.4f px\n", maxbox, maxlmk);
    }

    if (!ok) return 1;
    std::printf("yunet OK: %zu faces, raw heads + decode + e2e within tolerance\n", golden_n);
    return 0;
}
