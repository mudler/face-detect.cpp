// Task 3.1 gate: SCRFD conv backbone + PAFPN heads vs the RAW per-stride ONNX
// head outputs (scrfd_out_0..8) dumped by scripts/gen_baseline.py.
//
// The graph is gated from the reference's EXACT letterboxed 640x640 RGB pixels
// (scrfd_letterbox_rgb), not from a fresh JPEG decode, so the conv graph is
// isolated from JPEG-decoder / resize differences. stb_image vs libjpeg decode
// drifts 1-3 LSB, which propagates to ~1e-1 in the raw bbox heads and would
// swamp a 1e-3 graph gate (same precedent as the align gate's src_image dump).
// scrfd_letterbox itself is exercised below as a non-gating sanity check.
#include "scrfd_graph.hpp"
#include "model_loader.hpp"
#include "preprocess.hpp"
#include "image_io.hpp"
#include "backend.hpp"
#include "parity.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    // Default to CPU so the conv graph matches the onnxruntime CPU reference
    // closely, but RESPECT an externally-set FACEDETECT_DEVICE (overwrite=0) so
    // GPU verification can run the same gates on CUDA.
    setenv("FACEDETECT_DEVICE", "cpu", /*overwrite=*/0);
    fdtest::BackendGuard backend_guard;

    const char* gguf = std::getenv("FACEDETECT_TEST_GGUF");
    const char* base = std::getenv("FACEDETECT_TEST_BASELINE");
    const char* img  = std::getenv("FACEDETECT_TEST_IMAGE");
    if (!gguf || !base || !img) { std::fprintf(stderr, "env unset; skip\n"); return 77; }

    fd::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "load gguf failed\n"); return 1; }

    // Reconstruct the reference's letterboxed 640x640 RGB image (the exact pixels
    // the ONNX session consumed) from the baseline.
    std::vector<float> lbf; std::vector<int64_t> lbsh;
    if (!fdtest::load_baseline(base, "scrfd_letterbox_rgb", lbf, lbsh)) return 77;
    const int S = (int)ml.config().det_input_size;
    fd::Image letterboxed;
    letterboxed.width = S; letterboxed.height = S;
    letterboxed.rgb.resize(lbf.size());
    for (size_t i = 0; i < lbf.size(); ++i)
        letterboxed.rgb[i] = (uint8_t)(lbf[i] < 0 ? 0 : (lbf[i] > 255 ? 255 : lbf[i] + 0.5f));

    bool ok = true;

    // Blob gate (REQUIRED, per the Task 3.1 review): to_blob(reference pixels)
    // must equal the reference input blob. A missing tensor skips cleanly (77);
    // it must never silently pass (return 0 without comparing).
    std::vector<float> blob_ref; std::vector<int64_t> blob_sh;
    if (!fdtest::load_baseline(base, "scrfd_input_blob", blob_ref, blob_sh)) return 77;
    std::vector<float> blob = fd::to_blob(letterboxed, S, 127.5f, 128.0f, false);
    ok &= fdtest::compare(blob, blob_ref, "scrfd_input_blob", 1e-4f, 1e-4f);

    auto outs = fd::scrfd_forward(ml, letterboxed, fd::global_backend());
    if (outs.size() != 3) { std::fprintf(stderr, "scrfd_forward returned %zu strides\n", outs.size()); return 1; }

    // ONNX output order: [score s8,s16,s32, bbox s8,s16,s32, kps s8,s16,s32].
    const std::vector<float>* got[9] = {
        &outs[0].score, &outs[1].score, &outs[2].score,
        &outs[0].bbox,  &outs[1].bbox,  &outs[2].bbox,
        &outs[0].kps,   &outs[1].kps,   &outs[2].kps,
    };
    const char* labels[9] = {
        "scrfd_score_s8", "scrfd_score_s16", "scrfd_score_s32",
        "scrfd_bbox_s8",  "scrfd_bbox_s16",  "scrfd_bbox_s32",
        "scrfd_kps_s8",   "scrfd_kps_s16",   "scrfd_kps_s32",
    };
    for (int i = 0; i < 9; ++i) {
        std::vector<float> ref; std::vector<int64_t> sh;
        char name[32]; std::snprintf(name, sizeof(name), "scrfd_out_%d", i);
        if (!fdtest::load_baseline(base, name, ref, sh)) return 77;
        // INTERMEDIATE raw per-stride head outputs: strict 1e-3 on CPU, looser
        // on GPU (FP reduction-order non-determinism).
        ok &= fdtest::compare(*got[i], ref, labels[i],
                              fdtest::intermediate_atol(1e-3f), 1e-3f);
    }

    // det_scale gate (hardened from a sanity print, per the Task 3.1 review):
    // scrfd_letterbox over a REAL stb decode must reproduce the reference
    // det_scale exactly. det_scale is geometry-only (orig dims, not pixels), so
    // it is decoder-independent and must match to <=1e-4.
    std::vector<float> scale_ref; std::vector<int64_t> ssh;
    if (!fdtest::load_baseline(base, "scrfd_det_scale", scale_ref, ssh) || scale_ref.empty()) return 77;
    fd::Image src; float det_scale = 0.f;
    if (!fd::load_image_rgb(img, src)) { std::fprintf(stderr, "load image failed\n"); return 1; }
    fd::Image lb; fd::scrfd_letterbox(src, S, lb, det_scale);
    std::vector<float> scale_got{det_scale};
    ok &= fdtest::compare(scale_got, scale_ref, "scrfd_det_scale", 1e-4f, 1e-4f);

    return ok ? 0 : 1;
}
