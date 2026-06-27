// Task 6.2 parity gate: the MiniFASNet anti-spoof ensemble (liveness) vs the
// insightface/Silent-Face reference, dumped by scripts/gen_baseline.py.
//
// Two complementary gates, mirroring the genderage / ArcFace methodology:
//
//   A. REAL PRODUCTION PATH (libjpeg decode -> scrfd_detect -> per-model
//      scale-expanded 80x80 crop -> graph -> ensemble softmax): the averaged
//      "real" probability on the PRIMARY (largest-area) face, gated loosely vs
//      golden antispoof_real_prob (this port's own detection box drifts the crop
//      sub-pixel, so the gate is on the verdict + a 0.05 envelope, not tight).
//
//   B. ISOLATED PATH (feed the reference's exact antispoof_crop_i -> graph): the
//      raw per-model 3 logits gated vs golden antispoof_logits_i (tol 1e-2), and
//      the averaged-softmax real prob recomputed in C++ vs golden (tol 1e-3).
//      Feeding the reference crop isolates the conv/PReLU/SE/BN/MatMul graph from
//      the host-side crop+resize, exactly as the embedding test feeds aligned_crop.
//
// CPU is forced so the graph matches the onnxruntime CPU reference.
#include "model.hpp"
#include "model_loader.hpp"
#include "antispoof_graph.hpp"
#include "detect.hpp"
#include "backend.hpp"
#include "image_io.hpp"
#include "preprocess.hpp"
#include "parity.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::array<double, 3> softmax3(const std::vector<float>& l) {
    const float mx = std::max({l[0], l[1], l[2]});
    double e[3], s = 0.0;
    for (int k = 0; k < 3; ++k) { e[k] = std::exp((double)l[k] - mx); s += e[k]; }
    return {e[0] / s, e[1] / s, e[2] / s};
}

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

    // Anti-spoof goldens are optional (a pack/baseline without MiniFASNet).
    std::vector<float> ref_prob; std::vector<int64_t> sh;
    if (!fdtest::load_baseline(base, "antispoof_real_prob", ref_prob, sh) ||
        ref_prob.empty()) {
        std::fprintf(stderr, "antispoof goldens absent; skip\n");
        return 77;
    }
    const float gold_prob = ref_prob[0];
    const bool gold_real = gold_prob >= 0.5f;
    std::fprintf(stderr, "[GOLDEN] real_prob=%.5f is_real=%d\n", gold_prob, (int)gold_real);

    auto m = fd::Model::load(gguf);
    if (!m) { std::fprintf(stderr, "load gguf failed\n"); return 1; }
    if (!m->config().antispoof_present) {
        std::fprintf(stderr, "pack has no anti-spoof ensemble; skip\n");
        return 77;
    }

    bool ok = true;

    // --- A. Real production path: detect primary -> ensemble real prob. --------
    fd::Image src;
    if (!fd::load_image_rgb(img, src)) { std::fprintf(stderr, "load image failed\n"); return 1; }
    std::vector<fd::Detection> dets;
    try { dets = m->detect(src); }
    catch (const std::exception& e) { std::fprintf(stderr, "detect threw: %s\n", e.what()); return 1; }
    if (dets.empty()) { std::fprintf(stderr, "no faces detected\n"); return 1; }
    const fd::Detection& primary = *std::max_element(
        dets.begin(), dets.end(), [](const fd::Detection& a, const fd::Detection& b) {
            return (a.x2 - a.x1) * (a.y2 - a.y1) < (b.x2 - b.x1) * (b.y2 - b.y1);
        });
    const float prod_prob = fd::antispoof_real_prob(*m, src, primary);
    const bool prod_real = m->is_real(src, primary);
    std::fprintf(stderr, "[PRODUCTION] real_prob=%.5f is_real=%d\n", prod_prob, (int)prod_real);
    bool prod_ok = (prod_real == gold_real) && (std::abs(prod_prob - gold_prob) <= 0.05f);
    std::fprintf(stderr, "[PRODUCTION] %s\n", prod_ok ? "OK" : "FAIL");
    ok &= prod_ok;

    // --- B. Isolated path: reference 80x80 crops -> graph -> logits + softmax. -
    fd::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "load gguf (loader) failed\n"); return 1; }
    const int size = (int)ml.config().antispoof_input_size;  // 80
    const size_t n_models = ml.config().antispoof_scales.size();
    std::array<double, 3> accum{0.0, 0.0, 0.0};
    bool iso_have = true;
    for (size_t i = 0; i < n_models; ++i) {
        std::vector<float> crop_f, ref_logits; std::vector<int64_t> csh, lsh;
        const std::string cn = "antispoof_crop_" + std::to_string(i);
        const std::string ln = "antispoof_logits_" + std::to_string(i);
        if (!fdtest::load_baseline(base, cn, crop_f, csh) ||
            !fdtest::load_baseline(base, ln, ref_logits, lsh) || ref_logits.size() != 3) {
            iso_have = false; break;
        }
        fd::Image crop;
        crop.width = size; crop.height = size; crop.rgb.resize(crop_f.size());
        for (size_t j = 0; j < crop_f.size(); ++j)
            crop.rgb[j] = (uint8_t)(crop_f[j] < 0 ? 0 : (crop_f[j] > 255 ? 255 : crop_f[j] + 0.5f));
        // Raw BGR planes (swap_rb=true on the RGB Image), no mean/std.
        std::vector<float> blob = fd::to_blob(crop, size, 0.0f, 1.0f, /*swap_rb=*/true);
        std::vector<float> got = fd::minifasnet_forward(ml, (int)i, blob, size,
                                                        fd::global_backend());
        const std::string lbl = "antispoof_logits_" + std::to_string(i);
        // INTERMEDIATE per-model raw logits: strict 1e-2 on CPU, looser on GPU
        // (FP reduction-order non-determinism).
        ok &= fdtest::compare(got, ref_logits, lbl.c_str(),
                              fdtest::intermediate_atol(1e-2f), 0.0f);
        const std::array<double, 3> sm = softmax3(got);
        for (int k = 0; k < 3; ++k) accum[k] += sm[k];
    }
    if (iso_have && n_models > 0) {
        for (int k = 0; k < 3; ++k) accum[k] /= (double)n_models;
        const float iso_prob = (float)accum[1];
        std::fprintf(stderr, "[ISOLATED] real_prob=%.5f (golden=%.5f)\n", iso_prob, gold_prob);
        bool iso_ok = std::abs(iso_prob - gold_prob) <= 1e-3f;
        std::fprintf(stderr, "[ISOLATED] %s\n", iso_ok ? "OK" : "FAIL");
        ok &= iso_ok;
    } else {
        std::fprintf(stderr, "[ISOLATED] crops/logits absent; skip isolated gate\n");
    }

    return ok ? 0 : 1;
}
