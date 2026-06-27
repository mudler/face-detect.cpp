// Task 6.1 parity gate: the genderage head (age + gender) vs the insightface
// reference, dumped by scripts/gen_baseline.py.
//
// Two complementary gates, mirroring the ArcFace embedding methodology:
//
//   A. REAL PRODUCTION PATH (libjpeg decode -> scrfd_detect -> per-face genderage
//      warp -> graph -> decode): Model::analyze on the pinned fixture, gated on
//      the PRIMARY (largest-area) face: gender EXACT vs golden, age within 1 year.
//      This path uses THIS port's own detection box, so the 96x96 crop drifts
//      sub-pixel from the reference - robust on the decoded integer age/gender but
//      not a tight numeric gate (same rationale as the embedding production path).
//
//   B. ISOLATED PATH (feed the reference's exact genderage_crop -> graph): the raw
//      3-float `fc1` output [g0, g1, age_raw] gated vs golden genderage_out, AND
//      the decoded gender (exact) + age (within 1). Feeding the reference crop
//      isolates the conv/BN/pool/FC graph numerics from the host-side bbox warp,
//      exactly as the embedding test feeds the reference aligned_crop.
//
// CPU is forced so the conv/BN/matmul graph matches the onnxruntime CPU reference.
#include "model.hpp"
#include "model_loader.hpp"
#include "genderage_graph.hpp"
#include "detect.hpp"
#include "backend.hpp"
#include "image_io.hpp"
#include "parity.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
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

    // The genderage goldens are optional in the baseline; skip cleanly if absent
    // (e.g. a pack without the `ga.` head).
    std::vector<float> ref_age, ref_gender; std::vector<int64_t> sh;
    if (!fdtest::load_baseline(base, "age", ref_age, sh) ||
        !fdtest::load_baseline(base, "gender", ref_gender, sh)) {
        std::fprintf(stderr, "genderage goldens absent; skip\n");
        return 77;
    }
    const int gold_age = (int)std::lround(ref_age[0]);
    const char gold_gender = (std::lround(ref_gender[0]) == 1) ? 'M' : 'F';
    std::fprintf(stderr, "[GOLDEN] age=%d gender=%c\n", gold_age, gold_gender);

    bool ok = true;

    // --- A. Real production path: analyze -> primary (largest-area) face. -------
    auto m = fd::Model::load(gguf);
    if (!m) { std::fprintf(stderr, "load gguf failed\n"); return 1; }
    if (!m->config().genderage_present) {
        std::fprintf(stderr, "pack has no genderage head; skip\n");
        return 77;
    }
    fd::Image src;
    if (!fd::load_image_rgb(img, src)) { std::fprintf(stderr, "load image failed\n"); return 1; }
    std::vector<fd::Face> faces;
    try {
        faces = m->analyze(src);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "analyze threw: %s\n", e.what());
        return 1;
    }
    if (faces.empty()) { std::fprintf(stderr, "no faces detected\n"); return 1; }
    const fd::Face& primary = *std::max_element(
        faces.begin(), faces.end(), [](const fd::Face& a, const fd::Face& b) {
            return (a.det.x2 - a.det.x1) * (a.det.y2 - a.det.y1) <
                   (b.det.x2 - b.det.x1) * (b.det.y2 - b.det.y1);
        });
    std::fprintf(stderr, "[PRODUCTION] age=%d gender=%c\n", primary.age, primary.gender);
    bool prod_ok = (primary.gender == gold_gender) &&
                   (std::abs(primary.age - gold_age) <= 1);
    std::fprintf(stderr, "[PRODUCTION] %s\n", prod_ok ? "OK" : "FAIL");
    ok &= prod_ok;

    // --- B. Isolated path: reference genderage_crop -> graph -> raw + decode. ---
    fd::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "load gguf (loader) failed\n"); return 1; }
    std::vector<float> crop_f; std::vector<int64_t> csh;
    std::vector<float> ref_out; std::vector<int64_t> osh;
    if (fdtest::load_baseline(base, "genderage_crop", crop_f, csh) &&
        fdtest::load_baseline(base, "genderage_out", ref_out, osh) &&
        ref_out.size() == 3) {
        const int sz = (int)ml.config().genderage_input_size;  // 96
        fd::Image crop;
        crop.width = sz; crop.height = sz; crop.rgb.resize(crop_f.size());
        for (size_t i = 0; i < crop_f.size(); ++i)
            crop.rgb[i] = (uint8_t)(crop_f[i] < 0 ? 0 :
                                    (crop_f[i] > 255 ? 255 : crop_f[i] + 0.5f));
        std::vector<float> got = fd::genderage_forward(ml, crop, fd::global_backend());
        // INTERMEDIATE numeric gate on the raw 3-float fc1, fed the SAME pixels:
        // strict 1e-2 on CPU, looser on GPU (FP reduction-order non-determinism).
        ok &= fdtest::compare(got, ref_out, "genderage_out",
                              fdtest::intermediate_atol(1e-2f), 0.0f);
        const int g = (got[1] > got[0]) ? 1 : 0;
        const int a = (int)std::lround(got[2] * 100.0f);
        const char gc = (g == 1) ? 'M' : 'F';
        std::fprintf(stderr, "[ISOLATED] age=%d gender=%c\n", a, gc);
        bool iso_ok = (gc == gold_gender) && (std::abs(a - gold_age) <= 1);
        std::fprintf(stderr, "[ISOLATED] %s\n", iso_ok ? "OK" : "FAIL");
        ok &= iso_ok;
    } else {
        std::fprintf(stderr, "[ISOLATED] genderage_crop/out absent; skip isolated gate\n");
    }

    return ok ? 0 : 1;
}
