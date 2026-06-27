// Task 8.3 gate: the SFace recognizer (OpenCV-Zoo, Apache-2.0, 128-d) - the
// commercial-friendly alternative to the non-commercial insightface ArcFace,
// paired with the YuNet detector. Gated vs the cv2.FaceRecognizerSF reference
// dumped by scripts/gen_baseline.py (sface_aligned_crop = FaceRecognizerSF.alignCrop,
// sface_embedding = FaceRecognizerSF.feature, plus sface_src_rgb + sface_landmarks).
//
// Two gates:
//   (a) ALIGN     - fd::norm_crop on the reference's EXACT source pixels +
//       landmarks must reproduce cv2 FaceRecognizerSF.alignCrop (SFace shares the
//       5-point arcface_dst template). A few-LSB slack: this cross-checks two
//       different reference solvers (insightface umeyama vs OpenCV's
//       getSimilarityTransformMatrix), which agree to ~2 LSB.
//   (b) EMBEDDING - fed the reference's EXACT aligned crop (isolating the conv
//       graph from any alignment drift), fd::sface_feature must reproduce the raw
//       128-d cv2 feature to max|d| <= 1e-3, and the L2-normalized embedding's
//       cosine vs the (normalized) golden must be >= 0.9999.
#include "sface_graph.hpp"
#include "align.hpp"
#include "image_io.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "parity.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main() {
    // Default to CPU so the conv graph matches the onnxruntime CPU reference, but
    // RESPECT an externally-set FACEDETECT_DEVICE (overwrite=0) so GPU
    // verification can run the same FINAL-output gates on CUDA (the raw feature
    // max|d| <= 1e-3 and embedding cosine >= 0.9999 bounds hold on both devices).
    setenv("FACEDETECT_DEVICE", "cpu", /*overwrite=*/0);
    fdtest::BackendGuard backend_guard;

    const char* gguf = std::getenv("FACEDETECT_TEST_GGUF");
    const char* base = std::getenv("FACEDETECT_TEST_BASELINE");
    if (!gguf || !base) { std::fprintf(stderr, "env unset; skip\n"); return 77; }

    fd::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "load gguf failed\n"); return 1; }
    if (ml.config().recognizer != "sface") {
        std::fprintf(stderr, "not an sface pack; skip\n"); return 77;
    }
    if (ml.config().embed_dim != 128) {
        std::fprintf(stderr, "embed_dim %u != 128\n", ml.config().embed_dim); return 1;
    }

    // Golden tensors.
    std::vector<float> src_f, lmk_f, crop_f, emb_ref;
    std::vector<int64_t> ssh, lsh, csh, esh;
    if (!fdtest::load_baseline(base, "sface_aligned_crop", crop_f, csh)) return 77;
    if (!fdtest::load_baseline(base, "sface_embedding", emb_ref, esh)) return 77;
    if (!fdtest::load_baseline(base, "sface_landmarks", lmk_f, lsh)) return 77;
    if (emb_ref.size() != 128) {
        std::fprintf(stderr, "golden embedding size %zu != 128\n", emb_ref.size());
        return 1;
    }

    const int CS = 112;
    bool ok = true;

    // --- (a) ALIGN gate: norm_crop on the reference source + landmarks vs alignCrop.
    if (fdtest::load_baseline(base, "sface_src_rgb", src_f, ssh) && ssh.size() == 3) {
        const int H = (int)ssh[0], W = (int)ssh[1];
        std::vector<uint8_t> bytes(src_f.size());
        for (size_t i = 0; i < src_f.size(); ++i) bytes[i] = (uint8_t)(src_f[i] + 0.5f);
        fd::Image src;
        if (!fd::image_from_rgb(bytes.data(), W, H, src)) {
            std::fprintf(stderr, "src_rgb build failed\n"); return 1;
        }
        fd::Landmarks5 lmk;
        for (int k = 0; k < 5; ++k) { lmk[k][0] = lmk_f[k*2]; lmk[k][1] = lmk_f[k*2+1]; }
        fd::Image got;
        if (!fd::norm_crop(src, lmk, got, CS)) {
            std::fprintf(stderr, "norm_crop failed\n"); return 1;
        }
        std::vector<float> got_f(got.rgb.begin(), got.rgb.end());
        // Cross-reference slack: two different similarity solvers agree to ~2 LSB.
        ok &= fdtest::compare(got_f, crop_f, "sface_aligned_crop", /*atol=*/4.0f, 0.0f);
    } else {
        std::fprintf(stderr, "[sface] no sface_src_rgb; align gate skipped\n");
    }

    // --- (b) EMBEDDING gate (isolated): feed the reference's EXACT aligned crop.
    fd::Image crop;
    crop.width = CS; crop.height = CS;
    crop.rgb.resize(crop_f.size());
    for (size_t i = 0; i < crop_f.size(); ++i)
        crop.rgb[i] = (uint8_t)(crop_f[i] < 0 ? 0 : (crop_f[i] > 255 ? 255 : crop_f[i] + 0.5f));

    std::vector<float> raw = fd::sface_feature(ml, crop, fd::global_backend());
    if (raw.size() != 128) {
        std::fprintf(stderr, "sface_feature size %zu != 128\n", raw.size()); return 1;
    }
    // Raw feature vs cv2.FaceRecognizerSF.feature: max|d| <= 1e-3.
    ok &= fdtest::compare(raw, emb_ref, "sface_feature_raw", /*atol=*/1e-3f, /*rtol=*/0.0f);

    // L2-normalized embedding cosine vs the (normalized) golden: >= 0.9999.
    std::vector<float> emb = fd::sface_embed(ml, crop, fd::global_backend());
    const double cos = fdtest::cosine(emb, emb_ref);
    std::fprintf(stderr, "[sface] embedding cosine=%.6f (gate >= 0.9999)\n", cos);
    if (!(cos >= 0.9999)) ok = false;

    // sface_embed must be L2-normalized.
    double n2 = 0.0; for (float v : emb) n2 += (double)v * v;
    std::fprintf(stderr, "[sface] ||sface_embed||=%.6f (expect 1.0)\n", std::sqrt(n2));
    if (std::fabs(std::sqrt(n2) - 1.0) > 1e-4) ok = false;

    if (!ok) return 1;
    std::printf("sface OK: 128-d embedding cosine=%.6f within tolerance\n", cos);
    return 0;
}
