// Dense-landmark parity gate: the insightface 2d106det (106-pt 2D) and 1k3d68
// (68-pt 3D) heads vs the reference dumped by scripts/gen_baseline.py.
//
// ENGINE-LEVEL only: no LocalAI proto RPC / API endpoint consumes dense landmarks
// yet (the Detect RPC returns only the 5-point SCRFD kps), so this gates the C++
// engine directly. ISOLATED methodology (mirrors the genderage gate): feed the
// reference's EXACT 192x192 crop into the head, gate the raw `fc1` output tightly,
// then decode + map back to image space with the reference's OWN forward affine M
// (dumped) and gate the final points within ~1px of the reference. Feeding the
// reference crop + M isolates the conv graph + decode from the host-side bbox warp.
//
// CPU is forced so the conv/BN/matmul graph matches the onnxruntime CPU reference.
#include "model_loader.hpp"
#include "landmark_graph.hpp"
#include "detect.hpp"
#include "backend.hpp"
#include "image_io.hpp"
#include "parity.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

// Gate one head (2d/3d). Returns: 1 ok, 0 fail, -1 absent (skip this head).
int gate_head(fd::ModelLoader& ml, const std::string& base, bool three_d,
              const char* tag, int dim) {
    std::vector<float> crop_f, raw_ref, M_ref, pts_ref;
    std::vector<int64_t> sh;
    const std::string kcrop = std::string("landmark_") + tag + "_crop";
    const std::string kraw  = std::string("landmark_") + tag + "_raw";
    const std::string kM    = std::string("landmark_") + tag + "_M";
    const std::string kpts  = std::string("landmark_") + tag + "_points";
    if (!fdtest::load_baseline(base, kcrop, crop_f, sh) ||
        !fdtest::load_baseline(base, kraw, raw_ref, sh) ||
        !fdtest::load_baseline(base, kM, M_ref, sh) ||
        !fdtest::load_baseline(base, kpts, pts_ref, sh)) {
        std::fprintf(stderr, "[%s] landmark goldens absent; skip\n", tag);
        return -1;
    }
    const fd::FaceConfig::LandmarkHead& h =
        three_d ? ml.config().landmark_3d : ml.config().landmark_2d;
    if (!h.present) { std::fprintf(stderr, "[%s] head absent in pack; skip\n", tag); return -1; }

    const int sz = (int)h.input_size;  // 192
    fd::Image crop;
    crop.width = sz; crop.height = sz; crop.rgb.resize(crop_f.size());
    for (size_t i = 0; i < crop_f.size(); ++i)
        crop.rgb[i] = (uint8_t)(crop_f[i] < 0 ? 0 : (crop_f[i] > 255 ? 255 : crop_f[i] + 0.5f));

    std::vector<float> raw = fd::landmark_forward(ml, three_d, crop, fd::global_backend());
    // INTERMEDIATE raw fc1 gate: normalized landmark coords (~[-1,1]); 1.5e-2 on
    // CPU keeps the implied pixel drift (x input_size/2 = 96) well under the 1px
    // image-space gate, looser on GPU (FP reduction-order non-determinism).
    bool ok = fdtest::compare(raw, raw_ref, (std::string(tag) + "_raw").c_str(),
                              fdtest::intermediate_atol(1.5e-2f), 0.0f);

    // Decode crop-space, then map to image space with the reference's OWN M.
    std::array<float, 6> M{ M_ref[0], M_ref[1], M_ref[2], M_ref[3], M_ref[4], M_ref[5] };
    std::vector<fd::LandmarkPoint> cp = fd::landmark_decode_crop(ml, three_d, raw);
    std::vector<fd::LandmarkPoint> ip = fd::landmark_to_image(cp, M);
    std::vector<float> got;
    got.reserve(ip.size() * dim);
    for (const fd::LandmarkPoint& p : ip) {
        got.push_back(p.x);
        got.push_back(p.y);
        if (dim == 3) got.push_back(p.z);
    }
    // Image-space points within ~1px (the headline gate). z (3D) is in the same
    // scaled units the reference reports, so the same tolerance applies.
    ok &= fdtest::compare(got, pts_ref, (std::string(tag) + "_points(px)").c_str(), 1.0f, 0.0f);

    // End-to-end host-warp gate: the above feeds the reference's OWN crop + M, so
    // fd::landmark_crop (bbox -> 192 scale-about-center warp) is otherwise UNGATED.
    // Reconstruct the detection bbox + source pixels and exercise landmark_crop,
    // asserting it reproduces (a) the reference forward affine M and (b) the
    // reference crop. Gracefully skipped when the baseline predates the bbox /
    // src_image tensors (older goldens still gate the network + decode above).
    std::vector<float> bbox_ref, src_f;
    std::vector<int64_t> bsh, ish;
    if (fdtest::load_baseline(base, std::string("landmark_") + tag + "_bbox", bbox_ref, bsh) &&
        bbox_ref.size() == 4 &&
        fdtest::load_baseline(base, "src_image", src_f, ish) && ish.size() == 3) {
        const int H = (int)ish[0], W = (int)ish[1];
        std::vector<uint8_t> bytes(src_f.size());
        for (size_t i = 0; i < src_f.size(); ++i) bytes[i] = (uint8_t)(src_f[i] + 0.5f);
        fd::Image src;
        if (fd::image_from_rgb(bytes.data(), W, H, src)) {
            fd::Detection d;
            d.x1 = bbox_ref[0]; d.y1 = bbox_ref[1];
            d.x2 = bbox_ref[2]; d.y2 = bbox_ref[3];
            std::array<float, 6> Mc{};
            fd::Image hc;
            if (fd::landmark_crop(src, d, sz, Mc, hc)) {
                // Forward affine M: pure arithmetic, gate it tightly.
                std::vector<float> Mgot(Mc.begin(), Mc.end());
                ok &= fdtest::compare(Mgot, M_ref, (std::string(tag) + "_crop_M").c_str(),
                                      1e-4f, 0.0f);
                // The host warp vs cv2.warpAffine: same atol=1 the align gate uses
                // (bilinear LSB drift), against the reference's dumped crop.
                std::vector<float> hc_f(hc.rgb.begin(), hc.rgb.end());
                ok &= fdtest::compare(hc_f, crop_f, (std::string(tag) + "_crop(px)").c_str(),
                                      1.0f, 0.0f);
            } else {
                std::fprintf(stderr, "[%s] landmark_crop failed\n", tag);
                ok = false;
            }
        }
    } else {
        std::fprintf(stderr, "[%s] bbox/src_image golden absent; landmark_crop ungated\n", tag);
    }
    return ok ? 1 : 0;
}

} // namespace

int main() {
    // Default to CPU so the conv/BN/matmul graph matches the onnxruntime CPU
    // reference, but RESPECT an externally-set FACEDETECT_DEVICE (overwrite=0) so
    // GPU verification can run the same gates on CUDA.
    setenv("FACEDETECT_DEVICE", "cpu", /*overwrite=*/0);
    fdtest::BackendGuard backend_guard;

    const char* gguf = std::getenv("FACEDETECT_LANDMARKS_GGUF");
    const char* base = std::getenv("FACEDETECT_TEST_BASELINE");
    if (!gguf || !base) { std::fprintf(stderr, "env unset; skip\n"); return 77; }

    fd::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "load landmarks gguf failed\n"); return 1; }

    int r2 = gate_head(ml, base, /*three_d=*/false, "2d", 2);
    int r3 = gate_head(ml, base, /*three_d=*/true,  "3d", 3);

    if (r2 == -1 && r3 == -1) { std::fprintf(stderr, "no landmark goldens; skip\n"); return 77; }
    bool ok = (r2 != 0) && (r3 != 0);   // -1 (absent) counts as not-failed
    return ok ? 0 : 1;
}
