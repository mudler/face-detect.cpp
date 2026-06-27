#include "align.hpp"
#include "image_io.hpp"
#include "parity.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Alignment parity test (skeleton). When a baseline produced by
// scripts/gen_baseline.py is available (path via FACEDETECT_TEST_BASELINE), this
// will compare fd::norm_crop's aligned 112x112 crop against the reference
// insightface norm_crop output (max-abs-diff gate, see docs/parity.md). Until
// the baseline asset and the norm_crop implementation exist, it skips (RC 77).
int main() {
    const char* baseline = std::getenv("FACEDETECT_TEST_BASELINE");
    if (!baseline) {
        std::fprintf(stderr, "FACEDETECT_TEST_BASELINE not set; skipping\n");
        return 77;
    }

    // The baseline carries the source landmarks of the primary face and the
    // reference 112x112 aligned crop. Reconstruct norm_crop from the SAME source
    // image + landmarks and diff against aligned_crop (max-abs-diff <= 1 on 8-bit).
    std::vector<float> ref_crop, ref_lmk;
    std::vector<int64_t> sh, lsh;
    if (!fdtest::load_baseline(baseline, "aligned_crop", ref_crop, sh)) {
        std::fprintf(stderr, "baseline missing 'aligned_crop'; skipping\n");
        return 77;
    }
    // Prefer the primary face's exact landmarks (the ones that produced the
    // golden crop); det_landmarks is in score order, so for multi-face images its
    // first row is not the primary (largest-by-area) face.
    if (!fdtest::load_baseline(baseline, "primary_landmarks", ref_lmk, lsh) &&
        !fdtest::load_baseline(baseline, "det_landmarks", ref_lmk, lsh)) {
        std::fprintf(stderr, "baseline missing landmarks; skipping\n");
        return 77;
    }
    // Source pixels: prefer the reference's own decoded RGB (baseline 'src_image')
    // so this gate isolates the umeyama transform + bilinear warp from JPEG
    // decoder differences (stb_image vs OpenCV libjpeg drift 1-3 LSB, which would
    // otherwise swamp the <=1 gate and tell us nothing about the alignment math).
    // Fall back to decoding FACEDETECT_TEST_IMAGE when the baseline predates the
    // src_image tensor.
    fd::Image src;
    std::vector<float> src_f; std::vector<int64_t> ish;
    if (fdtest::load_baseline(baseline, "src_image", src_f, ish) && ish.size() == 3) {
        const int H = (int)ish[0], W = (int)ish[1];
        std::vector<uint8_t> bytes(src_f.size());
        for (size_t i = 0; i < src_f.size(); ++i) bytes[i] = (uint8_t)(src_f[i] + 0.5f);
        if (!fd::image_from_rgb(bytes.data(), W, H, src)) {
            std::fprintf(stderr, "src_image build failed; skip\n");
            return 77;
        }
    } else {
        const char* imgpath = std::getenv("FACEDETECT_TEST_IMAGE");
        if (!imgpath) { std::fprintf(stderr, "FACEDETECT_TEST_IMAGE unset; skip\n"); return 77; }
        if (!fd::load_image_rgb(imgpath, src)) {
            std::fprintf(stderr, "cannot load %s; skip\n", imgpath);
            return 77;
        }
    }
    fd::Landmarks5 lmk;  // primary face = first row of det_landmarks [N,5,2]
    for (int k = 0; k < 5; ++k) { lmk[k][0] = ref_lmk[k*2]; lmk[k][1] = ref_lmk[k*2+1]; }
    fd::Image got;
    if (!fd::norm_crop(src, lmk, got, 112)) { std::fprintf(stderr, "norm_crop failed\n"); return 1; }
    std::vector<float> got_f(got.rgb.begin(), got.rgb.end());
    return fdtest::compare(got_f, ref_crop, "aligned_crop", /*atol=*/1.0f, /*rtol=*/0.0f) ? 0 : 1;
}
