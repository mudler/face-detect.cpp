// Task 4.4 decode-parity gate: the C++ JPEG decode (fd::load_image_rgb) must
// reproduce OpenCV cv2.imread pixel-for-pixel, because every downstream stage
// (scrfd_detect -> norm_crop -> arcface_embed) is fed from it and the strict
// end-to-end embedding gate (cosine >= 0.9999 AND max|d| <= 1e-3) only holds
// when the decoder matches the reference at its source.
//
// The golden `src_image` tensor (dumped by scripts/gen_baseline.py) is exactly
// cv2.imread(path) converted BGR->RGB, i.e. libjpeg-turbo with JDCT_ISLOW IDCT +
// fancy chroma upsampling (the OpenCV/libjpeg defaults). fd::load_image_rgb now
// decodes JPEG through the same vendored libjpeg-turbo, so this is expected to be
// bit-exact (max|d| == 0). stb_image, by contrast, uses a different IDCT /
// upsampling and drifts 1-3 LSB per pixel (the RED state this gate replaced).
#include "image_io.hpp"
#include "parity.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    const char* base = std::getenv("FACEDETECT_TEST_BASELINE");
    const char* img  = std::getenv("FACEDETECT_TEST_IMAGE");
    if (!base || !img) { std::fprintf(stderr, "env unset; skip\n"); return 77; }

    // Golden cv2.imread pixels, RGB float [H,W,3].
    std::vector<float> ref; std::vector<int64_t> sh;
    if (!fdtest::load_baseline(base, "src_image", ref, sh) || sh.size() != 3) {
        std::fprintf(stderr, "baseline missing 'src_image'; skip\n");
        return 77;
    }
    const int H = (int)sh[0], W = (int)sh[1];

    // Real C++ decode of the same JPEG.
    fd::Image got;
    if (!fd::load_image_rgb(img, got)) { std::fprintf(stderr, "decode failed\n"); return 1; }
    if (got.width != W || got.height != H) {
        std::fprintf(stderr, "dim mismatch got=%dx%d golden=%dx%d\n",
                     got.width, got.height, W, H);
        return 1;
    }

    // load_image_rgb returns RGB; src_image is RGB. Same channel order, compare
    // directly at bit-exact tolerance (atol 0).
    std::vector<float> got_f(got.rgb.begin(), got.rgb.end());
    bool ok = fdtest::compare(got_f, ref, "decode_src_image", /*atol=*/0.0f, /*rtol=*/0.0f);
    return ok ? 0 : 1;
}
