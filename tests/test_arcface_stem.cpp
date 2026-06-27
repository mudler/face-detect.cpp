// Task 4.1 gate: ArcFace IResNet stem (first conv + folded BN + PReLU) vs the
// reference's stem PReLU output (arcface_stem_out, ONNX value '479') dumped by
// scripts/gen_baseline.py.
//
// The stem is gated from the reference's EXACT aligned 112x112 crop
// (aligned_crop, RGB float), isolating the conv graph from any upstream
// detection/alignment drift (same precedent as the SCRFD graph gate feeding the
// reference letterbox pixels). The C++ blob is built with to_blob(swap_rb=false)
// so its R,G,B plane order matches the reference's blobFromImage(swapRB=True) on
// the BGR crop - identical to the SCRFD detector blob convention.
#include "arcface_graph.hpp"
#include "model_loader.hpp"
#include "preprocess.hpp"
#include "image_io.hpp"
#include "backend.hpp"
#include "parity.hpp"
#include "ggml.h"
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
    if (!gguf || !base) { std::fprintf(stderr, "env unset; skip\n"); return 77; }

    fd::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "load gguf failed\n"); return 1; }

    // Reconstruct the reference's aligned 112x112 RGB crop (the exact pixels the
    // recognizer consumed). A missing tensor skips cleanly (77).
    std::vector<float> af; std::vector<int64_t> ash;
    if (!fdtest::load_baseline(base, "aligned_crop", af, ash)) return 77;
    const int N = (int)ml.config().rec_input_size;  // 112
    fd::Image aligned;
    aligned.width = N; aligned.height = N;
    aligned.rgb.resize(af.size());
    for (size_t i = 0; i < af.size(); ++i)
        aligned.rgb[i] = (uint8_t)(af[i] < 0 ? 0 : (af[i] > 255 ? 255 : af[i] + 0.5f));

    bool ok = true;

    // Blob gate (REQUIRED): to_blob(reference crop) must equal the reference
    // ArcFace input blob. swap_rb=false -> R,G,B planes (see file header).
    std::vector<float> blob = fd::to_blob(aligned, N, 127.5f, 127.5f, /*swap_rb=*/false);
    std::vector<float> blob_ref; std::vector<int64_t> blob_sh;
    if (!fdtest::load_baseline(base, "arcface_input_blob", blob_ref, blob_sh)) return 77;
    ok &= fdtest::compare(blob, blob_ref, "arcface_input_blob", 1e-4f, 1e-4f);

    // Run the stem as a ggml graph over the blob and capture its output.
    std::vector<float> stem_got, dummy;
    const bool computed = fd::global_backend().compute(
        [&](ggml_context* ctx) -> ggml_tensor* {
            const int64_t ne_in[4] = {N, N, 3, 1};
            ggml_tensor* x = fd::graph_input_tensor(ctx, GGML_TYPE_F32, 4, ne_in,
                                                    blob.data(), blob.size() * sizeof(float));
            std::vector<std::vector<float>> keep;
            ggml_tensor* s = fd::arcface_stem(ctx, ml, x, keep);
            fd::capture_graph_output(s, &stem_got);
            return s;
        }, dummy);
    if (!computed) { std::fprintf(stderr, "arcface_stem graph compute failed\n"); return 1; }

    std::vector<float> ref; std::vector<int64_t> sh;
    if (!fdtest::load_baseline(base, "arcface_stem_out", ref, sh)) return 77;
    // INTERMEDIATE (stem PReLU) gate: strict 1e-3 on CPU, looser on GPU where FP
    // reduction-order non-determinism inflates per-stage diffs (final embedding
    // parity, gated elsewhere, still holds at cosine >= 0.9999).
    ok &= fdtest::compare(stem_got, ref, "arcface_stem",
                          fdtest::intermediate_atol(1e-3f), 1e-3f);

    return ok ? 0 : 1;
}
