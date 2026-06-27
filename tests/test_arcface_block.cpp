// Task 4.2 gate: the first ArcFace IResNet IR residual block (layer1.0) vs the
// reference's block output (arcface_stage1_block0_out, ONNX value '489') dumped
// by scripts/gen_baseline.py.
//
// The block is gated from the reference's EXACT stem output (arcface_stem_out,
// ONNX '479'), isolating the residual block graph (bn1 -> conv1 -> prelu ->
// conv2(stride 2) + downsample shortcut) from any stem drift - the same
// component-isolation precedent as the 4.1 stem gate (which fed the reference's
// aligned crop). The C++ stem is already GREEN at ~1e-7, so feeding the golden
// stem here pins the diff to the block alone.
#include "arcface_graph.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "parity.hpp"
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    // Default to CPU so the conv/BN graph matches the onnxruntime CPU reference,
    // but RESPECT an externally-set FACEDETECT_DEVICE (overwrite=0) so GPU
    // verification can run the same gates on CUDA.
    setenv("FACEDETECT_DEVICE", "cpu", /*overwrite=*/0);
    fdtest::BackendGuard backend_guard;

    const char* gguf = std::getenv("FACEDETECT_TEST_GGUF");
    const char* base = std::getenv("FACEDETECT_TEST_BASELINE");
    if (!gguf || !base) { std::fprintf(stderr, "env unset; skip\n"); return 77; }

    fd::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "load gguf failed\n"); return 1; }

    // Block input = the reference's stem PReLU output [1,64,112,112] (NCHW), which
    // flattens to the same order as the ggml input tensor ne {112,112,64,1}.
    std::vector<float> stem; std::vector<int64_t> stem_sh;
    if (!fdtest::load_baseline(base, "arcface_stem_out", stem, stem_sh)) return 77;
    const int64_t W = 112, H = 112, C = 64;
    if ((int64_t)stem.size() != W * H * C) {
        std::fprintf(stderr, "unexpected stem size %zu\n", stem.size()); return 1;
    }

    // `keep` owns the block's host-side scalar buffers (BN eps) and MUST outlive
    // compute: graph_input_tensor defers the host->device copy until after the
    // graph is allocated, so the backing storage cannot be lambda-local.
    std::vector<std::vector<float>> keep;
    std::vector<float> got, dummy;
    const bool computed = fd::global_backend().compute(
        [&](ggml_context* ctx) -> ggml_tensor* {
            const int64_t ne_in[4] = {W, H, C, 1};
            ggml_tensor* x = fd::graph_input_tensor(ctx, GGML_TYPE_F32, 4, ne_in,
                                                    stem.data(), stem.size() * sizeof(float));
            ggml_tensor* y = fd::arcface_ir_block(ctx, ml, x, fd::rec_block(0, 0), keep);
            fd::capture_graph_output(y, &got);
            return y;
        }, dummy);
    if (!computed) { std::fprintf(stderr, "ir_block graph compute failed\n"); return 1; }

    std::vector<float> ref; std::vector<int64_t> sh;
    if (!fdtest::load_baseline(base, "arcface_stage1_block0_out", ref, sh)) return 77;
    // Block output is [1,64,56,56] (stride-2 downsample); INTERMEDIATE gate, tol
    // 1e-2 per the brief on CPU (the composed conv/BN graph lands far tighter),
    // looser on GPU for FP reduction-order non-determinism.
    bool ok = fdtest::compare(got, ref, "arcface_stage1_block0",
                              fdtest::intermediate_atol(1e-2f), 1e-2f);

    return ok ? 0 : 1;
}
