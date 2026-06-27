#pragma once
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Header-only golden-compare harness, ported from parakeet.cpp. Tests load
// reference tensors dumped by scripts/gen_baseline.py (the insightface reference
// pipeline) and diff the C++ stage outputs against them. Tests that need a
// baseline/model that is absent return 77 (ctest SKIP_RETURN_CODE) so CI without
// the reference assets skips cleanly instead of failing.

// fd::shutdown_backend() is defined in src/backend.cpp; forward-declared here so
// the BackendGuard below can free the process-global backend without pulling in
// backend.hpp. Tests instantiating BackendGuard already link the facedetect lib.
namespace fd { void shutdown_backend(); }

namespace fdtest {

// True when the active compute device (FACEDETECT_DEVICE) is a non-CPU device.
// On a GPU, FP reduction-order / FMA non-determinism inflates INTERMEDIATE
// (per-stage / encoder-style) tensor diffs well past the strict CPU bound, while
// the FINAL outputs (embedding cosine >= 0.9999, max|d| <= 1e-3) stay in
// tolerance. So intermediate-tensor gates loosen on GPU; final gates never do.
// Unset env (the in-test default before setenv) is treated as CPU.
inline bool device_is_gpu() {
    const char* dev = std::getenv("FACEDETECT_DEVICE");
    if (!dev || dev[0] == '\0') return false;
    return std::strcmp(dev, "cpu") != 0 && std::strcmp(dev, "CPU") != 0;
}

// Abs tolerance for an INTERMEDIATE-tensor gate: the strict `cpu_atol` on CPU,
// a looser `gpu_atol` (default 5e-2) on a GPU device. The looser bound applies
// ONLY to intermediate compares; FINAL-output gates keep their strict tolerance
// on both devices. Never tighter than the CPU value.
inline float intermediate_atol(float cpu_atol, float gpu_atol = 5e-2f) {
    return device_is_gpu() ? std::max(cpu_atol, gpu_atol) : cpu_atol;
}

// RAII guard: free the process-global fd::Backend at scope exit, BEFORE static
// destructors run. On a GPU device the CUDA driver may tear down before the
// ggml buffer destructor, aborting the process with "CUDA error: driver
// shutting down" even though every parity comparison already passed; freeing the
// backend here (mirrors voice-detect.cpp's explicit shutdown_backend()) makes
// GPU runs exit cleanly. No-op when no backend was ever created (idempotent).
struct BackendGuard { ~BackendGuard() { fd::shutdown_backend(); } };

// Load an f32 tensor (flattened, row-major) by name from a baseline gguf.
inline bool load_baseline(const std::string& path, const std::string& name,
                          std::vector<float>& out, std::vector<int64_t>& shape) {
    ggml_context* ctx = nullptr;
    gguf_init_params p{ /*no_alloc=*/false, /*ctx=*/&ctx };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) {
        std::fprintf(stderr, "[parity] failed to open baseline: %s\n", path.c_str());
        return false;
    }
    ggml_tensor* t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        std::fprintf(stderr, "[parity] tensor '%s' not found in %s\n", name.c_str(), path.c_str());
        gguf_free(g);
        ggml_free(ctx);
        return false;
    }
    shape.clear();
    for (int i = ggml_n_dims(t) - 1; i >= 0; --i) shape.push_back(t->ne[i]);
    size_t n = (size_t)ggml_nelements(t);
    out.resize(n);
    std::memcpy(out.data(), t->data, n * sizeof(float));
    gguf_free(g);
    ggml_free(ctx);
    return true;
}

// Compare got vs ref; returns true if all elements are within tolerance.
// Prints max/mean abs diff, worst element, and OK/FAIL to stderr.
inline bool compare(const std::vector<float>& got, const std::vector<float>& ref,
                    const char* label, float atol, float rtol) {
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "[%s] size mismatch got=%zu ref=%zu\n",
                     label, got.size(), ref.size());
        return false;
    }
    double maxabs = 0.0, sumabs = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        double d = std::fabs((double)got[i] - (double)ref[i]);
        sumabs += d;
        if (d > maxabs) { maxabs = d; worst = i; }
    }
    double mean = sumabs / (got.size() ? got.size() : 1);
    bool ok = true;
    for (size_t i = 0; i < got.size() && ok; ++i) {
        double tol = (double)atol + (double)rtol * std::fabs((double)ref[i]);
        if (std::fabs((double)got[i] - (double)ref[i]) > tol) ok = false;
    }
    std::fprintf(stderr,
        "[%s] n=%zu max|d|=%.3e mean|d|=%.3e (worst@%zu got=%.5f ref=%.5f) -> %s\n",
        label, got.size(), maxabs, mean, worst,
        got[worst], ref[worst], ok ? "OK" : "FAIL");
    return ok;
}

// Cosine similarity between two equal-length vectors (the embedding parity gate
// in docs/parity.md uses cosine >= 0.9999). Returns -2 on size mismatch.
inline double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return -2.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na == 0.0 || nb == 0.0) return -2.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

} // namespace fdtest
