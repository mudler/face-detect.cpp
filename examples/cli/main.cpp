#include "facedetect.h"
#include "facedetect_capi.h"
#include "model.hpp"
#include "model_loader.hpp"
#include "image_io.hpp"
#include "align.hpp"         // fd::warp_affine (pre-aligned-crop landmark path)
#include "detect.hpp"        // fd::Detection
#include "landmark_graph.hpp"
#include "arcface_graph.hpp" // fd::arcface_embed (recognizer-only bench)
#include "sface_graph.hpp"   // fd::sface_embed   (recognizer-only bench)
#include "backend.hpp"      // fd::set_num_threads, fd::shutdown_backend, fd::global_backend

#include <algorithm>
#include <array>
#include <chrono>

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const char* kUsage =
    "usage:\n"
    "  facedetect-cli info <model.gguf>\n"
    "  facedetect-cli embed   --model <model.gguf> --input <img> [--json] [--threads N]\n"
    "  facedetect-cli detect  --model <model.gguf> --input <img> [--threads N]\n"
    "  facedetect-cli verify  --model <model.gguf> --a <imgA> --b <imgB> "
    "[--threshold T] [--anti-spoof] [--threads N]\n"
    "  facedetect-cli analyze --model <model.gguf> --input <img> [--threads N]\n"
    "  facedetect-cli landmarks --model <landmarks.gguf> --input <img> "
    "[--3d] [--detector <det.gguf>] [--json] [--threads N]\n"
    "  facedetect-cli bench   --model <model.gguf> --input <img> "
    "[--mode pipeline|recognizer|detect|analyze] [--n N] [--threads N]\n";

int cmd_info(const char* path) {
    fd::ModelLoader ml;
    if (!ml.load(path)) { std::fprintf(stderr, "failed to load %s\n", path); return 1; }
    const fd::FaceConfig& c = ml.config();
    std::printf("face-detect.cpp %s\n", facedetect_version());
    std::printf("model: %s\n", path);
    std::printf("  arch            : %s\n", c.arch.c_str());
    std::printf("  detector        : %s (input %u, anchors %u)\n",
                c.detector.c_str(), c.det_input_size, c.det_num_anchors);
    std::printf("  det strides     : [");
    for (size_t i = 0; i < c.det_strides.size(); ++i)
        std::printf("%s%d", i ? "," : "", c.det_strides[i]);
    std::printf("]\n");
    std::printf("  det thresholds  : score=%.2f nms=%.2f\n", c.det_score_thresh, c.det_nms_thresh);
    std::printf("  recognizer      : %s (input %u, dim %u)\n",
                c.recognizer.c_str(), c.rec_input_size, c.embed_dim);
    std::printf("  verify thresh   : %.2f\n", c.verify_threshold);
    std::printf("  genderage       : %s\n", c.genderage_present ? "present" : "absent");
    std::printf("  anti-spoof      : %s\n", c.antispoof_present ? "present" : "absent");
    return 0;
}

// Shared flag parsing for --model/--input/--a/--b/--threshold/--anti-spoof/
// --threads/--json/--reps. Unknown flags are ignored.
struct Args {
    std::string model, input, a, b, json, detector, mode;
    float threshold = 0.0f;
    bool anti_spoof = false;
    bool json_out = false;
    bool three_d = false;
    int threads = 0;
    int reps = 10;
    int n = 0;  // bench timed passes (overrides --reps when > 0)
};

Args parse(int argc, char** argv) {
    Args r;
    for (int i = 0; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) r.model = argv[++i];
        else if (!std::strcmp(argv[i], "--input") && i + 1 < argc) r.input = argv[++i];
        else if (!std::strcmp(argv[i], "--a") && i + 1 < argc) r.a = argv[++i];
        else if (!std::strcmp(argv[i], "--b") && i + 1 < argc) r.b = argv[++i];
        else if (!std::strcmp(argv[i], "--threshold") && i + 1 < argc) r.threshold = (float)std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--anti-spoof")) r.anti_spoof = true;
        else if (!std::strcmp(argv[i], "--detector") && i + 1 < argc) r.detector = argv[++i];
        else if (!std::strcmp(argv[i], "--3d")) r.three_d = true;
        else if (!std::strcmp(argv[i], "--json")) r.json_out = true;
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) r.threads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) r.reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--n") && i + 1 < argc) r.n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) r.mode = argv[++i];
    }
    return r;
}

int cmd_embed(int argc, char** argv) {
    Args a = parse(argc, argv);
    if (a.model.empty() || a.input.empty()) {
        std::fprintf(stderr, "usage: facedetect-cli embed --model <m.gguf> --input <img> [--json]\n");
        return 2;
    }
    if (a.threads > 0) fd::set_num_threads(a.threads);
    facedetect_ctx* ctx = facedetect_capi_load(a.model.c_str());
    if (!ctx) { std::fprintf(stderr, "facedetect-cli: failed to load model %s\n", a.model.c_str()); return 1; }
    float* vec = nullptr; int dim = 0;
    int rc = facedetect_capi_embed_path(ctx, a.input.c_str(), &vec, &dim);
    if (rc != 0) {
        std::fprintf(stderr, "facedetect-cli embed: %s\n", facedetect_capi_last_error(ctx));
        facedetect_capi_free(ctx);
        return 1;
    }
    if (a.json_out) {
        std::printf("{\"dim\":%d,\"embedding\":[", dim);
        for (int i = 0; i < dim; ++i) std::printf("%s%.6f", i ? "," : "", vec[i]);
        std::printf("]}\n");
    } else {
        for (int i = 0; i < dim; ++i) std::printf("%s%.6f", i ? " " : "", vec[i]);
        std::printf("\n");
    }
    facedetect_capi_free_vec(vec);
    facedetect_capi_free(ctx);
    return 0;
}

int cmd_detect(int argc, char** argv) {
    Args a = parse(argc, argv);
    if (a.model.empty() || a.input.empty()) {
        std::fprintf(stderr, "usage: facedetect-cli detect --model <m.gguf> --input <img>\n");
        return 2;
    }
    if (a.threads > 0) fd::set_num_threads(a.threads);
    facedetect_ctx* ctx = facedetect_capi_load(a.model.c_str());
    if (!ctx) { std::fprintf(stderr, "facedetect-cli: failed to load model %s\n", a.model.c_str()); return 1; }
    char* json = facedetect_capi_detect_path_json(ctx, a.input.c_str());
    if (!json) {
        std::fprintf(stderr, "facedetect-cli detect: %s\n", facedetect_capi_last_error(ctx));
        facedetect_capi_free(ctx);
        return 1;
    }
    std::printf("%s\n", json);
    facedetect_capi_free_string(json);
    facedetect_capi_free(ctx);
    return 0;
}

int cmd_verify(int argc, char** argv) {
    Args a = parse(argc, argv);
    if (a.model.empty() || a.a.empty() || a.b.empty()) {
        std::fprintf(stderr, "usage: facedetect-cli verify --model <m.gguf> --a <imgA> --b <imgB> "
                             "[--threshold T] [--anti-spoof]\n");
        return 2;
    }
    if (a.threads > 0) fd::set_num_threads(a.threads);
    facedetect_ctx* ctx = facedetect_capi_load(a.model.c_str());
    if (!ctx) { std::fprintf(stderr, "facedetect-cli: failed to load model %s\n", a.model.c_str()); return 1; }
    float dist = 0.0f; int verified = 0;
    int rc = facedetect_capi_verify_paths(ctx, a.a.c_str(), a.b.c_str(), a.threshold,
                                          a.anti_spoof ? 1 : 0, &dist, &verified);
    if (rc != 0) {
        std::fprintf(stderr, "facedetect-cli verify: %s\n", facedetect_capi_last_error(ctx));
        facedetect_capi_free(ctx);
        return 1;
    }
    std::printf("{\"distance\":%.4f,\"verified\":%s}\n", dist, verified ? "true" : "false");
    facedetect_capi_free(ctx);
    return 0;
}

int cmd_analyze(int argc, char** argv) {
    Args a = parse(argc, argv);
    if (a.model.empty() || a.input.empty()) {
        std::fprintf(stderr, "usage: facedetect-cli analyze --model <m.gguf> --input <img>\n");
        return 2;
    }
    if (a.threads > 0) fd::set_num_threads(a.threads);
    facedetect_ctx* ctx = facedetect_capi_load(a.model.c_str());
    if (!ctx) { std::fprintf(stderr, "facedetect-cli: failed to load model %s\n", a.model.c_str()); return 1; }
    char* json = facedetect_capi_analyze_path_json(ctx, a.input.c_str());
    if (!json) {
        std::fprintf(stderr, "facedetect-cli analyze: %s\n", facedetect_capi_last_error(ctx));
        facedetect_capi_free(ctx);
        return 1;
    }
    std::printf("%s\n", json);
    facedetect_capi_free_string(json);
    facedetect_capi_free(ctx);
    return 0;
}

// Dense-landmark heads (2d106det 106-pt 2D / 1k3d68 68-pt 3D). ENGINE-LEVEL only:
// no LocalAI RPC consumes these, so this exercises the heads directly through the
// C++ engine. Two modes:
//   * --detector <det.gguf>: detect the primary (largest-area) face with a real
//     detector pack, crop+align via insightface Landmark.get geometry, regress, and
//     emit IMAGE-space points.
//   * no detector: treat --input as an already-aligned face, plain-resize it to the
//     head's 192 input, regress, and emit CROP-space points (no inverse transform).
int cmd_landmarks(int argc, char** argv) {
    Args a = parse(argc, argv);
    if (a.model.empty() || a.input.empty()) {
        std::fprintf(stderr, "usage: facedetect-cli landmarks --model <landmarks.gguf> "
                             "--input <img> [--3d] [--detector <det.gguf>] [--json]\n");
        return 2;
    }
    if (a.threads > 0) fd::set_num_threads(a.threads);
    fd::ModelLoader ml;
    if (!ml.load(a.model.c_str())) {
        std::fprintf(stderr, "facedetect-cli landmarks: failed to load %s\n", a.model.c_str());
        return 1;
    }
    const fd::FaceConfig::LandmarkHead& h =
        a.three_d ? ml.config().landmark_3d : ml.config().landmark_2d;
    if (!h.present) {
        std::fprintf(stderr, "facedetect-cli landmarks: pack has no %s landmark head\n",
                     a.three_d ? "3D" : "2D");
        return 1;
    }
    fd::Image img;
    if (!fd::load_image_rgb(a.input.c_str(), img)) {
        std::fprintf(stderr, "facedetect-cli landmarks: failed to read %s\n", a.input.c_str());
        return 1;
    }

    std::vector<fd::LandmarkPoint> pts;
    const char* space = "image";
    try {
        if (!a.detector.empty()) {
            auto det = fd::Model::load(a.detector);
            if (!det) {
                std::fprintf(stderr, "facedetect-cli landmarks: failed to load detector %s\n",
                             a.detector.c_str());
                return 1;
            }
            std::vector<fd::Detection> dets = det->detect(img);
            if (dets.empty()) {
                std::fprintf(stderr, "facedetect-cli landmarks: no face detected\n");
                return 1;
            }
            const fd::Detection& primary = *std::max_element(
                dets.begin(), dets.end(), [](const fd::Detection& x, const fd::Detection& y) {
                    return (x.x2 - x.x1) * (x.y2 - x.y1) < (y.x2 - y.x1) * (y.y2 - y.y1);
                });
            pts = fd::landmarks(ml, a.three_d, img, primary, fd::global_backend());
        } else {
            // No detector: treat the input as a pre-aligned crop, plain-resize to the
            // head input, and emit crop-space points.
            const int sz = (int)h.input_size;
            const std::array<float, 6> M{ (float)sz / img.width, 0.0f, 0.0f,
                                          0.0f, (float)sz / img.height, 0.0f };
            fd::Image crop;
            if (!fd::warp_affine(img, M, crop, sz, sz)) {
                std::fprintf(stderr, "facedetect-cli landmarks: resize failed\n");
                return 1;
            }
            std::vector<float> raw = fd::landmark_forward(ml, a.three_d, crop, fd::global_backend());
            pts = fd::landmark_decode_crop(ml, a.three_d, raw);
            space = "crop";
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "facedetect-cli landmarks: %s\n", e.what());
        return 1;
    }

    const int dim = (int)h.dim;
    if (a.json_out) {
        std::printf("{\"head\":\"%dd_%u\",\"space\":\"%s\",\"points\":[",
                    dim, h.num_points, space);
        for (size_t i = 0; i < pts.size(); ++i) {
            if (dim == 3)
                std::printf("%s[%.4f,%.4f,%.4f]", i ? "," : "", pts[i].x, pts[i].y, pts[i].z);
            else
                std::printf("%s[%.4f,%.4f]", i ? "," : "", pts[i].x, pts[i].y);
        }
        std::printf("]}\n");
    } else {
        std::fprintf(stderr, "landmarks %dd_%u (%s space): %zu points\n",
                     dim, h.num_points, space, pts.size());
        for (size_t i = 0; i < pts.size(); ++i) {
            if (dim == 3) std::printf("%.4f %.4f %.4f\n", pts[i].x, pts[i].y, pts[i].z);
            else          std::printf("%.4f %.4f\n", pts[i].x, pts[i].y);
        }
    }
    return 0;
}

// Time one face stage end to end over N passes (warmup excluded) and print a
// machine-parseable `<mode>: <ms> ms/image over <N> passes` line that
// scripts/bench_compare.py scrapes. Modes:
//   pipeline   - full detect -> align -> ArcFace/SFace embed (Model::embed)
//   recognizer - the recognizer graph ALONE on an aligned crop (the --input is
//                treated as a pre-aligned face: plain-resized to rec_input_size,
//                then arcface_embed / sface_embed), isolating the embed graph
//                from detection + alignment (latency is content-independent, so a
//                plain-resized photo is a valid stand-in for an aligned crop)
//   detect     - SCRFD/YuNet detection only (Model::detect)
//   analyze    - detect + genderage (Model::analyze)
int cmd_bench(int argc, char** argv) {
    Args a = parse(argc, argv);
    if (a.model.empty() || a.input.empty()) {
        std::fprintf(stderr, "usage: facedetect-cli bench --model <m.gguf> --input <img> "
                             "[--mode pipeline|recognizer|detect|analyze] [--n N]\n");
        return 2;
    }
    if (a.threads > 0) fd::set_num_threads(a.threads);
    const std::string mode = a.mode.empty() ? "pipeline" : a.mode;
    const int n = a.n > 0 ? a.n : a.reps;

    std::unique_ptr<fd::Model> model = fd::Model::load(a.model);
    if (!model) { std::fprintf(stderr, "facedetect-cli bench: failed to load %s\n", a.model.c_str()); return 1; }
    fd::Image img;
    if (!fd::load_image_rgb(a.input.c_str(), img)) {
        std::fprintf(stderr, "facedetect-cli bench: failed to read %s\n", a.input.c_str());
        return 1;
    }

    // Recognizer-only mode: build the aligned crop ONCE (plain-resize), so the
    // timed loop measures only the embed graph, not the per-iteration resize.
    fd::Image crop;
    if (mode == "recognizer") {
        const int sz = (int)model->config().rec_input_size;
        const std::array<float, 6> M{ (float)sz / img.width, 0.0f, 0.0f,
                                      0.0f, (float)sz / img.height, 0.0f };
        if (!fd::warp_affine(img, M, crop, sz, sz)) {
            std::fprintf(stderr, "facedetect-cli bench: resize failed\n");
            return 1;
        }
    }

    const bool sface = model->config().recognizer == "sface";
    auto once = [&]() {
        if (mode == "pipeline")        { volatile auto v = model->embed(img); (void)v; }
        else if (mode == "recognizer") {
            volatile auto v = sface ? fd::sface_embed(model->loader(), crop, fd::global_backend())
                                    : fd::arcface_embed(model->loader(), crop, fd::global_backend());
            (void)v;
        }
        else if (mode == "detect")     { volatile auto v = model->detect(img); (void)v; }
        else if (mode == "analyze")    { volatile auto v = model->analyze(img); (void)v; }
        else throw std::runtime_error("unknown --mode '" + mode + "'");
    };

    try {
        once();  // warmup (excluded)
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) once();
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / n;
        std::fprintf(stderr, "device=%s mode=%s n=%d\n",
                     fd::global_backend().device_name(), mode.c_str(), n);
        std::printf("%s: %.3f ms/image over %d passes\n", mode.c_str(), ms, n);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "facedetect-cli bench: %s\n", e.what());
        return 1;
    }
    return 0;
}

// Run a subcommand, then free the process-global backend while the GPU driver is
// still alive (avoids CUDA shutdown-ordering aborts at static destruction).
int run_and_shutdown(int (*fn)(int, char**), int argc, char** argv) {
    int rc = fn(argc, argv);
    fd::shutdown_backend();
    return rc;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && !std::strcmp(argv[1], "info"))
        return run_and_shutdown([](int, char** a) { return cmd_info(a[0]); }, 1, argv + 2);
    if (argc >= 2 && !std::strcmp(argv[1], "embed"))
        return run_and_shutdown(cmd_embed, argc - 2, argv + 2);
    if (argc >= 2 && !std::strcmp(argv[1], "detect"))
        return run_and_shutdown(cmd_detect, argc - 2, argv + 2);
    if (argc >= 2 && !std::strcmp(argv[1], "verify"))
        return run_and_shutdown(cmd_verify, argc - 2, argv + 2);
    if (argc >= 2 && !std::strcmp(argv[1], "analyze"))
        return run_and_shutdown(cmd_analyze, argc - 2, argv + 2);
    if (argc >= 2 && !std::strcmp(argv[1], "landmarks"))
        return run_and_shutdown(cmd_landmarks, argc - 2, argv + 2);
    if (argc >= 2 && !std::strcmp(argv[1], "bench"))
        return run_and_shutdown(cmd_bench, argc - 2, argv + 2);
    std::fprintf(stderr, "%s", kUsage);
    return 2;
}
