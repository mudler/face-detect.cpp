#include "facedetect_capi.h"
#include "model.hpp"
#include "image_io.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

// ABI version. Bump on breaking changes (see facedetect_capi.h changelog).
// v1: initial surface.
#define FACEDETECT_CAPI_ABI_VERSION 1

// The opaque context: a loaded model pack plus a buffer for the last error.
struct facedetect_ctx {
    std::unique_ptr<fd::Model> model;
    std::string last_error;
};

namespace {

// Record a last-error message on the context (no-op on NULL ctx).
void set_error(facedetect_ctx* ctx, const std::string& msg) {
    if (ctx) ctx->last_error = msg;
}

// malloc a NUL-terminated copy of `s` so a C consumer frees it with free()
// (matching facedetect_capi_free_string). Returns NULL on OOM.
char* dup_to_c(const std::string& s) {
    char* buf = static_cast<char*>(std::malloc(s.size() + 1));
    if (!buf) return nullptr;
    std::memcpy(buf, s.data(), s.size());
    buf[s.size()] = '\0';
    return buf;
}

// malloc a copy of a float vector for return across the C boundary. Returns NULL
// on OOM or empty input.
float* dup_vec(const std::vector<float>& v) {
    if (v.empty()) return nullptr;
    float* buf = static_cast<float*>(std::malloc(v.size() * sizeof(float)));
    if (!buf) return nullptr;
    std::memcpy(buf, v.data(), v.size() * sizeof(float));
    return buf;
}

// Append `s` to `out` as a JSON string literal (with surrounding quotes),
// escaping ", \\, and control characters.
void json_escape(std::string& out, const std::string& s) {
    out += '"';
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x", (unsigned char)c);
                    out += b;
                } else {
                    out += c;
                }
        }
    }
    out += '"';
}

// Serialize a list of faces (detection-only fields) to the detect JSON shape.
std::string detections_to_json(const std::vector<fd::Detection>& dets) {
    std::string out = "{\"faces\":[";
    for (size_t i = 0; i < dets.size(); ++i) {
        const fd::Detection& d = dets[i];
        if (i) out += ',';
        char b[256];
        std::snprintf(b, sizeof(b),
                      "{\"score\":%.4f,\"box\":[%.2f,%.2f,%.2f,%.2f],\"landmarks\":[",
                      d.score, d.x1, d.y1, d.x2, d.y2);
        out += b;
        for (int k = 0; k < 5; ++k) {
            if (k) out += ',';
            std::snprintf(b, sizeof(b), "[%.2f,%.2f]", d.landmarks[k][0], d.landmarks[k][1]);
            out += b;
        }
        out += "]}";
    }
    out += "]}";
    return out;
}

} // namespace

extern "C" int facedetect_capi_abi_version(void) { return FACEDETECT_CAPI_ABI_VERSION; }

extern "C" facedetect_ctx* facedetect_capi_load(const char* gguf_path) {
    if (!gguf_path) return nullptr;
    try {
        auto ctx = std::make_unique<facedetect_ctx>();
        ctx->model = fd::Model::load(gguf_path);
        if (!ctx->model) return nullptr;   // load failure: nothing to return
        return ctx.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void facedetect_capi_free(facedetect_ctx* ctx) { delete ctx; }

extern "C" const char* facedetect_capi_last_error(facedetect_ctx* ctx) {
    return ctx ? ctx->last_error.c_str() : "";
}

extern "C" void facedetect_capi_free_string(char* s) { std::free(s); }

extern "C" void facedetect_capi_free_vec(float* v) { std::free(v); }

extern "C" int facedetect_capi_embed_path(facedetect_ctx* ctx, const char* image_path,
                                          float** out_vec, int* out_dim) {
    if (!ctx || !ctx->model || !image_path || !out_vec || !out_dim) return 1;
    try {
        fd::Image img;
        if (!fd::load_image_rgb(image_path, img)) {
            set_error(ctx, std::string("failed to load image: ") + image_path);
            return 2;
        }
        std::vector<float> emb = ctx->model->embed(img);
        float* buf = dup_vec(emb);
        if (!buf) { set_error(ctx, "out of memory"); return 3; }
        *out_vec = buf;
        *out_dim = (int)emb.size();
        return 0;
    } catch (const std::exception& e) {
        set_error(ctx, e.what());
        return 4;
    } catch (...) {
        set_error(ctx, "unknown error");
        return 4;
    }
}

extern "C" int facedetect_capi_embed_rgb(facedetect_ctx* ctx, const unsigned char* rgb,
                                         int width, int height, float** out_vec,
                                         int* out_dim) {
    if (!ctx || !ctx->model || !rgb || !out_vec || !out_dim) return 1;
    try {
        fd::Image img;
        if (!fd::image_from_rgb(rgb, width, height, img)) {
            set_error(ctx, "invalid rgb image dimensions");
            return 2;
        }
        std::vector<float> emb = ctx->model->embed(img);
        float* buf = dup_vec(emb);
        if (!buf) { set_error(ctx, "out of memory"); return 3; }
        *out_vec = buf;
        *out_dim = (int)emb.size();
        return 0;
    } catch (const std::exception& e) {
        set_error(ctx, e.what());
        return 4;
    } catch (...) {
        set_error(ctx, "unknown error");
        return 4;
    }
}

extern "C" char* facedetect_capi_detect_path_json(facedetect_ctx* ctx,
                                                  const char* image_path) {
    if (!ctx || !ctx->model || !image_path) return nullptr;
    try {
        fd::Image img;
        if (!fd::load_image_rgb(image_path, img)) {
            set_error(ctx, std::string("failed to load image: ") + image_path);
            return nullptr;
        }
        std::vector<fd::Detection> dets = ctx->model->detect(img);
        return dup_to_c(detections_to_json(dets));
    } catch (const std::exception& e) {
        set_error(ctx, e.what());
        return nullptr;
    } catch (...) {
        set_error(ctx, "unknown error");
        return nullptr;
    }
}

extern "C" int facedetect_capi_verify_paths(facedetect_ctx* ctx, const char* a,
                                            const char* b, float threshold,
                                            int anti_spoof, float* out_distance,
                                            int* out_verified) {
    if (!ctx || !ctx->model || !a || !b || !out_distance || !out_verified) return 1;
    try {
        const float thr = threshold > 0.0f ? threshold : ctx->model->config().verify_threshold;
        fd::Image ia, ib;
        if (!fd::load_image_rgb(a, ia)) { set_error(ctx, std::string("failed to load image: ") + a); return 2; }
        if (!fd::load_image_rgb(b, ib)) { set_error(ctx, std::string("failed to load image: ") + b); return 2; }
        std::vector<float> ea = ctx->model->embed(ia);
        std::vector<float> eb = ctx->model->embed(ib);
        double dot = 0.0;
        const size_t n = ea.size() < eb.size() ? ea.size() : eb.size();
        for (size_t i = 0; i < n; ++i) dot += (double)ea[i] * (double)eb[i];
        const float dist = (float)(1.0 - dot);
        *out_distance = dist;
        int verified = dist <= thr ? 1 : 0;
        // Anti-spoof veto: when requested and the pack carries the MiniFASNet
        // ensemble, a face judged fake in EITHER image forces not-verified.
        if (verified && anti_spoof != 0 && ctx->model->config().antispoof_present) {
            auto live = [&](const fd::Image& im) -> bool {
                std::vector<fd::Detection> d = ctx->model->detect(im);
                if (d.empty()) return false;  // no face -> cannot prove liveness
                const fd::Detection& primary = *std::max_element(
                    d.begin(), d.end(), [](const fd::Detection& x, const fd::Detection& y) {
                        return (x.x2 - x.x1) * (x.y2 - x.y1) < (y.x2 - y.x1) * (y.y2 - y.y1);
                    });
                return ctx->model->is_real(im, primary);
            };
            if (!live(ia) || !live(ib)) verified = 0;
        }
        *out_verified = verified;
        return 0;
    } catch (const std::exception& e) {
        set_error(ctx, e.what());
        return 4;
    } catch (...) {
        set_error(ctx, "unknown error");
        return 4;
    }
}

extern "C" char* facedetect_capi_analyze_path_json(facedetect_ctx* ctx,
                                                   const char* image_path) {
    if (!ctx || !ctx->model || !image_path) return nullptr;
    try {
        fd::Image img;
        if (!fd::load_image_rgb(image_path, img)) {
            set_error(ctx, std::string("failed to load image: ") + image_path);
            return nullptr;
        }
        std::vector<fd::Face> faces = ctx->model->analyze(img);
        std::string out = "{\"faces\":[";
        for (size_t i = 0; i < faces.size(); ++i) {
            const fd::Face& f = faces[i];
            if (i) out += ',';
            char b[256];
            std::snprintf(b, sizeof(b),
                          "{\"score\":%.4f,\"box\":[%.2f,%.2f,%.2f,%.2f],\"age\":%d,\"gender\":",
                          f.det.score, f.det.x1, f.det.y1, f.det.x2, f.det.y2, f.age);
            out += b;
            const std::string g(1, f.gender);
            json_escape(out, g);
            out += '}';
        }
        out += "]}";
        return dup_to_c(out);
    } catch (const std::exception& e) {
        set_error(ctx, e.what());
        return nullptr;
    } catch (...) {
        set_error(ctx, "unknown error");
        return nullptr;
    }
}
