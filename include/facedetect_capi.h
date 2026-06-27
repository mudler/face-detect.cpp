#ifndef FACEDETECT_CAPI_H
#define FACEDETECT_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

// Flat C-API for face-detect.cpp - designed for dlopen / cgo / purego (LocalAI).
//
// All functions are extern "C" and never let a C++ exception cross the
// boundary. A model pack is loaded ONCE into an opaque `facedetect_ctx` and
// reused across calls. Returned strings are malloc'd UTF-8 owned by the caller
// and must be released with facedetect_capi_free_string; returned float vectors
// are malloc'd and released with facedetect_capi_free_vec.
//
// This is the surface LocalAI's face-recognition (biometric) backend dlopens,
// replacing the Python `insightface` backend. The full pipeline is: decode
// image -> SCRFD detect -> 5-landmark similarity-transform align to 112x112
// (insightface norm_crop) -> ArcFace embed -> L2-normalized 512-d embedding,
// plus verify / analyze / detect and an optional MiniFASNet anti-spoof veto.

// Opaque face-recognition context (wraps a loaded model pack + last-error buffer).
typedef struct facedetect_ctx facedetect_ctx;

// ABI version of this header/implementation. Bump on any breaking change to the
// function signatures or semantics below. Additive changes (new functions) are
// fine without a bump.
//
// CHANGELOG
// ---------
// v1: initial surface - load/free, last-error, embed_path / embed_rgb (+
//     free_vec), detect_path_json, verify_paths, analyze_path_json.
int facedetect_capi_abi_version(void);

// Load a GGUF model pack (SCRFD detector + ArcFace recognizer, and optionally
// the genderage + MiniFASNet anti-spoof heads). Returns an owning context, or
// NULL on failure. The returned context must be released with
// facedetect_capi_free.
facedetect_ctx* facedetect_capi_load(const char* gguf_path);

// Free a context obtained from facedetect_capi_load. Safe on NULL.
void facedetect_capi_free(facedetect_ctx* ctx);

// Human-readable description of the last error on `ctx`, or "" if none. The
// returned pointer is owned by the context and valid until the next call on it
// (or until facedetect_capi_free). Returns "" if `ctx` is NULL.
const char* facedetect_capi_last_error(facedetect_ctx* ctx);

// Free a string previously returned by facedetect_capi_detect_path_json /
// facedetect_capi_analyze_path_json. Safe on NULL.
void facedetect_capi_free_string(char* s);

// Free a float vector previously returned via facedetect_capi_embed_path /
// facedetect_capi_embed_rgb. Safe on NULL.
void facedetect_capi_free_vec(float* v);

// Detect the primary (largest / highest-scoring) face in an image FILE, align it
// to 112x112 (insightface norm_crop) and run ArcFace, returning the
// L2-normalized embedding. On success returns 0 and writes a malloc'd float
// array of length `*out_dim` (typically 512) to `*out_vec` (release with
// facedetect_capi_free_vec). On error returns nonzero, sets the context's last
// error (see facedetect_capi_last_error), and leaves `*out_vec`/`*out_dim`
// unchanged.
int facedetect_capi_embed_path(facedetect_ctx* ctx, const char* image_path,
                               float** out_vec, int* out_dim);

// Like facedetect_capi_embed_path but from raw, tightly-packed 8-bit RGB pixels
// (`rgb`, `width`*`height`*3 bytes, row-major, no padding). Same return /
// ownership contract.
int facedetect_capi_embed_rgb(facedetect_ctx* ctx, const unsigned char* rgb,
                              int width, int height, float** out_vec, int* out_dim);

// Detect ALL faces in an image FILE, returning a malloc'd UTF-8 JSON document
// (free with facedetect_capi_free_string) of the form:
//
//   {"faces":[{"score":0.9971,
//              "box":[x1,y1,x2,y2],
//              "landmarks":[[x,y],[x,y],[x,y],[x,y],[x,y]]}, ...]}
//
// where "box" is the detection rectangle in pixel coordinates and "landmarks"
// are the 5 facial keypoints (left eye, right eye, nose, left mouth, right
// mouth), the same order insightface emits. Returns NULL on error and sets the
// context's last error.
char* facedetect_capi_detect_path_json(facedetect_ctx* ctx, const char* image_path);

// Verify whether two images contain the same identity. Each image is run through
// the detect+align+embed pipeline (primary face); the cosine DISTANCE
// (1 - cosine_similarity) between the two embeddings is written to
// `*out_distance` and `*out_verified` is set to 1 iff the distance is <=
// `threshold` (insightface buffalo default 0.35; pass <=0 to use the default).
// When `anti_spoof` is nonzero and the pack carries the MiniFASNet ensemble, a
// spoofed face vetoes the match (`*out_verified` forced to 0). On success
// returns 0; on error returns nonzero, sets the context's last error, and
// leaves the out-params unchanged.
int facedetect_capi_verify_paths(facedetect_ctx* ctx, const char* a, const char* b,
                                 float threshold, int anti_spoof,
                                 float* out_distance, int* out_verified);

// Run age/gender analysis on every detected face in an image FILE, returning a
// malloc'd UTF-8 JSON document (free with facedetect_capi_free_string):
//
//   {"faces":[{"score":0.9971,
//              "box":[x1,y1,x2,y2],
//              "age":31,
//              "gender":"M"}, ...]}
//
// "age" is the rounded predicted age in years and "gender" is "M"/"F" from the
// genderage head. Returns NULL on error and sets the context's last error.
char* facedetect_capi_analyze_path_json(facedetect_ctx* ctx, const char* image_path);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // FACEDETECT_CAPI_H
