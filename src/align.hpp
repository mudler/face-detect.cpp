#pragma once
#include <array>
#include "image_io.hpp"

namespace fd {

// The 5 facial landmarks in insightface order: left eye, right eye, nose tip,
// left mouth corner, right mouth corner. Each is an (x, y) pixel coordinate in
// the source image.
using Landmarks5 = std::array<std::array<float, 2>, 5>;

// The canonical 112x112 ArcFace reference landmarks (insightface
// `arcface_dst`). The similarity transform maps a face's detected landmarks
// onto these so every aligned crop is in the same canonical frame.
extern const Landmarks5 kArcFaceRefLandmarks112;

// Align a face to a `size`x`size` (default 112) RGB crop via a similarity
// transform estimated from the 5 landmarks, reproducing insightface
// `norm_crop`/`estimate_norm` EXACTLY (umeyama similarity fit + warpAffine,
// bilinear sampling, border replicate). On success returns true and fills
// `out` with the aligned crop.
//
// TODO(impl): implement the umeyama similarity estimate (the 2x3 affine M) and
// the bilinear warp. This MUST be byte/near-byte identical to insightface's
// skimage SimilarityTransform path - parity is gated in docs/parity.md (aligned
// crop max-abs-diff). Until then this is a stub that returns false.
bool norm_crop(const Image& src, const Landmarks5& lmk, Image& out, int size = 112);

// OpenCV-faithful affine warp shared by every aligned crop (norm_crop's umeyama
// alignment and the genderage scale-about-center box crop). `M` is the FORWARD
// 2x3 affine [m00 m01 m02; m10 m11 m12] mapping source -> output pixels (the same
// convention cv2.warpAffine takes); it is inverted internally for backward
// bilinear sampling that reproduces cv2.warpAffine INTER_LINEAR (5-bit sub-pixel
// fixed point, BORDER_CONSTANT 0) to <=1 LSB. Fills `out` (out_w x out_h RGB);
// returns false on a degenerate (non-invertible) M.
bool warp_affine(const Image& src, const std::array<float, 6>& M, Image& out,
                 int out_w, int out_h);

} // namespace fd
