#include "preprocess.hpp"
#include <cstdio>
#include <cmath>
int main() {
    fd::Image img; img.width = 2; img.height = 1; img.rgb = {10,20,30, 40,50,60};
    // ArcFace params: (p-127.5)/127.5, swapRB=true (emit B,G,R per pixel), NCHW.
    auto blob = fd::to_blob(img, 2, 127.5f, 127.5f, true);
    // NCHW [1,3,1,2]; channel 0 (R-as-output-first after swap = B). swapRB on RGB
    // input means output channel order is B,G,R, i.e. plane0=B, plane1=G, plane2=R.
    // plane0 (B) pixel0 = (30-127.5)/127.5
    if (std::fabs(blob[0] - (30-127.5f)/127.5f) > 1e-5f) { std::fprintf(stderr,"p0\n"); return 1; }
    if (std::fabs(blob[2] - (20-127.5f)/127.5f) > 1e-5f) { std::fprintf(stderr,"g\n"); return 1; }
    if (std::fabs(blob[4] - (10-127.5f)/127.5f) > 1e-5f) { std::fprintf(stderr,"r\n"); return 1; }
    std::printf("preprocess OK\n"); return 0;
}
