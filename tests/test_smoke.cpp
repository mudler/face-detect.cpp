#include "facedetect.h"
#include "facedetect_capi.h"
#include <cstdio>
#include <cstring>

int main() {
    const char* v = facedetect_version();
    if (v == nullptr || std::strlen(v) == 0) {
        std::fprintf(stderr, "version string is empty\n");
        return 1;
    }
    if (facedetect_capi_abi_version() < 1) {
        std::fprintf(stderr, "bad abi version\n");
        return 1;
    }
    std::printf("face-detect.cpp version: %s (abi %d)\n", v, facedetect_capi_abi_version());
    return 0;
}
