#include "detect.hpp"
#include <cstdio>
#include <vector>

// Host-side NMS is implemented (no model needed), so this is a real behavioral
// test, not a skip. Two heavily-overlapping boxes + one disjoint box: greedy
// IoU NMS must keep the higher-scoring of the overlap pair and the disjoint box.
int main() {
    std::vector<fd::Detection> dets;
    fd::Detection a; a.x1 = 0;   a.y1 = 0;   a.x2 = 10;  a.y2 = 10;  a.score = 0.9f;
    fd::Detection b; b.x1 = 1;   b.y1 = 1;   b.x2 = 11;  b.y2 = 11;  b.score = 0.8f; // ~IoU>0.4 with a
    fd::Detection c; c.x1 = 100; c.y1 = 100; c.x2 = 110; c.y2 = 110; c.score = 0.7f; // disjoint
    dets = {a, b, c};

    std::vector<int> keep = fd::nms(dets, 0.4f);
    if (keep.size() != 2) {
        std::fprintf(stderr, "expected 2 kept, got %zu\n", keep.size());
        return 1;
    }
    // Highest score (a, idx 0) must be first; the disjoint box (c, idx 2) kept.
    if (keep[0] != 0) { std::fprintf(stderr, "expected idx 0 first, got %d\n", keep[0]); return 1; }
    bool has_c = false;
    for (int k : keep) if (k == 2) has_c = true;
    if (!has_c) { std::fprintf(stderr, "disjoint box not kept\n"); return 1; }

    std::printf("nms OK: kept %zu boxes\n", keep.size());
    return 0;
}
