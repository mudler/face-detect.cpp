#include "model.hpp"
#include "common.hpp"
#include "align.hpp"
#include "detect.hpp"
#include "arcface_graph.hpp"
#include "sface_graph.hpp"
#include "genderage_graph.hpp"
#include "antispoof_graph.hpp"
#include "backend.hpp"

#include <algorithm>
#include <stdexcept>

namespace fd {

std::unique_ptr<Model> Model::load(const std::string& gguf_path) {
    auto m = std::unique_ptr<Model>(new Model());
    if (!m->loader_.load(gguf_path)) {
        FD_LOG("Model::load: failed to load %s", gguf_path.c_str());
        return nullptr;
    }
    return m;
}

std::vector<Detection> Model::detect(const Image& img) const {
    if (img.empty()) throw std::runtime_error("facedetect: empty image");
    // Detector graph stage is deferred; scrfd_detect is a stub for now.
    return scrfd_detect(loader_, img);
}

std::vector<float> Model::embed(const Image& img) const {
    if (img.empty()) throw std::runtime_error("facedetect: empty image");
    // Pick the primary (largest-area) detection, norm_crop to the 112x112
    // aligned RGB crop, then run the ArcFace IResNet50 embed graph (which is
    // L2-normalized internally), matching insightface's primary.normed_embedding.
    std::vector<Detection> dets = scrfd_detect(loader_, img);
    if (dets.empty()) throw std::runtime_error("facedetect: no face detected");
    const Detection& primary = *std::max_element(
        dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
            return (a.x2 - a.x1) * (a.y2 - a.y1) < (b.x2 - b.x1) * (b.y2 - b.y1);
        });
    Image aligned;
    if (!norm_crop(img, primary.landmarks, aligned, (int)loader_.config().rec_input_size))
        throw std::runtime_error("facedetect: alignment failed");
    // SFace (OpenCV-Zoo, Apache) shares the 5-point arcface_dst template that
    // norm_crop already uses (OpenCV FaceRecognizerSF::alignCrop), so the aligned
    // crop is the same; only the recognizer graph + preprocessing differ. Dispatch
    // on the configured recognizer.
    if (loader_.config().recognizer == "sface")
        return sface_embed(loader_, aligned, global_backend());
    return arcface_embed(loader_, aligned, global_backend());
}

std::vector<Face> Model::analyze(const Image& img) const {
    if (img.empty()) throw std::runtime_error("facedetect: empty image");
    // Per detected face, run the genderage head when the pack carries it (the `ga.`
    // weights). Anti-spoof remains a later stage. Each face is independently warped
    // to the 96x96 genderage crop (its own detection box), so multi-face images get
    // per-face age/gender rather than only the primary's.
    std::vector<Detection> dets = scrfd_detect(loader_, img);
    const bool have_ga = loader_.config().genderage_present;
    std::vector<Face> faces;
    faces.reserve(dets.size());
    for (const Detection& d : dets) {
        Face f;
        f.det = d;
        if (have_ga) {
            std::pair<int, int> ga = genderage(loader_, img, d, global_backend());
            f.gender = (ga.first == 1) ? 'M' : 'F';
            f.age = ga.second;
        }
        faces.push_back(std::move(f));
    }
    return faces;
}

bool Model::is_real(const Image& img, const Detection& d) const {
    if (!loader_.config().antispoof_present) return true;  // no models -> no veto
    return antispoof_score(*this, img, d) >= 0.5f;
}

} // namespace fd
