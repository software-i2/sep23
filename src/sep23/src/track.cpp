// Copyright by BeeX [2026]
#include <sep23/track.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace sep23 {
namespace {

bool finite(const Eigen::Vector3f &p) { return std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z()) && p.z() < 0.0f; }

const cv::Size kWindow(21, 21);  // KLT window
const int      kEdgePx = 10;     // corners kept this far inside the gate: a window across its edge is dragged by what is past it
const float    kBackPx = 1.0f;   // forward-backward check (Kalal et al. 2010): a corner flowed back further off was dragged

}  // namespace

void Tracker::start(const Eigen::Vector3d &target) {
    active_ = true, seeded_ = covered_ = false;
    start_target_ = target_ = target;
    target_px_ = start_px_  = project(target);
    points_.clear(), start_depth_.clear();
    depth_change_ = 0.0;
    smoothed_     = Eigen::Vector3d::Zero();
}

// camera_link looks down -z with y up, so depth is -z and image v grows as y falls.
Eigen::Vector3d Tracker::unproject(const Eigen::Vector2d &px, double depth) const {
    return {(px.x() - camera_.cx) * depth / camera_.fx, -(px.y() - camera_.cy) * depth / camera_.fy, -depth};
}

Eigen::Vector2d Tracker::project(const Eigen::Vector3d &p) const {
    return {camera_.cx + camera_.fx * p.x() / -p.z(), camera_.cy - camera_.fy * p.y() / -p.z()};
}

cv::Mat Tracker::gateMask(const std::vector<Eigen::Vector3f> &points) const {
    cv::Mat         mask(camera_.height, camera_.width, CV_8U, cv::Scalar(0));
    const double    depth = -target_.z();
    const float     r2    = static_cast<float>(s_.roi_radius * s_.roi_radius);
    const Eigen::Vector3f target = target_.cast<float>();
    for (int v = 0; v < camera_.height; ++v) {
        uint8_t *row = mask.ptr<uint8_t>(v);
        for (int u = 0; u < camera_.width; ++u) {
            const Eigen::Vector3f &p = points[static_cast<size_t>(v) * camera_.width + u];
            row[u] = finite(p) && std::fabs(-p.z() - depth) <= s_.depth_gate && (p - target).squaredNorm() <= r2 ? 255 : 0;
        }
    }
    return mask;
}

// One depth per pixel, NaN where the cloud has none.
std::vector<double> Tracker::depthsAt(const std::vector<cv::Point2f> &px, const std::vector<Eigen::Vector3f> &points) const {
    std::vector<double> out(px.size(), std::nan(""));
    for (size_t i = 0; i < px.size(); ++i) {
        const int u = static_cast<int>(std::lround(px[i].x)), v = static_cast<int>(std::lround(px[i].y));
        if (u >= 0 && v >= 0 && u < camera_.width && v < camera_.height) {
            const Eigen::Vector3f &q = points[static_cast<size_t>(v) * camera_.width + u];
            if (finite(q)) {
                out[i] = -q.z();
            }
        }
    }
    return out;
}

void Tracker::seed(const cv::Mat &gray, const cv::Mat &mask, const std::vector<Eigen::Vector3f> &points) {
    // Corners already tracked keep their place; new ones fill the rest of the gate away from them.
    cv::Mat free;
    cv::erode(mask, free, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * kEdgePx + 1, 2 * kEdgePx + 1)));
    for (const cv::Point2f &p : points_) {
        cv::circle(free, p, 7, cv::Scalar(0), -1);
    }
    std::vector<cv::Point2f> fresh;
    const int                want = s_.max_points - static_cast<int>(points_.size());
    if (want > 0) {
        cv::goodFeaturesToTrack(gray, fresh, want, 0.01, 7, free);
    }
    points_.insert(points_.end(), fresh.begin(), fresh.end());
    // Each new corner's depth at the start: what it reads now, less how far the target's depth has changed since.
    for (const double d : depthsAt(fresh, points)) {
        start_depth_.push_back(d - depth_change_);
    }
}

TrackResult Tracker::update(const cv::Mat &gray, const std::vector<Eigen::Vector3f> &points) {
    TrackResult r;
    r.target_px = target_px_, r.start_px = start_px_, r.target = start_target_ + smoothed_, r.offset = smoothed_;
    if (!active_ || points.size() != static_cast<size_t>(camera_.width) * camera_.height || gray.cols != camera_.width
        || gray.rows != camera_.height) {
        return r;
    }
    r.mask            = gateMask(points);
    const double area = cv::countNonZero(r.mask);
    if (!seeded_) {
        seed(gray, r.mask, points);
        if (points_.size() < 3) {
            points_.clear(), start_depth_.clear();
            return r;  // nothing textured at the target's depth yet; try the next frame
        }
        previous_      = gray.clone();
        face_at_start_ = area;
        seeded_ = r.ok = r.reseeded = true;
        return r;
    }
    // With most of the face hidden, the few corners left, far from the target, swing it wildly: stop rather than follow them.
    r.face = area / face_at_start_;
    if (covered_ || r.face < s_.min_face) {
        covered_ = r.covered = true;
        return r;
    }

    std::vector<cv::Point2f> moved, back;
    std::vector<uint8_t>     status, back_status;
    std::vector<float>       error;
    if (!points_.empty()) {
        cv::calcOpticalFlowPyrLK(previous_, gray, points_, moved, status, error, kWindow, 3);
        cv::calcOpticalFlowPyrLK(gray, previous_, moved, back, back_status, error, kWindow, 3);
    }
    // Only points that flow back where they came from and land inside the gate, clear of its edge, count: the arm in front
    // of the mine, the seabed behind it and corners dragged by an occluder's edge fall out here.
    cv::Mat inside;
    cv::erode(r.mask, inside, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * kEdgePx + 1, 2 * kEdgePx + 1)));
    std::vector<double> from_depth;
    for (size_t i = 0; i < points_.size(); ++i) {
        const cv::Point2i at(static_cast<int>(std::lround(moved[i].x)), static_cast<int>(std::lround(moved[i].y)));
        const bool        returned = back_status[i] && cv::norm(back[i] - points_[i]) < kBackPx;
        if (status[i] && returned && at.x >= 0 && at.y >= 0 && at.x < camera_.width && at.y < camera_.height && inside.at<uint8_t>(at)) {
            r.from.push_back(points_[i]);
            r.to.push_back(moved[i]);
            from_depth.push_back(start_depth_[i]);
        } else {
            ++r.lost;
        }
    }

    std::vector<cv::Point2f> kept;
    std::vector<double>      kept_depth;
    if (r.from.size() >= 3) {
        cv::Mat         inliers;
        const cv::Mat   a = cv::estimateAffinePartial2D(r.from, r.to, inliers, cv::RANSAC, s_.ransac_px, 2000, 0.99);
        if (!a.empty()) {
            r.inlier.assign(inliers.begin<uint8_t>(), inliers.end<uint8_t>());
            for (size_t i = 0; i < r.to.size(); ++i) {
                if (r.inlier[i]) {
                    kept.push_back(r.to[i]);
                    kept_depth.push_back(from_depth[i]);
                }
            }
            const Eigen::Vector2d was = target_px_;
            target_px_ = {a.at<double>(0, 0) * was.x() + a.at<double>(0, 1) * was.y() + a.at<double>(0, 2),
                          a.at<double>(1, 0) * was.x() + a.at<double>(1, 1) * was.y() + a.at<double>(1, 2)};
            r.moved_px = target_px_ - was;
            r.rotation = std::atan2(a.at<double>(1, 0), a.at<double>(0, 0));
            r.scale    = std::hypot(a.at<double>(0, 0), a.at<double>(1, 0));
        }
    }
    r.inlier.resize(r.to.size(), 0);
    // The target's depth moves by the inliers' median change in their own depth, which no mix of corners can bias.
    const std::vector<double> depths = depthsAt(kept, points);
    std::vector<double>       change;
    for (size_t i = 0; i < kept.size(); ++i) {
        if (!std::isnan(depths[i]) && !std::isnan(kept_depth[i])) {
            change.push_back(depths[i] - kept_depth[i]);
        }
    }
    if (kept.size() >= 3 && !change.empty()) {
        depth_change_ = median(change);
        r.depth       = -start_target_.z() + depth_change_;
        target_       = unproject(target_px_, r.depth);
        smoothed_     = s_.ema_alpha * (target_ - start_target_) + (1.0 - s_.ema_alpha) * smoothed_;
        r.ok          = true;
    }
    points_      = kept;
    start_depth_ = kept_depth;
    if (static_cast<int>(points_.size()) < s_.min_points) {
        seed(gray, gateMask(points), points);
        r.reseeded = true;
    }
    previous_ = gray.clone();
    r.target_px = target_px_, r.target = start_target_ + smoothed_, r.offset = smoothed_;
    return r;
}

cv::Mat drawTrack(const cv::Mat &bgr, const TrackResult &r, TrackView view, const Eigen::Vector3d &aim, const Eigen::Vector2d &aim_px,
                  const std::string &steering) {
    cv::Mat out = bgr.clone();
    const cv::Scalar green(60, 200, 60), red(40, 40, 230), amber(0, 190, 255), cyan(230, 220, 40), magenta(230, 60, 230);
    if (view == TrackView::MASK && !r.mask.empty()) {
        cv::Mat dim = out * 0.35, tint(out.size(), out.type(), green);
        cv::addWeighted(out, 0.75, tint, 0.25, 0.0, tint);
        dim.copyTo(out);
        tint.copyTo(out, r.mask);
    }
    for (size_t i = 0; i < r.to.size(); ++i) {
        if (view == TrackView::KLT) {
            cv::line(out, r.from[i], r.to[i], amber, 1, cv::LINE_AA);
            cv::circle(out, r.to[i], 2, amber, -1, cv::LINE_AA);
        } else if (view == TrackView::RANSAC) {
            cv::circle(out, r.to[i], 3, r.inlier[i] ? green : red, -1, cv::LINE_AA);
        }
    }
    // Every view: where tracking started (ring), the target now (cross) and the way between; with steering, where the arm aims.
    const auto      px = [](const Eigen::Vector2d &p) { return cv::Point(static_cast<int>(std::lround(p.x())), static_cast<int>(std::lround(p.y()))); };
    const cv::Point c = px(r.target_px), s = px(r.start_px);
    const cv::Scalar mark = r.covered ? red : r.ok ? cyan : amber;
    cv::circle(out, s, 10, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::arrowedLine(out, s, c, mark, 2, cv::LINE_AA, 0, 0.2);
    cv::drawMarker(out, c, mark, cv::MARKER_CROSS, 24, 2, cv::LINE_AA);
    cv::putText(out, "target", c + cv::Point(10, -10), cv::FONT_HERSHEY_SIMPLEX, 0.5, mark, 1, cv::LINE_AA);
    if (!steering.empty()) {
        const cv::Point a = px(aim_px);
        cv::line(out, a, c, magenta, 1, cv::LINE_AA);
        cv::drawMarker(out, a, magenta, cv::MARKER_DIAMOND, 18, 2, cv::LINE_AA);
        cv::putText(out, "aim", a + cv::Point(10, 14), cv::FONT_HERSHEY_SIMPLEX, 0.5, magenta, 1, cv::LINE_AA);
    }

    // Text: only what to act on. Markers already label target/aim, so no legend.
    std::vector<std::string> lines;
    char                     line[200];
    if (!steering.empty()) {
        const Eigen::Vector3d gap = 1000 * (r.target - aim);
        std::snprintf(line, sizeof(line), "target-aim %+.0f %+.0f %+.0f mm (%.0f)", gap.x(), gap.y(), gap.z(), gap.norm());
        lines.push_back(line);
    }
    const size_t inliers = static_cast<size_t>(std::count(r.inlier.begin(), r.inlier.end(), 1));
    std::snprintf(line, sizeof(line), "moved %+.0f %+.0f %+.0f mm  %zu/%zu%s", 1000 * r.offset.x(), 1000 * r.offset.y(), 1000 * r.offset.z(),
                  inliers, r.to.size(), r.covered ? "  COVERED" : r.ok ? "" : r.to.empty() ? "  seeding" : "  held");
    lines.push_back(line);
    if (!steering.empty()) {
        lines.push_back(steering);
    }
    for (size_t i = 0; i < lines.size(); ++i) {
        const cv::Point at(12, 26 + 24 * static_cast<int>(i));
        cv::putText(out, lines[i], at, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
        cv::putText(out, lines[i], at, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    return out;
}

}  // namespace sep23
