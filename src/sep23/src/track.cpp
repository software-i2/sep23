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

double median(std::vector<double> v) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

bool finite(const Eigen::Vector3f &p) { return std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z()) && p.z() < 0.0f; }

}  // namespace

void Tracker::start(const Eigen::Vector3d &target) {
    active_ = true, seeded_ = calibrated_ = false;
    start_target_ = target_ = target;
    target_px_              = project(target);
    points_.clear();
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

std::vector<double> Tracker::depthsAt(const std::vector<cv::Point2f> &px, const std::vector<Eigen::Vector3f> &points) const {
    std::vector<double> out;
    for (const cv::Point2f &p : px) {
        const int u = static_cast<int>(std::lround(p.x)), v = static_cast<int>(std::lround(p.y));
        if (u >= 0 && v >= 0 && u < camera_.width && v < camera_.height) {
            const Eigen::Vector3f &q = points[static_cast<size_t>(v) * camera_.width + u];
            if (finite(q)) {
                out.push_back(-q.z());
            }
        }
    }
    return out;
}

void Tracker::seed(const cv::Mat &gray, const cv::Mat &mask) {
    // Corners already tracked keep their place; new ones fill the rest of the gate away from them.
    cv::Mat free = mask.clone();
    for (const cv::Point2f &p : points_) {
        cv::circle(free, p, 7, cv::Scalar(0), -1);
    }
    std::vector<cv::Point2f> fresh;
    const int                want = s_.max_points - static_cast<int>(points_.size());
    if (want > 0) {
        cv::goodFeaturesToTrack(gray, fresh, want, 0.01, 7, free);
    }
    points_.insert(points_.end(), fresh.begin(), fresh.end());
}

TrackResult Tracker::update(const cv::Mat &gray, const std::vector<Eigen::Vector3f> &points) {
    TrackResult r;
    r.target_px = target_px_, r.target = target_, r.offset = target_ - start_target_;
    if (!active_ || points.size() != static_cast<size_t>(camera_.width) * camera_.height || gray.cols != camera_.width
        || gray.rows != camera_.height) {
        return r;
    }
    r.mask = gateMask(points);
    if (!seeded_) {
        seed(gray, r.mask);
        if (points_.size() < 3) {
            points_.clear();
            return r;  // nothing textured at the target's depth yet; try the next frame
        }
        previous_ = gray.clone();
        seeded_ = r.ok = r.reseeded = true;
        return r;
    }

    std::vector<cv::Point2f> moved;
    std::vector<uint8_t>     status;
    std::vector<float>       error;
    if (!points_.empty()) {
        cv::calcOpticalFlowPyrLK(previous_, gray, points_, moved, status, error, cv::Size(21, 21), 3);
    }
    // Only points that land back inside the gate count: the arm in front of the mine, the seabed behind it, fall out here.
    for (size_t i = 0; i < points_.size(); ++i) {
        const cv::Point2i at(static_cast<int>(std::lround(moved[i].x)), static_cast<int>(std::lround(moved[i].y)));
        if (status[i] && at.x >= 0 && at.y >= 0 && at.x < camera_.width && at.y < camera_.height && r.mask.at<uint8_t>(at)) {
            r.from.push_back(points_[i]);
            r.to.push_back(moved[i]);
        } else {
            ++r.lost;
        }
    }

    std::vector<cv::Point2f> kept;
    if (r.from.size() >= 3) {
        cv::Mat         inliers;
        const cv::Mat   a = cv::estimateAffinePartial2D(r.from, r.to, inliers, cv::RANSAC, s_.ransac_px, 2000, 0.99);
        if (!a.empty()) {
            r.inlier.assign(inliers.begin<uint8_t>(), inliers.end<uint8_t>());
            for (size_t i = 0; i < r.to.size(); ++i) {
                if (r.inlier[i]) {
                    kept.push_back(r.to[i]);
                }
            }
            const Eigen::Vector2d was = target_px_;
            target_px_ = {a.at<double>(0, 0) * was.x() + a.at<double>(0, 1) * was.y() + a.at<double>(0, 2),
                          a.at<double>(1, 0) * was.x() + a.at<double>(1, 1) * was.y() + a.at<double>(1, 2)};
        }
    }
    r.inlier.resize(r.to.size(), 0);
    const std::vector<double> depths = depthsAt(kept, points);
    if (kept.size() >= 3 && !depths.empty()) {
        // Taken from the first consensus rather than the seeds, so an arm already inside the gate cannot bias it.
        if (!calibrated_) {
            target_above_ = -target_.z() - median(depths);
            calibrated_   = true;
        }
        target_ = unproject(target_px_, median(depths) + target_above_);
        r.ok    = true;
    }
    points_ = kept;
    if (static_cast<int>(points_.size()) < s_.min_points) {
        seed(gray, gateMask(points));
        r.reseeded = true;
    }
    previous_ = gray.clone();
    r.target_px = target_px_, r.target = target_, r.offset = target_ - start_target_;
    return r;
}

cv::Mat drawTrack(const cv::Mat &bgr, const TrackResult &r, TrackView view) {
    cv::Mat out = bgr.clone();
    const cv::Scalar green(60, 200, 60), red(40, 40, 230), amber(0, 190, 255), cyan(230, 220, 40);
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
    const cv::Point c(static_cast<int>(std::lround(r.target_px.x())), static_cast<int>(std::lround(r.target_px.y())));
    cv::circle(out, c, 8, r.ok ? cyan : red, 2, cv::LINE_AA);
    const size_t inliers = static_cast<size_t>(std::count(r.inlier.begin(), r.inlier.end(), 1));
    char         line[160];
    std::snprintf(line, sizeof(line), "%s  %zu/%zu inliers, %zu lost%s  offset %+.1f %+.1f %+.1f mm",
                  view == TrackView::KLT ? "klt" : view == TrackView::RANSAC ? "ransac" : "mask", inliers, r.to.size(), r.lost,
                  r.reseeded ? ", reseeded" : "", 1000 * r.offset.x(), 1000 * r.offset.y(), 1000 * r.offset.z());
    cv::putText(out, line, cv::Point(12, 26), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(out, line, cv::Point(12, 26), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    return out;
}

}  // namespace sep23
