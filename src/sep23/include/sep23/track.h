// Copyright by BeeX [2026]
#pragma once

#include <sep23/cloud.h>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <vector>

namespace sep23 {

struct TrackSettings {
    double depth_gate = 0.0;  // metres either side of the target's depth
    double roi_radius = 0.0;  // metres from the target: keeps a tilted seabed's slice of the gate out
    int    min_points = 0;    // fewer inliers than this and corners are detected again
    int    max_points = 0;
    double ransac_px  = 0.0;
};

// What one frame of tracking found. Points are pixels; the target is in camera_link (x right, y up, looking down -z).
struct TrackResult {
    bool                     ok = false;
    Eigen::Vector2d          target_px = Eigen::Vector2d::Zero();
    Eigen::Vector3d          target    = Eigen::Vector3d::Zero();
    Eigen::Vector3d          offset    = Eigen::Vector3d::Zero();  // target now minus target when tracking started
    std::vector<cv::Point2f> from, to;                             // every flowed point, previous frame then this one
    std::vector<uint8_t>     inlier;                               // per flowed point: in the 2D consensus
    size_t                   lost = 0;                             // points KLT or the gate dropped this frame
    bool                     reseeded = false;
    cv::Mat                  mask;                                 // 255 inside the depth gate and ROI
};

// Follows the target through the arm's occlusion: KLT on the full-resolution image inside a depth gate around the target,
// one 2D similarity by RANSAC moves the target pixel, and only the inliers' cloud points give it depth.
class Tracker {
public:
    Tracker(const TrackSettings &s, const CameraModel &camera) : s_(s), camera_(camera) {}

    // Target in camera_link; the next frame seeds the corners.
    void start(const Eigen::Vector3d &target);
    void stop() { active_ = false; }
    bool active() const { return active_; }

    // Grey image and its organised cloud (one camera_link point per pixel, row-major).
    TrackResult update(const cv::Mat &gray, const std::vector<Eigen::Vector3f> &points);

private:
    cv::Mat              gateMask(const std::vector<Eigen::Vector3f> &points) const;
    std::vector<double>  depthsAt(const std::vector<cv::Point2f> &px, const std::vector<Eigen::Vector3f> &points) const;
    Eigen::Vector3d      unproject(const Eigen::Vector2d &px, double depth) const;
    Eigen::Vector2d      project(const Eigen::Vector3d &p) const;
    void                 seed(const cv::Mat &gray, const cv::Mat &mask);

    TrackSettings            s_;
    CameraModel              camera_;
    bool                     active_ = false, seeded_ = false, calibrated_ = false;
    Eigen::Vector3d          start_target_ = Eigen::Vector3d::Zero(), target_ = Eigen::Vector3d::Zero();
    Eigen::Vector2d          target_px_ = Eigen::Vector2d::Zero();
    double                   target_above_ = 0.0;  // the target's depth minus its first inliers' median: a handle stands proud
    cv::Mat                  previous_;
    std::vector<cv::Point2f> points_;
};

enum class TrackView { RANSAC, KLT, MASK };

// The debug overlay for one view, on a copy of the colour image.
cv::Mat drawTrack(const cv::Mat &bgr, const TrackResult &r, TrackView view);

}  // namespace sep23
