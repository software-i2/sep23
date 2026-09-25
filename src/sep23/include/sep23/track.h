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
    double min_face   = 0.0;  // share of the face seen at the start below which tracking stops for good
    double ema_alpha  = 1.0;  // each frame's weight in the running average of the offset; 1 takes every frame raw
};

// What one frame of tracking found. Points are pixels; the target is in camera_link (x right, y up, looking down -z).
struct TrackResult {
    bool                     ok = false;
    Eigen::Vector2d          target_px = Eigen::Vector2d::Zero();
    Eigen::Vector2d          start_px  = Eigen::Vector2d::Zero();       // where the target was when tracking started
    Eigen::Vector3d          target    = Eigen::Vector3d::Zero();
    Eigen::Vector3d          offset    = Eigen::Vector3d::Zero();  // target now minus at the start, averaged over frames (ema_alpha)
    std::vector<cv::Point2f> from, to;                             // every flowed point, previous frame then this one
    std::vector<uint8_t>     inlier;                               // per flowed point: in the 2D consensus
    size_t                   lost = 0;                             // points KLT or the gate dropped this frame
    Eigen::Vector2d          moved_px = Eigen::Vector2d::Zero();   // how this frame's RANSAC similarity moved the target pixel
    double                   rotation = 0.0, scale = 1.0;          // that similarity's turn (rad) and zoom
    double                   depth = 0.0;                          // the target's depth, this frame
    bool                     reseeded = false;
    cv::Mat                  mask;                                 // 255 inside the depth gate and ROI: the face tracked
    double                   face    = 1.0;                        // the mask's area over its area when tracking started
    bool                     covered = false;                      // too little face left: stopped until the next start
};

// Follows the target through the arm's occlusion: KLT on the full-resolution image inside a depth gate around the target,
// one 2D similarity by RANSAC moves the target pixel, and the inliers' own depth changes since they were seeded move its
// depth. Not their median depth: the corners left on a tilted face change with what hides it, and that would move the target.
class Tracker {
public:
    Tracker(const TrackSettings &s, const CameraModel &camera) : s_(s), camera_(camera) {}

    // Target in camera_link; the next frame seeds the corners.
    void start(const Eigen::Vector3d &target);
    Eigen::Vector2d project(const Eigen::Vector3d &p) const;  // camera_link to pixel
    void stop() { active_ = false; }
    bool active() const { return active_; }

    // Grey image and its organised cloud (one camera_link point per pixel, row-major).
    TrackResult update(const cv::Mat &gray, const std::vector<Eigen::Vector3f> &points);

private:
    cv::Mat              gateMask(const std::vector<Eigen::Vector3f> &points) const;
    std::vector<double>  depthsAt(const std::vector<cv::Point2f> &px, const std::vector<Eigen::Vector3f> &points) const;
    Eigen::Vector3d      unproject(const Eigen::Vector2d &px, double depth) const;
    void                 seed(const cv::Mat &gray, const cv::Mat &mask, const std::vector<Eigen::Vector3f> &points);

    TrackSettings            s_;
    CameraModel              camera_;
    bool                     active_ = false, seeded_ = false, covered_ = false;
    double                   face_at_start_ = 0.0;
    Eigen::Vector3d          start_target_ = Eigen::Vector3d::Zero(), target_ = Eigen::Vector3d::Zero();
    Eigen::Vector2d          target_px_ = Eigen::Vector2d::Zero(), start_px_ = target_px_;
    cv::Mat                  previous_;
    std::vector<cv::Point2f> points_;
    std::vector<double>      start_depth_;  // per point: its depth when seeded, less the depth change up to then (NaN: none)
    double                   depth_change_ = 0.0;                   // the inliers' median depth change since the start
    Eigen::Vector3d          smoothed_     = Eigen::Vector3d::Zero();  // the offset's running average
};

enum class TrackView { RANSAC, KLT, MASK };

// The debug overlay for one view, on a copy of the colour image: the frame's decision, and with `steering` set, where the arm
// aims (`aim` in camera_link, `aim_px` its pixel), the target-aim gap and what steering did.
cv::Mat drawTrack(const cv::Mat &bgr, const TrackResult &r, TrackView view, const Eigen::Vector3d &aim, const Eigen::Vector2d &aim_px,
                  const std::string &steering);

}  // namespace sep23
