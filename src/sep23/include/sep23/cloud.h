// Copyright by BeeX [2026]
#pragma once

#include <Eigen/Geometry>

#include <cstdint>
#include <string>
#include <vector>

namespace sep23 {

// Where the jaws could close, as vision offers it.
struct GraspPose {
    Eigen::Vector3d point, bar, approach;
};

// A box of voxels. Cell index = x + nx * (y + ny * z).
struct VoxelBox {
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    double          voxel  = 1.0;
    long            nx = 0, ny = 0, nz = 0;

    long            count() const { return nx * ny * nz; }
    long            index(long x, long y, long z) const { return x + nx * (y + ny * z); }
    long            cellOf(const Eigen::Vector3d &p) const;  // -1 outside
    Eigen::Vector3d centre(long cell) const;
    void            coords(long cell, long &x, long &y, long &z) const;
};

// Occupied cells in the arm base frame; handle cells are the part the blades may close on.
struct ObstacleMap {
    VoxelBox              box;
    std::vector<uint32_t> obstacle, handle;
};

struct CameraModel {
    int    width = 0, height = 0;
    double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
};

// One capture: an organised cloud (one point per pixel, row-major) and its grasp poses, in the camera frame.
struct Frame {
    std::vector<Eigen::Vector3f> points;
    std::vector<GraspPose>       poses;
    Eigen::Isometry3d            camera_to_arm = Eigen::Isometry3d::Identity();
};

struct CloudSettings {
    bool   outlier_filter = true, handle_carving = true, corridor_carving = true;
    bool   candidate_averaging = true, obstacle_averaging = true;
    double voxel = 0.0, crop_radius = 0.0;
    double depth_tolerance = 0.0;
    int    min_agreeing_neighbours = 0, slope_window_px = 0;
    int    min_points_per_voxel = 0;
    double free_space_tolerance = 0.0, max_ray_stretch = 1.0;
    double handle_radius = 0.0, bar_gap = 0.0;
    double corridor_length = 0.0, corridor_radius = 0.0;
    int    consensus_min_frames = 0;
    double match = 0.0, max_axis = 0.0, outlier = 0.0, max_spread = 0.0, duplicate = 0.0;
    int    vote_min_frames = 0, vote_radius = 0;
};

// What a look produced, in the arm base frame.
struct Scene {
    std::vector<GraspPose> poses;       // newest frame, uncropped
    std::vector<GraspPose> candidates;  // what the arm could reach after a park move
    ObstacleMap            map;
    std::string            summary;
};

// Frames oldest first; each stage follows its switch in `s`.
Scene processFrames(const std::vector<Frame> &frames, const CameraModel &camera, const CloudSettings &s);

// Spots on the bar that at least min_frames frames agree on, averaged. Frames oldest first.
std::vector<GraspPose> agreeOnSpots(const std::vector<std::vector<GraspPose>> &frames, const CloudSettings &s,
                                    std::string &summary);

// The middle value of a non-empty list.
double median(std::vector<double> v);

// The point nearest `p` on the handle the poses draw: each pose stands for its bar out to `half_length` either side. Not empty.
Eigen::Vector3d nearestOnHandle(const std::vector<GraspPose> &poses, double half_length, const Eigen::Vector3d &p);

}  // namespace sep23
