// Copyright by BeeX [2026]
#pragma once

#include <sep23/arm.h>
#include <sep23/cloud.h>

namespace sep23 {

// The vehicle's underside in arm_base: inside the x-y footprint, everything below floor_z is hull.
struct Hull {
    double min_x = 0.0, max_x = 0.0, min_y = 0.0, max_y = 0.0, floor_z = 0.0;
};

bool hitsHull(const Body &body, const Hull &hull);

// Blade keep-out from obstacle cells: covers the gap between blade samples and a voxel centre's reach to its corners.
inline double bladeRadius(double voxel, double blade_step) { return (voxel + blade_step) * std::sqrt(3.0) / 2.0; }

// The obstacle map inflated for lookups; anything outside the map is free.
class ObstacleGrid {
public:
    // Links keep link_radius from every occupied cell; blades keep blade_radius from obstacles and stay out of handle cells.
    ObstacleGrid(const ObstacleMap &map, double link_radius, double blade_radius);

    bool linkBlocked(const Eigen::Vector3d &p) const { return blocked(link_, p); }
    bool bladeBlocked(const Eigen::Vector3d &p) const { return blocked(blade_, p); }

    // Where the arm would stand, relative to where the map was built; the map stays put.
    void setQueryToMap(const Eigen::Isometry3d &t) { query_to_map_ = t; }

private:
    bool blocked(const std::vector<uint8_t> &layer, const Eigen::Vector3d &p) const;

    VoxelBox             box_;
    Eigen::Isometry3d    query_to_map_ = Eigen::Isometry3d::Identity();
    std::vector<uint8_t> link_, blade_;
};

enum class Verdict { CLEAR, JOINT_LIMIT, HULL, OBSTACLE };

// Whether a posture is inside the limits, off the hull and clear of the map.
class Collision {
public:
    // A blade_stride above 1 skips blade samples: it can miss contact but never invent it.
    Collision(const Arm &arm, const ObstacleGrid &grid, const Hull &hull, double link_step, size_t blade_stride = 1)
            : arm_(arm), grid_(grid), hull_(hull), link_step_(link_step), blade_stride_(blade_stride) {}

    Verdict check(const Joints &q);
    Verdict sweep(const Joints &a, const Joints &b, double step);

    const Arm &arm() const { return arm_; }

private:
    const Arm          &arm_;
    const ObstacleGrid &grid_;
    Hull                hull_;
    double              link_step_;
    size_t              blade_stride_;
    Body                body_;
};

}  // namespace sep23
