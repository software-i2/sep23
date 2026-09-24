// Copyright by BeeX [2026]
#include <sep23/collision.h>

#include <algorithm>
#include <stdexcept>

namespace sep23 {
namespace {

bool insideHull(const Eigen::Vector3d &p, const Hull &h) {
    return p.z() < h.floor_z && p.x() >= h.min_x && p.x() <= h.max_x && p.y() >= h.min_y && p.y() <= h.max_y;
}

// Clips the parameter range [lo, hi] of a segment to lower <= from + t * span <= upper.
bool clip(double from, double span, double lower, double upper, double &lo, double &hi) {
    if (std::fabs(span) < 1e-12) {
        return from >= lower && from <= upper;
    }
    double enter = (lower - from) / span, leave = (upper - from) / span;
    if (enter > leave) {
        std::swap(enter, leave);
    }
    lo = std::max(lo, enter);
    hi = std::min(hi, leave);
    return lo <= hi;
}

// A link can cross the footprint with both ends outside it, so the whole segment is clipped.
bool linkInsideHull(const Segment &s, const Hull &h) {
    const Eigen::Vector3d span = s.b - s.a;
    double                lo = 0.0, hi = 1.0;
    return clip(s.a.x(), span.x(), h.min_x, h.max_x, lo, hi) && clip(s.a.y(), span.y(), h.min_y, h.max_y, lo, hi)
           && clip(s.a.z(), span.z(), -1e9, h.floor_z, lo, hi) && lo < hi;
}

void inflate(const VoxelBox &box, const std::vector<uint32_t> &cells, double radius, std::vector<uint8_t> &layer) {
    const int                       reach = static_cast<int>(std::floor(radius / box.voxel));
    std::vector<std::array<int, 3>> offsets;
    for (int z = -reach; z <= reach; ++z) {
        for (int y = -reach; y <= reach; ++y) {
            for (int x = -reach; x <= reach; ++x) {
                if ((x * x + y * y + z * z) * box.voxel * box.voxel <= radius * radius) {
                    offsets.push_back({x, y, z});
                }
            }
        }
    }
    for (const uint32_t cell : cells) {
        long cx = 0, cy = 0, cz = 0;
        box.coords(cell, cx, cy, cz);
        for (const auto &o : offsets) {
            const long x = cx + o[0], y = cy + o[1], z = cz + o[2];
            if (x >= 0 && y >= 0 && z >= 0 && x < box.nx && y < box.ny && z < box.nz) {
                layer[box.index(x, y, z)] = 1;
            }
        }
    }
}

}  // namespace

bool hitsHull(const Body &body, const Hull &hull) {
    if (insideHull(body.throat, hull) || insideHull(body.tip, hull)) {
        return true;
    }
    for (const Segment &s : body.links) {
        if (linkInsideHull(s, hull)) {
            return true;
        }
    }
    return std::any_of(body.blades.begin(), body.blades.end(), [&](const Eigen::Vector3d &p) { return insideHull(p, hull); });
}

ObstacleGrid::ObstacleGrid(const ObstacleMap &map, double link_radius, double blade_radius) : box_(map.box) {
    if (!(box_.voxel > 0.0)) {
        throw std::invalid_argument("the obstacle map has no voxel size");
    }
    const size_t count = static_cast<size_t>(box_.count());
    for (const std::vector<uint32_t> *cells : {&map.obstacle, &map.handle}) {
        if (std::any_of(cells->begin(), cells->end(), [&](uint32_t c) { return c >= count; })) {
            throw std::invalid_argument("the obstacle map lists a cell outside its own box");
        }
    }
    std::vector<uint32_t> occupied = map.obstacle;
    occupied.insert(occupied.end(), map.handle.begin(), map.handle.end());
    link_.assign(count, 0);
    inflate(box_, occupied, link_radius, link_);
    blade_.assign(count, 0);
    inflate(box_, map.obstacle, blade_radius, blade_);
    for (const uint32_t cell : map.handle) {
        blade_[cell] = 1;
    }
}

bool ObstacleGrid::blocked(const std::vector<uint8_t> &layer, const Eigen::Vector3d &p) const {
    const long cell = box_.cellOf(query_to_map_ * p);
    return cell >= 0 && layer[cell] != 0;
}

Verdict Collision::check(const Joints &q) {
    if (!arm_.withinLimits(q)) {
        return Verdict::JOINT_LIMIT;
    }
    arm_.body(q, body_, blade_stride_);
    if (hitsHull(body_, hull_)) {
        return Verdict::HULL;
    }
    for (const Segment &s : body_.links) {
        const int steps = std::max(1, static_cast<int>(std::ceil((s.b - s.a).norm() / link_step_)));
        for (int n = 0; n <= steps; ++n) {
            if (grid_.linkBlocked(s.a + (s.b - s.a) * (static_cast<double>(n) / steps))) {
                return Verdict::OBSTACLE;
            }
        }
    }
    for (const Eigen::Vector3d &p : body_.blades) {
        if (grid_.bladeBlocked(p)) {
            return Verdict::OBSTACLE;
        }
    }
    return Verdict::CLEAR;
}

bool Collision::segmentClear(const Joints &a, const Joints &b, double step) {
    const int steps = std::max(1, static_cast<int>(std::ceil(largestMove(a, b) / step)));
    for (int n = 1; n < steps; ++n) {
        Joints q;
        for (int j = 0; j < JOINT_COUNT; ++j) {
            q[j] = a[j] + (b[j] - a[j]) * n / steps;
        }
        if (check(q) != Verdict::CLEAR) {
            return false;
        }
    }
    return true;
}

}  // namespace sep23
