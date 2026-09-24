// Copyright by BeeX [2026]
#include <sep23/cloud.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

namespace sep23 {

long VoxelBox::cellOf(const Eigen::Vector3d &p) const {
    const Eigen::Vector3d l = (p - origin) / voxel;
    const double          x = std::floor(l.x()), y = std::floor(l.y()), z = std::floor(l.z());
    if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) {
        return -1;
    }
    return index(static_cast<long>(x), static_cast<long>(y), static_cast<long>(z));
}

void VoxelBox::coords(long cell, long &x, long &y, long &z) const {
    x = cell % nx;
    y = (cell / nx) % ny;
    z = cell / (nx * ny);
}

Eigen::Vector3d VoxelBox::centre(long cell) const {
    long x = 0, y = 0, z = 0;
    coords(cell, x, y, z);
    return origin + voxel * Eigen::Vector3d(x + 0.5, y + 0.5, z + 0.5);
}

namespace {

constexpr double kShortest = 1e-9;

Eigen::Vector3d unit(const Eigen::Vector3d &v) { return v / std::max(v.norm(), kShortest); }

// Depth of each pixel of an organised cloud; NaN where the camera saw nothing.
class DepthView {
public:
    DepthView(const Frame &f, const CameraModel &c) : frame_(f), camera_(c) {}
    double at(int u, int v) const {
        const float z = frame_.points[static_cast<size_t>(v) * camera_.width + u].z();
        return std::isfinite(z) && -z > kShortest ? -z : std::numeric_limits<double>::quiet_NaN();
    }
    bool seen(int u, int v) const { return std::isfinite(at(u, v)); }
    bool inside(int u, int v) const { return u >= 0 && v >= 0 && u < camera_.width && v < camera_.height; }

private:
    const Frame       &frame_;
    const CameraModel &camera_;
};

// Pixels whose neighbours lie on the same local plane; the rest are flying pixels.
std::vector<uint8_t> supportedPixels(const DepthView &depth, const CameraModel &c, const CloudSettings &s) {
    const int W = c.width, H = c.height;
    // Summed-area tables over seen pixels, one row and column larger than the image.
    enum { N, U, V, UU, VV, UV, D, DU, DV, SUMS };
    using Sums = std::array<double, SUMS>;
    std::vector<Sums> t(static_cast<size_t>(W + 1) * (H + 1), Sums{});
    const auto        at = [&](int u, int v) { return static_cast<size_t>(v) * (W + 1) + u; };
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            const double z  = depth.seen(u, v) ? depth.at(u, v) : 0.0;
            const Sums   px = z > 0.0 ? Sums{1.0, 1.0 * u, 1.0 * v, 1.0 * u * u, 1.0 * v * v, 1.0 * u * v, z, z * u, z * v} : Sums{};
            const size_t i  = at(u + 1, v + 1);
            const size_t l = i - 1, up = i - (W + 1), ul = up - 1;
            for (int k = 0; k < SUMS; ++k) {
                t[i][k] = px[k] + t[l][k] + t[up][k] - t[ul][k];
            }
        }
    }

    const int            half = s.slope_window_px / 2;
    std::vector<uint8_t> out(static_cast<size_t>(W) * H, 0);
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            if (!depth.seen(u, v)) {
                continue;
            }
            const int u0 = std::max(0, u - half), v0 = std::max(0, v - half);
            const int u1 = std::min(W - 1, u + half), v1 = std::min(H - 1, v + half);
            Sums      w;
            for (int k = 0; k < SUMS; ++k) {
                w[k] = t[at(u1 + 1, v1 + 1)][k] - t[at(u1 + 1, v0)][k] - t[at(u0, v1 + 1)][k] + t[at(u0, v0)][k];
            }
            // A least-squares plane over the seen pixels, about their own centroid: holes and the image edge move it off (u, v).
            const double mu = w[U] / w[N], mv = w[V] / w[N];
            const double uu = w[UU] - w[U] * mu, vv = w[VV] - w[V] * mv, uv = w[UV] - w[U] * mv;
            const double zu = w[DU] - w[D] * mu, zv = w[DV] - w[D] * mv;
            const double det      = uu * vv - uv * uv;  // at least 1/count unless the seen pixels lie on one line
            const bool   planar   = det > 1e-2;
            const double slope_u  = planar ? (zu * vv - zv * uv) / det : 0.0;
            const double slope_v  = planar ? (zv * uu - zu * uv) / det : 0.0;
            int          agreeing = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if ((dx || dy) && depth.inside(u + dx, v + dy) && depth.seen(u + dx, v + dy)) {
                        const double expected = depth.at(u, v) + slope_u * dx + slope_v * dy;
                        agreeing += std::fabs(depth.at(u + dx, v + dy) - expected) <= s.depth_tolerance;
                    }
                }
            }
            out[static_cast<size_t>(v) * W + u] = agreeing >= s.min_agreeing_neighbours;
        }
    }
    return out;
}

// True when the surface measured in the cell's pixel lies clearly behind the cell centre.
bool seenThrough(const Eigen::Vector3d &centre, const DepthView &depth, const CameraModel &c, const CloudSettings &s,
                 double half_diagonal) {
    const double range = -centre.z();
    if (!(range > kShortest)) {
        return false;
    }
    const int u = static_cast<int>(std::lround(c.cx + c.fx * centre.x() / range));
    const int v = static_cast<int>(std::lround(c.cy - c.fy * centre.y() / range));
    if (!depth.inside(u, v) || !depth.seen(u, v)) {
        return false;
    }
    // A cell centre half a diagonal off a grazing surface reads far in front of it, so stretch the slack by 1/cos(tilt).
    const double z      = depth.at(u, v);
    double       across = 0.0, down = 0.0;
    if (depth.inside(u - 1, v) && depth.inside(u + 1, v) && depth.seen(u - 1, v) && depth.seen(u + 1, v)) {
        across = 0.5 * (depth.at(u + 1, v) - depth.at(u - 1, v)) * c.fx / z;
    }
    if (depth.inside(u, v - 1) && depth.inside(u, v + 1) && depth.seen(u, v - 1) && depth.seen(u, v + 1)) {
        down = 0.5 * (depth.at(u, v + 1) - depth.at(u, v - 1)) * c.fy / z;
    }
    const double stretch = std::min(std::sqrt(1.0 + across * across + down * down), s.max_ray_stretch);
    return range < z - (s.free_space_tolerance + half_diagonal * stretch);
}

std::vector<uint32_t> occupiedCells(const Frame &f, const CameraModel &c, const VoxelBox &box,
                                    const std::vector<uint8_t> &handle, const CloudSettings &s) {
    const DepthView            depth(f, c);
    const std::vector<uint8_t> supported = s.outlier_filter ? supportedPixels(depth, c, s) : std::vector<uint8_t>();
    std::vector<uint8_t>       hits(static_cast<size_t>(box.count()), 0);
    std::vector<long>          touched;
    for (size_t i = 0; i < f.points.size(); ++i) {
        if (!f.points[i].allFinite()) {
            continue;
        }
        const long cell = box.cellOf(f.camera_to_arm * f.points[i].cast<double>());
        if (cell < 0 || (!supported.empty() && !handle[cell] && !supported[i])) {
            continue;
        }
        if (hits[cell] == 0) {
            touched.push_back(cell);
        }
        hits[cell] = static_cast<uint8_t>(std::min(hits[cell] + 1, 255));
    }

    const Eigen::Isometry3d arm_to_camera = f.camera_to_arm.inverse();
    const double            half_diagonal = 0.5 * std::sqrt(3.0) * box.voxel;
    std::vector<uint32_t>   out;
    for (const long cell : touched) {
        if (hits[cell] >= s.min_points_per_voxel || handle[cell]
            || !seenThrough(arm_to_camera * box.centre(cell), depth, c, s, half_diagonal)) {
            out.push_back(static_cast<uint32_t>(cell));
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Cells occupied in at least min_frames frames, where a frame votes for every cell within `radius` of one it saw.
std::vector<uint32_t> voteOccupied(const std::vector<std::vector<uint32_t>> &frames, const VoxelBox &box, int min_frames,
                                   int radius) {
    std::vector<uint8_t>  votes(static_cast<size_t>(box.count()), 0), marked(votes);
    std::vector<uint32_t> measured;
    for (size_t k = 0; k < frames.size(); ++k) {
        const uint8_t mark = static_cast<uint8_t>(k + 1);
        for (const uint32_t cell : frames[k]) {
            measured.push_back(cell);
            long cx = 0, cy = 0, cz = 0;
            box.coords(cell, cx, cy, cz);
            for (long z = std::max(0L, cz - radius); z <= std::min(box.nz - 1, cz + radius); ++z) {
                for (long y = std::max(0L, cy - radius); y <= std::min(box.ny - 1, cy + radius); ++y) {
                    for (long x = std::max(0L, cx - radius); x <= std::min(box.nx - 1, cx + radius); ++x) {
                        const long n = box.index(x, y, z);
                        if (marked[n] != mark) {
                            marked[n] = mark;
                            ++votes[n];
                        }
                    }
                }
            }
        }
    }
    std::sort(measured.begin(), measured.end());
    measured.erase(std::unique(measured.begin(), measured.end()), measured.end());
    std::vector<uint32_t> out;
    for (const uint32_t cell : measured) {
        if (votes[cell] >= min_frames) {
            out.push_back(cell);
        }
    }
    return out;
}

// Marks cells whose centre lies within `radius` of the segment a-b.
void markCapsule(const VoxelBox &box, const Eigen::Vector3d &a, const Eigen::Vector3d &b, double radius,
                 std::vector<uint8_t> &mask) {
    const Eigen::Vector3d lo = (a.cwiseMin(b) - Eigen::Vector3d::Constant(radius) - box.origin) / box.voxel;
    const Eigen::Vector3d hi = (a.cwiseMax(b) + Eigen::Vector3d::Constant(radius) - box.origin) / box.voxel;
    const long            size[3] = {box.nx, box.ny, box.nz};
    const auto clamp = [&](double v, int axis) { return std::min(std::max(static_cast<long>(std::floor(v)), 0L), size[axis] - 1); };
    const Eigen::Vector3d ab      = b - a;
    const double          length2 = std::max(ab.squaredNorm(), kShortest * kShortest);
    for (long z = clamp(lo.z(), 2); z <= clamp(hi.z(), 2); ++z) {
        for (long y = clamp(lo.y(), 1); y <= clamp(hi.y(), 1); ++y) {
            for (long x = clamp(lo.x(), 0); x <= clamp(hi.x(), 0); ++x) {
                const long            cell = box.index(x, y, z);
                const Eigen::Vector3d p    = box.centre(cell);
                const double          t    = std::min(std::max((p - a).dot(ab) / length2, 0.0), 1.0);
                if ((p - (a + t * ab)).norm() <= radius) {
                    mask[cell] = 1;
                }
            }
        }
    }
}

// Cells around the handle: near each pose, and along the bar between poses close enough to be one bar.
std::vector<uint8_t> handleRegion(const VoxelBox &box, const std::vector<GraspPose> &poses, const CloudSettings &s) {
    std::vector<uint8_t> mask(static_cast<size_t>(box.count()), 0);
    for (size_t i = 0; i < poses.size(); ++i) {
        markCapsule(box, poses[i].point, poses[i].point, s.handle_radius, mask);
        const double gap = i > 0 ? (poses[i].point - poses[i - 1].point).norm() : 0.0;
        if (gap > kShortest && gap <= s.bar_gap) {
            markCapsule(box, poses[i - 1].point, poses[i].point, s.handle_radius, mask);
        }
    }
    return mask;
}

// Cells in the corridor running back from each pose along its approach.
void markCorridors(const VoxelBox &box, const std::vector<GraspPose> &poses, const CloudSettings &s, std::vector<uint8_t> &mask) {
    for (const GraspPose &p : poses) {
        if (p.approach.norm() > kShortest) {
            markCapsule(box, p.point, p.point - unit(p.approach) * s.corridor_length, s.corridor_radius, mask);
        }
    }
}

VoxelBox fitBox(const std::vector<Frame> &frames, const CloudSettings &s) {
    Eigen::Vector3d lo = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity()), hi = -lo;
    for (const Frame &f : frames) {
        for (const Eigen::Vector3f &p : f.points) {
            const Eigen::Vector3d q = f.camera_to_arm * p.cast<double>();
            if (p.allFinite() && q.norm() <= s.crop_radius) {
                lo = lo.cwiseMin(q);
                hi = hi.cwiseMax(q);
            }
        }
        // Cropped like the points: one stray pose metres out would grow every per-cell buffer by the cube of its distance.
        for (const GraspPose &g : f.poses) {
            const Eigen::Vector3d q = f.camera_to_arm * g.point;
            if (q.norm() <= s.crop_radius) {
                lo = lo.cwiseMin(q);
                hi = hi.cwiseMax(q);
            }
        }
    }
    VoxelBox box;
    box.voxel = s.voxel;
    if (!lo.allFinite()) {
        box.nx = box.ny = box.nz = 1;
        return box;
    }
    // Carving reaches past the measured points, so the box has to hold the regions too.
    const double margin = std::max(s.handle_radius, s.corridor_length + s.corridor_radius) + s.voxel;
    box.origin          = lo - Eigen::Vector3d::Constant(margin);
    const Eigen::Vector3d span = (hi - lo + Eigen::Vector3d::Constant(2.0 * margin)) / s.voxel;
    box.nx = std::max(1L, static_cast<long>(std::ceil(span.x())));
    box.ny = std::max(1L, static_cast<long>(std::ceil(span.y())));
    box.nz = std::max(1L, static_cast<long>(std::ceil(span.z())));
    return box;
}

// Direction of the bar at pose i: towards its joined neighbours, or its own bar axis at a loose end.
Eigen::Vector3d barDirection(const std::vector<GraspPose> &p, size_t i, double gap) {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    const auto      joined = [&](size_t a) { const double l = (p[a + 1].point - p[a].point).norm(); return l > kShortest && l <= gap; };
    if (i > 0 && joined(i - 1)) {
        sum += unit(p[i].point - p[i - 1].point);
    }
    if (i + 1 < p.size() && joined(i)) {
        sum += unit(p[i + 1].point - p[i].point);
    }
    return sum.norm() > kShortest ? unit(sum) : unit(p[i].bar);
}

// The point of one frame's bar nearest the anchor, if within reach and running the same way.
bool nearestOnBar(const Eigen::Vector3d &anchor, const Eigen::Vector3d &direction, const std::vector<GraspPose> &p,
                  const CloudSettings &s, GraspPose &out) {
    const double cos_limit = std::cos(s.max_axis);
    double       best      = s.match;
    bool         found     = false;
    for (size_t i = 0; i < p.size(); ++i) {
        if (std::fabs(barDirection(p, i, s.bar_gap).dot(direction)) >= cos_limit && (p[i].point - anchor).norm() <= best) {
            best  = (p[i].point - anchor).norm();
            out   = p[i];
            found = true;
        }
        if (i + 1 == p.size()) {
            break;
        }
        const Eigen::Vector3d ab     = p[i + 1].point - p[i].point;
        const double          length = ab.norm();
        if (length <= kShortest || length > s.bar_gap || std::fabs(ab.dot(direction)) < cos_limit * length) {
            continue;
        }
        const double          t     = std::min(std::max((anchor - p[i].point).dot(ab) / (length * length), 0.0), 1.0);
        const Eigen::Vector3d point = p[i].point + t * ab;
        if ((point - anchor).norm() <= best) {
            const Eigen::Vector3d next_bar = p[i + 1].bar.dot(p[i].bar) >= 0.0 ? p[i + 1].bar : -p[i + 1].bar;
            best  = (point - anchor).norm();
            out   = {point, (1.0 - t) * p[i].bar + t * next_bar, (1.0 - t) * p[i].approach + t * p[i + 1].approach};
            found = true;
        }
    }
    return found;
}

double median(std::vector<double> v) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

}  // namespace

std::vector<GraspPose> agreeOnSpots(const std::vector<std::vector<GraspPose>> &frames, const CloudSettings &s,
                                    std::string &summary) {
    std::vector<GraspPose>       spots;
    std::vector<Eigen::Vector3d> tried;
    int                          too_few = 0, outliers = 0, spread = 0;
    const auto near = [&](const Eigen::Vector3d &p) {
        return std::any_of(tried.begin(), tried.end(), [&](const Eigen::Vector3d &q) { return (q - p).norm() <= s.duplicate; });
    };

    // Anchors from the newest frame first, so the spots follow the most recent view.
    for (size_t n = 0; n < frames.size(); ++n) {
        const std::vector<GraspPose> &anchors = frames[(frames.size() - 1 + n) % frames.size()];
        for (size_t i = 0; i < anchors.size(); ++i) {
            if (near(anchors[i].point)) {
                continue;
            }
            tried.push_back(anchors[i].point);
            const Eigen::Vector3d  direction = barDirection(anchors, i, s.bar_gap);
            std::vector<GraspPose> found;
            for (const std::vector<GraspPose> &frame : frames) {
                GraspPose hit;
                if (nearestOnBar(anchors[i].point, direction, frame, s, hit)) {
                    found.push_back(hit);
                }
            }
            if (static_cast<int>(found.size()) < s.consensus_min_frames) {
                ++too_few;
                continue;
            }
            std::vector<double> xs, ys, zs;
            for (const GraspPose &g : found) {
                xs.push_back(g.point.x());
                ys.push_back(g.point.y());
                zs.push_back(g.point.z());
            }
            const Eigen::Vector3d  middle(median(xs), median(ys), median(zs));
            std::vector<GraspPose> inliers;
            for (const GraspPose &g : found) {
                if ((g.point - middle).norm() <= s.outlier) {
                    inliers.push_back(g);
                }
            }
            if (static_cast<int>(inliers.size()) < s.consensus_min_frames) {
                ++outliers;
                continue;
            }
            Eigen::Vector3d centre = Eigen::Vector3d::Zero(), bar = centre, approach = centre;
            for (const GraspPose &g : inliers) {
                centre += g.point / static_cast<double>(inliers.size());
                bar += g.bar.dot(inliers.front().bar) >= 0.0 ? g.bar : -g.bar;
                approach += g.approach;
            }
            double squared = 0.0;
            for (const GraspPose &g : inliers) {
                squared += (g.point - centre).squaredNorm();
            }
            if (std::sqrt(squared / inliers.size()) > s.max_spread) {
                ++spread;
                continue;
            }
            tried.push_back(centre);
            spots.push_back({centre, unit(bar), unit(approach)});
        }
    }
    char line[160];
    std::snprintf(line, sizeof(line), "%zu spots agree; refused %d seen in too few frames, %d with outliers, %d too spread",
                  spots.size(), too_few, outliers, spread);
    summary = line;
    return spots;
}

Scene processFrames(const std::vector<Frame> &frames, const CameraModel &camera, const CloudSettings &s) {
    Scene scene;
    if (frames.empty()) {
        scene.summary = "no frames";
        return scene;
    }
    scene.map.box       = fitBox(frames, s);
    const VoxelBox &box = scene.map.box;

    std::vector<std::vector<GraspPose>> poses(frames.size()), reachable(frames.size());
    for (size_t k = 0; k < frames.size(); ++k) {
        for (const GraspPose &g : frames[k].poses) {
            const Eigen::Isometry3d &t = frames[k].camera_to_arm;
            poses[k].push_back({t * g.point, t.linear() * g.bar, t.linear() * g.approach});
            if (poses[k].back().point.norm() <= s.crop_radius) {
                reachable[k].push_back(poses[k].back());
            }
        }
    }
    scene.poses = poses.back();

    std::string averaged = "newest frame only";
    scene.candidates     = s.candidate_averaging ? agreeOnSpots(reachable, s, averaged) : reachable.back();

    // Every frame's poses keep their own surface when averaging, the newest frame's otherwise.
    std::vector<GraspPose> carve_from = scene.candidates;
    for (size_t k = s.obstacle_averaging ? 0 : frames.size() - 1; k < frames.size(); ++k) {
        carve_from.insert(carve_from.end(), poses[k].begin(), poses[k].end());
    }
    std::vector<uint8_t> handle = s.handle_carving ? handleRegion(box, carve_from, s)
                                                   : std::vector<uint8_t>(static_cast<size_t>(box.count()), 0);
    std::vector<std::vector<uint32_t>> occupied;
    for (size_t k = s.obstacle_averaging ? 0 : frames.size() - 1; k < frames.size(); ++k) {
        occupied.push_back(occupiedCells(frames[k], camera, box, handleRegion(box, poses[k], s), s));
    }
    const std::vector<uint32_t> cells =
            s.obstacle_averaging ? voteOccupied(occupied, box, s.vote_min_frames, s.vote_radius) : occupied.back();
    if (s.corridor_carving) {
        markCorridors(box, carve_from, s, handle);
    }
    for (const uint32_t cell : cells) {
        (handle[cell] ? scene.map.handle : scene.map.obstacle).push_back(cell);
    }

    char line[320];
    std::snprintf(line, sizeof(line),
                  "%zu frames; newest has %zu poses; %zu candidates (%s); %zu obstacle and %zu handle cells in %ldx%ldx%ld of %.1f mm",
                  frames.size(), scene.poses.size(), scene.candidates.size(), averaged.c_str(), scene.map.obstacle.size(),
                  scene.map.handle.size(), box.nx, box.ny, box.nz, box.voxel * 1000.0);
    scene.summary = line;
    return scene;
}

}  // namespace sep23
