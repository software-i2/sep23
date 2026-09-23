// Copyright by BeeX [2026]
#include <sep23/park.h>
#include <sep23/plan.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <set>

namespace sep23 {

Eigen::Isometry3d toIsometry(const VehiclePose &p) {
    Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
    t.linear()          = Eigen::AngleAxisd(p.yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    t.translation()     = Eigen::Vector3d(p.x, p.y, p.z);
    return t;
}

VehiclePose compose(const VehiclePose &from, const VehiclePose &move) {
    const double c = std::cos(from.yaw), s = std::sin(from.yaw);
    return {from.x + c * move.x - s * move.y, from.y + s * move.x + c * move.y, from.z + move.z, from.yaw + move.yaw};
}

ReachMap::ReachMap(const Arm &arm, double along, double floor_z, double cell)
        : cell_(cell), z_min_(floor_z), base_lo_(arm.lower(BASE)), base_hi_(arm.upper(BASE)) {
    const double span = arm.reach(along);
    nr_               = static_cast<int>(std::ceil(span / cell)) + 1;
    nz_               = static_cast<int>(std::ceil((span - floor_z) / cell)) + 1;
    // Solved facing the middle of the base window, so the base limit never refuses a cell.
    const double         facing = 0.5 * (base_lo_ + base_hi_);
    const Joints         seed{};
    std::vector<bool>    solved(static_cast<size_t>(nr_) * nz_, false);
    for (int iz = 0; iz < nz_; ++iz) {
        for (int ir = 0; ir < nr_; ++ir) {
            const double    r = (ir + 0.5) * cell, z = z_min_ + (iz + 0.5) * cell;
            const Eigen::Vector3d target(r * std::cos(facing), r * std::sin(facing), z);
            Joints          q;
            solved[iz * nr_ + ir] = arm.solve(target, along, false, seed, q) == Ik::SOLVED
                                    || arm.solve(target, along, true, seed, q) == Ik::SOLVED;
        }
    }
    // Dilated by one cell on purpose: a false negative loses a parking pose for good, a false positive costs one exact check.
    cells_.assign(solved.size(), false);
    for (int iz = 0; iz < nz_; ++iz) {
        for (int ir = 0; ir < nr_; ++ir) {
            const bool up = iz + 1 < nz_ && solved[(iz + 1) * nr_ + ir], down = iz > 0 && solved[(iz - 1) * nr_ + ir];
            const bool out = ir + 1 < nr_ && solved[iz * nr_ + ir + 1], in = ir > 0 && solved[iz * nr_ + ir - 1];
            cells_[iz * nr_ + ir] = solved[iz * nr_ + ir] || up || down || out || in;
        }
    }
}

bool ReachMap::reachable(double radial, double z) const {
    const int ir = static_cast<int>(std::floor(radial / cell_)), iz = static_cast<int>(std::floor((z - z_min_) / cell_));
    return ir >= 0 && iz >= 0 && ir < nr_ && iz < nz_ && cells_[iz * nr_ + ir];
}

bool ReachMap::baseAngleFor(double azimuth) const {
    for (int turns = -1; turns <= 1; ++turns) {
        const double a = azimuth + 2.0 * M_PI * turns;
        if (a >= base_lo_ && a <= base_hi_) {
            return true;
        }
    }
    return false;
}

ParkSearch::ParkSearch(const Arm &arm, const Hull &hull, const ParkSettings &s, const Eigen::Isometry3d &vehicle_to_arm,
                       const Eigen::Vector3d &camera_in_vehicle)
        : arm_(arm),
          hull_(hull),
          s_(s),
          vehicle_to_arm_(vehicle_to_arm),
          camera_in_vehicle_(camera_in_vehicle),
          reach_(arm, arm.mountDistance() + s.grasp_point_from_mount, hull.floor_z, s.reach_cell) {}

Eigen::Isometry3d ParkSearch::armMove(const VehiclePose &move) const {
    return vehicle_to_arm_ * toIsometry(move) * vehicle_to_arm_.inverse();
}

int ParkSearch::admitted(const std::vector<GraspPose> &grasps, const VehiclePose &move) const {
    const Eigen::Isometry3d to_moved = armMove(move).inverse();
    const Eigen::Vector3d   camera   = vehicle_to_arm_ * toIsometry(move) * camera_in_vehicle_;
    int                     count    = 0;
    for (const GraspPose &g : grasps) {
        const double standoff = (g.point - camera).norm();
        if (standoff < s_.standoff_min || standoff > s_.standoff_max) {
            continue;
        }
        const Eigen::Vector3d p = to_moved * g.point;
        count += reach_.baseAngleFor(std::atan2(p.y(), p.x())) && reach_.reachable(std::hypot(p.x(), p.y()), p.z());
    }
    return count;
}

void ParkSearch::shortlist(const std::vector<GraspPose> &grasps, std::vector<Scored> &out) const {
    out.clear();
    if (grasps.empty()) {
        return;
    }
    // A move that leaves the whole cloud outside the arm's reach sphere cannot admit anything.
    Eigen::Vector3d centre = Eigen::Vector3d::Zero();
    for (const GraspPose &g : grasps) {
        centre += vehicle_to_arm_.inverse() * g.point / static_cast<double>(grasps.size());
    }
    double spread = 0.0;
    for (const GraspPose &g : grasps) {
        spread = std::max(spread, (vehicle_to_arm_.inverse() * g.point - centre).norm());
    }
    const double cull = arm_.reach(arm_.mountDistance() + s_.grasp_point_from_mount) + spread
                        + vehicle_to_arm_.translation().norm() + s_.coarse_step;

    std::set<std::array<long, 4>> seen;
    const auto                    once = [&](const VehiclePose &m) {
        const std::array<long, 4> key = {std::lround(m.x / s_.fine_step), std::lround(m.y / s_.fine_step),
                                         std::lround(m.z / s_.fine_step), std::lround(m.yaw / s_.fine_yaw)};
        return seen.insert(key).second;
    };
    const auto range = [](double half, double step) {
        std::vector<double> v;
        for (double x = -half; x <= half + 1e-9; x += step) {
            v.push_back(x);
        }
        return v;
    };

    std::vector<Scored> coarse;
    for (const double x : range(s_.box_xy, s_.coarse_step)) {
        for (const double y : range(s_.box_xy, s_.coarse_step)) {
            for (const double z : range(s_.box_z, s_.coarse_step)) {
                if ((centre - Eigen::Vector3d(x, y, z)).norm() > cull) {
                    continue;
                }
                for (const double yaw : range(s_.box_yaw, s_.coarse_yaw)) {
                    const VehiclePose m{x, y, z, yaw};
                    coarse.push_back({m, admitted(grasps, m)});
                }
            }
        }
    }
    std::stable_sort(coarse.begin(), coarse.end(), [](const Scored &a, const Scored &b) { return a.admitted > b.admitted; });

    const size_t refine = std::min<size_t>(static_cast<size_t>(std::max(1, s_.refine_count)), coarse.size());
    for (size_t i = 0; i < refine; ++i) {
        for (const double dx : range(s_.coarse_step, s_.fine_step)) {
            for (const double dy : range(s_.coarse_step, s_.fine_step)) {
                for (const double dz : range(s_.coarse_step, s_.fine_step)) {
                    for (const double dyaw : range(s_.coarse_yaw, s_.fine_yaw)) {
                        const VehiclePose &c = coarse[i].move;
                        const VehiclePose  m{c.x + dx, c.y + dy, c.z + dz, c.yaw + dyaw};
                        const double eps = 1e-9;
                        if (std::fabs(m.x) > s_.box_xy + eps || std::fabs(m.y) > s_.box_xy + eps || std::fabs(m.z) > s_.box_z + eps
                            || std::fabs(m.yaw) > s_.box_yaw + eps || !once(m)) {
                            continue;
                        }
                        const int held = admitted(grasps, m);
                        if (held > 0) {
                            out.push_back({m, held});
                        }
                    }
                }
            }
        }
    }
    // Every coarse move that admits anything, not only the refined few: reach alone favours the deepest, worst standoffs.
    for (const Scored &c : coarse) {
        if (c.admitted > 0 && once(c.move)) {
            out.push_back(c);
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const Scored &a, const Scored &b) { return a.admitted > b.admitted; });
    out.resize(std::min(out.size(), static_cast<size_t>(s_.screen_count)));
}

ParkSearch::Holds ParkSearch::verify(const std::vector<GraspPose> &grasps, const VehiclePose &move, ObstacleGrid &grid,
                                       const Joints &home, size_t stride, bool routes) const {
    const Eigen::Isometry3d to_moved = armMove(move).inverse();
    grid.setQueryToMap(armMove(move));
    Collision collision(arm_, grid, hull_, s_.link_step, stride);
    const double along = arm_.mountDistance() + s_.grasp_point_from_mount;
    Holds        out;
    for (size_t i = 0; i < grasps.size(); ++i) {
        const GraspPose        moved{to_moved * grasps[i].point, to_moved.linear() * grasps[i].bar, Eigen::Vector3d::Zero()};
        std::vector<GraspGoal> goals;
        graspGoals(moved, i, home, along, collision, goals);
        if (goals.empty()) {
            continue;
        }
        ++out.held;
        out.routes += routes && std::any_of(goals.begin(), goals.end(), [&](const GraspGoal &g) {
                          return collision.segmentClear(home, g.joints, s_.edge_step);
                      });
    }
    return out;
}

bool ParkSearch::transitClear(const VehiclePose &to, ObstacleGrid &grid, const Joints &home) const {
    Collision collision(arm_, grid, hull_, s_.link_step, static_cast<size_t>(s_.screen_blade_stride));
    for (int n = 0; n <= s_.transit_samples; ++n) {
        const double f = static_cast<double>(n) / s_.transit_samples;
        grid.setQueryToMap(armMove({to.x * f, to.y * f, to.z * f, to.yaw * f}));
        if (collision.check(home) != Verdict::CLEAR) {
            return false;
        }
    }
    return true;
}

ParkChoice ParkSearch::choose(const std::vector<GraspPose> &grasps, const ObstacleMap &map, const Joints &home,
                              const std::function<bool()> &cancelled) const {
    const auto          started = std::chrono::steady_clock::now();
    const auto          elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); };
    std::vector<Scored> options;
    shortlist(grasps, options);
    const size_t searched = options.size();
    const double search_s = elapsed();

    ObstacleGrid grid(map, s_.link_radius, bladeRadius(map.box.voxel, s_.blade_step));
    const size_t stride = static_cast<size_t>(s_.screen_blade_stride);

    // Screened at the blade stride: optimistic, so the best few are rechecked exactly.
    std::vector<std::pair<int, VehiclePose>> ranked;
    size_t                                   screened = 0;
    for (const Scored &o : options) {
        if (elapsed() - search_s > s_.screen_budget_s || cancelled()) {
            break;
        }
        ++screened;
        const int held = verify(grasps, o.move, grid, home, stride, false).held;
        if (held > 0) {
            ranked.emplace_back(held, o.move);
        }
    }
    // Among equal holds the shorter drive wins: less time for the scene to move away from the snapshot.
    const auto travel = [](const VehiclePose &m) { return std::hypot(std::hypot(m.x, m.y), m.z); };
    std::stable_sort(ranked.begin(), ranked.end(), [&](const auto &a, const auto &b) {
        return a.first != b.first ? a.first > b.first : travel(a.second) < travel(b.second);
    });

    // Routes first, holds second: a pose with more holds and no straight route from home is the no-path case.
    // Exact slots go to poses the vehicle can actually drive to; a refused drive does not use one up.
    ParkChoice best;
    int        blocked = 0, exact = 0;
    for (size_t i = 0; i < ranked.size() && exact < s_.exact_count && !cancelled(); ++i) {
        const VehiclePose &m = ranked[i].second;
        if (!transitClear(m, grid, home)) {
            ++blocked;
            continue;
        }
        ++exact;
        const int held = verify(grasps, m, grid, home, 1, false).held;
        if (held == 0) {
            continue;
        }
        const int routes = verify(grasps, m, grid, home, stride, true).routes;
        if (best.decision == ParkDecision::NOWHERE || routes > best.routes || (routes == best.routes && held > best.held)) {
            best = {ParkDecision::MOVE, m, held, routes, ""};
        }
    }

    // Staying needs no drive, so it wins ties; the drive is time the scene spends moving away from the snapshot.
    const Holds stay = verify(grasps, VehiclePose(), grid, home, 1, true);
    if (stay.held > 0 && (stay.routes > best.routes || (stay.routes == best.routes && stay.held >= best.held))) {
        best = {ParkDecision::STAY, VehiclePose(), stay.held, stay.routes, ""};
    }
    // A move without a straight route is only refused when staying still reaches something.
    if (best.decision == ParkDecision::MOVE && best.routes == 0 && stay.held > 0) {
        best = {ParkDecision::STAY, VehiclePose(), stay.held, stay.routes, ""};
    }

    char line[320];
    std::snprintf(line, sizeof(line),
                  "%zu moves searched in %.1f s, %zu screened, %zu hold something, %d refused for dragging the arm through the scene, %d checked exactly; "
                  "staying holds %d routes %d; %s (%.2f %.2f %.2f m, %.0f deg) holds %d routes %d",
                  searched, search_s, screened, ranked.size(), blocked, exact, stay.held, stay.routes,
                  best.decision == ParkDecision::MOVE ? "moving" : best.decision == ParkDecision::STAY ? "staying" : "nowhere",
                  best.move.x, best.move.y, best.move.z, deg(best.move.yaw), best.held, best.routes);
    best.summary = line;
    return best;
}

}  // namespace sep23
