// Copyright by BeeX [2026]
// Runtime: how long each stage of one pick takes, on a synthetic look built with robot.yaml and pick.yaml values.
// A mine face 0.40 m below the camera with a handle bar 6 cm in front of it; not a real scene, so compare runs, not absolutes.
#include <sep23/park.h>
#include <sep23/plan.h>
#include <sep23/track.h>

#include "common.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <random>

using namespace sep23;

namespace {

// Runs `f` `runs` times and prints the median and best wall time of one call, in ms.
template <class F>
void timeIt(const char *stage, int runs, F &&f) {
    std::vector<double> ms;
    for (int i = 0; i < runs; ++i) {
        const auto started = std::chrono::steady_clock::now();
        f();
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
    }
    std::sort(ms.begin(), ms.end());
    std::printf("  %-34s %10.3f ms median %10.3f ms best  (%d runs)\n", stage, ms[ms.size() / 2], ms.front(), runs);
}

Eigen::Isometry3d fromXyzRpy(const Eigen::Vector3d &xyz, const Eigen::Vector3d &rpy_deg) {
    Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
    t.linear()          = (Eigen::AngleAxisd(rad(rpy_deg.z()), Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(rad(rpy_deg.y()), Eigen::Vector3d::UnitY())
                  * Eigen::AngleAxisd(rad(rpy_deg.x()), Eigen::Vector3d::UnitX()))
                         .toRotationMatrix();
    t.translation() = xyz;
    return t;
}

// The arm as config/robot.yaml describes it.
ArmConfig shippedArm() {
    ArmConfig a;
    a.geometry = geometryFromUrdf(expandedUrdf(), kNames);
    a.offset   = {{rad(2.403), 0.0, 0.0, 0.0}};
    a.sign     = {{1, 1, 1, -1}};
    a.min      = {{0.0, 0.0, 0.0, 0.0}};
    a.max      = {{rad(349.6), rad(200.0), rad(184.5), rad(184.5)}};
    JawShape &jaw             = a.jaw;
    jaw.open_len              = 0.007;
    jaw.mount_to_throat       = 0.0300;
    jaw.mount_to_tip          = 0.0969;
    jaw.palm_length           = 0.00975;
    jaw.hinge_roll_at_zero    = rad(-60.0);
    jaw.blade_rotation_per_m  = 51.0;
    jaw.hinge_offset_closing  = 0.0155;
    jaw.hinge_offset_approach = 0.0069;
    jaw.blades = {{-0.0050, 0.0000, -0.0252, 0.0051, 0.0057}, {0.0000, 0.0050, -0.0252, 0.0058, 0.0057},
                  {0.0050, 0.0100, -0.0202, 0.0063, 0.0057},  {0.0100, 0.0150, -0.0165, 0.0066, 0.0057},
                  {0.0150, 0.0200, -0.0153, 0.0067, 0.0049},  {0.0200, 0.0250, -0.0144, 0.0067, 0.0049},
                  {0.0250, 0.0300, -0.0138, 0.0065, 0.0049},  {0.0300, 0.0350, -0.0137, 0.0060, 0.0049},
                  {0.0350, 0.0400, -0.0139, 0.0052, 0.0049},  {0.0400, 0.0450, -0.0145, 0.0043, 0.0049},
                  {0.0450, 0.0500, -0.0156, 0.0030, 0.0049},  {0.0500, 0.0550, -0.0168, 0.0015, 0.0049},
                  {0.0550, 0.0600, -0.0184, -0.0003, 0.0049}, {0.0600, 0.0650, -0.0202, -0.0024, 0.0049},
                  {0.0650, 0.0700, -0.0211, -0.0048, 0.0049}, {0.0700, 0.0750, -0.0211, -0.0076, 0.0049},
                  {0.0750, 0.0800, -0.0211, -0.0107, 0.0049}, {0.0800, 0.0852, -0.0211, -0.0143, 0.0049}};
    return a;
}

CloudSettings cloudSettings() {
    CloudSettings s;
    s.voxel                   = 0.0025;
    s.crop_radius             = 1.0;
    s.depth_tolerance         = 0.003;
    s.min_agreeing_neighbours = 5;
    s.slope_window_px         = 5;
    s.min_points_per_voxel    = 2;
    s.free_space_tolerance    = 0.005;
    s.max_ray_stretch         = 1.0 / std::cos(rad(85.0));
    s.handle_radius           = 0.01;
    s.bar_gap                 = 0.015;
    s.corridor_length         = 0.05;
    s.corridor_radius         = 0.01;
    s.consensus_min_frames    = 2;
    s.match                   = 0.010;
    s.max_axis                = rad(20.0);
    s.outlier                 = 0.006;
    s.max_spread              = 0.003;
    s.duplicate               = 0.005;
    s.vote_min_frames         = 2;
    s.vote_radius             = 1;
    return s;
}

ParkSettings parkSettings() {
    ParkSettings p;
    p.box_xy = 1.0, p.box_z = 0.5, p.box_yaw = rad(45.0);
    p.coarse_step = 0.100, p.coarse_yaw = rad(9.0), p.fine_step = 0.025, p.fine_yaw = rad(3.0);
    p.refine_count = 4;
    p.standoff_min = 0.30, p.standoff_max = 0.40;
    p.screen_count = 3000, p.screen_budget_s = 10.0, p.screen_blade_stride = 20, p.exact_count = 10, p.transit_samples = 16;
    p.reach_cell             = 0.002;
    p.grasp_point_from_mount = 0.050, p.link_radius = 0.020, p.link_step = 0.002, p.blade_step = 0.0025;
    p.swing_band             = rad(5.0);
    return p;
}

PlanSettings planSettings() {
    PlanSettings s;
    s.grasp_point_from_mount = 0.050;
    s.budget_s = 10.0, s.goal_budget_s = 0.5;
    s.range = 1.0, s.edge_step = rad(1.0);
    s.joint_speed = rad(0.5) * 20.0;
    return s;
}

const double kFace = 0.40, kBar = 0.34, kBarRadius = 0.008, kBarHalf = 0.06;  // camera_link depths and bar size, metres

// One organised cloud in camera_link (x right, y up, looking down -z): the face, the bar along x in front of it, a little noise.
std::vector<Eigen::Vector3f> look(const CameraModel &camera, std::mt19937 &rng) {
    std::normal_distribution<double> noise(0.0, 0.0005);
    std::vector<Eigen::Vector3f>     points(static_cast<size_t>(camera.width) * camera.height);
    for (int v = 0; v < camera.height; ++v) {
        for (int u = 0; u < camera.width; ++u) {
            const Eigen::Vector3d ray((u - camera.cx) / camera.fx, -(v - camera.cy) / camera.fy, -1.0);
            double                depth = kFace;
            const double a = ray.y() * ray.y() + 1.0, b = -2.0 * kBar, c = kBar * kBar - kBarRadius * kBarRadius;  // ray meets bar
            const double disc = b * b - 4.0 * a * c;
            if (disc >= 0.0) {
                const double t = (-b - std::sqrt(disc)) / (2.0 * a);
                depth          = std::fabs(t * ray.x()) <= kBarHalf ? t : depth;
            }
            points[static_cast<size_t>(v) * camera.width + u] = (ray * (depth + noise(rng))).cast<float>();
        }
    }
    return points;
}

}  // namespace

int main() {
    const CameraModel       camera{800, 600, 479.2662, 479.2662, 397.9827, 322.8405};
    const Eigen::Isometry3d camera_to_vehicle = fromXyzRpy({0.35, -0.05, -0.05}, {0, 90, 0}) * fromXyzRpy({0, 0, 0}, {90, 0, -90});
    const Eigen::Isometry3d arm_to_vehicle    = fromXyzRpy({0.35, 0.0, -0.05}, {180, 0, 0});
    const Hull              hull{-0.80, 0.10, -0.40, 0.40, -1.0e-6};
    const ParkSettings      park_settings = parkSettings();
    const PlanSettings      plan_settings = planSettings();
    const auto              never         = [] { return false; };

    std::printf("startup\n");
    std::unique_ptr<Arm> arm;
    timeIt("arm: build from URDF (xacro)", 3, [&] { arm.reset(new Arm(shippedArm(), park_settings.blade_step)); });
    std::unique_ptr<ParkSearch> park;
    timeIt("park: reach map", 3, [&] {
        park.reset(new ParkSearch(*arm, hull, park_settings, arm_to_vehicle.inverse(), camera_to_vehicle.translation()));
    });
    const Joints home = arm->toModel({{0.0, rad(90.0), 0.0, 0.0}});

    // The look: 5 frames, grasp poses every centimetre along the bar, approach into the face.
    std::mt19937       rng(1);
    CloudSettings      cloud = cloudSettings();
    std::vector<Frame> frames(5);
    cloud.crop_radius += arm->reach(arm->tipDistance());
    for (Frame &f : frames) {
        f.points        = look(camera, rng);
        f.camera_to_arm = arm_to_vehicle.inverse() * camera_to_vehicle;
        for (double x = -0.05; x <= 0.05 + 1e-9; x += 0.01) {
            f.poses.push_back({Eigen::Vector3d(x, 0.0, -kBar), Eigen::Vector3d::UnitX(), -Eigen::Vector3d::UnitZ()});
        }
    }

    std::printf("one look (%dx%d, %zu frames)\n", camera.width, camera.height, frames.size());
    Scene scene;
    timeIt("cloud: processFrames", 5, [&] { scene = processFrames(frames, camera, cloud); });
    std::printf("    -> %zu candidates, %zu obstacle cells, %zu handle cells\n", scene.candidates.size(), scene.map.obstacle.size(),
                scene.map.handle.size());
    std::unique_ptr<ObstacleGrid> grid;
    timeIt("collision: inflate obstacle grid", 5, [&] {
        grid.reset(new ObstacleGrid(scene.map, park_settings.link_radius, bladeRadius(scene.map.box.voxel, park_settings.blade_step)));
    });
    ParkChoice choice;
    timeIt("park: choose", 3, [&] { choice = park->choose(scene.candidates, scene.map, home, never); });
    std::printf("    -> %s\n", choice.summary.c_str());
    Collision collision(*arm, *grid, hull, park_settings.link_step);
    Plan      plan;
    timeIt("plan: planGrasp", 3, [&] { plan = planGrasp(scene.candidates, home, collision, plan_settings, never); });
    std::printf("    -> %s\n", plan.ok ? "planned" : plan.summary.c_str());

    // Tracking: the face and bar textured, drifting 1.5 px a frame, as the tracker sees it between arm ticks.
    cv::Mat texture(camera.height + 200, camera.width + 200, CV_8U);
    cv::randu(texture, 0, 255);
    cv::GaussianBlur(texture, texture, cv::Size(0, 0), 2.0);
    Tracker tracker({0.05, 0.25, 15, 150, 2.5, 0.3, 0.3}, camera);
    tracker.start(Eigen::Vector3d(0.0, 0.0, -kBar));
    int k = 0;
    std::printf("per frame\n");
    timeIt("track: update", 50, [&] {
        const int shift = static_cast<int>(1.5 * k++) % 200;
        tracker.update(texture(cv::Rect(shift, 100, camera.width, camera.height)), frames[k % frames.size()].points);
    });

    // The inner loops park and plan spend their time in.
    std::printf("inner loops\n");
    const Eigen::Vector3d target = scene.candidates.empty() ? Eigen::Vector3d(0.0, -0.05, 0.3) : scene.candidates.front().point;
    const double          along  = arm->mountDistance() + park_settings.grasp_point_from_mount;
    Joints                q;
    timeIt("arm: solve (IK) x10000", 1, [&] {
        for (int i = 0; i < 10000; ++i) {
            arm->solve(target, along, i % 2, home, q);
        }
    });
    timeIt("collision: check x10000", 1, [&] {
        for (int i = 0; i < 10000; ++i) {
            collision.check(home);
        }
    });
    return 0;
}
