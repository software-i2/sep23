// Copyright by BeeX [2026]
#include <sep23/bpl.h>
#include <sep23/follow.h>
#include <sep23/fsm.h>
#include <sep23/park.h>
#include <sep23/track.h>

#include <gtest/gtest.h>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <memory>
#include <random>

using namespace sep23;

namespace {

const std::array<std::string, JOINT_COUNT> kNames = {"axis_e", "axis_d", "axis_c", "axis_b"};

std::string expandedUrdf() {
    const std::string command = "xacro " SEP23_DIR "/urdf/robot.urdf.xacro robot_yaml:=" SEP23_DIR "/config/robot.yaml";
    std::unique_ptr<FILE, int (*)(FILE *)> pipe(popen(command.c_str(), "r"), pclose);
    std::string                            out;
    char                                   buffer[4096];
    while (pipe && fgets(buffer, sizeof(buffer), pipe.get())) {
        out += buffer;
    }
    return out;
}

// The shipped arm with no calibration offsets, so it can be compared with the URDF directly.
ArmConfig testArm(const std::string &urdf) {
    ArmConfig c;
    c.geometry = geometryFromUrdf(urdf, kNames);
    c.sign     = {{1, 1, 1, -1}};
    c.min      = {{0.0, 0.0, 0.0, 0.0}};
    c.max      = {{rad(349.6), rad(200.0), rad(184.5), rad(184.5)}};
    c.jaw.mount_to_tip = 0.0969;
    c.jaw.open_width   = 0.007;
    c.jaw.blades       = {{0.0, 0.05, -0.02, 0.005, 0.005}};
    return c;
}

Joints randomJoints(const Arm &arm, std::mt19937 &rng) {
    Joints q;
    for (int j = 0; j < JOINT_COUNT; ++j) {
        q[j] = std::uniform_real_distribution<double>(arm.lower(j), arm.upper(j))(rng);
    }
    return q;
}

}  // namespace

TEST(Fsm, WalksThePickAndLatchesStops) {
    EXPECT_EQ(nextState(State::READY, Event::START), State::STREAM);
    EXPECT_EQ(nextState(State::STREAM, Event::STREAMING), State::COLLECT);
    EXPECT_EQ(nextState(State::COLLECT, Event::NO_CANDIDATES_SPOT), State::RESURVEY);
    EXPECT_EQ(nextState(State::PICKSPOT, Event::SPOT_UNCHANGED), State::COLLECT);
    EXPECT_EQ(nextState(State::GOTOGRASP, Event::STALLED), State::RETARGET);
    EXPECT_EQ(nextState(State::GOTOGRASP, Event::COLLIDED), State::ESTOP);
    EXPECT_EQ(nextState(State::PICKGRASP, Event::START), State::PICKGRASP);
    EXPECT_EQ(nextState(State::READY, Event::FAILURE), State::READY);
    EXPECT_EQ(nextState(State::ESTOP, Event::START), State::STREAM);
}

TEST(Follower, ReachesAndCatchesAStuckJoint) {
    FollowSettings s{rad(1.0), rad(0.5), 5.0, rad(0.2), 0.2, 3};
    PathFollower   f(s);
    const Joints   start{}, goal{{rad(10), 0, 0, 0}};
    Joints         target;
    bool           send = false;

    f.load({start, goal});
    Following state = Following::SENDING;
    for (int t = 0; t < 100 && state != Following::REACHED; ++t) {
        state = f.tick(t * 0.05, target, send);
        if (send) {
            f.measure(target);
        }
    }
    EXPECT_EQ(state, Following::REACHED);

    // The same reading judged every tick must not count as a joint that stopped.
    f.load({start, goal});
    f.measure(start);
    for (int t = 0; t < 30; ++t) {
        EXPECT_NE(f.tick(t * 0.05, target, send), Following::BLOCKED);
    }

    // New readings that never move are a joint against something.
    f.load({start, goal});
    for (int t = 0; t < 30 && state != Following::BLOCKED; ++t) {
        f.measure(start);
        state = f.tick(t * 0.05, target, send);
    }
    EXPECT_EQ(state, Following::BLOCKED);
    EXPECT_EQ(f.blockedJoint(), BASE);
}

// The largest move sets the step count; every joint covers its share per step, so all arrive on the same tick.
TEST(Follower, JointsArriveTogether) {
    FollowSettings s{rad(0.5), rad(0.5), 5.0, rad(0.2), 0.2, 3};
    PathFollower   f(s);
    const Joints   start{}, goal{{rad(10), rad(-4), rad(1), 0}};
    Joints         target, previous = start;
    bool           send = false;

    f.load({start, goal});
    int sent = 0;
    for (int t = 0; t < 100; ++t) {
        f.tick(t * 0.05, target, send);
        if (!send) {
            break;
        }
        ++sent;
        for (int j = 0; j < JOINT_COUNT; ++j) {
            EXPECT_NEAR(target[j] - previous[j], goal[j] / 20, 1e-9);  // 10 deg / 0.5 deg = 20 steps
        }
        previous = target;
    }
    EXPECT_EQ(sent, 20);
    for (int j = 0; j < JOINT_COUNT; ++j) {
        EXPECT_NEAR(target[j], goal[j], 1e-9);
    }
}

TEST(Bpl, FramesRoundTripAndRejectCorruption) {
    std::vector<uint8_t> stream = bpl::encode(5, bpl::POSITION, 1.5f);
    EXPECT_EQ(stream.back(), 0x00);
    EXPECT_EQ(std::count(stream.begin(), stream.end(), 0x00), 1);
    const std::vector<uint8_t> second = bpl::encode(2, bpl::MODE, std::vector<uint8_t>{0x00, 0x00});
    stream.insert(stream.end(), second.begin(), second.end());

    bpl::Reader              reader;
    std::vector<bpl::Packet> got = reader.feed(stream.data(), stream.size());
    ASSERT_EQ(got.size(), 2u);
    float value = 0.0f;
    std::memcpy(&value, got[0].data.data(), sizeof(float));
    EXPECT_EQ(got[0].device, 5);
    EXPECT_FLOAT_EQ(value, 1.5f);
    EXPECT_EQ(got[1].data, (std::vector<uint8_t>{0x00, 0x00}));

    std::vector<uint8_t> broken = bpl::encode(5, bpl::POSITION, 1.5f);
    broken[2] ^= 0x01;
    EXPECT_TRUE(reader.feed(broken.data(), broken.size()).empty());
}

TEST(Arm, ForwardKinematicsMatchesTheUrdf) {
    const std::string urdf = expandedUrdf();
    ASSERT_FALSE(urdf.empty());
    KDL::Tree  tree;
    KDL::Chain chain;
    ASSERT_TRUE(kdl_parser::treeFromString(urdf, tree));
    ASSERT_TRUE(tree.getChain("arm_base", "ee_base_link", chain));
    ASSERT_EQ(chain.getNrOfJoints(), 4u);
    KDL::ChainFkSolverPos_recursive fk(chain);

    const Arm    arm(testArm(urdf), 0.005);
    std::mt19937 rng(7);
    for (int n = 0; n < 1000; ++n) {
        const Joints    reported = arm.toReported(randomJoints(arm, rng));
        KDL::JntArray   q(4);
        KDL::Frame      mount;
        for (int j = 0; j < JOINT_COUNT; ++j) {
            q(j) = reported[j];
        }
        ASSERT_GE(fk.JntToCart(q, mount), 0);
        const Joints          model = arm.toModel(reported);
        const Eigen::Vector3d ours  = arm.points(model).mount;
        const Eigen::Vector3d axis  = arm.axes(model).approach;
        EXPECT_NEAR((ours - Eigen::Vector3d(mount.p.x(), mount.p.y(), mount.p.z())).norm(), 0.0, 1e-5);
        EXPECT_NEAR(axis.dot(Eigen::Vector3d(mount.M.UnitZ().x(), mount.M.UnitZ().y(), mount.M.UnitZ().z())), 1.0, 1e-6);
    }
}

// Reach-back postures are left out on purpose: the IK never proposes them, they jam the upper arm on the housing.
TEST(Arm, InverseKinematicsRecoversEveryFrontFacingPosture) {
    const Arm    arm(testArm(expandedUrdf()), 0.005);
    const double along = arm.mountDistance() + 0.06;
    std::mt19937 rng(11);
    int          front = 0;
    for (int n = 0; n < 1000; ++n) {
        const Joints          q      = randomJoints(arm, rng);
        const Eigen::Vector3d target = arm.points(q).wrist + arm.axes(q).approach * along;
        if (target.x() * std::cos(q[BASE]) + target.y() * std::sin(q[BASE]) < 0.0) {
            continue;
        }
        ++front;
        bool matched = false;
        for (const bool up : {false, true}) {
            Joints out;
            if (arm.solve(target, along, up, q, out) == Ik::SOLVED) {
                matched |= (arm.points(out).wrist + arm.axes(out).approach * along - target).norm() < 1e-6;
            }
        }
        EXPECT_TRUE(matched) << "posture " << n;
    }
    EXPECT_GT(front, 200);
}

TEST(Arm, TellsWhatTheJawCaught) {
    ArmConfig c           = testArm(expandedUrdf());
    const Arm             arm(c, 0.005);
    const Joints          q{{0.3, 1.2, 0.8, -0.5}};
    const JawAxes         a     = arm.axes(q);
    const Eigen::Vector3d grasp = arm.points(q).mount + 0.06 * a.approach;
    const auto            caught = [&](const Eigen::Vector3d &p) { return Arm::caught(arm.inJaw(q, p), 0.07, 0.01); };
    EXPECT_TRUE(caught(grasp));
    EXPECT_TRUE(caught(grasp + 0.006 * a.hinge + 0.007 * a.closing));
    EXPECT_FALSE(caught(grasp + 0.011 * a.closing)) << "off centre";
    EXPECT_FALSE(caught(grasp + 0.011 * a.approach)) << "past 70 mm";
    EXPECT_FALSE(caught(grasp - 0.07 * a.approach)) << "behind the mount";
}

TEST(Collision, InflatesObstaclesAndKeepsHandleExact) {
    ObstacleMap map;
    map.box.voxel  = 0.01;
    map.box.nx     = map.box.ny = map.box.nz = 21;
    map.obstacle   = {static_cast<uint32_t>(map.box.index(10, 10, 10))};
    map.handle     = {static_cast<uint32_t>(map.box.index(2, 2, 2))};
    ObstacleGrid grid(map, 0.03, 0.015);
    const Eigen::Vector3d obstacle = map.box.centre(map.box.index(10, 10, 10));
    EXPECT_TRUE(grid.linkBlocked(obstacle + Eigen::Vector3d(0.03, 0, 0)));
    EXPECT_FALSE(grid.linkBlocked(obstacle + Eigen::Vector3d(0.045, 0, 0)));
    EXPECT_TRUE(grid.bladeBlocked(obstacle + Eigen::Vector3d(0.01, 0, 0)));
    EXPECT_FALSE(grid.bladeBlocked(obstacle + Eigen::Vector3d(0.03, 0, 0)));
    const Eigen::Vector3d handle = map.box.centre(map.box.index(2, 2, 2));
    EXPECT_TRUE(grid.bladeBlocked(handle));
    EXPECT_FALSE(grid.bladeBlocked(handle + Eigen::Vector3d(0.01, 0, 0)));
    EXPECT_TRUE(grid.linkBlocked(handle + Eigen::Vector3d(0.02, 0, 0)));
}

TEST(Park, ReachMapNeverRefusesWhatTheIkSolves) {
    const Arm      arm(testArm(expandedUrdf()), 0.005);
    const double   along = arm.mountDistance() + 0.06;
    const ReachMap reach(arm, along, -0.05, 0.002);
    std::mt19937   rng(3);
    std::uniform_real_distribution<double> coord(-0.45, 0.45);
    int solvable = 0;
    for (int n = 0; n < 20000; ++n) {
        const Eigen::Vector3d p(coord(rng), coord(rng), coord(rng));
        Joints                q;
        const bool solved = arm.solve(p, along, false, Joints{}, q) == Ik::SOLVED || arm.solve(p, along, true, Joints{}, q) == Ik::SOLVED;
        if (solved && p.z() > -0.05) {
            ++solvable;
            EXPECT_TRUE(reach.reachable(std::hypot(p.x(), p.y()), p.z()) && reach.baseAngleFor(std::atan2(p.y(), p.x())));
        }
    }
    EXPECT_GT(solvable, 500);
}

// A still face tilted away from the camera, then its near half hidden: the corners left sit deeper, so a median depth over
// them would move the target ~25 mm. Each corner's own depth change is zero, and so must the target's be.
TEST(Track, HidingPartOfATiltedFaceLeavesTheTargetStill) {
    CameraModel camera{320, 240, 250.0, 250.0, 160.0, 120.0};
    TrackSettings s{0.05, 0.25, 15, 150, 2.5, 0.0};
    cv::Mat       face(camera.height, camera.width, CV_8U);
    cv::randu(face, 0, 255);
    cv::GaussianBlur(face, face, cv::Size(0, 0), 2.0);
    cv::normalize(face, face, 0, 255, cv::NORM_MINMAX);
    const auto depth = [](int u) { return 0.40 + 0.0005 * (u - 160); };
    std::vector<Eigen::Vector3f> points(static_cast<size_t>(camera.width) * camera.height);
    const auto look = [&](bool hide) {
        for (int v = 0; v < camera.height; ++v) {
            for (int u = 0; u < camera.width; ++u) {
                const double d = depth(u);
                points[static_cast<size_t>(v) * camera.width + u] =
                        hide && u < 100 ? Eigen::Vector3f::Constant(NAN)
                                        : Eigen::Vector3f(static_cast<float>((u - camera.cx) * d / camera.fx),
                                                          static_cast<float>(-(v - camera.cy) * d / camera.fy), static_cast<float>(-d));
            }
        }
    };
    Tracker tracker(s, camera);
    tracker.start(Eigen::Vector3d((80.0 - camera.cx) * depth(80) / camera.fx, 0.0, -depth(80)));
    TrackResult r;
    for (int k = 0; k < 6; ++k) {
        look(k >= 3);
        r = tracker.update(face, points);
        ASSERT_TRUE(r.ok) << "frame " << k;
    }
    EXPECT_LT(r.offset.norm(), 0.002) << "the target moved " << 1000 * r.offset.transpose() << " mm with nothing moving";
}

// A mine face drifting across the image, an arm patch inside the depth gate moving the other way, a seabed behind the gate.
TEST(Track, FollowsTheMineThroughAnArmInsideTheGate) {
    CameraModel camera{320, 240, 250.0, 250.0, 160.0, 120.0};
    TrackSettings s{0.05, 0.25, 15, 150, 2.5, 0.5};
    const auto    texture = [&] {
        cv::Mat t(480, 640, CV_8U);
        cv::randu(t, 0, 255);
        cv::GaussianBlur(t, t, cv::Size(0, 0), 2.0);
        cv::normalize(t, t, 0, 255, cv::NORM_MINMAX);
        return t;
    };
    const cv::Mat mine = texture(), arm = texture(), seabed = texture();
    const double  dx = 1.5, dy = -1.0;  // the mine's drift per frame, pixels
    const auto    frame = [&](int k, cv::Mat &gray, std::vector<Eigen::Vector3f> &points) {
        gray.create(camera.height, camera.width, CV_8U);
        points.resize(static_cast<size_t>(camera.width) * camera.height);
        for (int v = 0; v < camera.height; ++v) {
            for (int u = 0; u < camera.width; ++u) {
                double       depth;
                uint8_t      value;
                const bool   in_arm = u >= 150 && u < 200 && v >= 60 && v < 200;
                if (u >= 260) {
                    depth = 0.50, value = seabed.at<uint8_t>(v + 100, u + 100);
                } else if (in_arm) {
                    depth = 0.42, value = arm.at<uint8_t>(v + 100 - 2 * k, u + 100 + 3 * k);
                } else {
                    depth = 0.40, value = mine.at<uint8_t>(static_cast<int>(std::lround(v + 100 - k * dy)), static_cast<int>(std::lround(u + 100 - k * dx)));
                }
                gray.at<uint8_t>(v, u) = value;
                points[static_cast<size_t>(v) * camera.width + u] =
                        Eigen::Vector3f(static_cast<float>((u - camera.cx) * depth / camera.fx), static_cast<float>(-(v - camera.cy) * depth / camera.fy),
                                        static_cast<float>(-depth));
            }
        }
    };

    Tracker tracker(s, camera);
    const Eigen::Vector2d start_px(80.0, 120.0);
    tracker.start(Eigen::Vector3d((start_px.x() - camera.cx) * 0.40 / camera.fx, -(start_px.y() - camera.cy) * 0.40 / camera.fy, -0.40));
    cv::Mat                      gray;
    std::vector<Eigen::Vector3f> points;
    TrackResult                  r;
    size_t                       arm_rejected = 0;
    const int                    frames = 10;
    for (int k = 0; k <= frames; ++k) {
        frame(k, gray, points);
        r = tracker.update(gray, points);
        ASSERT_TRUE(r.ok) << "frame " << k;
        for (size_t i = 0; i < r.to.size(); ++i) {
            arm_rejected += !r.inlier[i] && r.to[i].x >= 150 && r.to[i].x < 200;
        }
        for (const cv::Point2f &p : r.to) {
            EXPECT_LT(p.x, 260.0f) << "a seabed corner passed the depth gate";
        }
    }
    // Rounded pixel sampling makes the drift jitter by half a pixel; the RANSAC fit averages it out.
    EXPECT_NEAR(r.target_px.x(), start_px.x() + frames * dx, 1.0);
    EXPECT_NEAR(r.target_px.y(), start_px.y() + frames * dy, 1.0);
    // camera_link: +x right, +y up, so drifting right and up in the image is +x and +y.
    EXPECT_NEAR(r.offset.x(), frames * dx * 0.40 / camera.fx, 0.002);
    EXPECT_NEAR(r.offset.y(), -frames * dy * 0.40 / camera.fy, 0.002);
    EXPECT_NEAR(r.offset.z(), 0.0, 0.002);
    EXPECT_GT(arm_rejected, 0u) << "RANSAC never rejected an arm corner";
    EXPECT_FALSE(r.covered);

    // The mine and the arm hidden: tracking stops rather than follow what is left, and stays stopped once they are back.
    frame(frames + 1, gray, points);
    for (int v = 0; v < camera.height; ++v) {
        for (int u = 0; u < 260; ++u) {
            points[static_cast<size_t>(v) * camera.width + u] = Eigen::Vector3f::Constant(NAN);
        }
    }
    r = tracker.update(gray, points);
    EXPECT_TRUE(r.covered && !r.ok) << "face " << r.face;
    frame(frames + 2, gray, points);
    EXPECT_TRUE(tracker.update(gray, points).covered);
}

TEST(Cloud, ConsensusKeepsWhatFramesAgreeOn) {
    CloudSettings s;
    s.bar_gap = 0.015;
    s.consensus_min_frames = 2;
    s.match = 0.01;
    s.max_axis = rad(20);
    s.outlier = 0.006;
    s.max_spread = 0.003;
    s.duplicate = 0.005;
    std::vector<std::vector<GraspPose>> frames(3);
    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < 10; ++i) {
            frames[k].push_back({Eigen::Vector3d(0.01 * i + 0.003 * k, 0.0005 * k, 0.3), Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitZ()});
        }
    }
    std::string                  summary;
    const std::vector<GraspPose> spots = agreeOnSpots(frames, s, summary);
    ASSERT_FALSE(spots.empty()) << summary;
    for (const GraspPose &g : spots) {
        EXPECT_NEAR(g.point.y(), 0.0005, 0.001);
        EXPECT_NEAR(g.point.z(), 0.3, 1e-9);
    }
    EXPECT_TRUE(agreeOnSpots({frames[0]}, s, summary).empty()) << "one frame cannot agree with itself";
}

TEST(Cloud, HandleRunsAlongEachPoseBar) {
    const Eigen::Vector3d        x = Eigen::Vector3d::UnitX(), z = Eigen::Vector3d::UnitZ();
    const std::vector<GraspPose> bar = {{{0.05, 0.0, 0.0}, x, z}, {{0.0, 0.0, 0.0}, -x, z}};  // any order, either way along
    EXPECT_LT((nearestOnHandle(bar, 0.0075, {0.004, 0.002, 0.0}) - Eigen::Vector3d(0.004, 0.0, 0.0)).norm(), 1e-12);
    EXPECT_LT((nearestOnHandle(bar, 0.0075, {0.02, 0.002, 0.0}) - Eigen::Vector3d(0.0075, 0.0, 0.0)).norm(), 1e-12) << "past the bar's end";
}

// A tilted wall with scattered missing pixels and one pixel floating in front of it: the filter keeps the wall, drops the spike.
TEST(Cloud, FlyingPixelFilterKeepsASurfaceWithHoles) {
    const CameraModel camera{80, 60, 100.0, 100.0, 40.0, 30.0};
    CloudSettings     s;
    s.voxel                   = 0.0025;
    s.crop_radius             = 1.0;
    s.depth_tolerance         = 0.003;
    s.min_agreeing_neighbours = 5;
    s.slope_window_px         = 5;
    s.min_points_per_voxel    = 1;
    s.free_space_tolerance    = 0.005;
    s.max_ray_stretch         = 10.0;
    s.handle_carving = s.corridor_carving = s.candidate_averaging = s.obstacle_averaging = false;

    Frame        f;
    std::mt19937 rng(5);
    for (int v = 0; v < camera.height; ++v) {
        for (int u = 0; u < camera.width; ++u) {
            const double depth = u == 40 && v == 30 ? 0.35 : 0.40 + 0.0005 * u;
            const bool   hole  = std::uniform_real_distribution<double>(0.0, 1.0)(rng) < 0.05;
            f.points.push_back(hole ? Eigen::Vector3f::Constant(NAN)
                                    : Eigen::Vector3f(static_cast<float>((u - camera.cx) * depth / camera.fx),
                                                      static_cast<float>(-(v - camera.cy) * depth / camera.fy), static_cast<float>(-depth)));
        }
    }
    const Eigen::Vector3d spike = f.points[30 * camera.width + 40].cast<double>();
    ASSERT_TRUE(spike.allFinite());
    const auto obstacles = [&](bool filter) {
        s.outlier_filter = filter;
        return processFrames({f}, camera, s).map;
    };
    const ObstacleMap all = obstacles(false), kept = obstacles(true);
    const uint32_t    spike_cell = static_cast<uint32_t>(all.box.cellOf(spike));
    EXPECT_TRUE(std::binary_search(all.obstacle.begin(), all.obstacle.end(), spike_cell));
    EXPECT_FALSE(std::binary_search(kept.obstacle.begin(), kept.obstacle.end(), spike_cell)) << "the spike passed the filter";
    EXPECT_GT(kept.obstacle.size(), 0.95 * all.obstacle.size()) << "holes cost the wall around them";
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
