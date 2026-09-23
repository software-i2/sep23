// Copyright by BeeX [2026]
#include <sep23/bpl.h>
#include <sep23/follow.h>
#include <sep23/fsm.h>
#include <sep23/park.h>

#include <gtest/gtest.h>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl_parser/kdl_parser.hpp>

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

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
