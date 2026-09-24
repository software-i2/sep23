// Copyright by BeeX [2026]
#pragma once

#include <sep23/arm.h>

#include <vector>

namespace sep23 {

struct FollowSettings {
    double max_step = 0.0, arrival_tolerance = 0.0, arrival_timeout_s = 0.0;
    double blocked_min_step = 0.0, blocked_follow_fraction = 0.0;
    int    blocked_strikes = 0;
};

enum class Following { SENDING, SETTLING, REACHED, STALLED, BLOCKED };

// Sends one waypoint per tick and judges from the joint readings whether the arm keeps up.
class PathFollower {
public:
    explicit PathFollower(const FollowSettings &s) : s_(s) {}

    // Cuts the lines between corners into waypoints; the first corner is where the arm already is.
    void load(const std::vector<Joints> &corners);
    // A new joint reading; only readings not yet judged may decide whether a joint keeps up.
    void measure(const Joints &q);
    void loseMeasurement();
    Following tick(double now_s, Joints &target, bool &send);

    int    blockedJoint() const { return blocked_joint_; }

private:
    bool arrived() const;
    int  jointNotFollowing() const;
    int  jointNotClosingIn() const;
    bool strike(int joint);

    FollowSettings      s_;
    std::vector<Joints> waypoints_;
    size_t              next_          = 0;
    Following           state_         = Following::REACHED;
    double              settle_start_s_ = 0.0;
    Joints              now_{}, previous_{};
    bool                have_now_ = false, have_previous_ = false, unjudged_ = false;
    int                 strikes_ = 0, blocked_joint_ = -1;
};

}  // namespace sep23
