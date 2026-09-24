// Copyright by BeeX [2026]
#include <sep23/follow.h>

#include <algorithm>

namespace sep23 {

void PathFollower::load(const std::vector<Joints> &corners) {
    waypoints_.clear();
    for (size_t k = 1; k < corners.size(); ++k) {
        const int steps = std::max(1, static_cast<int>(std::ceil(largestMove(corners[k - 1], corners[k]) / s_.max_step)));
        for (int n = 1; n <= steps; ++n) {
            Joints w;
            for (int j = 0; j < JOINT_COUNT; ++j) {
                w[j] = corners[k - 1][j] + (corners[k][j] - corners[k - 1][j]) * n / steps;
            }
            waypoints_.push_back(w);
        }
    }
    next_          = 0;
    strikes_       = 0;
    blocked_joint_ = -1;
    have_now_ = have_previous_ = unjudged_ = false;
    state_ = waypoints_.empty() ? Following::REACHED : Following::SENDING;
}

void PathFollower::measure(const Joints &q) {
    now_      = q;
    have_now_ = unjudged_ = true;
}

Following PathFollower::tick(double now_s, Joints &target, bool &send) {
    send = false;
    // Judging the same reading twice reads as a joint that stopped, and enough of those is a false collision.
    const bool judge = unjudged_;
    unjudged_        = false;
    if (state_ == Following::SENDING) {
        if (judge && strike(jointNotFollowing())) {
            return state_;
        }
        target = waypoints_[next_++];
        send   = true;
        if (next_ >= waypoints_.size()) {
            settle_start_s_ = now_s;
            state_          = Following::SETTLING;
        }
        return state_;
    }
    if (state_ != Following::SETTLING) {
        return state_;
    }
    if (arrived()) {
        state_ = Following::REACHED;
    } else if (!(judge && strike(jointNotClosingIn())) && now_s - settle_start_s_ > s_.arrival_timeout_s) {
        state_ = Following::STALLED;
    }
    return state_;
}

bool PathFollower::arrived() const {
    if (!have_now_) {
        return false;
    }
    for (int j = 0; j < JOINT_COUNT; ++j) {
        if (std::fabs(now_[j] - waypoints_.back()[j]) > s_.arrival_tolerance) {
            return false;
        }
    }
    return true;
}

// While sending: a joint asked to move that barely moved since the last reading.
int PathFollower::jointNotFollowing() const {
    if (!have_now_ || !have_previous_ || next_ < 2) {
        return -1;
    }
    for (int j = 0; j < JOINT_COUNT; ++j) {
        const double asked = std::fabs(waypoints_[next_ - 1][j] - waypoints_[next_ - 2][j]);
        if (asked >= s_.blocked_min_step && std::fabs(now_[j] - previous_[j]) < s_.blocked_follow_fraction * asked) {
            return j;
        }
    }
    return -1;
}

// While settling: a joint still far from the goal that barely closed in since the last reading.
int PathFollower::jointNotClosingIn() const {
    if (!have_now_ || !have_previous_) {
        return -1;
    }
    const Joints &goal = waypoints_.back();
    for (int j = 0; j < JOINT_COUNT; ++j) {
        const double gap_before = std::fabs(goal[j] - previous_[j]);
        const double owed       = std::min(gap_before - s_.arrival_tolerance, s_.max_step);
        if (owed >= s_.blocked_min_step && gap_before - std::fabs(goal[j] - now_[j]) < s_.blocked_follow_fraction * owed) {
            return j;
        }
    }
    return -1;
}

// Counts consecutive readings with a joint not following; enough of them means the arm hit something.
bool PathFollower::strike(int joint) {
    if (joint >= 0 && ++strikes_ >= s_.blocked_strikes) {
        blocked_joint_ = joint;
        state_         = Following::BLOCKED;
        return true;
    }
    if (joint < 0) {
        strikes_ = 0;
    }
    previous_      = now_;
    have_previous_ = have_now_;
    return false;
}

}  // namespace sep23
