// Copyright by BeeX [2026]
#include <sep23/plan.h>

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>

namespace sep23 {
namespace ob = ompl::base;
namespace og = ompl::geometric;

const char *outcomeName(Outcome o) {
    switch (o) {
    case Outcome::UNREACHABLE: return "unreachable";
    case Outcome::JOINT_LIMIT: return "joint_limit";
    case Outcome::NO_JAW_ROLL: return "no_jaw_roll";
    case Outcome::HULL: return "hull";
    case Outcome::OBSTACLE: return "obstacle";
    case Outcome::NOT_PLANNED: return "not_planned";
    case Outcome::NO_PATH: return "no_path";
    case Outcome::OK: return "ok";
    }
    return "unknown";
}

double jointTravel(const Joints &a, const Joints &b) {
    double sum = 0.0;
    for (int j = 0; j < JOINT_COUNT; ++j) {
        sum += std::fabs(b[j] - a[j]);
    }
    return sum;
}

double largestMove(const Joints &a, const Joints &b) {
    double largest = 0.0;
    for (int j = 0; j < JOINT_COUNT; ++j) {
        largest = std::max(largest, std::fabs(b[j] - a[j]));
    }
    return largest;
}

Outcome graspGoals(const GraspPose &candidate, size_t index, const Joints &start, double along, Collision &collision,
                   std::vector<GraspGoal> &goals) {
    const Arm &arm      = collision.arm();
    Outcome    furthest = Outcome::UNREACHABLE;
    const auto reached  = [&](Outcome o) { furthest = std::max(furthest, o); };
    bool       found    = false;
    for (const bool elbow_up : {false, true}) {
        Joints   q;
        const Ik ik = arm.solve(candidate.point, along, elbow_up, start, q);
        if (ik == Ik::JOINT_LIMIT) {
            reached(Outcome::JOINT_LIMIT);
        }
        double rolls[2];
        if (ik != Ik::SOLVED) {
            continue;
        }
        if (!arm.rollsAcrossBar(q, candidate.bar, rolls)) {
            reached(Outcome::NO_JAW_ROLL);
            continue;
        }
        for (const double roll : rolls) {
            if (!arm.fitIntoLimits(WRIST, roll, start[WRIST], q[WRIST])) {
                reached(Outcome::NO_JAW_ROLL);
                continue;
            }
            const Verdict v = collision.check(q);
            if (v == Verdict::CLEAR) {
                goals.push_back({index, q, jointTravel(start, q)});
                found = true;
            } else {
                reached(v == Verdict::HULL ? Outcome::HULL : v == Verdict::OBSTACLE ? Outcome::OBSTACLE : Outcome::JOINT_LIMIT);
            }
        }
    }
    return found ? Outcome::NOT_PLANNED : furthest;
}

namespace {

Joints toJoints(const ob::State *s) {
    const double *v = s->as<ob::RealVectorStateSpace::StateType>()->values;
    return {{v[0], v[1], v[2], v[3]}};
}

// One RRTConnect query; the start is taken as valid so the arm can always leave where it is.
bool connect(const Joints &start, const Joints &goal, Collision &collision, const PlanSettings &s, double budget_s,
             const std::function<bool()> &cancelled, std::vector<Joints> &path) {
    if (collision.segmentClear(start, goal, s.edge_step)) {
        path = {start, goal};
        return true;
    }
    auto               space = std::make_shared<ob::RealVectorStateSpace>(JOINT_COUNT);
    ob::RealVectorBounds bounds(JOINT_COUNT);
    for (int j = 0; j < JOINT_COUNT; ++j) {
        bounds.setLow(j, collision.arm().lower(j));
        bounds.setHigh(j, collision.arm().upper(j));
    }
    space->setBounds(bounds);

    og::SimpleSetup setup(space);
    setup.setStateValidityChecker([&](const ob::State *state) {
        const Joints q = toJoints(state);
        return q == start || collision.check(q) == Verdict::CLEAR;
    });
    setup.getSpaceInformation()->setStateValidityCheckingResolution(s.edge_step / space->getMaximumExtent());
    auto planner = std::make_shared<og::RRTConnect>(setup.getSpaceInformation());
    planner->setRange(s.range);
    setup.setPlanner(planner);

    ob::ScopedState<> from(space), to(space);
    for (int j = 0; j < JOINT_COUNT; ++j) {
        from[j] = start[j];
        to[j]   = goal[j];
    }
    setup.setStartAndGoalStates(from, to);

    const auto started = std::chrono::steady_clock::now();
    const ob::PlannerTerminationCondition stop = ob::plannerOrTerminationCondition(
            ob::timedPlannerTerminationCondition(budget_s), ob::PlannerTerminationCondition(cancelled));
    if (setup.solve(stop) != ob::PlannerStatus::EXACT_SOLUTION) {
        return false;
    }
    const double spent = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    setup.simplifySolution(std::max(0.0, budget_s - spent));
    path.clear();
    for (const ob::State *state : setup.getSolutionPath().getStates()) {
        path.push_back(toJoints(state));
    }
    return true;
}

}  // namespace

Plan planGrasp(const std::vector<GraspPose> &candidates, const Joints &start, Collision &collision, const PlanSettings &s,
               const std::function<bool()> &cancelled) {
    const auto started   = std::chrono::steady_clock::now();
    const auto remaining = [&] { return s.budget_s - std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); };
    const double along = collision.arm().mountDistance() + s.grasp_point_from_mount;

    Plan plan;
    plan.outcomes.assign(candidates.size(), Outcome::UNREACHABLE);
    std::vector<GraspGoal> goals;
    for (size_t i = 0; i < candidates.size(); ++i) {
        plan.outcomes[i] = graspGoals(candidates[i], i, start, along, collision, goals);
    }
    std::sort(goals.begin(), goals.end(), [](const GraspGoal &a, const GraspGoal &b) { return a.travel < b.travel; });

    for (const GraspGoal &goal : goals) {
        if (plan.ok || cancelled() || remaining() < s.goal_budget_s) {
            break;
        }
        if (connect(start, goal.joints, collision, s, std::min(s.goal_budget_s, remaining()), cancelled, plan.path)) {
            plan.ok        = true;
            plan.candidate = goal.candidate;
            plan.outcomes[goal.candidate] = Outcome::OK;
            for (size_t k = 1; k < plan.path.size(); ++k) {
                plan.travel += jointTravel(plan.path[k - 1], plan.path[k]);
            }
        } else if (plan.outcomes[goal.candidate] != Outcome::OK) {
            plan.outcomes[goal.candidate] = Outcome::NO_PATH;
        }
    }

    std::map<std::string, int> tally;
    for (const Outcome o : plan.outcomes) {
        ++tally[outcomeName(o)];
    }
    char line[160];
    const double spent = s.budget_s - remaining();
    if (plan.ok) {
        std::snprintf(line, sizeof(line), "chose candidate %zu of %zu, %.3f rad of travel over %zu corners (%.2f s):",
                      plan.candidate, candidates.size(), plan.travel, plan.path.size(), spent);
    } else {
        std::snprintf(line, sizeof(line), "no path to any of %zu candidates (%.2f s):", candidates.size(), spent);
    }
    plan.summary = line;
    for (const auto &t : tally) {
        plan.summary += " " + std::to_string(t.second) + " " + t.first;
    }
    return plan;
}

}  // namespace sep23
