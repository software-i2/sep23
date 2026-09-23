// Copyright by BeeX [2026]
#pragma once

#include <sep23/collision.h>

#include <functional>

namespace sep23 {

// How one candidate fared, ordered by how far the checks got.
enum class Outcome { UNREACHABLE, JOINT_LIMIT, NO_JAW_ROLL, HULL, OBSTACLE, NOT_PLANNED, NO_PATH, OK };

const char *outcomeName(Outcome o);

// A posture that holds a candidate and is itself allowed.
struct GraspGoal {
    size_t candidate = 0;
    Joints joints{};
    double travel = 0.0;  // summed joint travel from the start
};

// Adds every allowed holding posture; returns NOT_PLANNED if any, else the furthest failure.
Outcome graspGoals(const GraspPose &candidate, size_t index, const Joints &start, double along, Collision &collision,
                   std::vector<GraspGoal> &goals);

double jointTravel(const Joints &a, const Joints &b);

struct PlanSettings {
    double grasp_point_from_mount = 0.0;
    double budget_s = 0.0, goal_budget_s = 0.0;
    double range = 0.0, edge_step = 0.0;
};

struct Plan {
    bool                 ok        = false;
    size_t               candidate = 0;
    std::vector<Joints>  path;  // model radians, starting at the start
    double               travel = 0.0;
    std::vector<Outcome> outcomes;
    std::string          summary;
};

// Plans to the cheapest-looking holding postures in turn with OMPL RRTConnect and keeps the first that joins.
Plan planGrasp(const std::vector<GraspPose> &candidates, const Joints &start, Collision &collision, const PlanSettings &s,
               const std::function<bool()> &cancelled);

}  // namespace sep23
