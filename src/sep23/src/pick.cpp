// Copyright by BeeX [2026]
#include <sep23/pick.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace sep23 {

Pick::Pick(const PickConfig &c, const Arm &arm, PickIO &io)
        : c_(c),
          arm_(arm),
          io_(io),
          park_(arm, c.hull, c.park, c.arm_to_vehicle.inverse(), c.camera_to_vehicle.translation()),
          follower_(c.follow),
          tracker_(c.track, c.camera) {
    c_.cloud.crop_radius += arm_.reach(arm_.tipDistance());
    seedPlanner(static_cast<unsigned>(c_.seed));
    entered_ = io_.now();
    publishState("waiting for start");
}

void Pick::tick() {
    if (stop_requested_.exchange(false)) {
        if (isWorking(state_)) {
            apply(Event::STOP, "stopped on request");
        } else if (!io_.standby()) {  // after a pick the arm still holds its last posture
            io_.log(Level::ERROR, "stop: standby failed, the arm may still be powered");
        }
    }
    if (start_requested_.exchange(false)) {
        if (isWorking(state_)) {
            io_.log(Level::WARN, std::string("start ignored: already in ") + stateName(state_));
        } else {
            park_attempt_ = look_attempt_ = 0;
            grabbed_         = false;
            surveying_since_ = reparking_since_ = 0.0;
            run_started_                        = io_.now();
            io_.showMap(ObstacleMap(), Eigen::Isometry3d::Identity(), false);
            io_.showPath({});
            if (c_.track_enabled) {
                io_.clearTrack();
            }
            c_.track_steer = steer_requested_;
            apply(Event::START, c_.track_steer ? "started, steering" : "started, not steering");
        }
    }
    std::string message;
    const Event event = step(message);
    if (!stop_requested_) {
        apply(event, message);
    }
    if (++ticks_ % std::max(1, static_cast<int>(c_.loop_hz / 2.0)) == 0) {
        publishBody();
    }
}

// --- requests ---

void Pick::requestStart(bool steer) {
    steer_requested_ = steer;
    start_requested_ = true;
}

bool Pick::reset(std::string &message) {
    const bool ok = !working_;
    message       = ok ? "vehicle back at the origin" : "a pick is running, stop it first";
    if (ok) {
        setVehicle(VehiclePose());
        std::string homed;
        message += io_.home(homed) ? ", arm " + homed : ", arm not homed: the driver did not answer";
    }
    return ok;
}

// --- sensors ---

void Pick::cloudSeen() {
    const double now = io_.now();
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    frame_seen_ = now;
}

void Pick::addFrame(Frame frame) {
    const Eigen::Isometry3d vehicle_to_world = toIsometry(vehicle());
    const Eigen::Isometry3d camera_to_world  = c_.camera_fixed_in_world ? c_.camera_to_vehicle : vehicle_to_world * c_.camera_to_vehicle;
    frame.camera_to_arm = (vehicle_to_world * c_.arm_to_vehicle).inverse() * camera_to_world;

    std::lock_guard<std::mutex> lock(sensor_mutex_);
    frames_.push_back(std::move(frame));
    while (frames_.size() > static_cast<size_t>(c_.frames_needed)) {
        frames_.pop_front();
    }
}

void Pick::addPoseless() {
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    ++poseless_;
}

void Pick::joints(const Joints &reported, double stamp) {
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    joints_       = reported;
    joints_stamp_ = stamp;
}

void Pick::jaw(double position, double stamp) {
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    jaw_       = position;
    jaw_stamp_ = stamp;
}

bool Pick::wantsTrackFrame(bool viewer) {
    std::lock_guard<std::mutex> lock(track_mutex_);
    return tracker_.active() || viewer;
}

// While the arm moves blind, follow the target in the image; with track/steer, steer() moves the goal with it.
cv::Mat Pick::trackFrame(const cv::Mat &bgr_in, const std::vector<Eigen::Vector3f> &points, bool viewer) {
    std::lock_guard<std::mutex> lock(track_mutex_);
    if (!tracker_.active()) {  // nothing is tracked between blind motions: say so rather than show a bare camera feed
        if (!viewer) {
            return cv::Mat();
        }
        cv::Mat           bgr  = bgr_in.clone();
        const std::string idle = "idle";
        cv::putText(bgr, idle, cv::Point(12, 26), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
        cv::putText(bgr, idle, cv::Point(12, 26), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        return bgr;
    }
    cv::Mat gray;
    cv::cvtColor(bgr_in, gray, cv::COLOR_BGR2GRAY);
    const bool was_covered = track_.covered;
    track_                 = tracker_.update(gray, points);
    if (track_.covered && !was_covered) {
        char line[120];
        std::snprintf(line, sizeof(line), "only %.0f%% of the mine face is visible: tracking stopped, the arm finishes open loop", 100 * track_.face);
        io_.log(Level::WARN, line);
    }
    trail_.push_back(track_.target);
    io_.showTrack(track_, trail_);
    if (!viewer) {
        return cv::Mat();
    }
    // Where the arm aims: the planned target moved by the shift steering last took, in camera_link.
    const Eigen::Vector3d aim = track_.target - track_.offset + scene_camera_to_arm_.linear().transpose() * steer_aim_;
    return drawTrack(bgr_in, track_, view_.load(), aim, tracker_.project(aim), c_.track_steer ? steer_note_ : "");
}

VehiclePose Pick::vehicle() {
    std::lock_guard<std::mutex> lock(vehicle_mutex_);
    return vehicle_;
}

void Pick::setVehicle(const VehiclePose &v) {
    std::lock_guard<std::mutex> lock(vehicle_mutex_);
    vehicle_ = v;
}

// Reported radians, if newer than the timeout.
bool Pick::freshJoints(Joints &q, double &stamp) {
    const double now = io_.now();
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    if (joints_stamp_ == 0.0 || now - joints_stamp_ > c_.joint_state_timeout_s) {
        return false;
    }
    q     = joints_;
    stamp = joints_stamp_;
    return true;
}

// --- state machine ---

void Pick::apply(Event event, const std::string &message) {
    const State next = nextState(state_, event);
    if (next == state_) {
        return;
    }
    if (next == State::COLLECT) {
        spot_phase_ = state_ == State::STREAM || state_ == State::RESURVEY || state_ == State::REPARK;
        if (!spot_phase_) {
            io_.showMap(ObstacleMap(), Eigen::Isometry3d::Identity(), true);
        }
    }
    enter(next, message);
}

void Pick::enter(State next, std::string message) {
    const State  left  = state_;
    const double now   = io_.now();
    const double spent = now - entered_;
    state_             = next;
    working_           = isWorking(next);
    entered_           = now;
    if (next != State::GOTOGRASP && next != State::CLOSEJAW) {
        std::lock_guard<std::mutex> lock(track_mutex_);
        tracker_.stop();
    }
    char line[200];
    switch (next) {
    case State::GOTOGRASP: {
        // follow the chosen candidate from where the grasp look saw it.
        if (!c_.track_enabled) {
            break;
        }
        std::lock_guard<std::mutex> lock(track_mutex_);
        tracker_.start(scene_camera_to_arm_.inverse() * scene_.candidates[plan_.candidate].point);
        track_ = TrackResult();
        trail_.clear();
        steer_note_ = "on the plan", steer_aim_ = Eigen::Vector3d::Zero();
        break;
    }
    case State::COLLECT: {
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        frames_.clear();
        poseless_ = 0;
        break;
    }
    case State::PICKSPOT:
        surveying_since_ = 0.0;
        break;
    case State::GOTOSPOT: {
        ++park_attempt_;
        reparking_since_      = 0.0;
        drive_from_           = vehicle();
        const double distance = std::hypot(std::hypot(drive_to_.x - drive_from_.x, drive_to_.y - drive_from_.y), drive_to_.z - drive_from_.z);
        const double seconds  = std::max(distance / c_.speed, std::fabs(drive_to_.yaw - drive_from_.yaw) / c_.yaw_speed);
        drive_steps_          = std::max(1, static_cast<int>(std::ceil(seconds * c_.loop_hz)));
        drive_step_           = 0;
        break;
    }
    case State::CLOSEJAW: {
        std::string why;
        if (!io_.closeJaw(why)) {
            enter(State::FAIL, "could not close the jaw: " + why);
            return;
        }
        jaw_closed_at_   = io_.now();
        jaw_last_seen_   = 0.0;
        jaw_still_since_ = 0.0;
        break;
    }
    case State::RESURVEY:
        if (surveying_since_ == 0.0) {
            surveying_since_ = now;
        }
        std::snprintf(line, sizeof(line), "; nothing to park for after %.0f of %.0f s", now - surveying_since_, c_.spot_search_s);
        message += line;
        break;
    case State::RETARGET:
        ++look_attempt_;
        std::snprintf(line, sizeof(line), "; look %u of %d from this spot gave no grasp", look_attempt_, c_.retarget_attempts);
        message += line;
        break;
    case State::REPARK:
        look_attempt_ = 0;
        if (reparking_since_ == 0.0) {
            reparking_since_ = now;
        }
        std::snprintf(line, sizeof(line), "; %u of %d parking spots used, surveying for somewhere else", park_attempt_, c_.park_attempts);
        message += line;
        break;
    case State::ESTOP:
        if (!io_.standby()) {
            message += "; standby failed, the arm may still be powered";
        }
        break;
    default:
        break;
    }
    publishState(message, left, spent);
}

Event Pick::step(std::string &message) {
    switch (state_) {
    case State::STREAM: return checkStream(message);
    case State::COLLECT: return checkCollect(message);
    case State::PROCESS: return process(message);
    case State::PICKSPOT: return pickSpot(message);
    case State::GOTOSPOT: return drive(message);
    case State::PICKGRASP: return pickGrasp(message);
    case State::GOTOGRASP: return follow(message);
    case State::CLOSEJAW: return checkJaw(message);
    case State::RESURVEY: return retry(surveying_since_, c_.spot_search_s, false, Event::SURVEY_AGAIN, Event::OUT_OF_TIME, message);
    case State::RETARGET:
        return retry(0.0, 0.0, look_attempt_ >= static_cast<uint32_t>(c_.retarget_attempts), Event::LOOK_AGAIN, Event::OUT_OF_LOOKS, message);
    case State::REPARK:
        return retry(reparking_since_, c_.repark_search_s, park_attempt_ >= static_cast<uint32_t>(c_.park_attempts), Event::PARK_AGAIN,
                     Event::OUT_OF_PARKS, message);
    default: return Event::NONE;
    }
}

// Looks again at once unless a count or a clock has run out.
Event Pick::retry(double since, double budget_s, bool out_of_tries, Event again, Event done, std::string &message) {
    if (out_of_tries || (since != 0.0 && io_.now() - since >= budget_s)) {
        message = out_of_tries ? "out of tries" : "out of time";
        return done;
    }
    message = "looking again";
    return again;
}

// A camera already producing when start was pressed counts.
Event Pick::checkStream(std::string &message) {
    double seen = 0.0;
    {
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        seen = frame_seen_;
    }
    const double now = io_.now();
    if (seen != 0.0 && now - seen < c_.stream_timeout_s) {
        message = "the camera is producing";
        return Event::STREAMING;
    }
    if (now - entered_ >= c_.stream_timeout_s) {
        message = "no camera frame in " + std::to_string(c_.stream_timeout_s) + " s";
        return Event::NO_STREAM;
    }
    return Event::NONE;
}

// Silence is a fault; frames that carry no grasp poses are a survey with nothing in view.
Event Pick::checkCollect(std::string &message) {
    size_t have = 0, poseless = 0;
    {
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        have     = frames_.size();
        poseless = poseless_;
    }
    if (have >= static_cast<size_t>(c_.frames_needed)) {
        message = std::to_string(have) + " frames in";
        return Event::FRAMES_IN;
    }
    if (io_.now() - entered_ < c_.frame_timeout_s) {
        return Event::NONE;
    }
    if (have == 0 && poseless > 0) {
        message = "no handle in view: " + std::to_string(poseless) + " frames carried no grasp poses";
        return spot_phase_ ? Event::NO_CANDIDATES_SPOT : Event::NO_CANDIDATES_GRASP;
    }
    message = "only " + std::to_string(have) + " of " + std::to_string(c_.frames_needed) + " frames in " + std::to_string(c_.frame_timeout_s)
              + " s";
    return Event::FAILURE;
}

Event Pick::process(std::string &message) {
    std::vector<Frame> frames;
    {
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        frames.assign(frames_.begin(), frames_.end());
    }
    const auto started = std::chrono::steady_clock::now();
    scene_             = processFrames(frames, c_.camera, c_.cloud);
    if (!frames.empty()) {
        scene_camera_to_arm_ = frames.back().camera_to_arm;
    }
    scene_to_world_ = armToWorld();
    char took[32];
    std::snprintf(took, sizeof(took), " (%.2f s)", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    message = scene_.summary + took;
    io_.showMap(scene_.map, scene_to_world_, spot_phase_);
    io_.showGrasps(scene_, scene_to_world_, -1);
    if (scene_.candidates.empty()) {
        return spot_phase_ ? Event::NO_CANDIDATES_SPOT : Event::NO_CANDIDATES_GRASP;
    }
    return spot_phase_ ? Event::CANDIDATES_SPOT : Event::CANDIDATES_GRASP;
}

Event Pick::pickSpot(std::string &message) {
    Joints reported = c_.home;
    double stamp    = 0.0;
    freshJoints(reported, stamp);
    ParkChoice choice;
    try {
        choice = park_.choose(scene_.candidates, scene_.map, arm_.toModel(reported), [this] { return stop_requested_.load(); });
    } catch (const std::invalid_argument &e) {
        message = std::string("the obstacle map is not usable: ") + e.what();
        return Event::FAILURE;
    }
    message = choice.summary;
    if (choice.decision == ParkDecision::MOVE) {
        drive_to_ = compose(vehicle(), choice.move);
        return Event::SPOT_CHOSEN;
    }
    return choice.decision == ParkDecision::STAY ? Event::SPOT_UNCHANGED : Event::NO_SPOT;
}

Event Pick::drive(std::string &message) {
    const double f = static_cast<double>(++drive_step_) / drive_steps_;
    setVehicle({drive_from_.x + (drive_to_.x - drive_from_.x) * f, drive_from_.y + (drive_to_.y - drive_from_.y) * f,
                drive_from_.z + (drive_to_.z - drive_from_.z) * f, drive_from_.yaw + (drive_to_.yaw - drive_from_.yaw) * f});
    if (drive_step_ < drive_steps_) {
        return Event::NONE;
    }
    message = "arrived";
    return Event::ARRIVED;
}

Event Pick::pickGrasp(std::string &message) {
    Joints reported;
    double stamp = 0.0;
    if (!freshJoints(reported, stamp)) {
        message = "no fresh joint readings to plan from";
        return Event::FAILURE;
    }
    Joints start = arm_.toModel(reported);
    for (int j = 0; j < JOINT_COUNT; ++j) {
        if (start[j] < arm_.lower(j) - c_.start_tolerance || start[j] > arm_.upper(j) + c_.start_tolerance) {
            message = std::string("the arm is outside its ") + JOINT_KEYS[j] + " limit";
            return Event::FAILURE;
        }
    }
    start = arm_.clamped(start);
    try {
        grid_.reset(new ObstacleGrid(scene_.map, c_.park.link_radius, bladeRadius(scene_.map.box.voxel, c_.park.blade_step)));
        Collision collision(arm_, *grid_, c_.hull, c_.park.link_step);
        plan_ = planGrasp(scene_.candidates, start, collision, c_.plan, [this] { return stop_requested_.load(); });
    } catch (const std::invalid_argument &e) {
        message = std::string("the obstacle map is not usable: ") + e.what();
        return Event::FAILURE;
    }
    attempted_ = aimed_ = replanned_ = Eigen::Vector3d::Zero();
    near_since_                      = 0.0;
    retargets_                       = 0;
    steer_tries_.clear();
    message = plan_.summary;
    io_.showGrasps(scene_, scene_to_world_, plan_.ok ? static_cast<int>(plan_.candidate) : -1);
    publishPath();
    if (!plan_.ok) {
        return Event::NO_PLAN;
    }
    follower_.load(plan_.path);
    last_stamp_ = 0.0;
    return Event::PLAN_FOUND;
}

// Until the face is covered (track/min_face_visible), the goal and the obstacles shift with the tracked target (the scene
// taken as rigid) and the arm is sent to no posture that hits them there. Once the tracked target is track/retarget_m from
// where the arm is heading, or the next posture is blocked, it goes straight to the shifted goal if clear, else plans
// around for track/replan_budget_s. Returns true when the arm must hold: its next posture is blocked and nothing clear was
// found. Shifts past track/max_shift_m are taken as tracking failures.
bool Pick::steer() {
    Eigen::Vector3d offset;
    bool            ok = false;
    {
        std::lock_guard<std::mutex> lock(track_mutex_);
        if (track_.covered) {
            steer_note_ = "stopped, face covered: open loop to the last goal";
            return false;
        }
        ok     = track_.ok;
        offset = scene_camera_to_arm_.linear() * track_.offset;  // the last good shift while a frame finds no consensus
    }
    Joints now;
    double stamp = 0.0;
    if (offset.norm() > c_.track_max_shift) {
        noteSteer("ignoring a tracked shift past max_shift_m");
        return false;
    }
    if (!freshJoints(now, stamp)) {
        return false;
    }
    now = arm_.clamped(arm_.toModel(now));  // the planner takes no start past a limit
    grid_->setQueryToMap(Eigen::Isometry3d(Eigen::Translation3d(-offset)));  // a point near the moved scene, in the map
    Collision     collision(arm_, *grid_, c_.hull, c_.park.link_step);
    Joints        next;
    const Verdict ahead   = follower_.upcoming(next) ? collision.check(next) : Verdict::CLEAR;
    const bool    blocked = ahead == Verdict::OBSTACLE || ahead == Verdict::HULL;
    const char   *hits    = ahead == Verdict::HULL ? "hull" : "obstacle";
    // Tried once per tracker frame (each moves the offset).
    if (offset == attempted_ || (!blocked && (!ok || (offset - aimed_).norm() < c_.track_retarget))) {
        return blocked;
    }
    attempted_         = offset;
    const auto give_up = [&](const std::string &why) {
        ++steer_tries_[why];
        noteSteer(blocked ? std::string("HOLDING, next posture hits the ") + hits + "; " + why : "staying on course; " + why);
        return blocked;
    };

    GraspPose target = scene_.candidates[plan_.candidate];
    target.point += offset;
    std::vector<GraspGoal> goals;
    const Outcome o = graspGoals(target, 0, plan_.path.back(), arm_.mountDistance() + c_.plan.grasp_point_from_mount, collision, goals);
    if (goals.empty()) {
        return give_up(std::string("no holding posture (") + outcomeName(o) + ")");
    }
    std::sort(goals.begin(), goals.end(), [](const GraspGoal &a, const GraspGoal &b) { return a.swing < b.swing; });
    std::vector<Joints> path;
    Verdict             v = Verdict::CLEAR;
    for (size_t i = 0; i < goals.size() && path.empty(); ++i) {
        v = collision.sweep(now, goals[i].joints, c_.plan.edge_step);
        if (v == Verdict::CLEAR) {
            path = {now, goals[i].joints};
        }
    }
    // Nothing straight: plan around it. The arm waits while that runs: every frame when blocked, else once per retarget_m.
    const char *how = "straight";
    if (path.empty() && (blocked || (offset - replanned_).norm() >= c_.track_retarget)) {
        replanned_      = offset;
        PlanSettings s  = c_.plan;
        s.budget_s      = c_.track_replan_s;
        s.goal_budget_s = 0.5 * c_.track_replan_s;
        const Plan p    = planGrasp({target}, now, collision, s, [this] { return stop_requested_.load(); });
        if (!p.ok) {
            return give_up("replan found no way");
        }
        path = p.path;
        how  = "replanned around";
    }
    if (path.empty()) {
        return give_up(std::string("straight way blocked (") + (v == Verdict::HULL ? "hull" : v == Verdict::OBSTACLE ? "obstacle" : "joint_limit") + ")");
    }
    ++steer_tries_[std::string(blocked ? "avoided: " : "retargeted: ") + how];
    plan_.path = path;
    aimed_     = offset;
    follower_.load(plan_.path);
    last_stamp_ = 0.0;
    ++retargets_;
    publishPath();
    char line[120];
    std::snprintf(line, sizeof(line), "%s%s to the target shifted %.0f mm", blocked ? "avoided the obstacle, " : "", how, 1000 * offset.norm());
    noteSteer(line);
    return false;
}

// What steering last did and the shift it aims at, for the debug image.
void Pick::noteSteer(const std::string &note) {
    std::lock_guard<std::mutex> lock(track_mutex_);
    steer_note_ = note;
    steer_aim_  = aimed_;
}

Event Pick::follow(std::string &message) {
    const bool hold = c_.track_enabled && c_.track_steer && steer();
    Joints     reported;
    double     stamp = 0.0;
    // Without readings a joint against something goes unnoticed, so the arm holds its last target instead of moving on blind.
    if (!freshJoints(reported, stamp)) {
        message = "joint readings went stale mid-motion, so contact cannot be judged; holding the last target";
        return Event::FAILURE;
    }
    if (stamp != last_stamp_) {
        last_stamp_ = stamp;
        const Joints q = arm_.toModel(reported);
        follower_.measure(q);
        const Eigen::Vector3d grasp = arm_.points(q).mount + arm_.axes(q).approach * c_.plan.grasp_point_from_mount;
        const double          off   = (grasp - (scene_.candidates[plan_.candidate].point + aimed_)).norm();
        if (off > c_.close_within) {
            near_since_ = 0.0;
        } else if (near_since_ == 0.0) {
            near_since_ = stamp;
        } else if (stamp - near_since_ >= c_.close_hold_s) {
            char line[120];
            std::snprintf(line, sizeof(line), "held %.1f mm from the aimed target for %.1f s", 1000 * off, c_.close_hold_s);
            message = line + motionReport();
            return Event::REACHED;
        }
    }
    Joints          target;
    bool            send  = false;
    const Following state = hold ? Following::SENDING : follower_.tick(io_.now(), target, send);
    if (send) {
        io_.sendTargets(arm_.toReported(target));
    }
    switch (state) {
    case Following::REACHED:
        message = "reached the end of the path" + motionReport();
        return Event::REACHED;
    case Following::STALLED:
        io_.standby();
        message = "did not arrive within the arrival timeout, arm released";
        return Event::STALLED;
    case Following::BLOCKED:
        message = c_.joint_names[follower_.blockedJoint()] + " stopped following, so the arm hit something" + motionReport();
        return Event::COLLIDED;
    default:
        return Event::NONE;
    }
}

// What the tracker and steering did on the way there.
std::string Pick::motionReport() {
    if (!c_.track_enabled) {
        return "";
    }
    std::lock_guard<std::mutex> lock(track_mutex_);
    char                        line[120];
    std::snprintf(line, sizeof(line), "; the tracker saw the target move %.1f mm (%s)", 1000 * track_.offset.norm(),
                  track_.covered ? "face covered, open loop" : track_.ok ? "tracking" : "not tracking");
    std::string out = line;
    if (c_.track_steer) {
        const double behind = (scene_camera_to_arm_.linear() * track_.offset - aimed_).norm();
        std::snprintf(line, sizeof(line), "; %d retargets left the goal %.1f mm from it", retargets_, 1000 * behind);
        out += line;
        for (const auto &b : steer_tries_) {
            out += "; " + std::to_string(b.second) + " tries: " + b.first;
        }
    }
    return out;
}

// Settled once the jaw reading holds still; stopping short of closed means it holds something.
Event Pick::checkJaw(std::string &message) {
    double jaw = 0.0, stamp = 0.0;
    {
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        jaw   = jaw_;
        stamp = jaw_stamp_;
    }
    char line[160];
    if (stamp > jaw_closed_at_ && stamp != jaw_last_seen_) {
        jaw_last_seen_ = stamp;
        if (jaw_still_since_ == 0.0 || std::fabs(jaw - jaw_still_) > c_.jaw_settle_tolerance) {
            jaw_still_       = jaw;
            jaw_still_since_ = stamp;
        } else if (stamp - jaw_still_since_ >= c_.jaw_settle_time_s) {
            grabbed_ = jaw > c_.jaw_closed + c_.jaw_grabbed_margin;
            std::snprintf(line, sizeof(line), "%s: the jaw settled at %.1f mm", grabbed_ ? "holding something" : "closed on nothing", jaw * 1000.0);
            message = line;
            return Event::JAW_SETTLED;
        }
    }
    if (io_.now() - entered_ >= c_.jaw_timeout_s) {
        grabbed_ = false;
        message  = "the jaw did not settle, so there is no grip verdict";
        return Event::JAW_SETTLED;
    }
    return Event::NONE;
}

// --- output ---

void Pick::publishState(const std::string &message, State left, double spent) {
    io_.status({state_, message, park_attempt_, grabbed_});
    char timing[80] = "";
    if (spent >= 0.0) {
        std::snprintf(timing, sizeof(timing), "[%s %.1fs, run %.1fs] ", stateName(left), spent, run_started_ == 0.0 ? 0.0 : io_.now() - run_started_);
    }
    const Level level = state_ == State::FAIL || state_ == State::ESTOP                                     ? Level::ERROR
                        : state_ == State::RESURVEY || state_ == State::RETARGET || state_ == State::REPARK ? Level::WARN
                                                                                                             : Level::INFO;
    io_.log(level, std::string(stateName(state_)) + ": " + timing + message);
}

void Pick::publishPath() {
    std::vector<Eigen::Vector3d> tip;
    for (size_t k = 1; k < plan_.path.size(); ++k) {
        for (int n = 0; n <= 20; ++n) {
            Joints q;
            for (int j = 0; j < JOINT_COUNT; ++j) {
                q[j] = plan_.path[k - 1][j] + (plan_.path[k][j] - plan_.path[k - 1][j]) * n / 20.0;
            }
            tip.push_back(arm_.points(q).tip);
        }
    }
    io_.showPath(tip);
}

void Pick::publishBody() {
    Joints reported;
    double stamp = 0.0;
    if (freshJoints(reported, stamp)) {
        Body body;
        arm_.body(arm_.toModel(reported), body, 4);
        io_.showBody(body);
    }
}

}  // namespace sep23
