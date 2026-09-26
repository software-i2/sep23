// Copyright by BeeX [2026]
#pragma once

#include <sep23/log.h>

#include <sep23/follow.h>
#include <sep23/fsm.h>
#include <sep23/park.h>
#include <sep23/plan.h>
#include <sep23/track.h>

#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace sep23 {

struct PickConfig {
    std::string                          world_frame, vehicle_frame, arm_frame, camera_frame;
    std::array<std::string, JOINT_COUNT> joint_names;
    std::string                          jaw_name;
    double                               jaw_closed = 0.0;
    Joints                               home{};  // reported radians, assumed when no joint readings have arrived
    CameraModel                          camera;
    Eigen::Isometry3d                    camera_to_vehicle = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d                    arm_to_vehicle    = Eigen::Isometry3d::Identity();
    bool                                 camera_fixed_in_world = false;  // a replayed bag or the synthetic scene saw it from one place
    Hull                                 hull;
    CloudSettings                        cloud;
    int                                  frames_needed = 1, approach_column = 0, bar_column = 1;
    double                               frame_timeout_s = 0.0;
    ParkSettings                         park;
    double                               speed = 0.0, yaw_speed = 0.0;
    PlanSettings                         plan;
    int                                  seed = 0;
    double                               start_tolerance = 0.0;
    FollowSettings                       follow;
    TrackSettings                        track;
    bool                                 track_enabled = false, track_steer = false;
    double                               track_retarget = 0.0, track_max_shift = 0.0, track_replan_s = 0.0;
    double                               loop_hz = 0.0, joint_state_timeout_s = 0.0;
    double                               stream_timeout_s = 0.0, spot_search_s = 0.0, repark_search_s = 0.0;
    int                                  retarget_attempts = 0, park_attempts = 0;
    double                               jaw_settle_tolerance = 0.0, jaw_settle_time_s = 0.0, jaw_grabbed_margin = 0.0, jaw_timeout_s = 0.0;
    double                               close_within = 0.0, close_hold_s = 0.0;
};

struct PickStatus {
    State       state = State::READY;
    std::string message;
    uint32_t    park_attempt = 0;  // parking spots driven to this pick
    bool        grabbed      = false;
};

// Everything the pick needs from outside it. roswrapper.cpp implements it over ROS1; the pick never sees a message type.
class PickIO {
public:
    virtual ~PickIO() = default;

    virtual double now()                                       = 0;  // seconds, on the clock the sensor stamps are on
    virtual void   log(Level level, const std::string &text) = 0;

    // The arm driver.
    virtual void sendTargets(const Joints &reported) = 0;  // the driver's radians, in joint order
    virtual bool closeJaw(std::string &why)          = 0;
    virtual bool standby()                           = 0;
    virtual bool home(std::string &message)          = 0;  // false when the driver did not answer

    // What the pick shows; nothing here feeds back into it.
    virtual void status(const PickStatus &s)                                                              = 0;
    virtual void showMap(const ObstacleMap &map, const Eigen::Isometry3d &map_to_world, bool park)       = 0;
    virtual void showGrasps(const Scene &scene, const Eigen::Isometry3d &scene_to_world, int chosen)     = 0;
    virtual void showPath(const std::vector<Eigen::Vector3d> &tip)                                        = 0;  // arm frame; empty wipes it
    virtual void showBody(const Body &body)                                                               = 0;  // arm frame
    virtual void showTrack(const TrackResult &r, const std::vector<Eigen::Vector3d> &trail)               = 0;  // camera frame
    virtual void clearTrack()                                                                             = 0;
};

// Runs the pick after one start: survey, park, survey again from there, plan, follow, close the jaw.
// tick() runs on one thread; the sensor and request calls may come from others.
class Pick {
public:
    // Adds the arm's reach to the crop radius and seeds the planner.
    Pick(const PickConfig &c, const Arm &arm, PickIO &io);

    const PickConfig &config() const { return c_; }

    void tick();

    // --- requests ---
    void requestStart(bool steer);  // `steer` replaces track/steer for this run
    void requestStop() { stop_requested_ = true; }
    // Puts the simulated vehicle back at the origin and the arm home, so a loop repeats a pick from the same place.
    bool reset(std::string &message);

    // --- sensors, stamps in seconds on io.now()'s clock ---
    void cloudSeen();
    void addFrame(Frame frame);  // its camera_to_arm is filled in here
    void addPoseless();          // a frame that carried no grasp poses
    void joints(const Joints &reported, double stamp);
    void jaw(double position, double stamp);
    // Whether a tracker frame is wanted at all: while tracking, or for a viewer.
    bool wantsTrackFrame(bool viewer);
    // Tracks one colour frame and its organised cloud in the camera frame; returns the debug view when `viewer`, else nothing.
    cv::Mat trackFrame(const cv::Mat &bgr, const std::vector<Eigen::Vector3f> &points, bool viewer);
    void    setView(TrackView v) { view_ = v; }

    VehiclePose vehicle();

private:
    void setVehicle(const VehiclePose &v);
    bool freshJoints(Joints &q, double &stamp);
    Eigen::Isometry3d armToWorld() { return toIsometry(vehicle()) * c_.arm_to_vehicle; }

    void  apply(Event event, const std::string &message);
    void  enter(State next, std::string message);
    Event step(std::string &message);
    Event retry(double since, double budget_s, bool out_of_tries, Event again, Event done, std::string &message);
    Event checkStream(std::string &message);
    Event checkCollect(std::string &message);
    Event process(std::string &message);
    Event pickSpot(std::string &message);
    Event drive(std::string &message);
    Event pickGrasp(std::string &message);
    bool  steer();
    void  noteSteer(const std::string &note);
    Event follow(std::string &message);
    std::string motionReport();
    Event checkJaw(std::string &message);

    void publishState(const std::string &message, State left = State::READY, double spent = -1.0);
    void publishPath();
    void publishBody();

    PickConfig   c_;
    const Arm   &arm_;
    PickIO      &io_;
    ParkSearch   park_;
    PathFollower follower_;

    std::mutex                   track_mutex_;
    Tracker                      tracker_;
    TrackResult                  track_;
    std::vector<Eigen::Vector3d> trail_;  // the tracked target over this blind motion
    std::atomic<TrackView>       view_{TrackView::RANSAC};
    std::string                  steer_note_;                                // what steering last did
    Eigen::Vector3d              steer_aim_ = Eigen::Vector3d::Zero();       // its shift, arm frame

    std::mutex        sensor_mutex_;
    std::deque<Frame> frames_;
    size_t            poseless_ = 0;
    double            frame_seen_ = 0.0, joints_stamp_ = 0.0, jaw_stamp_ = 0.0;  // 0: never
    Joints            joints_{};
    double            jaw_ = 0.0;

    std::mutex  vehicle_mutex_;
    VehiclePose vehicle_;

    std::atomic<bool> start_requested_{false}, stop_requested_{false}, working_{false}, steer_requested_{false};
    State             state_ = State::READY;
    bool              spot_phase_ = true, grabbed_ = false;
    uint32_t          park_attempt_ = 0, look_attempt_ = 0;
    long              ticks_ = 0;
    // Seconds on io.now()'s clock; 0 is unset.
    double            entered_ = 0.0, run_started_ = 0.0, surveying_since_ = 0.0, reparking_since_ = 0.0, last_stamp_ = 0.0;
    Scene             scene_;
    Eigen::Isometry3d scene_to_world_ = Eigen::Isometry3d::Identity();
    Plan              plan_;
    Eigen::Isometry3d scene_camera_to_arm_ = Eigen::Isometry3d::Identity();  // the newest frame of the look
    Eigen::Vector3d   attempted_           = Eigen::Vector3d::Zero();      // the last tracked shift tried, arm frame
    Eigen::Vector3d   aimed_               = Eigen::Vector3d::Zero();      // the shift the goal actually took
    int               retargets_           = 0;
    Eigen::Vector3d   replanned_           = Eigen::Vector3d::Zero();      // the tracked shift last planned around
    std::map<std::string, int>    steer_tries_;                            // what steering tries came to this motion
    std::unique_ptr<ObstacleGrid> grid_;  // the grasp look's obstacles, inflated once; steering shifts it by a query offset
    VehiclePose       drive_from_, drive_to_;
    int               drive_steps_ = 1, drive_step_ = 0;
    double            jaw_closed_at_ = 0.0, jaw_last_seen_ = 0.0, jaw_still_since_ = 0.0, near_since_ = 0.0;
    double            jaw_still_ = 0.0;
};

}  // namespace sep23
