// Copyright by BeeX [2026]
#include "markers.h"

#include <sep23/PickState.h>
#include <sep23/follow.h>
#include <sep23/fsm.h>
#include <sep23/park.h>
#include <sep23/params.h>
#include <sep23/plan.h>
#include <sep23/track.h>

#include <geometry_msgs/PoseArray.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <ompl/util/Console.h>
#include <ompl/util/RandomNumbers.h>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/String.h>
#include <std_srvs/Trigger.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <atomic>
#include <cstring>
#include <deque>
#include <map>

using namespace sep23;

namespace {

std::vector<Eigen::Vector3f> cloudPoints(const sensor_msgs::PointCloud2 &cloud) {
    std::vector<Eigen::Vector3f> points;
    points.reserve(static_cast<size_t>(cloud.width) * cloud.height);
    sensor_msgs::PointCloud2ConstIterator<float> x(cloud, "x"), y(cloud, "y"), z(cloud, "z");
    for (; x != x.end(); ++x, ++y, ++z) {
        points.emplace_back(*x, *y, *z);
    }
    return points;
}

bool toBgr(const sensor_msgs::Image &image, cv::Mat &bgr) {
    const int type = image.encoding == "mono8" ? CV_8UC1 : image.encoding == "bgr8" || image.encoding == "rgb8" ? CV_8UC3 : -1;
    if (type < 0) {
        return false;
    }
    const cv::Mat view(static_cast<int>(image.height), static_cast<int>(image.width), type, const_cast<uint8_t *>(image.data.data()), image.step);
    if (image.encoding == "mono8") {
        cv::cvtColor(view, bgr, cv::COLOR_GRAY2BGR);
    } else if (image.encoding == "rgb8") {
        cv::cvtColor(view, bgr, cv::COLOR_RGB2BGR);
    } else {
        bgr = view.clone();
    }
    return true;
}

sensor_msgs::Image toImage(const cv::Mat &m, const std_msgs::Header &header, const std::string &encoding) {
    sensor_msgs::Image out;
    out.header   = header;
    out.height   = static_cast<uint32_t>(m.rows);
    out.width    = static_cast<uint32_t>(m.cols);
    out.encoding = encoding;
    out.step     = static_cast<uint32_t>(m.cols * m.elemSize());
    out.data.resize(static_cast<size_t>(out.step) * m.rows);
    for (int r = 0; r < m.rows; ++r) {
        std::memcpy(&out.data[static_cast<size_t>(r) * out.step], m.ptr(r), out.step);
    }
    return out;
}

struct PickConfig {
    std::string                          world_frame, vehicle_frame, arm_frame, camera_frame;
    std::array<std::string, JOINT_COUNT> joint_names;
    std::string                          jaw_name;
    double                               jaw_closed = 0.0;
    Joints                               home{};  // reported radians, assumed when no joint_states have arrived
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
    double                               start_tolerance = 0.0;
    FollowSettings                       follow;
    TrackSettings                        track;
    bool                                 track_enabled = false, track_steer = false;
    double                               track_retarget = 0.0, track_max_shift = 0.0, track_replan_s = 0.0;
    double                               loop_hz = 0.0, joint_state_timeout_s = 0.0;
    double                               stream_timeout_s = 0.0, spot_search_s = 0.0, repark_search_s = 0.0;
    int                                  retarget_attempts = 0, park_attempts = 0;
    double                               jaw_settle_tolerance = 0.0, jaw_settle_time_s = 0.0, jaw_grabbed_margin = 0.0, jaw_timeout_s = 0.0;
    double                               catch_reach = 0.0, catch_off_centre = 0.0, close_within = 0.0, close_hold_s = 0.0;
};

Eigen::Isometry3d fromXyzRpy(const std::vector<double> &xyz, const std::vector<double> &rpy_deg) {
    Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
    t.linear()          = (Eigen::AngleAxisd(rad(rpy_deg[2]), Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(rad(rpy_deg[1]), Eigen::Vector3d::UnitY())
                  * Eigen::AngleAxisd(rad(rpy_deg[0]), Eigen::Vector3d::UnitX()))
                         .toRotationMatrix();
    t.translation() = Eigen::Vector3d(xyz[0], xyz[1], xyz[2]);
    return t;
}

ArmConfig loadArm(Params &robot, const std::string &urdf, PickConfig &c) {
    ArmConfig a;
    for (int j = 0; j < JOINT_COUNT; ++j) {
        const std::string key = JOINT_KEYS[j];
        c.joint_names[j]      = robot.text("arm/joint_names/" + key);
        const std::vector<double> limits = robot.numbers("arm/limits_deg/" + key, 2);
        a.min[j]    = rad(limits[0]);
        a.max[j]    = rad(limits[1]);
        a.offset[j] = rad(robot.number("arm/zero_offset_deg/" + key));
        a.sign[j]   = robot.number("arm/direction_sign/" + key);
        c.home[j]   = rad(robot.number("arm/home_deg/" + key));
    }
    c.jaw_name   = robot.text("arm/joint_names/jaw");
    c.jaw_closed = robot.numbers("jaws/limits_m", 2)[0];
    JawShape &jaw             = a.jaw;
    jaw.open_width            = robot.number("jaws/open_m");
    jaw.mount_to_throat       = robot.number("jaws/mount_to_throat_m");
    jaw.mount_to_tip          = robot.number("jaws/mount_to_tip_m");
    jaw.palm_length           = robot.number("jaws/palm_length_m");
    jaw.hinge_roll_at_zero    = rad(robot.number("jaws/hinge_roll_at_wrist_zero_deg"));
    jaw.blade_rotation_per_m  = robot.number("jaws/blade_rotation_rad_per_m");
    jaw.hinge_offset_closing  = robot.number("jaws/hinge_offset_closing_m");
    jaw.hinge_offset_approach = robot.number("jaws/hinge_offset_approach_m");
    for (const std::vector<double> &r : robot.rows("jaws/blade_profile_m", 5)) {
        jaw.blades.push_back({r[0], r[1], r[2], r[3], r[4]});
    }
    a.geometry = geometryFromUrdf(urdf, c.joint_names);
    return a;
}

void loadPick(Params &robot, Params &pick, PickConfig &c) {
    c.world_frame   = robot.text("frames/world");
    c.vehicle_frame = robot.text("frames/vehicle");
    c.arm_frame     = robot.text("frames/arm");
    c.camera_frame  = robot.text("camera/frame");
    c.camera        = {robot.whole("camera/intrinsics/width_px"), robot.whole("camera/intrinsics/height_px"),
                       robot.number("camera/intrinsics/fx_px"),   robot.number("camera/intrinsics/fy_px"),
                       robot.number("camera/intrinsics/cx_px"),   robot.number("camera/intrinsics/cy_px")};
    c.camera_to_vehicle = fromXyzRpy(robot.numbers("camera/mount_xyz_m", 3), robot.numbers("camera/mount_rpy_deg", 3))
                          * fromXyzRpy({0, 0, 0}, robot.numbers("camera/optical_rpy_deg", 3));
    c.arm_to_vehicle = fromXyzRpy(robot.numbers("vehicle/arm_mount_xyz_m", 3), robot.numbers("vehicle/arm_mount_rpy_deg", 3));
    c.hull = {robot.number("vehicle/hull_footprint_m/min_x"), robot.number("vehicle/hull_footprint_m/max_x"),
              robot.number("vehicle/hull_footprint_m/min_y"), robot.number("vehicle/hull_footprint_m/max_y"),
              robot.number("vehicle/hull_floor_z_m")};
    robot.require(c.camera.width > 0 && c.camera.height > 0, "camera/intrinsics", "a positive image size");
    robot.require(c.hull.max_x > c.hull.min_x && c.hull.max_y > c.hull.min_y, "vehicle/hull_footprint_m", "max above min");

    c.loop_hz               = pick.number("loop_hz");
    c.joint_state_timeout_s = pick.number("joint_state_timeout_s");
    c.camera_fixed_in_world = pick.flag("camera_fixed_in_world");
    c.stream_timeout_s      = pick.number("task/stream_timeout_s");
    c.spot_search_s         = pick.number("task/spot_search_s");
    c.repark_search_s       = pick.number("task/repark_search_s");
    c.retarget_attempts     = pick.whole("task/retarget_attempts");
    c.park_attempts         = pick.whole("task/park_attempts");
    c.jaw_settle_tolerance  = pick.number("jaw/settle_tolerance_m");
    c.jaw_settle_time_s     = pick.number("jaw/settle_time_s");
    c.jaw_grabbed_margin    = pick.number("jaw/grabbed_margin_m");
    c.jaw_timeout_s         = pick.number("jaw/timeout_s");
    c.catch_reach           = pick.number("jaw/catch_reach_m");
    c.catch_off_centre      = pick.number("jaw/catch_off_centre_m");
    c.close_within          = pick.number("jaw/close_within_m");
    c.close_hold_s          = pick.number("jaw/close_hold_s");
    pick.require(c.loop_hz > 0.0 && c.joint_state_timeout_s > 0.0, "loop_hz", "positive, with a positive joint_state_timeout_s");
    pick.require(c.retarget_attempts > 0 && c.park_attempts > 0, "task", "positive retarget_attempts and park_attempts");
    pick.require(c.jaw_timeout_s > c.jaw_settle_time_s, "jaw/timeout_s", "longer than jaw/settle_time_s");

    CloudSettings &s        = c.cloud;
    s.outlier_filter        = pick.flag("cloud/outlier_filter");
    s.handle_carving        = pick.flag("cloud/handle_carving");
    s.corridor_carving      = pick.flag("cloud/corridor_carving");
    s.candidate_averaging   = pick.flag("cloud/candidate_averaging");
    s.obstacle_averaging    = pick.flag("cloud/obstacle_averaging");
    const int frames        = pick.whole("cloud/frames");
    c.frames_needed         = s.candidate_averaging || s.obstacle_averaging ? frames : 1;
    c.frame_timeout_s       = pick.number("cloud/frame_timeout_s");
    s.voxel                 = pick.number("cloud/voxel_m");
    s.crop_radius           = pick.number("cloud/crop_margin_m");  // the arm's reach is added once the arm is built
    c.approach_column       = pick.whole("cloud/approach_column");
    c.bar_column            = pick.whole("cloud/bar_column");
    s.depth_tolerance       = pick.number("cloud/flying_pixel/depth_tolerance_m");
    s.min_agreeing_neighbours = pick.whole("cloud/flying_pixel/min_agreeing_neighbours");
    s.slope_window_px       = pick.whole("cloud/flying_pixel/slope_window_px");
    s.min_points_per_voxel  = pick.whole("cloud/occupancy/min_points_per_voxel");
    s.free_space_tolerance  = pick.number("cloud/occupancy/free_space_tolerance_m");
    const double tilt       = pick.number("cloud/occupancy/max_surface_tilt_deg");
    s.max_ray_stretch       = 1.0 / std::max(std::cos(rad(tilt)), 1e-3);
    s.handle_radius         = pick.number("cloud/handle_radius_m");
    s.bar_gap               = pick.number("cloud/bar_gap_m");
    s.corridor_length       = pick.number("cloud/corridor/length_m");
    s.corridor_radius       = pick.number("cloud/corridor/radius_m");
    s.consensus_min_frames  = pick.whole("cloud/consensus/min_frames");
    s.match                 = pick.number("cloud/consensus/match_m");
    s.max_axis              = rad(pick.number("cloud/consensus/max_axis_deg"));
    s.outlier               = pick.number("cloud/consensus/outlier_m");
    s.max_spread            = pick.number("cloud/consensus/max_spread_m");
    s.duplicate             = pick.number("cloud/consensus/duplicate_m");
    s.vote_min_frames       = pick.whole("cloud/vote/min_frames");
    s.vote_radius           = pick.whole("cloud/vote/radius_voxels");
    pick.require(frames > 0 && s.consensus_min_frames <= frames && s.vote_min_frames <= frames && s.vote_min_frames > 0,
                 "cloud/frames", "positive and at least consensus/min_frames and vote/min_frames");
    pick.require(s.voxel > 0.0 && c.frame_timeout_s > 0.0, "cloud/voxel_m", "positive, with a positive frame_timeout_s");
    pick.require(s.slope_window_px >= 3 && s.slope_window_px % 2 == 1, "cloud/flying_pixel/slope_window_px", "odd and at least 3");
    pick.require(tilt >= 0.0 && tilt < 90.0, "cloud/occupancy/max_surface_tilt_deg", "between 0 and 90");
    pick.require(c.approach_column >= 0 && c.approach_column <= 2 && c.bar_column >= 0 && c.bar_column <= 2
                         && c.bar_column != c.approach_column,
                 "cloud/bar_column", "0, 1 or 2 and not the approach column");

    ParkSettings &p          = c.park;
    p.box_xy                 = pick.number("park/box_xy_m");
    p.box_z                  = pick.number("park/box_z_m");
    p.box_yaw                = rad(pick.number("park/box_yaw_deg"));
    p.coarse_step            = pick.number("park/coarse_step_m");
    p.coarse_yaw             = rad(pick.number("park/coarse_step_deg"));
    p.fine_step              = pick.number("park/fine_step_m");
    p.fine_yaw               = rad(pick.number("park/fine_step_deg"));
    p.refine_count           = pick.whole("park/refine_count");
    const std::vector<double> standoff = pick.numbers("park/standoff_m", 2);
    p.standoff_min           = standoff[0];
    p.standoff_max           = standoff[1];
    p.screen_count           = pick.whole("park/screen_count");
    p.screen_budget_s        = pick.number("park/screen_budget_s");
    p.screen_blade_stride    = pick.whole("park/screen_blade_stride");
    p.exact_count            = pick.whole("park/exact_count");
    p.transit_samples        = pick.whole("park/transit_samples");
    p.reach_cell             = pick.number("park/reach_cell_m");
    p.grasp_point_from_mount = pick.number("collision/grasp_point_from_mount_m");
    p.link_radius            = pick.number("collision/link_radius_m");
    p.link_step              = pick.number("collision/link_step_m");
    p.blade_step             = pick.number("collision/blade_step_m");
    p.swing_band             = rad(pick.number("park/swing_band_deg"));
    c.speed                  = pick.number("park/speed_m_s");
    c.yaw_speed              = rad(pick.number("park/yaw_speed_deg_s"));
    pick.require(p.fine_step > 0.0 && p.fine_step <= p.coarse_step && p.fine_yaw > 0.0 && p.fine_yaw <= p.coarse_yaw,
                 "park/fine_step_m", "positive and no larger than the coarse steps");
    pick.require(p.standoff_min >= 0.0 && p.standoff_min < p.standoff_max, "park/standoff_m", "increasing and not negative");
    pick.require(p.screen_count > 0 && p.screen_blade_stride > 0 && p.exact_count > 0 && p.transit_samples > 0 && p.refine_count > 0,
                 "park", "positive counts");
    pick.require(p.reach_cell > 0.0 && c.speed > 0.0 && c.yaw_speed > 0.0, "park", "positive reach_cell_m, speed_m_s and yaw_speed_deg_s");
    pick.require(p.swing_band >= 0.0, "park/swing_band_deg", "not negative");

    PlanSettings &plan          = c.plan;
    plan.grasp_point_from_mount = p.grasp_point_from_mount;
    plan.budget_s               = pick.number("plan/budget_s");
    plan.goal_budget_s          = pick.number("plan/goal_budget_s");
    plan.range                  = rad(pick.number("plan/range_deg"));
    plan.edge_step              = rad(pick.number("plan/edge_step_deg"));
    c.start_tolerance           = rad(pick.number("plan/start_tolerance_deg"));
    pick.require(plan.goal_budget_s > 0.0 && plan.budget_s >= plan.goal_budget_s && plan.edge_step > 0.0 && plan.range > 0.0,
                 "plan", "positive budgets, range and edge step, with budget_s at least goal_budget_s");

    FollowSettings &f          = c.follow;
    f.max_step                 = rad(pick.number("follow/max_step_deg"));
    f.arrival_tolerance        = rad(pick.number("follow/arrival_tolerance_deg"));
    f.arrival_timeout_s        = pick.number("follow/arrival_timeout_s");
    f.blocked_min_step         = rad(pick.number("follow/blocked_min_step_deg"));
    f.blocked_follow_fraction  = pick.number("follow/blocked_follow_fraction");
    f.blocked_strikes          = pick.whole("follow/blocked_strikes");
    pick.require(f.max_step > 0.0 && f.arrival_tolerance > 0.0 && f.blocked_strikes > 0, "follow", "positive step, tolerance and strikes");
    plan.joint_speed = f.max_step * c.loop_hz;

    c.track_enabled   = pick.flag("track/enabled");
    c.track_steer     = pick.flag("track/steer");
    c.track_retarget    = pick.number("track/retarget_m");
    c.track_max_shift = pick.number("track/max_shift_m");
    c.track_replan_s  = pick.number("track/replan_budget_s");
    pick.require(c.track_replan_s > 0.0, "track/replan_budget_s", "positive");
    pick.require(c.track_retarget > 0.0 && c.track_max_shift > c.track_retarget, "track/max_shift_m", "above a positive retarget_m");
    TrackSettings &t = c.track;
    t.depth_gate     = pick.number("track/depth_gate_m");
    t.roi_radius     = pick.number("track/roi_radius_m");
    t.min_points     = pick.whole("track/min_points");
    t.max_points     = pick.whole("track/max_points");
    t.ransac_px      = pick.number("track/ransac_px");
    t.min_face       = pick.number("track/min_face_visible");
    t.ema_alpha      = pick.number("track/ema_alpha");
    pick.require(t.depth_gate > 0.0 && t.roi_radius > 0.0 && t.ransac_px > 0.0 && t.min_points >= 3 && t.max_points > t.min_points,
                 "track", "positive gate, radius and threshold, with max_points above a min_points of at least 3");
    pick.require(t.min_face >= 0.0 && t.min_face < 1.0, "track/min_face_visible", "at least 0 and below 1");
    pick.require(t.ema_alpha > 0.0 && t.ema_alpha <= 1.0, "track/ema_alpha", "above 0 and at most 1");
}

// Runs the pick after one start: survey, park, survey again from there, plan, follow, close the jaw.
class Pick {
public:
    Pick(const PickConfig &c, const Arm &arm)
            : c_(c),
              arm_(arm),
              park_(arm, c.hull, c.park, c.arm_to_vehicle.inverse(), c.camera_to_vehicle.translation()),
              follower_(c.follow),
              cloud_sub_(nh_, "cloud", 1),
              poses_sub_(nh_, "grasp_poses", 1),
              sync_(cloud_sub_, poses_sub_, 5),
              track_sync_(5),
              tracker_(c.track, c.camera) {
        ros::NodeHandle pnh("~");
        sync_.registerCallback(&Pick::onFrame, this);
        cloud_sub_.registerCallback(&Pick::onCloudSeen, this);
        
        if (c_.track_enabled) {
            image_sub_.subscribe(nh_, "image", 1);
            track_sync_.connectInput(cloud_sub_, image_sub_);
            track_sync_.registerCallback(&Pick::onTrackFrame, this);
            view_sub_   = nh_.subscribe("tracker/view", 1, &Pick::onView, this);
            debug_pub_  = nh_.advertise<sensor_msgs::Image>("tracker/debug_image", 1);
            info_pub_   = nh_.advertise<sensor_msgs::CameraInfo>("tracker/camera_info", 1);  // Foxglove pairs it with the image
            shift_pub_  = nh_.advertise<visualization_msgs::MarkerArray>("tracker/target_shift", 1, true);
        }
        joints_sub_  = nh_.subscribe("joint_states", 1, &Pick::onJoints, this);
        targets_pub_ = nh_.advertise<sensor_msgs::JointState>("driver/joint_targets", 1);
        close_jaw_   = nh_.serviceClient<std_srvs::Trigger>("driver/close_jaw");
        standby_     = nh_.serviceClient<std_srvs::Trigger>("driver/standby");
        home_        = nh_.serviceClient<std_srvs::Trigger>("driver/home");
        state_pub_   = pnh.advertise<PickState>("state", 1, true);
        park_map_pub_ = pnh.advertise<sensor_msgs::PointCloud2>("park_map", 1, true);
        map_pub_     = pnh.advertise<sensor_msgs::PointCloud2>("pick_map", 1, true);
        grasp_pub_   = pnh.advertise<visualization_msgs::MarkerArray>("grasps", 1, true);
        path_pub_    = pnh.advertise<visualization_msgs::Marker>("path", 1, true);
        handle_pub_  = pnh.advertise<visualization_msgs::Marker>("handle_check", 1, true);
        body_pub_    = pnh.advertise<visualization_msgs::MarkerArray>("body", 1);
        hull_pub_    = pnh.advertise<visualization_msgs::Marker>("hull", 1, true);
        hull_pub_.publish(hullMarker(c_.hull, c_.arm_frame));
        services_ = {pnh.advertiseService("start", &Pick::onStart, this), pnh.advertiseService("stop", &Pick::onStop, this),
                     pnh.advertiseService("reset", &Pick::onReset, this)};
        tf_timer_ = nh_.createTimer(ros::Duration(1.0 / c_.loop_hz), [this](const ros::TimerEvent &) { broadcastVehicle(); });
        // A replayed camera stays put while the vehicle moves, so the URDF leaves it off the vehicle and it is pinned here.
        if (c_.camera_fixed_in_world) {
            geometry_msgs::TransformStamped t = tf2::eigenToTransform(c_.camera_to_vehicle);
            t.header.stamp    = ros::Time::now();
            t.header.frame_id = c_.world_frame;
            t.child_frame_id  = c_.camera_frame;
            static_broadcaster_.sendTransform(t);
        }
        publishState("waiting for start");
    }

    void tick() {
        if (stop_requested_.exchange(false)) {
            if (isWorking(state_)) {
                apply(Event::STOP, "stopped on request");
            } else if (!releaseArm()) {  // after a pick the arm still holds its last posture
                ROS_ERROR("[pick] stop: standby failed, the arm may still be powered");
            }
        }
        if (start_requested_.exchange(false)) {
            if (isWorking(state_)) {
                ROS_WARN("[pick] start ignored: already in %s", stateName(state_));
            } else {
                park_attempt_ = look_attempt_ = 0;
                grabbed_ = on_handle_ = false;
                surveying_since_ = reparking_since_ = ros::Time();
                run_started_                        = ros::Time::now();
                map_pub_.publish(mapCloud(ObstacleMap(), Eigen::Isometry3d::Identity(), c_.world_frame));
                clearTrails();
                ros::param::get("~track/steer", c_.track_steer);  // scripts/soak.sh sets it
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

private:
    // --- sensors ---

    void onCloudSeen(const sensor_msgs::PointCloud2::ConstPtr &) {
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        frame_seen_ = ros::Time::now();
    }

    // while the arm moves blind, follow the target in the image; with track/steer, steer() moves the goal with it.
    void onTrackFrame(const sensor_msgs::PointCloud2::ConstPtr &cloud, const sensor_msgs::Image::ConstPtr &image) {
        std::lock_guard<std::mutex> lock(track_mutex_);
        const bool viewer = debug_pub_.getNumSubscribers() > 0;  // Foxglove subscribes to what its panels show
        if (!tracker_.active() && !viewer) {
            return;
        }
        cv::Mat bgr;
        if (cloud->header.frame_id != c_.camera_frame || !toBgr(*image, bgr)) {
            ROS_WARN_THROTTLE(5.0, "[pick] frame dropped: expected a bgr8, rgb8 or mono8 image with a cloud in %s", c_.camera_frame.c_str());
            return;
        }
        if (!tracker_.active()) {  // nothing is tracked between blind motions: say so rather than show a bare camera feed
            const std::string idle = "idle";
            cv::putText(bgr, idle, cv::Point(12, 26), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
            cv::putText(bgr, idle, cv::Point(12, 26), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            debug_pub_.publish(toImage(bgr, image->header, "bgr8"));
            return;
        }
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        const bool was_covered = track_.covered;
        track_                 = tracker_.update(gray, cloudPoints(*cloud));
        if (track_.covered && !was_covered) {
            ROS_WARN("[pick] only %.0f%% of the mine face is visible: tracking stopped, the arm finishes open loop", 100 * track_.face);
        }

        // From where the arm is aiming to where tracker puts the target: the correction a closed loop would make.
        trail_.push_back(track_.target);
        visualization_msgs::Marker shift = marker(c_.camera_frame, "shift", visualization_msgs::Marker::ARROW, 1.0f, 0.3f, 0.9f, 1.0f, 0.003);
        shift.scale.y = 0.007, shift.scale.z = 0.01;  // shaft and head width, head length
        shift.points  = {toPoint(track_.target - track_.offset), toPoint(track_.target)};
        visualization_msgs::Marker trail = marker(c_.camera_frame, "trail", visualization_msgs::Marker::LINE_STRIP, 0.3f, 0.9f, 1.0f, 0.8f, 0.0015);
        for (const Eigen::Vector3d &p : trail_) {
            trail.points.push_back(toPoint(p));
        }
        visualization_msgs::Marker ball = marker(c_.camera_frame, "target", visualization_msgs::Marker::SPHERE, 1.0f, 0.55f, 0.0f, 1.0f, 0.012);
        ball.pose.position = toPoint(track_.target);
        visualization_msgs::MarkerArray markers;
        markers.markers = {shift, trail, ball};
        shift_pub_.publish(markers);
        if (viewer) {
            // Where the arm aims: the planned target moved by the shift steering last took, in camera_link.
            const Eigen::Vector3d aim = track_.target - track_.offset + scene_camera_to_arm_.linear().transpose() * steer_aim_;
            debug_pub_.publish(toImage(drawTrack(bgr, track_, view_.load(), aim, tracker_.project(aim), c_.track_steer ? steer_note_ : ""),
                                       image->header, "bgr8"));
            sensor_msgs::CameraInfo info;
            info.header = image->header;
            info.width = c_.camera.width, info.height = c_.camera.height;
            info.K = {c_.camera.fx, 0.0, c_.camera.cx, 0.0, c_.camera.fy, c_.camera.cy, 0.0, 0.0, 1.0};
            info.R = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
            info.P = {c_.camera.fx, 0.0, c_.camera.cx, 0.0, 0.0, c_.camera.fy, c_.camera.cy, 0.0, 0.0, 0.0, 1.0, 0.0};
            info_pub_.publish(info);
        }
    }

    void onView(const std_msgs::String::ConstPtr &msg) {
        if (msg->data == "ransac" || msg->data == "klt" || msg->data == "mask") {
            view_ = msg->data == "ransac" ? TrackView::RANSAC : msg->data == "klt" ? TrackView::KLT : TrackView::MASK;
        } else {
            ROS_WARN("[pick] view %s is none of ransac, klt, mask", msg->data.c_str());
        }
    }

    void onFrame(const sensor_msgs::PointCloud2::ConstPtr &cloud, const geometry_msgs::PoseArray::ConstPtr &poses) {
        if (cloud->header.frame_id != c_.camera_frame || poses->header.frame_id != c_.camera_frame
            || static_cast<long>(cloud->width) * cloud->height != static_cast<long>(c_.camera.width) * c_.camera.height) {
            ROS_WARN_THROTTLE(5.0, "[pick] frame dropped: expected an organised %dx%d cloud in %s", c_.camera.width, c_.camera.height,
                              c_.camera_frame.c_str());
            return;
        }
        if (poses->poses.empty()) {
            std::lock_guard<std::mutex> lock(sensor_mutex_);
            ++poseless_;
            return;
        }
        Frame frame;
        frame.points = cloudPoints(*cloud);
        for (const geometry_msgs::Pose &p : poses->poses) {
            const Eigen::Matrix3d r = Eigen::Quaterniond(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z).normalized().toRotationMatrix();
            frame.poses.push_back({Eigen::Vector3d(p.position.x, p.position.y, p.position.z), r.col(c_.bar_column), r.col(c_.approach_column)});
        }
        const Eigen::Isometry3d vehicle_to_world = toIsometry(vehicle());
        const Eigen::Isometry3d camera_to_world  = c_.camera_fixed_in_world ? c_.camera_to_vehicle : vehicle_to_world * c_.camera_to_vehicle;
        frame.camera_to_arm = (vehicle_to_world * c_.arm_to_vehicle).inverse() * camera_to_world;

        std::lock_guard<std::mutex> lock(sensor_mutex_);
        frames_.push_back(std::move(frame));
        while (frames_.size() > static_cast<size_t>(c_.frames_needed)) {
            frames_.pop_front();
        }
    }

    void onJoints(const sensor_msgs::JointState::ConstPtr &msg) {
        const auto at = [&](const std::string &name) {
            const size_t i = std::find(msg->name.begin(), msg->name.end(), name) - msg->name.begin();
            return i < msg->name.size() && i < msg->position.size() ? msg->position[i] : std::numeric_limits<double>::quiet_NaN();
        };
        Joints q;
        for (int j = 0; j < JOINT_COUNT; ++j) {
            q[j] = at(c_.joint_names[j]);
        }
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        if (std::all_of(q.begin(), q.end(), [](double v) { return std::isfinite(v); })) {
            joints_       = q;
            joints_stamp_ = msg->header.stamp;
        }
        if (std::isfinite(at(c_.jaw_name))) {
            jaw_       = at(c_.jaw_name);
            jaw_stamp_ = msg->header.stamp;
        }
    }

    // Reported radians, if newer than the timeout.
    bool freshJoints(Joints &q, ros::Time &stamp) {
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        if (joints_stamp_.isZero() || (ros::Time::now() - joints_stamp_).toSec() > c_.joint_state_timeout_s) {
            return false;
        }
        q     = joints_;
        stamp = joints_stamp_;
        return true;
    }

    VehiclePose vehicle() {
        std::lock_guard<std::mutex> lock(vehicle_mutex_);
        return vehicle_;
    }

    void setVehicle(const VehiclePose &v) {
        std::lock_guard<std::mutex> lock(vehicle_mutex_);
        vehicle_ = v;
    }

    Eigen::Isometry3d armToWorld() { return toIsometry(vehicle()) * c_.arm_to_vehicle; }

    // --- services ---

    bool onStart(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res) {
        start_requested_ = true;
        res.success      = true;
        res.message      = "start requested";
        return true;
    }

    bool onStop(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res) {
        stop_requested_ = true;
        res.success     = true;
        res.message     = "stop requested, the arm will be released";
        return true;
    }

    // Puts the simulated vehicle back at the origin and the arm home, so a soak repeats a pick from the same place.
    bool onReset(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res) {
        res.success = !working_;
        res.message = res.success ? "vehicle back at the origin" : "a pick is running, stop it first";
        if (res.success) {
            setVehicle(VehiclePose());
            std_srvs::Trigger call;
            res.message += home_.call(call) ? ", arm " + call.response.message : ", arm not homed: the driver did not answer";
        }
        return true;
    }

    // --- state machine ---

    void apply(Event event, const std::string &message) {
        const State next = nextState(state_, event);
        if (next == state_) {
            return;
        }
        if (next == State::COLLECT) {
            spot_phase_ = state_ == State::STREAM || state_ == State::RESURVEY || state_ == State::REPARK;
            if (!spot_phase_) {
                park_map_pub_.publish(mapCloud(ObstacleMap(), Eigen::Isometry3d::Identity(), c_.world_frame));
            }
        }
        enter(next, message);
    }

    void enter(State next, std::string message) {
        const State  left      = state_;
        const double spent     = (ros::Time::now() - entered_).toSec();
        state_                 = next;
        working_               = isWorking(next);
        entered_               = ros::Time::now();
        if (next != State::GOTOGRASP && next != State::CLOSEJAW) {
            std::lock_guard<std::mutex> lock(track_mutex_);
            tracker_.stop();
        }
        char         line[200];
        switch (next) {
        case State::GOTOGRASP: {
            {
                std::lock_guard<std::mutex> lock(sensor_mutex_);
                frames_.clear();  // what arrives during the motion is what handleCheck() judges the end against
            }
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
            surveying_since_ = ros::Time();
            break;
        case State::GOTOSPOT: {
            ++park_attempt_;
            reparking_since_          = ros::Time();
            drive_from_               = vehicle();
            const double distance     = std::hypot(std::hypot(drive_to_.x - drive_from_.x, drive_to_.y - drive_from_.y), drive_to_.z - drive_from_.z);
            const double seconds      = std::max(distance / c_.speed, std::fabs(drive_to_.yaw - drive_from_.yaw) / c_.yaw_speed);
            drive_steps_              = std::max(1, static_cast<int>(std::ceil(seconds * c_.loop_hz)));
            drive_step_               = 0;
            break;
        }
        case State::CLOSEJAW: {
            std_srvs::Trigger call;
            if (!close_jaw_.call(call) || !call.response.success) {
                enter(State::FAIL, "could not close the jaw: " + call.response.message);
                return;
            }
            jaw_closed_at_   = ros::Time::now();
            jaw_last_seen_   = ros::Time();
            jaw_still_since_ = ros::Time();
            break;
        }
        case State::RESURVEY:
            if (surveying_since_.isZero()) {
                surveying_since_ = ros::Time::now();
            }
            std::snprintf(line, sizeof(line), "; nothing to park for after %.0f of %.0f s", (ros::Time::now() - surveying_since_).toSec(),
                          c_.spot_search_s);
            message += line;
            break;
        case State::RETARGET:
            ++look_attempt_;
            std::snprintf(line, sizeof(line), "; look %u of %d from this spot gave no grasp", look_attempt_, c_.retarget_attempts);
            message += line;
            break;
        case State::REPARK:
            look_attempt_ = 0;
            if (reparking_since_.isZero()) {
                reparking_since_ = ros::Time::now();
            }
            std::snprintf(line, sizeof(line), "; %u of %d parking spots used, surveying for somewhere else", park_attempt_, c_.park_attempts);
            message += line;
            break;
        case State::ESTOP:
            if (!releaseArm()) {
                message += "; standby failed, the arm may still be powered";
            }
            break;
        default:
            break;
        }
        publishState(message, left, spent);
    }

    Event step(std::string &message) {
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
            return retry(ros::Time(), 0.0, look_attempt_ >= static_cast<uint32_t>(c_.retarget_attempts), Event::LOOK_AGAIN,
                         Event::OUT_OF_LOOKS, message);
        case State::REPARK:
            return retry(reparking_since_, c_.repark_search_s, park_attempt_ >= static_cast<uint32_t>(c_.park_attempts),
                         Event::PARK_AGAIN, Event::OUT_OF_PARKS, message);
        default: return Event::NONE;
        }
    }

    // Looks again at once unless a count or a clock has run out.
    Event retry(const ros::Time &since, double budget_s, bool out_of_tries, Event again, Event done, std::string &message) {
        if (out_of_tries || (!since.isZero() && (ros::Time::now() - since).toSec() >= budget_s)) {
            message = out_of_tries ? "out of tries" : "out of time";
            return done;
        }
        message = "looking again";
        return again;
    }

    // A camera already producing when start was pressed counts.
    Event checkStream(std::string &message) {
        ros::Time seen;
        {
            std::lock_guard<std::mutex> lock(sensor_mutex_);
            seen = frame_seen_;
        }
        if (!seen.isZero() && (ros::Time::now() - seen).toSec() < c_.stream_timeout_s) {
            message = "the camera is producing";
            return Event::STREAMING;
        }
        if ((ros::Time::now() - entered_).toSec() >= c_.stream_timeout_s) {
            message = "no camera frame in " + std::to_string(c_.stream_timeout_s) + " s";
            return Event::NO_STREAM;
        }
        return Event::NONE;
    }

    // Silence is a fault; frames that carry no grasp poses are a survey with nothing in view.
    Event checkCollect(std::string &message) {
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
        if ((ros::Time::now() - entered_).toSec() < c_.frame_timeout_s) {
            return Event::NONE;
        }
        if (have == 0 && poseless > 0) {
            message = "no handle in view: " + std::to_string(poseless) + " frames carried no grasp poses";
            return spot_phase_ ? Event::NO_CANDIDATES_SPOT : Event::NO_CANDIDATES_GRASP;
        }
        message = "only " + std::to_string(have) + " of " + std::to_string(c_.frames_needed) + " frames in "
                  + std::to_string(c_.frame_timeout_s) + " s";
        return Event::FAILURE;
    }

    Event process(std::string &message) {
        std::vector<Frame> frames;
        {
            std::lock_guard<std::mutex> lock(sensor_mutex_);
            frames.assign(frames_.begin(), frames_.end());
        }
        const auto started = ros::WallTime::now();
        scene_             = processFrames(frames, c_.camera, c_.cloud);
        if (!frames.empty()) {
            scene_camera_to_arm_ = frames.back().camera_to_arm;
        }
        scene_to_world_    = armToWorld();
        char took[32];
        std::snprintf(took, sizeof(took), " (%.2f s)", (ros::WallTime::now() - started).toSec());
        message = scene_.summary + took;
        (spot_phase_ ? park_map_pub_ : map_pub_).publish(mapCloud(scene_.map, scene_to_world_, c_.world_frame));
        publishGrasps(-1);
        if (scene_.candidates.empty()) {
            return spot_phase_ ? Event::NO_CANDIDATES_SPOT : Event::NO_CANDIDATES_GRASP;
        }
        return spot_phase_ ? Event::CANDIDATES_SPOT : Event::CANDIDATES_GRASP;
    }

    Event pickSpot(std::string &message) {
        Joints    reported = c_.home;
        ros::Time stamp;
        freshJoints(reported, stamp);
        ParkChoice choice;
        try {
            choice = park_.choose(scene_.candidates, scene_.map, arm_.toModel(reported), [this] { return stop_requested_.load(); });
        } catch (const std::invalid_argument &e) {
            message = std::string("the obstacle map is not usable: ") + e.what();
            return Event::FAILURE;
        }
        message        = choice.summary;
        if (choice.decision == ParkDecision::MOVE) {
            drive_to_ = compose(vehicle(), choice.move);
            return Event::SPOT_CHOSEN;
        }
        return choice.decision == ParkDecision::STAY ? Event::SPOT_UNCHANGED : Event::NO_SPOT;
    }

    Event drive(std::string &message) {
        const double f = static_cast<double>(++drive_step_) / drive_steps_;
        setVehicle({drive_from_.x + (drive_to_.x - drive_from_.x) * f, drive_from_.y + (drive_to_.y - drive_from_.y) * f,
                    drive_from_.z + (drive_to_.z - drive_from_.z) * f, drive_from_.yaw + (drive_to_.yaw - drive_from_.yaw) * f});
        if (drive_step_ < drive_steps_) {
            return Event::NONE;
        }
        message = "arrived";
        return Event::ARRIVED;
    }

    Event pickGrasp(std::string &message) {
        Joints    reported;
        ros::Time stamp;
        if (!freshJoints(reported, stamp)) {
            message = "no fresh joint_states to plan from";
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
        near_since_ = ros::Time();
        retargets_ = 0;
        steer_tries_.clear();
        message  = plan_.summary;
        publishGrasps(plan_.ok ? static_cast<int>(plan_.candidate) : -1);
        publishPath();
        if (!plan_.ok) {
            return Event::NO_PLAN;
        }
        follower_.load(plan_.path);
        last_stamp_ = ros::Time();
        return Event::PLAN_FOUND;
    }

    // Until the face is covered (track/min_face_visible), the goal and the obstacles shift with the tracked target (the scene
    // taken as rigid) and the arm is sent to no posture that hits them there. Once the tracked target is track/retarget_m from
    // where the arm is heading, or the next posture is blocked, it goes straight to the shifted goal if clear, else plans
    // around for track/replan_budget_s. Returns true when the arm must hold: its next posture is blocked and nothing clear was
    // found. Shifts past track/max_shift_m are taken as tracking failures.
    bool steer() {
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
        Joints    now;
        ros::Time stamp;
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
        attempted_ = offset;
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
        last_stamp_ = ros::Time();
        ++retargets_;
        publishPath();
        char line[120];
        std::snprintf(line, sizeof(line), "%s%s to the target shifted %.0f mm", blocked ? "avoided the obstacle, " : "", how, 1000 * offset.norm());
        noteSteer(line);
        return false;
    }

    // What steering last did and the shift it aims at, for the debug image.
    void noteSteer(const std::string &note) {
        std::lock_guard<std::mutex> lock(track_mutex_);
        steer_note_ = note;
        steer_aim_  = aimed_;
    }

    Event follow(std::string &message) {
        const bool hold = c_.track_enabled && c_.track_steer && steer();
        Joints    reported;
        ros::Time stamp;
        // Without readings a joint against something goes unnoticed, so the arm holds its last target instead of moving on blind.
        if (!freshJoints(reported, stamp)) {
            message = "joint_states went stale mid-motion, so contact cannot be judged; holding the last target";
            return Event::FAILURE;
        }
        if (stamp != last_stamp_) {
            last_stamp_ = stamp;
            const Joints          q     = arm_.toModel(reported);
            follower_.measure(q);
            const Eigen::Vector3d grasp = arm_.points(q).mount + arm_.axes(q).approach * c_.plan.grasp_point_from_mount;
            const double          off   = (grasp - (scene_.candidates[plan_.candidate].point + aimed_)).norm();
            if (off > c_.close_within) {
                near_since_ = ros::Time();
            } else if (near_since_.isZero()) {
                near_since_ = stamp;
            } else if ((stamp - near_since_).toSec() >= c_.close_hold_s) {
                char line[120];
                std::snprintf(line, sizeof(line), "held %.1f mm from the aimed target for %.1f s", 1000 * off, c_.close_hold_s);
                message = line + motionReport(q);
                return Event::REACHED;
            }
        }
        Joints          target;
        bool            send  = false;
        const Following state = hold ? Following::SENDING : follower_.tick(ros::Time::now().toSec(), target, send);
        if (send) {
            sensor_msgs::JointState msg;
            msg.header.stamp = ros::Time::now();
            const Joints out = arm_.toReported(target);
            msg.name.assign(c_.joint_names.begin(), c_.joint_names.end());
            msg.position.assign(out.begin(), out.end());
            targets_pub_.publish(msg);
        }
        switch (state) {
        case Following::REACHED:
            message = "reached the end of the path" + motionReport(arm_.toModel(reported));
            return Event::REACHED;
        case Following::STALLED:
            releaseArm();
            message = "did not arrive within the arrival timeout, arm released";
            return Event::STALLED;
        case Following::BLOCKED:
            message = c_.joint_names[follower_.blockedJoint()] + " stopped following, so the arm hit something" + motionReport(arm_.toModel(reported));
            return Event::COLLIDED;
        default:
            return Event::NONE;
        }
    }

    // Where the jaw is against the handle, and what the tracker and steering did on the way there.
    std::string motionReport(const Joints &q) {
        std::string out = handleCheck(q);
        if (!c_.track_enabled) {
            return out;
        }
        std::lock_guard<std::mutex> lock(track_mutex_);
        char                        line[120];
        std::snprintf(line, sizeof(line), "; the tracker saw the target move %.1f mm (%s)", 1000 * track_.offset.norm(),
                      track_.covered ? "face covered, open loop" : track_.ok ? "tracking" : "not tracking");
        out += line;
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
    Event checkJaw(std::string &message) {
        double    jaw = 0.0;
        ros::Time stamp;
        {
            std::lock_guard<std::mutex> lock(sensor_mutex_);
            jaw   = jaw_;
            stamp = jaw_stamp_;
        }
        char line[160];
        if (stamp > jaw_closed_at_ && stamp != jaw_last_seen_) {
            jaw_last_seen_ = stamp;
            if (jaw_still_since_.isZero() || std::fabs(jaw - jaw_still_) > c_.jaw_settle_tolerance) {
                jaw_still_       = jaw;
                jaw_still_since_ = stamp;
            } else if ((stamp - jaw_still_since_).toSec() >= c_.jaw_settle_time_s) {
                grabbed_ = jaw > c_.jaw_closed + c_.jaw_grabbed_margin;
                std::snprintf(line, sizeof(line), "%s: the jaw settled at %.1f mm", grabbed_ ? "holding something" : "closed on nothing",
                              jaw * 1000.0);
                message = line;
                return Event::JAW_SETTLED;
            }
        }
        if ((ros::Time::now() - entered_).toSec() >= c_.jaw_timeout_s) {
            grabbed_ = false;
            message  = "the jaw did not settle, so there is no grip verdict";
            return Event::JAW_SETTLED;
        }
        return Event::NONE;
    }

    // Pass or fail for the blind motion: whether the jaw ended with the handle between its open blades, the handle drawn
    // from the newest grasp poses seen during the motion. A replayed bag's poses are unoccluded, so they show where the
    // handle really went; a live camera's are not trustworthy with the arm in view.
    std::string handleCheck(const Joints &q) {
        on_handle_ = false;
        std::vector<GraspPose> handle;
        {
            std::lock_guard<std::mutex> lock(sensor_mutex_);
            if (frames_.empty()) {
                return "; no grasp poses during the motion to check the jaw against";
            }
            const Eigen::Isometry3d &t = frames_.back().camera_to_arm;
            for (const GraspPose &g : frames_.back().poses) {
                handle.push_back({t * g.point, t.linear() * g.bar, t.linear() * g.approach});
            }
        }
        const double          along   = c_.plan.grasp_point_from_mount;
        const Eigen::Vector3d grasp   = arm_.points(q).mount + arm_.axes(q).approach * along;
        const Eigen::Vector3d planned = scene_.candidates[plan_.candidate].point;
        const Eigen::Vector3d j       = arm_.inJaw(q, nearestOnHandle(handle, 0.5 * c_.cloud.bar_gap, grasp));
        on_handle_                    = Arm::caught(j, c_.catch_reach, c_.catch_off_centre);
        // The handle judged, as each pose's bar: green on it, red missed.
        visualization_msgs::Marker bars = marker(c_.arm_frame, "handle_check", visualization_msgs::Marker::LINE_LIST, on_handle_ ? 0.1f : 1.0f,
                                                 on_handle_ ? 1.0f : 0.1f, 0.1f, 1.0f, 0.002);
        for (const GraspPose &g : handle) {
            const Eigen::Vector3d half = 0.5 * c_.cloud.bar_gap * g.bar.normalized();
            bars.points.push_back(toPoint(g.point - half));
            bars.points.push_back(toPoint(g.point + half));
        }
        handle_pub_.publish(bars);
        char line[200];
        std::snprintf(line, sizeof(line),
                      "; %s: it sits %+.1f %+.1f %+.1f mm from the grasp point (approach, hinge, closing); the planned target is %.1f mm off it",
                      on_handle_ ? "ON THE HANDLE" : "MISSED THE HANDLE", 1000 * (j.x() - along), 1000 * j.y(), 1000 * j.z(),
                      1000 * (nearestOnHandle(handle, 0.5 * c_.cloud.bar_gap, planned) - planned).norm());
        return line;
    }

    bool releaseArm() {
        std_srvs::Trigger call;
        return standby_.call(call) && call.response.success;
    }

    // --- output ---

    void publishState(const std::string &message, State left = State::READY, double spent = -1.0) {
        PickState msg;
        msg.state        = stateName(state_);
        msg.message      = message;
        msg.park_attempt = park_attempt_;
        msg.grabbed      = grabbed_;
        msg.on_handle    = on_handle_;
        state_pub_.publish(msg);
        char timing[80] = "";
        if (spent >= 0.0) {
            std::snprintf(timing, sizeof(timing), "[%s %.1fs, run %.1fs] ", stateName(left), spent,
                          run_started_.isZero() ? 0.0 : (ros::Time::now() - run_started_).toSec());
        }
        if (state_ == State::FAIL || state_ == State::ESTOP) {
            ROS_ERROR("[pick] %s: %s%s", msg.state.c_str(), timing, message.c_str());
        } else if (state_ == State::RESURVEY || state_ == State::RETARGET || state_ == State::REPARK) {
            ROS_WARN("[pick] %s: %s%s", msg.state.c_str(), timing, message.c_str());
        } else {
            ROS_INFO("[pick] %s: %s%s", msg.state.c_str(), timing, message.c_str());
        }
    }

    // Vision's poses, the candidates, and the chosen one if any, in the world.
    void publishGrasps(int chosen) {
        visualization_msgs::MarkerArray out;
        visualization_msgs::Marker      poses = marker(c_.world_frame, "poses", visualization_msgs::Marker::SPHERE_LIST, 0.6f, 0.6f, 0.6f, 1.0f, 0.004);
        visualization_msgs::Marker      spots = marker(c_.world_frame, "candidates", visualization_msgs::Marker::SPHERE_LIST, 1.0f, 0.8f, 0.1f, 1.0f, 0.007);
        for (const GraspPose &g : scene_.poses) {
            poses.points.push_back(toPoint(scene_to_world_ * g.point));
        }
        for (const GraspPose &g : scene_.candidates) {
            spots.points.push_back(toPoint(scene_to_world_ * g.point));
        }
        visualization_msgs::Marker pick = marker(c_.world_frame, "chosen", visualization_msgs::Marker::LINE_LIST, 0.1f, 1.0f, 0.3f, 1.0f, 0.003);
        if (chosen >= 0) {
            const GraspPose &g = scene_.candidates[chosen];
            pick.points.push_back(toPoint(scene_to_world_ * Eigen::Vector3d(g.point - 0.03 * g.bar.normalized())));
            pick.points.push_back(toPoint(scene_to_world_ * Eigen::Vector3d(g.point + 0.03 * g.bar.normalized())));
        } else {
            pick.action = visualization_msgs::Marker::DELETE;
        }
        out.markers = {poses, spots, pick};
        grasp_pub_.publish(out);
    }

    void publishPath() {
        visualization_msgs::Marker line = marker(c_.arm_frame, "path", visualization_msgs::Marker::LINE_STRIP, 0.1f, 1.0f, 0.3f, 1.0f, 0.002);
        for (size_t k = 1; k < plan_.path.size(); ++k) {
            for (int n = 0; n <= 20; ++n) {
                Joints q;
                for (int j = 0; j < JOINT_COUNT; ++j) {
                    q[j] = plan_.path[k - 1][j] + (plan_.path[k][j] - plan_.path[k - 1][j]) * n / 20.0;
                }
                line.points.push_back(toPoint(arm_.points(q).tip));
            }
        }
        if (line.points.empty()) {
            line.action = visualization_msgs::Marker::DELETE;
        }
        path_pub_.publish(line);
    }

    // The last run's planned path and tracker trail are latched: wipe them so a new run starts on a clean view.
    void clearTrails() {
        visualization_msgs::Marker line = marker(c_.arm_frame, "path", visualization_msgs::Marker::LINE_STRIP, 0, 0, 0, 0, 0);
        line.action                      = visualization_msgs::Marker::DELETE;
        path_pub_.publish(line);
        line.ns = "handle_check";
        handle_pub_.publish(line);
        if (c_.track_enabled) {
            visualization_msgs::MarkerArray wipe;
            wipe.markers.push_back(marker(c_.camera_frame, "", visualization_msgs::Marker::ARROW, 0, 0, 0, 0, 0));
            wipe.markers[0].action = visualization_msgs::Marker::DELETEALL;
            shift_pub_.publish(wipe);
        }
    }

    void publishBody() {
        Joints    reported;
        ros::Time stamp;
        if (freshJoints(reported, stamp)) {
            Body body;
            arm_.body(arm_.toModel(reported), body, 4);
            body_pub_.publish(bodyMarkers(body, c_.park.link_radius, c_.arm_frame));
        }
    }

    void broadcastVehicle() {
        geometry_msgs::TransformStamped t = tf2::eigenToTransform(toIsometry(vehicle()));
        t.header.stamp                    = ros::Time::now();
        t.header.frame_id                 = c_.world_frame;
        t.child_frame_id                  = c_.vehicle_frame;
        broadcaster_.sendTransform(t);
    }

    PickConfig   c_;
    const Arm   &arm_;
    ParkSearch   park_;
    PathFollower follower_;

    ros::NodeHandle                                                              nh_;
    message_filters::Subscriber<sensor_msgs::PointCloud2>                        cloud_sub_;
    message_filters::Subscriber<geometry_msgs::PoseArray>                        poses_sub_;
    message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, geometry_msgs::PoseArray> sync_;
    message_filters::Subscriber<sensor_msgs::Image>                              image_sub_;
    message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, sensor_msgs::Image> track_sync_;
    ros::Subscriber                                                              view_sub_;
    ros::Publisher                                                               debug_pub_, info_pub_, shift_pub_;
    std::vector<Eigen::Vector3d>                                                 trail_;  // the tracked target over this blind motion
    std::mutex                                                                   track_mutex_;
    Tracker                                                                      tracker_;
    TrackResult                                                                  track_;
    std::atomic<TrackView>                                                       view_{TrackView::RANSAC};
    std::string                                                                  steer_note_;  // what steering last did
    Eigen::Vector3d                                                              steer_aim_ = Eigen::Vector3d::Zero();  // its shift, arm frame
    ros::Subscriber                   joints_sub_;
    ros::Publisher                    targets_pub_, state_pub_, park_map_pub_, map_pub_, grasp_pub_, path_pub_, handle_pub_, body_pub_, hull_pub_;
    ros::ServiceClient                close_jaw_, standby_, home_;
    std::vector<ros::ServiceServer>   services_;
    ros::Timer                        tf_timer_;
    tf2_ros::TransformBroadcaster     broadcaster_;
    tf2_ros::StaticTransformBroadcaster static_broadcaster_;

    std::mutex        sensor_mutex_;
    std::deque<Frame> frames_;
    size_t            poseless_ = 0;
    ros::Time         frame_seen_, joints_stamp_, jaw_stamp_;
    Joints            joints_{};
    double            jaw_ = 0.0;

    std::mutex  vehicle_mutex_;
    VehiclePose vehicle_;

    std::atomic<bool> start_requested_{false}, stop_requested_{false}, working_{false};
    State             state_ = State::READY;
    bool              spot_phase_ = true, grabbed_ = false, on_handle_ = false;
    uint32_t          park_attempt_ = 0, look_attempt_ = 0;
    long              ticks_ = 0;
    ros::Time         entered_ = ros::Time::now(), run_started_, surveying_since_, reparking_since_, last_stamp_;
    Scene             scene_;
    Eigen::Isometry3d scene_to_world_ = Eigen::Isometry3d::Identity();
    Plan              plan_;
    Eigen::Isometry3d scene_camera_to_arm_ = Eigen::Isometry3d::Identity();  // the newest frame of the look
    Eigen::Vector3d   attempted_           = Eigen::Vector3d::Zero();      // the last tracked shift tried, arm frame
    Eigen::Vector3d   aimed_               = Eigen::Vector3d::Zero();      // the shift the goal actually took
    int               retargets_             = 0;
    Eigen::Vector3d   replanned_           = Eigen::Vector3d::Zero();      // the tracked shift last planned around
    std::map<std::string, int> steer_tries_;                               // what steering tries came to this motion
    std::unique_ptr<ObstacleGrid> grid_;  // the grasp look's obstacles, inflated once; steering shifts it by a query offset
    VehiclePose       drive_from_, drive_to_;
    int               drive_steps_ = 1, drive_step_ = 0;
    ros::Time         jaw_closed_at_, jaw_last_seen_, jaw_still_since_, near_since_;
    double            jaw_still_ = 0.0;
};

}  // namespace

int main(int argc, char **argv) {
    ros::init(argc, argv, "pick");
    Params     robot("/robot", {"frames", "arm", "jaws", "camera", "vehicle"});
    Params     pick("/pick", {""});
    PickConfig config;
    loadPick(robot, pick, config);
    const int seed = pick.whole("plan/seed");

    std::string urdf;
    if (!ros::param::get("/robot_description", urdf)) {
        ROS_ERROR("[pick] /robot_description is not set");
        return 1;
    }
    std::unique_ptr<Arm> arm;
    try {
        arm.reset(new Arm(loadArm(robot, urdf, config), config.park.blade_step));
    } catch (const std::invalid_argument &e) {
        ROS_ERROR("[pick] the arm is not usable: %s", e.what());
        return 1;
    }
    config.cloud.crop_radius += arm->reach(arm->tipDistance());
    const std::string problems = robot.problems() + pick.problems();
    if (!problems.empty()) {
        ROS_ERROR("[pick] configuration is not usable:\n%s", problems.c_str());
        return 1;
    }

    ompl::RNG::setSeed(static_cast<std::uint_fast32_t>(seed));
    ompl::msg::setLogLevel(ompl::msg::LOG_WARN);
    Pick node(config, *arm);
    ROS_INFO("[pick] ready: crop radius %.2f m, %d frames per look", config.cloud.crop_radius, config.frames_needed);
    ros::AsyncSpinner spinner(2);
    spinner.start();
    for (ros::Rate rate(config.loop_hz); ros::ok(); rate.sleep()) {
        node.tick();
    }
    return 0;
}
