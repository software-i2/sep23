// Copyright by BeeX [2026]
// Everything ROS1 for the driver and pick executables: parameters, messages, topics, services and TF. Driver and Pick know
// none of it, so another middleware needs only a file like this one. CMake builds it twice: -DSEP23_DRIVER or -DSEP23_PICK
// picks the main.
#include <sep23/PickState.h>
#include <sep23/driver.h>
#include <sep23/pick.h>

#include <geometry_msgs/PoseArray.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <opencv2/imgproc.hpp>
#include <ros/param.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/String.h>
#include <std_srvs/Trigger.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>

using namespace sep23;

namespace {

// ============================================================================================================================
// parameters and markers
// ============================================================================================================================

// Strict reads from one namespace: nothing is defaulted and every problem is collected.
class Params {
public:
    // `owned` are the sub-namespaces this node answers for; unread keys under them are reported.
    Params(std::string ns, std::vector<std::string> owned) : ns_(std::move(ns)), owned_(std::move(owned)) {}

    double number(const std::string &key) {
        XmlRpc::XmlRpcValue v;
        double              out = 0.0;
        if (fetch(key, v) && !toNumber(v, out)) {
            fail(key, "a number");
        }
        return out;
    }

    int whole(const std::string &key) {
        XmlRpc::XmlRpcValue v;
        if (!fetch(key, v)) {
            return 0;
        }
        if (v.getType() != XmlRpc::XmlRpcValue::TypeInt) {
            fail(key, "a whole number");
            return 0;
        }
        return static_cast<int>(v);
    }

    bool flag(const std::string &key) {
        XmlRpc::XmlRpcValue v;
        if (!fetch(key, v)) {
            return false;
        }
        if (v.getType() != XmlRpc::XmlRpcValue::TypeBoolean) {
            fail(key, "true or false");
            return false;
        }
        return static_cast<bool>(v);
    }

    std::string text(const std::string &key) {
        XmlRpc::XmlRpcValue v;
        if (!fetch(key, v)) {
            return {};
        }
        if (v.getType() != XmlRpc::XmlRpcValue::TypeString) {
            fail(key, "text");
            return {};
        }
        return static_cast<std::string>(v);
    }

    std::vector<double> numbers(const std::string &key, size_t count) {
        XmlRpc::XmlRpcValue v;
        std::vector<double> out(count, 0.0);
        if (fetch(key, v) && !toNumbers(v, count, out)) {
            fail(key, "a list of " + std::to_string(count) + " numbers");
        }
        return out;
    }

    std::vector<std::vector<double>> rows(const std::string &key, size_t columns) {
        XmlRpc::XmlRpcValue              v;
        std::vector<std::vector<double>> out;
        if (!fetch(key, v)) {
            return out;
        }
        for (int i = 0; v.getType() == XmlRpc::XmlRpcValue::TypeArray && i < v.size(); ++i) {
            std::vector<double> row(columns, 0.0);
            if (!toNumbers(v[i], columns, row)) {
                break;
            }
            out.push_back(row);
        }
        if (out.empty() || static_cast<int>(out.size()) != v.size()) {
            fail(key, "rows of " + std::to_string(columns) + " numbers");
            out.clear();
        }
        return out;
    }

    void require(bool ok, const std::string &key, const std::string &rule) {
        if (!ok) {
            errors_.push_back(ns_ + "/" + key + " must be " + rule);
        }
    }

    // Failed reads and broken rules, then keys nothing read under the owned sub-namespaces.
    std::string problems() const {
        std::string out;
        for (const std::string &line : errors_) {
            out += "  " + line + "\n";
        }
        std::vector<std::string> names;
        ros::param::getParamNames(names);
        for (const std::string &name : names) {
            for (const std::string &sub : owned_) {
                const std::string prefix = sub.empty() ? ns_ + "/" : ns_ + "/" + sub + "/";
                if (name.compare(0, prefix.size(), prefix) == 0 && read_.count(name) == 0) {
                    out += "  " + name + " is set but nothing reads it\n";
                    break;
                }
            }
        }
        return out;
    }

private:
    bool fetch(const std::string &key, XmlRpc::XmlRpcValue &v) {
        const std::string name = ns_ + "/" + key;
        read_.insert(name);
        if (!ros::param::get(name, v)) {
            errors_.push_back(name + " is missing");
            return false;
        }
        return true;
    }

    void fail(const std::string &key, const std::string &expected) { errors_.push_back(ns_ + "/" + key + " must be " + expected); }

    static bool toNumber(XmlRpc::XmlRpcValue &v, double &out) {
        if (v.getType() == XmlRpc::XmlRpcValue::TypeDouble) {
            out = static_cast<double>(v);
            return true;
        }
        if (v.getType() == XmlRpc::XmlRpcValue::TypeInt) {
            out = static_cast<int>(v);
            return true;
        }
        return false;
    }

    static bool toNumbers(XmlRpc::XmlRpcValue &v, size_t count, std::vector<double> &out) {
        if (v.getType() != XmlRpc::XmlRpcValue::TypeArray || static_cast<size_t>(v.size()) != count) {
            return false;
        }
        for (int i = 0; i < v.size(); ++i) {
            if (!toNumber(v[i], out[i])) {
                return false;
            }
        }
        return true;
    }

    std::string              ns_;
    std::vector<std::string> owned_;
    std::set<std::string>    read_;
    std::vector<std::string> errors_;
};

#if defined(SEP23_PICK)
inline geometry_msgs::Point toPoint(const Eigen::Vector3d &v) {
    geometry_msgs::Point p;
    p.x = v.x();
    p.y = v.y();
    p.z = v.z();
    return p;
}

inline visualization_msgs::Marker marker(const std::string &frame, const std::string &ns, int type, float r, float g, float b,
                                         float a, double size) {
    visualization_msgs::Marker m;
    m.header.frame_id    = frame;
    m.header.stamp       = ros::Time::now();
    m.ns                 = ns;
    m.type               = type;
    m.action             = visualization_msgs::Marker::ADD;
    m.pose.orientation.w = 1.0;
    m.frame_locked       = true;  // follow the frame as the vehicle moves, not where it was when published
    m.scale.x = m.scale.y = m.scale.z = size;
    m.color.r = r;
    m.color.g = g;
    m.color.b = b;
    m.color.a = a;
    return m;
}

// Occupied cells as points in `frame`, with intensity 1 on handle cells.
inline sensor_msgs::PointCloud2 mapCloud(const ObstacleMap &map, const Eigen::Isometry3d &map_to_frame, const std::string &frame) {
    sensor_msgs::PointCloud2 cloud;
    cloud.header.frame_id = frame;
    cloud.header.stamp    = ros::Time::now();
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(4, "x", 1, sensor_msgs::PointField::FLOAT32, "y", 1, sensor_msgs::PointField::FLOAT32, "z", 1,
                                  sensor_msgs::PointField::FLOAT32, "intensity", 1, sensor_msgs::PointField::FLOAT32);
    modifier.resize(map.obstacle.size() + map.handle.size());
    sensor_msgs::PointCloud2Iterator<float> x(cloud, "x"), y(cloud, "y"), z(cloud, "z"), i(cloud, "intensity");
    for (const std::vector<uint32_t> *cells : {&map.obstacle, &map.handle}) {
        for (const uint32_t c : *cells) {
            const Eigen::Vector3d p = map_to_frame * map.box.centre(c);
            *x = static_cast<float>(p.x());
            *y = static_cast<float>(p.y());
            *z = static_cast<float>(p.z());
            *i = cells == &map.handle ? 1.0f : 0.0f;
            ++x;
            ++y;
            ++z;
            ++i;
        }
    }
    return cloud;
}

// Link capsules and blade samples of the collision body.
inline visualization_msgs::MarkerArray bodyMarkers(const Body &body, double link_radius, const std::string &frame) {
    visualization_msgs::MarkerArray out;
    int                             id = 0;
    for (const Segment &s : body.links) {
        visualization_msgs::Marker tube = marker(frame, "links", visualization_msgs::Marker::CYLINDER, 0.2f, 0.6f, 1.0f, 0.35f, 2.0 * link_radius);
        const Eigen::Vector3d      span = s.b - s.a;
        const Eigen::Quaterniond   turn = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), span.norm() > 0.0 ? span : Eigen::Vector3d::UnitZ());
        tube.id                         = id++;
        tube.pose.position              = toPoint(0.5 * (s.a + s.b));
        tube.pose.orientation   = tf2::toMsg(turn);
        tube.scale.z            = span.norm();
        out.markers.push_back(tube);
    }
    visualization_msgs::Marker blades = marker(frame, "blades", visualization_msgs::Marker::SPHERE_LIST, 0.2f, 0.6f, 1.0f, 0.35f, 0.003);
    for (const Eigen::Vector3d &p : body.blades) {
        blades.points.push_back(toPoint(p));
    }
    out.markers.push_back(blades);
    return out;
}

// The hull footprint as a flat plate at its floor height.
inline visualization_msgs::Marker hullMarker(const Hull &h, const std::string &frame) {
    visualization_msgs::Marker plate = marker(frame, "hull", visualization_msgs::Marker::TRIANGLE_LIST, 0.8f, 0.3f, 0.3f, 0.3f, 1.0);
    const Eigen::Vector3d      c[4]  = {{h.min_x, h.min_y, h.floor_z}, {h.max_x, h.min_y, h.floor_z}, {h.max_x, h.max_y, h.floor_z},
                                        {h.min_x, h.max_y, h.floor_z}};
    for (const int k : {0, 1, 2, 0, 2, 3}) {
        plate.points.push_back(toPoint(c[k]));
    }
    return plate;
}

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
    jaw.open_len              = robot.number("jaws/open_m");
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

    c.seed                  = pick.whole("plan/seed");
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
#endif

void rosLog(const char *node, Level level, const std::string &text) {
    switch (level) {
    case Level::INFO: ROS_INFO("[%s] %s", node, text.c_str()); break;
    case Level::WARN: ROS_WARN("[%s] %s", node, text.c_str()); break;
    case Level::ERROR: ROS_ERROR("[%s] %s", node, text.c_str()); break;
    }
}

#if defined(SEP23_DRIVER)
// ============================================================================================================================
// driver: joint_states out, joint targets in, and the jaw, home and standby as services.
// ============================================================================================================================

class DriverNode {
public:
    explicit DriverNode(Driver &driver) : driver_(driver) {
        ros::NodeHandle nh, pnh("~");
        pub_states_  = nh.advertise<sensor_msgs::JointState>("joint_states", 1);
        sub_targets_ = pnh.subscribe("joint_targets", 1, &DriverNode::onTargets, this);
        services_    = {pnh.advertiseService("home", &DriverNode::onHome, this), pnh.advertiseService("open_jaw", &DriverNode::onOpenJaw, this),
                        pnh.advertiseService("close_jaw", &DriverNode::onCloseJaw, this),
                        pnh.advertiseService("standby", &DriverNode::onStandby, this)};
    }

    void poll() {
        AxisReadings r;
        if (!driver_.poll(ros::Time::now().toSec(), r)) {
            ROS_WARN_THROTTLE(5.0, "[driver] not every joint has answered yet");
            return;
        }
        sensor_msgs::JointState msg;
        msg.header.stamp = ros::Time(r.stamp);
        for (int a = 0; a < AXES; ++a) {
            msg.name.push_back(driver_.axes()[a].name);
            msg.position.push_back(r.position[a]);
        }
        pub_states_.publish(msg);
    }

private:
    void onTargets(const sensor_msgs::JointState::ConstPtr &msg) {
        std::string why;
        if (!driver_.moveTo(msg->name, msg->position, why)) {
            ROS_WARN("[driver] %s", why.c_str());
        }
    }

    bool onHome(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res) {
        res.success = driver_.home(res.message);
        return true;
    }

    bool onOpenJaw(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res) {
        res.success = driver_.openJaw(res.message);
        return true;
    }

    bool onCloseJaw(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res) {
        res.success = driver_.closeJaw(res.message);
        return true;
    }

    bool onStandby(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res) {
        res.success = driver_.standby(res.message);
        rosLog("driver", res.success ? Level::INFO : Level::ERROR, res.message);
        return true;
    }

    Driver                         &driver_;
    ros::Publisher                  pub_states_;
    ros::Subscriber                 sub_targets_;
    std::vector<ros::ServiceServer> services_;
};

int runDriver(int argc, char **argv) {
    ros::init(argc, argv, "driver");
    Params robot("/robot", {"driver"});
    Params own("/driver", {""});

    std::array<Axis, AXES> axes;
    const std::string      keys[AXES] = {"base", "shoulder", "elbow", "wrist", "jaw"};
    const double joint_speed          = rad(robot.number("driver/joint_speed_deg_s"));
    for (int a = 0; a < AXES; ++a) {
        Axis &x  = axes[a];
        x.name   = robot.text("arm/joint_names/" + keys[a]);
        const int device = robot.whole("driver/device_ids/" + keys[a]);
        robot.require(device > 0 && device < 256, "driver/device_ids/" + keys[a], "between 1 and 255");
        x.device = static_cast<uint8_t>(device);
        if (a == JAW) {
            const std::vector<double> limits = robot.numbers("jaws/limits_m", 2);
            x.min           = limits[0];
            x.max           = limits[1];
            x.home          = robot.number("jaws/open_m");
            x.speed         = robot.number("driver/jaw_speed_m_s");
            x.wire_per_unit = 1000.0;  // the jaw speaks millimetres on the wire
        } else {
            const std::vector<double> limits = robot.numbers("arm/limits_deg/" + keys[a], 2);
            x.min   = rad(limits[0]);
            x.max   = rad(limits[1]);
            x.home  = rad(robot.number("arm/home_deg/" + keys[a]));
            x.speed = joint_speed;
        }
        robot.require(x.min < x.max && x.home >= x.min && x.home <= x.max, keys[a], "min below max with home inside");
    }
    const bool     simulated = own.flag("simulated");
    SerialSettings serial;
    serial.port             = robot.text("driver/serial_port");
    serial.baud             = robot.whole("driver/baud_rate");
    serial.reply_timeout_s  = robot.number("driver/reply_timeout_s");
    serial.connect_attempts = robot.whole("driver/connect_attempts");
    serial.connect_retry_s  = robot.number("driver/connect_retry_s");
    serial.wake_speed       = rad(robot.number("driver/wake_speed_deg_s"));
    const double poll_hz    = robot.number("driver/poll_hz");
    robot.require(poll_hz > 0.0 && serial.reply_timeout_s > 0.0 && serial.connect_attempts > 0, "driver",
                  "positive poll_hz, reply_timeout_s and connect_attempts");
    const std::string problems = robot.problems() + own.problems();
    if (!problems.empty()) {
        ROS_ERROR("[driver] configuration is not usable:\n%s", problems.c_str());
        return 1;
    }

    const Log                           log = [](Level level, const std::string &text) { rosLog("driver", level, text); };
    std::shared_ptr<SimulatedActuators> sim;
    std::shared_ptr<Actuators>          actuators;
    if (simulated) {
        sim       = simulateArm(axes);
        actuators = sim;
    } else {
        std::string why;
        actuators = connectArm(serial, axes, log, why);
        if (!actuators) {
            ROS_ERROR("[driver] %s", why.c_str());
            return 1;
        }
    }

    Driver      driver(axes, actuators, sim, poll_hz, log);
    std::string released;
    driver.standby(released);
    DriverNode node(driver);
    ROS_INFO("[driver] talking to %s at %.1f Hz", simulated ? "the simulated arm" : serial.port.c_str(), poll_hz);
    ros::AsyncSpinner spinner(2);
    spinner.start();
    for (ros::Rate rate(poll_hz); ros::ok(); rate.sleep()) {
        node.poll();
    }
    driver.standby(released);
    return 0;
}
#endif

#if defined(SEP23_PICK)
// ============================================================================================================================
// pick: PickIO over topics and the driver's services, and the sensors, requests and TF around Pick.
// ============================================================================================================================

class RosPickIO : public PickIO {
public:
    explicit RosPickIO(const PickConfig &c) : c_(c) {
        ros::NodeHandle nh, pnh("~");
        targets_pub_  = nh.advertise<sensor_msgs::JointState>("driver/joint_targets", 1);
        close_jaw_    = nh.serviceClient<std_srvs::Trigger>("driver/close_jaw");
        standby_      = nh.serviceClient<std_srvs::Trigger>("driver/standby");
        home_         = nh.serviceClient<std_srvs::Trigger>("driver/home");
        state_pub_    = pnh.advertise<PickState>("state", 1, true);
        park_map_pub_ = pnh.advertise<sensor_msgs::PointCloud2>("park_map", 1, true);
        map_pub_      = pnh.advertise<sensor_msgs::PointCloud2>("pick_map", 1, true);
        grasp_pub_    = pnh.advertise<visualization_msgs::MarkerArray>("grasps", 1, true);
        path_pub_     = pnh.advertise<visualization_msgs::Marker>("path", 1, true);
        body_pub_     = pnh.advertise<visualization_msgs::MarkerArray>("body", 1);
        hull_pub_     = pnh.advertise<visualization_msgs::Marker>("hull", 1, true);
        if (c_.track_enabled) {
            shift_pub_ = nh.advertise<visualization_msgs::MarkerArray>("tracker/target_shift", 1, true);
        }
        hull_pub_.publish(hullMarker(c_.hull, c_.arm_frame));
    }

    double now() override { return ros::Time::now().toSec(); }
    void   log(Level level, const std::string &text) override { rosLog("pick", level, text); }

    void sendTargets(const Joints &reported) override {
        sensor_msgs::JointState msg;
        msg.header.stamp = ros::Time::now();
        msg.name.assign(c_.joint_names.begin(), c_.joint_names.end());
        msg.position.assign(reported.begin(), reported.end());
        targets_pub_.publish(msg);
    }

    bool closeJaw(std::string &why) override {
        std_srvs::Trigger call;
        if (!close_jaw_.call(call)) {
            why = "the driver did not answer";
            return false;
        }
        why = call.response.message;
        return call.response.success;
    }

    bool standby() override {
        std_srvs::Trigger call;
        return standby_.call(call) && call.response.success;
    }

    bool home(std::string &message) override {
        std_srvs::Trigger call;
        if (!home_.call(call)) {
            return false;
        }
        message = call.response.message;
        return true;
    }

    void status(const PickStatus &s) override {
        PickState msg;
        msg.state        = stateName(s.state);
        msg.message      = s.message;
        msg.park_attempt = s.park_attempt;
        msg.grabbed      = s.grabbed;
        state_pub_.publish(msg);
    }

    void showMap(const ObstacleMap &map, const Eigen::Isometry3d &map_to_world, bool park) override {
        (park ? park_map_pub_ : map_pub_).publish(mapCloud(map, map_to_world, c_.world_frame));
    }

    // Vision's poses, the candidates, and the chosen one if any, in the world.
    void showGrasps(const Scene &scene, const Eigen::Isometry3d &scene_to_world, int chosen) override {
        visualization_msgs::MarkerArray out;
        visualization_msgs::Marker      poses = marker(c_.world_frame, "poses", visualization_msgs::Marker::SPHERE_LIST, 0.6f, 0.6f, 0.6f, 1.0f, 0.004);
        visualization_msgs::Marker      spots = marker(c_.world_frame, "candidates", visualization_msgs::Marker::SPHERE_LIST, 1.0f, 0.8f, 0.1f, 1.0f, 0.007);
        for (const GraspPose &g : scene.poses) {
            poses.points.push_back(toPoint(scene_to_world * g.point));
        }
        for (const GraspPose &g : scene.candidates) {
            spots.points.push_back(toPoint(scene_to_world * g.point));
        }
        visualization_msgs::Marker pick = marker(c_.world_frame, "chosen", visualization_msgs::Marker::LINE_LIST, 0.1f, 1.0f, 0.3f, 1.0f, 0.003);
        if (chosen >= 0) {
            const GraspPose &g = scene.candidates[chosen];
            pick.points.push_back(toPoint(scene_to_world * Eigen::Vector3d(g.point - 0.03 * g.bar.normalized())));
            pick.points.push_back(toPoint(scene_to_world * Eigen::Vector3d(g.point + 0.03 * g.bar.normalized())));
        } else {
            pick.action = visualization_msgs::Marker::DELETE;
        }
        out.markers = {poses, spots, pick};
        grasp_pub_.publish(out);
    }

    void showPath(const std::vector<Eigen::Vector3d> &tip) override {
        visualization_msgs::Marker line = marker(c_.arm_frame, "path", visualization_msgs::Marker::LINE_STRIP, 0.1f, 1.0f, 0.3f, 1.0f, 0.002);
        for (const Eigen::Vector3d &p : tip) {
            line.points.push_back(toPoint(p));
        }
        if (line.points.empty()) {
            line.action = visualization_msgs::Marker::DELETE;
        }
        path_pub_.publish(line);
    }

    void showBody(const Body &body) override { body_pub_.publish(bodyMarkers(body, c_.park.link_radius, c_.arm_frame)); }

    // From where the arm is aiming to where the tracker puts the target: the correction a closed loop would make.
    void showTrack(const TrackResult &r, const std::vector<Eigen::Vector3d> &trail) override {
        visualization_msgs::Marker shift = marker(c_.camera_frame, "shift", visualization_msgs::Marker::ARROW, 1.0f, 0.3f, 0.9f, 1.0f, 0.003);
        shift.scale.y = 0.007, shift.scale.z = 0.01;  // shaft and head width, head length
        shift.points  = {toPoint(r.target - r.offset), toPoint(r.target)};
        visualization_msgs::Marker line = marker(c_.camera_frame, "trail", visualization_msgs::Marker::LINE_STRIP, 0.3f, 0.9f, 1.0f, 0.8f, 0.0015);
        for (const Eigen::Vector3d &p : trail) {
            line.points.push_back(toPoint(p));
        }
        visualization_msgs::Marker ball = marker(c_.camera_frame, "target", visualization_msgs::Marker::SPHERE, 1.0f, 0.55f, 0.0f, 1.0f, 0.012);
        ball.pose.position              = toPoint(r.target);
        visualization_msgs::MarkerArray markers;
        markers.markers = {shift, line, ball};
        shift_pub_.publish(markers);
    }

    void clearTrack() override {
        visualization_msgs::MarkerArray wipe;
        wipe.markers.push_back(marker(c_.camera_frame, "", visualization_msgs::Marker::ARROW, 0, 0, 0, 0, 0));
        wipe.markers[0].action = visualization_msgs::Marker::DELETEALL;
        shift_pub_.publish(wipe);
    }

private:
    PickConfig         c_;
    ros::Publisher     targets_pub_, state_pub_, park_map_pub_, map_pub_, grasp_pub_, path_pub_, body_pub_, hull_pub_, shift_pub_;
    ros::ServiceClient close_jaw_, standby_, home_;
};

class PickNode {
public:
    PickNode(const PickConfig &config, const Arm &arm)
            : io_(config),
              pick_(config, arm, io_),
              cloud_sub_(nh_, "cloud", 1),
              poses_sub_(nh_, "grasp_poses", 1),
              sync_(cloud_sub_, poses_sub_, 5),
              track_sync_(5) {
        ros::NodeHandle pnh("~");
        const PickConfig &c = pick_.config();
        sync_.registerCallback(&PickNode::onFrame, this);
        cloud_sub_.registerCallback(&PickNode::onCloudSeen, this);
        if (c.track_enabled) {
            image_sub_.subscribe(nh_, "image", 1);
            track_sync_.connectInput(cloud_sub_, image_sub_);
            track_sync_.registerCallback(&PickNode::onTrackFrame, this);
            view_sub_  = nh_.subscribe("tracker/view", 1, &PickNode::onView, this);
            debug_pub_ = nh_.advertise<sensor_msgs::Image>("tracker/debug_image", 1);
            info_pub_  = nh_.advertise<sensor_msgs::CameraInfo>("tracker/camera_info", 1);  // Foxglove pairs it with the image
        }
        joints_sub_ = nh_.subscribe("joint_states", 1, &PickNode::onJoints, this);
        services_   = {pnh.advertiseService("start", &PickNode::onStart, this), pnh.advertiseService("stop", &PickNode::onStop, this),
                       pnh.advertiseService("reset", &PickNode::onReset, this)};
        tf_timer_   = nh_.createTimer(ros::Duration(1.0 / c.loop_hz), [this](const ros::TimerEvent &) { broadcastVehicle(); });
        // A replayed camera stays put while the vehicle moves, so the URDF leaves it off the vehicle and it is pinned here.
        if (c.camera_fixed_in_world) {
            geometry_msgs::TransformStamped t = tf2::eigenToTransform(c.camera_to_vehicle);
            t.header.stamp                    = ros::Time::now();
            t.header.frame_id                 = c.world_frame;
            t.child_frame_id                  = c.camera_frame;
            static_broadcaster_.sendTransform(t);
        }
    }

    const PickConfig &config() const { return pick_.config(); }
    void              tick() { pick_.tick(); }

private:
    void onCloudSeen(const sensor_msgs::PointCloud2::ConstPtr &) { pick_.cloudSeen(); }

    void onFrame(const sensor_msgs::PointCloud2::ConstPtr &cloud, const geometry_msgs::PoseArray::ConstPtr &poses) {
        const PickConfig &c = pick_.config();
        if (cloud->header.frame_id != c.camera_frame || poses->header.frame_id != c.camera_frame
            || static_cast<long>(cloud->width) * cloud->height != static_cast<long>(c.camera.width) * c.camera.height) {
            ROS_WARN_THROTTLE(5.0, "[pick] frame dropped: expected an organised %dx%d cloud in %s", c.camera.width, c.camera.height,
                              c.camera_frame.c_str());
            return;
        }
        if (poses->poses.empty()) {
            pick_.addPoseless();
            return;
        }
        Frame frame;
        frame.points = cloudPoints(*cloud);
        for (const geometry_msgs::Pose &p : poses->poses) {
            const Eigen::Matrix3d r = Eigen::Quaterniond(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z).normalized().toRotationMatrix();
            frame.poses.push_back({Eigen::Vector3d(p.position.x, p.position.y, p.position.z), r.col(c.bar_column), r.col(c.approach_column)});
        }
        pick_.addFrame(std::move(frame));
    }

    void onTrackFrame(const sensor_msgs::PointCloud2::ConstPtr &cloud, const sensor_msgs::Image::ConstPtr &image) {
        const PickConfig &c      = pick_.config();
        const bool        viewer = debug_pub_.getNumSubscribers() > 0;  // Foxglove subscribes to what its panels show
        if (!pick_.wantsTrackFrame(viewer)) {
            return;
        }
        cv::Mat bgr;
        if (cloud->header.frame_id != c.camera_frame || !toBgr(*image, bgr)) {
            ROS_WARN_THROTTLE(5.0, "[pick] frame dropped: expected a bgr8, rgb8 or mono8 image with a cloud in %s", c.camera_frame.c_str());
            return;
        }
        const cv::Mat view = pick_.trackFrame(bgr, cloudPoints(*cloud), viewer);
        if (view.empty()) {
            return;
        }
        debug_pub_.publish(toImage(view, image->header, "bgr8"));
        sensor_msgs::CameraInfo info;
        info.header = image->header;
        info.width = c.camera.width, info.height = c.camera.height;
        info.K = {c.camera.fx, 0.0, c.camera.cx, 0.0, c.camera.fy, c.camera.cy, 0.0, 0.0, 1.0};
        info.R = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        info.P = {c.camera.fx, 0.0, c.camera.cx, 0.0, 0.0, c.camera.fy, c.camera.cy, 0.0, 0.0, 0.0, 1.0, 0.0};
        info_pub_.publish(info);
    }

    void onView(const std_msgs::String::ConstPtr &msg) {
        if (msg->data == "ransac" || msg->data == "klt" || msg->data == "mask") {
            pick_.setView(msg->data == "ransac" ? TrackView::RANSAC : msg->data == "klt" ? TrackView::KLT : TrackView::MASK);
        } else {
            ROS_WARN("[pick] view %s is none of ransac, klt, mask", msg->data.c_str());
        }
    }

    void onJoints(const sensor_msgs::JointState::ConstPtr &msg) {
        const PickConfig &c  = pick_.config();
        const auto        at = [&](const std::string &name) {
            const size_t i = std::find(msg->name.begin(), msg->name.end(), name) - msg->name.begin();
            return i < msg->name.size() && i < msg->position.size() ? msg->position[i] : std::numeric_limits<double>::quiet_NaN();
        };
        Joints q;
        for (int j = 0; j < JOINT_COUNT; ++j) {
            q[j] = at(c.joint_names[j]);
        }
        if (std::all_of(q.begin(), q.end(), [](double v) { return std::isfinite(v); })) {
            pick_.joints(q, msg->header.stamp.toSec());
        }
        const double jaw = at(c.jaw_name);
        if (std::isfinite(jaw)) {
            pick_.jaw(jaw, msg->header.stamp.toSec());
        }
    }

    bool onStart(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res) {
        bool steer = pick_.config().track_steer;
        ros::param::get("~track/steer", steer);  // scripts/loop.sh sets it
        pick_.requestStart(steer);
        res.success = true;
        res.message = "start requested";
        return true;
    }

    bool onStop(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res) {
        pick_.requestStop();
        res.success = true;
        res.message = "stop requested, the arm will be released";
        return true;
    }

    bool onReset(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res) {
        res.success = pick_.reset(res.message);
        return true;
    }

    void broadcastVehicle() {
        const PickConfig               &c = pick_.config();
        geometry_msgs::TransformStamped t = tf2::eigenToTransform(toIsometry(pick_.vehicle()));
        t.header.stamp                    = ros::Time::now();
        t.header.frame_id                 = c.world_frame;
        t.child_frame_id                  = c.vehicle_frame;
        broadcaster_.sendTransform(t);
    }

    RosPickIO io_;
    Pick      pick_;

    ros::NodeHandle                                                                        nh_;
    message_filters::Subscriber<sensor_msgs::PointCloud2>                                  cloud_sub_;
    message_filters::Subscriber<geometry_msgs::PoseArray>                                  poses_sub_;
    message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, geometry_msgs::PoseArray> sync_;
    message_filters::Subscriber<sensor_msgs::Image>                                        image_sub_;
    message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, sensor_msgs::Image>       track_sync_;
    ros::Subscriber                                                                        view_sub_, joints_sub_;
    ros::Publisher                                                                         debug_pub_, info_pub_;
    std::vector<ros::ServiceServer>                                                        services_;
    ros::Timer                                                                             tf_timer_;
    tf2_ros::TransformBroadcaster                                                          broadcaster_;
    tf2_ros::StaticTransformBroadcaster                                                    static_broadcaster_;
};

int runPick(int argc, char **argv) {
    ros::init(argc, argv, "pick");
    Params     robot("/robot", {"frames", "arm", "jaws", "camera", "vehicle"});
    Params     pick("/pick", {""});
    PickConfig config;
    loadPick(robot, pick, config);

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
    const std::string problems = robot.problems() + pick.problems();
    if (!problems.empty()) {
        ROS_ERROR("[pick] configuration is not usable:\n%s", problems.c_str());
        return 1;
    }

    PickNode node(config, *arm);
    ROS_INFO("[pick] ready: crop radius %.2f m, %d frames per look", node.config().cloud.crop_radius, node.config().frames_needed);
    ros::AsyncSpinner spinner(2);
    spinner.start();
    for (ros::Rate rate(node.config().loop_hz); ros::ok(); rate.sleep()) {
        node.tick();
    }
    return 0;
}
#endif

}  // namespace

int main(int argc, char **argv) {
#if defined(SEP23_DRIVER)
    return runDriver(argc, argv);
#elif defined(SEP23_PICK)
    return runPick(argc, argv);
#else
#error "build roswrapper.cpp with SEP23_DRIVER or SEP23_PICK"
#endif
}
