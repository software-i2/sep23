// Copyright by BeeX [2026]
#include <sep23/arm.h>
#include <sep23/bpl.h>
#include <sep23/params.h>

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_srvs/Trigger.h>

#include <chrono>
#include <thread>

using namespace sep23;

namespace {

constexpr int AXES = JOINT_COUNT + 1;  // the arm joints, then the jaw
constexpr int JAW  = JOINT_COUNT;

// One actuator in ROS units: radians for rotary joints, metres for the jaw.
struct Axis {
    std::string name;
    uint8_t     device = 0;
    double      min = 0.0, max = 0.0, home = 0.0, speed = 0.0, wire_per_unit = 1.0;
};

// Publishes joint_states, takes joint targets and the jaw, home and standby calls.
class Driver {
public:
    Driver(std::array<Axis, AXES> axes, double jaw_open, std::shared_ptr<Actuators> actuators, double poll_hz)
            : axes_(std::move(axes)), jaw_open_(jaw_open), actuators_(std::move(actuators)), poll_hz_(poll_hz) {
        ros::NodeHandle nh, pnh("~");
        pub_states_  = nh.advertise<sensor_msgs::JointState>("joint_states", 1);
        sub_targets_ = pnh.subscribe("joint_targets", 1, &Driver::onTargets, this);
        services_    = {pnh.advertiseService("home", &Driver::onHome, this), pnh.advertiseService("open_jaw", &Driver::onOpenJaw, this),
                        pnh.advertiseService("close_jaw", &Driver::onCloseJaw, this),
                        pnh.advertiseService("standby", &Driver::onStandby, this)};
    }

    void poll() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int a = 0; a < AXES; ++a) {
            float wire = 0.0f;
            if (actuators_->position(axes_[a].device, wire)) {
                position_[a] = wire / axes_[a].wire_per_unit;
                read_at_[a]  = ros::Time::now();
            }
        }
        if (homing_) {
            stepHome();
        }
        // Stamped with the oldest reading, so a joint that stopped answering shows up as stale.
        ros::Time oldest = *std::min_element(read_at_.begin(), read_at_.end());
        if (oldest.isZero()) {
            ROS_WARN_THROTTLE(5.0, "[driver] not every joint has answered yet");
            return;
        }
        sensor_msgs::JointState msg;
        msg.header.stamp = oldest;
        for (int a = 0; a < AXES; ++a) {
            msg.name.push_back(axes_[a].name);
            msg.position.push_back(position_[a]);
        }
        pub_states_.publish(msg);
    }

    void releaseAll() {
        for (const Axis &a : axes_) {
            actuators_->standby(a.device);
        }
    }

private:
    void stepHome() {
        bool moving = false;
        for (int a = 0; a < AXES; ++a) {
            const double step = axes_[a].speed / poll_hz_, gap = axes_[a].home - position_[a];
            moving |= std::fabs(gap) > step;
            actuators_->command(axes_[a].device,
                                static_cast<float>((std::fabs(gap) <= step ? axes_[a].home : position_[a] + std::copysign(step, gap))
                                                   * axes_[a].wire_per_unit));
        }
        homing_ = moving;
        if (!homing_) {
            ROS_INFO("[driver] home reached");
        }
    }

    bool move(int a, double target, std::string &why) {
        if (!std::isfinite(target) || target < axes_[a].min || target > axes_[a].max) {
            why = axes_[a].name + " target " + std::to_string(target) + " is outside its limits";
            return false;
        }
        homing_ = false;
        if (!actuators_->command(axes_[a].device, static_cast<float>(target * axes_[a].wire_per_unit))) {
            why = "the command to " + axes_[a].name + " could not be written";
            return false;
        }
        return true;
    }

    // Every target in the message is checked before any is sent.
    void onTargets(const sensor_msgs::JointState::ConstPtr &msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<int>            index;
        for (size_t i = 0; i < msg->name.size() && msg->name.size() == msg->position.size(); ++i) {
            const auto found = std::find_if(axes_.begin(), axes_.end(), [&](const Axis &a) { return a.name == msg->name[i]; });
            if (found == axes_.end() || !(msg->position[i] >= found->min && msg->position[i] <= found->max)) {
                ROS_WARN("[driver] joint targets dropped: %s is unknown or outside its limits", msg->name[i].c_str());
                return;
            }
            index.push_back(static_cast<int>(found - axes_.begin()));
        }
        std::string why;
        for (size_t i = 0; i < index.size(); ++i) {
            if (!move(index[i], msg->position[i], why)) {
                ROS_WARN("[driver] %s", why.c_str());
            }
        }
    }

    bool onHome(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res) {
        std::lock_guard<std::mutex> lock(mutex_);
        res.success = std::none_of(read_at_.begin(), read_at_.end(), [](const ros::Time &t) { return t.isZero(); });
        res.message = res.success ? "moving home with the jaw open" : "not every joint has reported a position yet";
        homing_     = res.success;
        return true;
    }

    bool onOpenJaw(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res) {
        std::lock_guard<std::mutex> lock(mutex_);
        res.success = move(JAW, jaw_open_, res.message);
        return true;
    }

    bool onCloseJaw(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res) {
        std::lock_guard<std::mutex> lock(mutex_);
        res.success = move(JAW, axes_[JAW].min, res.message);
        return true;
    }

    bool onStandby(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res) {
        std::lock_guard<std::mutex> lock(mutex_);
        homing_     = false;
        res.success = true;
        for (const Axis &a : axes_) {
            res.success &= actuators_->standby(a.device);  // every joint is released even if one fails
        }
        res.message = res.success ? "every joint released" : "a standby command could not be written";
        if (res.success) {
            ROS_INFO("[driver] %s", res.message.c_str());
        } else {
            ROS_ERROR("[driver] %s", res.message.c_str());
        }
        return true;
    }

    std::array<Axis, AXES>          axes_;
    double                          jaw_open_;
    std::shared_ptr<Actuators>      actuators_;
    double                          poll_hz_;
    std::mutex                      mutex_;
    std::array<double, AXES>        position_{};
    std::array<ros::Time, AXES>     read_at_{};
    bool                            homing_ = false;
    ros::Publisher                  pub_states_;
    ros::Subscriber                 sub_targets_;
    std::vector<ros::ServiceServer> services_;
};

}  // namespace

int main(int argc, char **argv) {
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
    const bool        simulated = own.flag("simulated");
    const std::string port      = robot.text("driver/serial_port");
    const int         baud      = robot.whole("driver/baud_rate");
    const double      poll_hz   = robot.number("driver/poll_hz");
    const double      timeout   = robot.number("driver/reply_timeout_s");
    const int         attempts  = robot.whole("driver/connect_attempts");
    const double      retry_s   = robot.number("driver/connect_retry_s");
    const double      wake      = rad(robot.number("driver/wake_speed_deg_s"));
    robot.require(poll_hz > 0.0 && timeout > 0.0 && attempts > 0, "driver", "positive poll_hz, reply_timeout_s and connect_attempts");
    const std::string problems = robot.problems() + own.problems();
    if (!problems.empty()) {
        ROS_ERROR("[driver] configuration is not usable:\n%s", problems.c_str());
        return 1;
    }

    std::shared_ptr<Actuators> actuators;
    if (simulated) {
        std::vector<SimulatedActuators::Joint> joints;
        for (const Axis &a : axes) {
            joints.push_back({a.device, static_cast<float>(a.min * a.wire_per_unit), static_cast<float>(a.max * a.wire_per_unit),
                              static_cast<float>(a.home * a.wire_per_unit), static_cast<float>(a.speed * a.wire_per_unit)});
        }
        actuators = std::make_shared<SimulatedActuators>(joints);
    } else {
        auto serial = std::make_shared<SerialActuators>(port, baud, timeout);
        bool awake  = false;
        for (int n = 1; serial->error().empty() && !awake && n <= attempts; ++n) {
            awake = serial->wake(axes[WRIST].device, static_cast<float>(wake));
            if (!awake) {
                ROS_WARN("[driver] no answer from the arm, attempt %d of %d", n, attempts);
                std::this_thread::sleep_for(std::chrono::duration<double>(retry_s));
            }
        }
        if (!awake) {
            ROS_ERROR("[driver] the arm did not answer on %s %s", port.c_str(), serial->error().c_str());
            return 1;
        }
        actuators = serial;
    }

    Driver driver(axes, axes[JAW].home, actuators, poll_hz);
    driver.releaseAll();
    ROS_INFO("[driver] talking to %s at %.1f Hz", simulated ? "the simulated arm" : port.c_str(), poll_hz);
    ros::AsyncSpinner spinner(2);
    spinner.start();
    for (ros::Rate rate(poll_hz); ros::ok(); rate.sleep()) {
        driver.poll();
    }
    driver.releaseAll();
    return 0;
}
