// Copyright by BeeX [2026]
#pragma once

#include <sep23/log.h>

#include <sep23/arm.h>
#include <sep23/bpl.h>

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace sep23 {

constexpr int AXES = JOINT_COUNT + 1;  // the arm joints, then the jaw
constexpr int JAW  = JOINT_COUNT;

// One actuator in SI units: radians for rotary joints, metres for the jaw.
struct Axis {
    std::string name;
    uint8_t     device = 0;
    double      min = 0.0, max = 0.0, home = 0.0, speed = 0.0, wire_per_unit = 1.0;
};

struct SerialSettings {
    std::string port;
    int         baud = 0;
    double      reply_timeout_s = 0.0;
    int         connect_attempts = 0;
    double      connect_retry_s = 0.0;
    double      wake_speed = 0.0;  // rad/s
};

// The real arm, woken through the wrist; null with `why` set when it never answers.
std::shared_ptr<Actuators> connectArm(const SerialSettings &s, const std::array<Axis, AXES> &axes, const Log &log, std::string &why);
// Joints that walk to their targets, starting at home.
std::shared_ptr<SimulatedActuators> simulateArm(const std::array<Axis, AXES> &axes);

// Every axis position, stamped with the oldest reading so a joint that stopped answering shows up as stale.
struct AxisReadings {
    double                   stamp = 0.0;  // seconds on the caller's clock
    std::array<double, AXES> position{};
};

// The arm with no middleware: checks targets against the limits, ramps home, opens and closes the jaw, releases.
// Thread safe; every time is seconds on the caller's clock.
class Driver {
public:
    // `simulated` is the same actuators when they are simulated, so home can skip the ramp; null on the real arm.
    Driver(const std::array<Axis, AXES> &axes, std::shared_ptr<Actuators> actuators, std::shared_ptr<SimulatedActuators> simulated,
              double poll_hz, Log log);

    const std::array<Axis, AXES> &axes() const { return axes_; }

    // Reads every axis and steps a home ramp; false until every axis has answered once.
    bool poll(double now, AxisReadings &out);
    // Every target is checked before any is sent.
    bool moveTo(const std::vector<std::string> &names, const std::vector<double> &positions, std::string &why);
    bool home(std::string &message);
    bool openJaw(std::string &why) { return moveOne(JAW, axes_[JAW].home, why); }
    bool closeJaw(std::string &why) { return moveOne(JAW, axes_[JAW].min, why); }
    // Every joint is released even if one fails.
    bool standby(std::string &message);

private:
    bool moveOne(int a, double target, std::string &why);
    bool move(int a, double target, std::string &why);  // with the lock held
    void stepHome();

    std::array<Axis, AXES>              axes_;
    std::shared_ptr<Actuators>          actuators_;
    std::shared_ptr<SimulatedActuators> simulated_;
    double                              poll_hz_;
    Log                                 log_;
    std::mutex                          mutex_;
    std::array<double, AXES>            position_{};
    std::array<double, AXES>            read_at_{};  // 0 until the axis answers
    bool                                homing_ = false;
};

}  // namespace sep23
