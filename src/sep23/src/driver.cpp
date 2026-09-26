// Copyright by BeeX [2026]
#include <sep23/driver.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

namespace sep23 {

std::shared_ptr<Actuators> connectArm(const SerialSettings &s, const std::array<Axis, AXES> &axes, const Log &log, std::string &why) {
    auto serial = std::make_shared<SerialActuators>(s.port, s.baud, s.reply_timeout_s);
    bool awake  = false;
    for (int n = 1; serial->error().empty() && !awake && n <= s.connect_attempts; ++n) {
        awake = serial->wake(axes[WRIST].device, static_cast<float>(s.wake_speed));
        if (!awake) {
            log(Level::WARN, "no answer from the arm, attempt " + std::to_string(n) + " of " + std::to_string(s.connect_attempts));
            std::this_thread::sleep_for(std::chrono::duration<double>(s.connect_retry_s));
        }
    }
    if (!awake) {
        why = "the arm did not answer on " + s.port + " " + serial->error();
        return nullptr;
    }
    return serial;
}

std::shared_ptr<SimulatedActuators> simulateArm(const std::array<Axis, AXES> &axes) {
    std::vector<SimulatedActuators::Joint> joints;
    for (const Axis &a : axes) {
        joints.push_back({a.device, static_cast<float>(a.min * a.wire_per_unit), static_cast<float>(a.max * a.wire_per_unit),
                          static_cast<float>(a.home * a.wire_per_unit), static_cast<float>(a.speed * a.wire_per_unit)});
    }
    return std::make_shared<SimulatedActuators>(joints);
}

Driver::Driver(const std::array<Axis, AXES> &axes, std::shared_ptr<Actuators> actuators, std::shared_ptr<SimulatedActuators> simulated,
                     double poll_hz, Log log)
        : axes_(axes), actuators_(std::move(actuators)), simulated_(std::move(simulated)), poll_hz_(poll_hz), log_(std::move(log)) {}

bool Driver::poll(double now, AxisReadings &out) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int a = 0; a < AXES; ++a) {
        float wire = 0.0f;
        if (actuators_->position(axes_[a].device, wire)) {
            position_[a] = wire / axes_[a].wire_per_unit;
            read_at_[a]  = now;
        }
    }
    if (homing_) {
        stepHome();
    }
    out.stamp    = *std::min_element(read_at_.begin(), read_at_.end());
    out.position = position_;
    return out.stamp > 0.0;
}

bool Driver::moveTo(const std::vector<std::string> &names, const std::vector<double> &positions, std::string &why) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (names.size() != positions.size()) {
        why = "joint targets dropped: " + std::to_string(names.size()) + " names for " + std::to_string(positions.size()) + " positions";
        return false;
    }
    std::vector<int> index;
    for (size_t i = 0; i < names.size(); ++i) {
        const auto found = std::find_if(axes_.begin(), axes_.end(), [&](const Axis &a) { return a.name == names[i]; });
        if (found == axes_.end() || !(positions[i] >= found->min && positions[i] <= found->max)) {
            why = "joint targets dropped: " + names[i] + " is unknown or outside its limits";
            return false;
        }
        index.push_back(static_cast<int>(found - axes_.begin()));
    }
    bool ok = true;
    for (size_t i = 0; i < index.size(); ++i) {
        std::string failed;
        if (!move(index[i], positions[i], failed)) {
            why += (ok ? "" : "; ") + failed;
            ok = false;
        }
    }
    return ok;
}

bool Driver::home(std::string &message) {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool ok = std::none_of(read_at_.begin(), read_at_.end(), [](double t) { return t == 0.0; });
    message       = ok ? "moving home with the jaw open" : "not every joint has reported a position yet";
    homing_       = ok;
    if (ok && simulated_) {  // the simulated arm skips the ramp
        for (const Axis &a : axes_) {
            simulated_->place(a.device, static_cast<float>(a.home * a.wire_per_unit));
        }
        homing_ = false;
        message = "placed at home with the jaw open";
    }
    return ok;
}

bool Driver::standby(std::string &message) {
    std::lock_guard<std::mutex> lock(mutex_);
    homing_ = false;
    bool ok = true;
    for (const Axis &a : axes_) {
        ok &= actuators_->standby(a.device);
    }
    message = ok ? "every joint released" : "a standby command could not be written";
    return ok;
}

bool Driver::moveOne(int a, double target, std::string &why) {
    std::lock_guard<std::mutex> lock(mutex_);
    return move(a, target, why);
}

bool Driver::move(int a, double target, std::string &why) {
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

void Driver::stepHome() {
    bool moving = false;
    for (int a = 0; a < AXES; ++a) {
        const double step = axes_[a].speed / poll_hz_, gap = axes_[a].home - position_[a];
        moving |= std::fabs(gap) > step;
        actuators_->command(axes_[a].device, static_cast<float>((std::fabs(gap) <= step ? axes_[a].home : position_[a] + std::copysign(step, gap))
                                                                * axes_[a].wire_per_unit));
    }
    homing_ = moving;
    if (!homing_) {
        log_(Level::INFO, "home reached");
    }
}

}  // namespace sep23
