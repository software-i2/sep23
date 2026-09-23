// Copyright by BeeX [2026]
#include <sep23/arm.h>

#include <urdf/model.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace sep23 {
namespace {

constexpr double kTolerance = 1e-6;

urdf::JointConstSharedPtr joint(const urdf::Model &model, const std::string &name) {
    urdf::JointConstSharedPtr found = model.getJoint(name);
    if (!found) {
        throw std::invalid_argument("the URDF has no joint " + name);
    }
    return found;
}

void requireAxis(const urdf::Joint &j, double x, double y, double z) {
    if (std::fabs(j.axis.x - x) > kTolerance || std::fabs(j.axis.y - y) > kTolerance || std::fabs(j.axis.z - z) > kTolerance) {
        throw std::invalid_argument("joint " + j.name + " does not turn about the axis the analytic model assumes");
    }
}

// Offset of a joint from its parent, which the planar model needs to lie in the x-z plane.
urdf::Vector3 planarOffset(const urdf::Joint &j) {
    const urdf::Vector3 &p = j.parent_to_joint_origin_transform.position;
    if (std::fabs(p.y) > kTolerance) {
        throw std::invalid_argument("joint " + j.name + " is offset out of the arm plane");
    }
    return p;
}

double yawOf(const urdf::Joint &j) {
    double r = 0.0, p = 0.0, y = 0.0;
    j.parent_to_joint_origin_transform.rotation.getRPY(r, p, y);
    if (std::fabs(r) > kTolerance || std::fabs(p) > kTolerance) {
        throw std::invalid_argument("joint " + j.name + " is tilted out of the arm plane");
    }
    return y;
}

double wrapToPi(double a) { return std::atan2(std::sin(a), std::cos(a)); }

}  // namespace

ArmGeometry geometryFromUrdf(const std::string &urdf_xml, const std::array<std::string, JOINT_COUNT> &names) {
    urdf::Model model;
    if (!model.initString(urdf_xml)) {
        throw std::invalid_argument("robot_description is not a valid URDF");
    }
    const urdf::JointConstSharedPtr base     = joint(model, names[BASE]);
    const urdf::JointConstSharedPtr shoulder = joint(model, names[SHOULDER]);
    const urdf::JointConstSharedPtr elbow    = joint(model, names[ELBOW]);
    const urdf::JointConstSharedPtr wrist    = joint(model, names[WRIST]);
    const urdf::JointConstSharedPtr mount    = joint(model, "ee_joint");
    requireAxis(*base, 0, 0, 1);
    requireAxis(*shoulder, 0, 1, 0);
    requireAxis(*elbow, 0, 1, 0);
    requireAxis(*wrist, 0, 0, -1);

    ArmGeometry g;
    g.base_axis_z = planarOffset(*base).z;
    g.shoulder_x  = planarOffset(*shoulder).x;
    g.shoulder_z  = planarOffset(*shoulder).z;
    g.upper_x     = planarOffset(*elbow).x;
    g.upper_z     = planarOffset(*elbow).z;
    g.fore_x      = planarOffset(*wrist).x;
    g.fore_z      = planarOffset(*wrist).z;
    g.wrist_to_mount = planarOffset(*mount).z;
    yawOf(*shoulder);
    const double elbow_yaw = std::fabs(wrapToPi(yawOf(*elbow)));
    if (elbow_yaw < kTolerance) {
        g.elbow_sign = 1.0;
    } else if (std::fabs(elbow_yaw - M_PI) < 1e-4) {
        g.elbow_sign = -1.0;
    } else {
        throw std::invalid_argument("the elbow frame must be turned 0 or 180 deg");
    }
    return g;
}

Arm::Arm(const ArmConfig &config, double blade_step) : config_(config) {
    if (!(blade_step > 0.0) || config_.jaw.blades.empty()) {
        throw std::invalid_argument("the blade profile is empty or the blade step is not positive");
    }
    for (int j = 0; j < JOINT_COUNT; ++j) {
        if (std::fabs(config_.sign[j]) != 1.0 || !(config_.min[j] < config_.max[j])) {
            throw std::invalid_argument(std::string("joint ") + JOINT_KEYS[j] + " needs a sign of 1 or -1 and min below max");
        }
        const double a = config_.sign[j] * (config_.min[j] - config_.offset[j]);
        const double b = config_.sign[j] * (config_.max[j] - config_.offset[j]);
        lower_[j]      = std::min(a, b);
        upper_[j]      = std::max(a, b);
    }

    // Blade samples in the jaw mount frame: x down the approach, y along the hinge, z along the closing line.
    const JawShape &jaw = config_.jaw;
    const double    c   = std::cos(jaw.open_width * jaw.blade_rotation_per_m);
    const double    s   = std::sin(jaw.open_width * jaw.blade_rotation_per_m);
    const auto      steps = [&](double span) { return std::max(1, static_cast<int>(std::ceil(span / blade_step))); };
    for (const double side : {1.0, -1.0}) {
        for (const BladeBand &band : jaw.blades) {
            const int nh = steps(2.0 * band.hinge_half_width);
            const int nc = steps(band.closing_max - band.closing_min);
            const int na = steps(band.approach_max - band.approach_min);
            for (int i = 0; i <= nh; ++i) {
                const double hinge = -band.hinge_half_width + 2.0 * band.hinge_half_width * i / nh;
                for (int k = 0; k <= nc; ++k) {
                    const double closing = band.closing_min + (band.closing_max - band.closing_min) * k / nc;
                    for (int m = 0; m <= na; ++m) {
                        const double approach = band.approach_min + (band.approach_max - band.approach_min) * m / na;
                        blade_local_.emplace_back(jaw.hinge_offset_approach - closing * s + approach * c, hinge,
                                                  side * (jaw.hinge_offset_closing + closing * c + approach * s));
                    }
                }
            }
        }
    }
}

Joints Arm::toModel(const Joints &reported) const {
    Joints out;
    for (int j = 0; j < JOINT_COUNT; ++j) {
        out[j] = config_.sign[j] * (reported[j] - config_.offset[j]);
    }
    return out;
}

Joints Arm::toReported(const Joints &model) const {
    Joints out;
    for (int j = 0; j < JOINT_COUNT; ++j) {
        out[j] = config_.offset[j] + config_.sign[j] * model[j];
    }
    return out;
}

bool Arm::withinLimits(const Joints &q) const {
    for (int j = 0; j < JOINT_COUNT; ++j) {
        if (q[j] < lower_[j] || q[j] > upper_[j]) {
            return false;
        }
    }
    return true;
}

void Arm::forearm(double along, double &length, double &angle) const {
    const double x = config_.geometry.elbow_sign * config_.geometry.fore_x;
    const double z = config_.geometry.fore_z + along;
    length         = std::hypot(x, z);
    angle          = std::atan2(x, z);
}

double Arm::reach(double along) const {
    const ArmGeometry &g = config_.geometry;
    double             fore_length = 0.0, fore_angle = 0.0;
    forearm(along, fore_length, fore_angle);
    return std::hypot(g.shoulder_x, g.base_axis_z + g.shoulder_z) + upperLength() + fore_length;
}

ArmPoints Arm::points(const Joints &q) const {
    const ArmGeometry &g = config_.geometry;
    double             fore_length = 0.0, fore_angle = 0.0;
    forearm(0.0, fore_length, fore_angle);

    const double upper_angle = upperAngle() + q[SHOULDER];
    const double wrist_axis  = g.elbow_sign * q[ELBOW] + q[SHOULDER];
    const double elbow_x     = g.shoulder_x + upperLength() * std::sin(upper_angle);
    const double elbow_z     = g.shoulder_z + upperLength() * std::cos(upper_angle);
    const double wrist_x     = elbow_x + fore_length * std::sin(fore_angle + wrist_axis);
    const double wrist_z     = elbow_z + fore_length * std::cos(fore_angle + wrist_axis);

    // A point in the arm plane (x out, z up) turned about the base axis.
    const auto place = [&](double x, double z) {
        return Eigen::Vector3d(x * std::cos(q[BASE]), x * std::sin(q[BASE]), z + g.base_axis_z);
    };
    const auto along = [&](double d) { return place(wrist_x + d * std::sin(wrist_axis), wrist_z + d * std::cos(wrist_axis)); };

    return {place(g.shoulder_x, g.shoulder_z), place(elbow_x, elbow_z), place(wrist_x, wrist_z), along(mountDistance()),
            along(mountDistance() + config_.jaw.mount_to_throat), along(tipDistance())};
}

JawAxes Arm::axes(const Joints &q) const {
    const double wrist_axis = config_.geometry.elbow_sign * q[ELBOW] + q[SHOULDER];
    const double sa = std::sin(wrist_axis), ca = std::cos(wrist_axis);
    const double roll = config_.jaw.hinge_roll_at_zero + q[WRIST];
    const double sr = std::sin(roll), cr = std::cos(roll);
    const double sb = std::sin(q[BASE]), cb = std::cos(q[BASE]);
    const auto   turn = [&](double x, double y, double z) { return Eigen::Vector3d(x * cb - y * sb, x * sb + y * cb, z); };
    return {turn(sa, 0.0, ca), turn(cr * ca, sr, -cr * sa), turn(-sr * ca, cr, sr * sa)};
}

void Arm::body(const Joints &q, Body &out, size_t blade_stride) const {
    const ArmPoints p = points(q);
    const JawAxes   a = axes(q);
    out.links         = {{{p.shoulder, p.elbow}, {p.elbow, p.wrist}, {p.wrist, p.mount},
                          {p.mount, p.mount + a.approach * config_.jaw.palm_length}}};
    out.throat        = p.throat;
    out.tip           = p.tip;
    const size_t step = std::max<size_t>(1, blade_stride);
    out.blades.clear();
    for (size_t i = 0; i < blade_local_.size(); i += step) {
        const Eigen::Vector3d &l = blade_local_[i];
        out.blades.push_back(p.mount + a.approach * l.x() + a.hinge * l.y() + a.closing * l.z());
    }
}

bool Arm::fitIntoLimits(int joint, double angle, double seed, double &out) const {
    bool   found   = false;
    double nearest = std::numeric_limits<double>::max();
    for (int turns = -1; turns <= 1; ++turns) {
        const double candidate = angle + 2.0 * M_PI * turns;
        if (candidate >= lower_[joint] && candidate <= upper_[joint] && std::fabs(candidate - seed) < nearest) {
            nearest = std::fabs(candidate - seed);
            out     = candidate;
            found   = true;
        }
    }
    return found;
}

Ik Arm::solve(const Eigen::Vector3d &target, double along, bool elbow_up, const Joints &seed, Joints &out) const {
    const ArmGeometry &g      = config_.geometry;
    const double       radial = std::hypot(target.x(), target.y());
    const double       base   = radial > 1e-9 ? std::atan2(target.y(), target.x()) : seed[BASE];
    if (!fitIntoLimits(BASE, base, seed[BASE], out[BASE])) {
        return Ik::JOINT_LIMIT;
    }

    double fore_length = 0.0, fore_angle = 0.0;
    forearm(along, fore_length, fore_angle);
    const double upper_length = upperLength();
    const double x            = radial - g.shoulder_x;
    const double z            = target.z() - g.base_axis_z - g.shoulder_z;
    const double distance     = std::hypot(x, z);
    if (distance > upper_length + fore_length + 1e-9 || distance < std::fabs(upper_length - fore_length) - 1e-9) {
        return Ik::OUT_OF_REACH;
    }

    const double cos_bend = std::max(-1.0, std::min(1.0, (distance * distance - upper_length * upper_length - fore_length * fore_length)
                                                             / (2.0 * upper_length * fore_length)));
    const double bend = elbow_up ? std::acos(cos_bend) : -std::acos(cos_bend);
    const double upper_direction =
            std::atan2(x, z) - std::atan2(fore_length * std::sin(bend), upper_length + fore_length * std::cos(bend));
    const double shoulder = upper_direction - upperAngle();
    const double elbow    = g.elbow_sign * (upper_direction + bend - fore_angle - shoulder);
    if (!fitIntoLimits(SHOULDER, shoulder, seed[SHOULDER], out[SHOULDER]) || !fitIntoLimits(ELBOW, elbow, seed[ELBOW], out[ELBOW])) {
        return Ik::JOINT_LIMIT;
    }
    out[WRIST] = seed[WRIST];
    return Ik::SOLVED;
}

bool Arm::rollsAcrossBar(const Joints &q, const Eigen::Vector3d &bar, double rolls[2]) const {
    if (bar.norm() < 1e-3) {
        return false;
    }
    Joints at_zero = q;
    at_zero[WRIST] = 0.0;
    const JawAxes a = axes(at_zero);
    // The hinge at roll r is hinge(0) cos r + closing(0) sin r; lay it along the bar.
    const double u = a.hinge.dot(bar.normalized());
    const double v = a.closing.dot(bar.normalized());
    if (std::hypot(u, v) < 1e-3) {
        return false;
    }
    rolls[0] = std::atan2(v, u);
    rolls[1] = wrapToPi(rolls[0] + M_PI);
    return true;
}

}  // namespace sep23
