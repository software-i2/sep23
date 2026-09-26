// Copyright by BeeX [2026]
// Helpers shared by sanity.cpp and runtime.cpp.
#pragma once

#include <sep23/arm.h>

#include <cstdio>
#include <memory>
#include <string>

namespace sep23 {

const std::array<std::string, JOINT_COUNT> kNames = {"axis_e", "axis_d", "axis_c", "axis_b"};

// The robot URDF with config/robot.yaml filled in, as the launch file builds it.
inline std::string expandedUrdf() {
    const std::string command = "xacro " SEP23_DIR "/urdf/robot.urdf.xacro robot_yaml:=" SEP23_DIR "/config/robot.yaml";
    std::unique_ptr<FILE, int (*)(FILE *)> pipe(popen(command.c_str(), "r"), pclose);
    std::string                            out;
    char                                   buffer[4096];
    while (pipe && fgets(buffer, sizeof(buffer), pipe.get())) {
        out += buffer;
    }
    return out;
}

}  // namespace sep23
