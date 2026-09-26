// Copyright by BeeX [2026]
#pragma once

#include <functional>
#include <string>

namespace sep23 {

// driver and pick log through this; the wrapper decides where it goes.
enum class Level { INFO, WARN, ERROR };
using Log = std::function<void(Level, const std::string &)>;

}  // namespace sep23
