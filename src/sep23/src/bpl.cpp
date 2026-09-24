// Copyright by BeeX [2026]
#include <sep23/bpl.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <termios.h>
#include <thread>
#include <unistd.h>

namespace sep23 {
namespace bpl {
namespace {

constexpr size_t kTrailer     = 4;     // id, device, length, crc
constexpr size_t kMaxBuffered = 4096;  // a stuck line never delivers a zero byte

uint8_t crc8(const uint8_t *bytes, size_t count) {
    static const std::array<uint8_t, 256> table = [] {
        std::array<uint8_t, 256> t{};
        for (int i = 0; i < 256; ++i) {
            uint8_t v = static_cast<uint8_t>(i);
            for (int bit = 0; bit < 8; ++bit) {
                v = (v & 1u) ? static_cast<uint8_t>((v >> 1) ^ 0xB2) : static_cast<uint8_t>(v >> 1);
            }
            t[i] = v;
        }
        return t;
    }();
    uint8_t crc = 0x00;
    for (size_t i = 0; i < count; ++i) {
        crc = table[crc ^ bytes[i]];
    }
    return crc ^ 0xFF;
}

std::vector<uint8_t> cobsEncode(const std::vector<uint8_t> &in) {
    std::vector<uint8_t> out{0x00};
    size_t               code_at = 0;
    uint8_t              code    = 1;
    for (const uint8_t b : in) {
        if (b != 0x00) {
            out.push_back(b);
            if (++code != 0xFF) {
                continue;
            }
        }
        out[code_at] = code;
        code_at      = out.size();
        out.push_back(0x00);
        code = 1;
    }
    out[code_at] = code;
    return out;
}

bool cobsDecode(const std::vector<uint8_t> &in, std::vector<uint8_t> &out) {
    out.clear();
    for (size_t i = 0; i < in.size();) {
        const uint8_t code = in[i++];
        if (code == 0x00) {
            return false;
        }
        for (uint8_t k = 1; k < code; ++k, ++i) {
            if (i >= in.size()) {
                return false;
            }
            out.push_back(in[i]);
        }
        if (code != 0xFF && i < in.size()) {
            out.push_back(0x00);
        }
    }
    return true;
}

bool parse(const std::vector<uint8_t> &frame, Packet &out) {
    std::vector<uint8_t> body;
    if (!cobsDecode(frame, body) || body.size() < kTrailer || body[body.size() - 2] != body.size()
        || crc8(body.data(), body.size() - 1) != body.back()) {
        return false;
    }
    out.id     = body[body.size() - 4];
    out.device = body[body.size() - 3];
    out.data.assign(body.begin(), body.end() - kTrailer);
    return true;
}

}  // namespace

std::vector<uint8_t> encode(uint8_t device, uint8_t id, const std::vector<uint8_t> &data) {
    std::vector<uint8_t> body = data;
    body.push_back(id);
    body.push_back(device);
    body.push_back(static_cast<uint8_t>(data.size() + kTrailer));
    body.push_back(crc8(body.data(), body.size()));
    std::vector<uint8_t> frame = cobsEncode(body);
    frame.push_back(0x00);
    return frame;
}

std::vector<uint8_t> encode(uint8_t device, uint8_t id, float value) {
    std::vector<uint8_t> bytes(sizeof(float));
    std::memcpy(bytes.data(), &value, sizeof(float));
    return encode(device, id, bytes);
}

std::vector<Packet> Reader::feed(const uint8_t *bytes, size_t count) {
    buffer_.insert(buffer_.end(), bytes, bytes + count);
    if (buffer_.size() > kMaxBuffered) {
        buffer_.erase(buffer_.begin(), buffer_.end() - kMaxBuffered);
    }
    std::vector<Packet> packets;
    for (auto end = std::find(buffer_.begin(), buffer_.end(), 0); end != buffer_.end();
         end      = std::find(buffer_.begin(), buffer_.end(), 0)) {
        const std::vector<uint8_t> frame(buffer_.begin(), end);
        buffer_.erase(buffer_.begin(), end + 1);
        Packet p;
        if (!frame.empty() && parse(frame, p)) {
            packets.push_back(p);
        }
    }
    return packets;
}

}  // namespace bpl

namespace {

bool baudConstant(int baud, speed_t &out) {
    switch (baud) {
    case 9600: out = B9600; return true;
    case 19200: out = B19200; return true;
    case 38400: out = B38400; return true;
    case 57600: out = B57600; return true;
    case 115200: out = B115200; return true;
    case 230400: out = B230400; return true;
    default: return false;
    }
}

double steadySeconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

SerialActuators::SerialActuators(const std::string &port, int baud, double reply_timeout_s) : reply_timeout_s_(reply_timeout_s) {
    speed_t speed = 0;
    if (!baudConstant(baud, speed)) {
        error_ = "unsupported baud rate " + std::to_string(baud);
        return;
    }
    fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    termios tty{};
    if (fd_ < 0 || tcgetattr(fd_, &tty) != 0) {
        error_ = "cannot open " + port + ": " + std::strerror(errno);
        return;
    }
    cfmakeraw(&tty);  // any line discipline would corrupt binary packets
    cfsetispeed(&tty, speed);
    cfsetospeed(&tty, speed);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cc[VMIN] = tty.c_cc[VTIME] = 0;
    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        error_ = std::string("cannot configure ") + port + ": " + std::strerror(errno);
        ::close(fd_);
        fd_ = -1;
        return;
    }
    tcflush(fd_, TCIOFLUSH);
}

SerialActuators::~SerialActuators() {
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

bool SerialActuators::send(const std::vector<uint8_t> &frame) {
    return fd_ >= 0 && ::write(fd_, frame.data(), frame.size()) == static_cast<ssize_t>(frame.size());
}

bool SerialActuators::request(uint8_t device, uint8_t field, bpl::Packet &reply) {
    if (!send(bpl::encode(device, bpl::REQUEST, std::vector<uint8_t>{field}))) {
        return false;
    }
    uint8_t      chunk[256];
    const double deadline = steadySeconds() + reply_timeout_s_;
    while (steadySeconds() < deadline) {
        const ssize_t got = ::read(fd_, chunk, sizeof(chunk));
        if (got <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        for (const bpl::Packet &p : reader_.feed(chunk, static_cast<size_t>(got))) {
            if (p.device == device && p.id == field) {
                reply = p;
                return true;
            }
        }
    }
    return false;
}

bool SerialActuators::wake(uint8_t device, float speed) {
    std::lock_guard<std::mutex> lock(mutex_);
    bpl::Packet                 reply;
    return send(bpl::encode(device, bpl::VELOCITY, speed)) && request(device, bpl::MODE, reply) && !reply.data.empty()
           && reply.data[0] == bpl::VELOCITY_MODE;
}

bool SerialActuators::position(uint8_t device, float &value) {
    std::lock_guard<std::mutex> lock(mutex_);
    bpl::Packet                 reply;
    if (!request(device, bpl::POSITION, reply) || reply.data.size() < sizeof(float)) {
        return false;
    }
    std::memcpy(&value, reply.data.data(), sizeof(float));
    return true;
}

bool SerialActuators::command(uint8_t device, float value) {
    std::lock_guard<std::mutex> lock(mutex_);
    return send(bpl::encode(device, bpl::POSITION, value));
}

bool SerialActuators::standby(uint8_t device) {
    std::lock_guard<std::mutex> lock(mutex_);
    return send(bpl::encode(device, bpl::MODE, std::vector<uint8_t>{bpl::STANDBY_MODE}));
}

SimulatedActuators::SimulatedActuators(const std::vector<Joint> &joints) : last_s_(steadySeconds()) {
    for (const Joint &j : joints) {
        joints_.push_back({j, j.start, j.start});
    }
}

SimulatedActuators::State *SimulatedActuators::find(uint8_t device) {
    for (State &s : joints_) {
        if (s.joint.device == device) {
            return &s;
        }
    }
    return nullptr;
}

void SimulatedActuators::advance() {
    const double now = steadySeconds();
    const double dt  = std::min(now - last_s_, 1.0);  // a host stall does not teleport the joints
    last_s_          = now;
    for (State &s : joints_) {
        const float step = static_cast<float>(s.joint.speed * dt), error = s.target - s.position;
        s.position += std::fabs(error) <= step ? error : std::copysign(step, error);
    }
}

bool SimulatedActuators::position(uint8_t device, float &value) {
    std::lock_guard<std::mutex> lock(mutex_);
    advance();
    State *s = find(device);
    if (s) {
        value = s->position;
    }
    return s != nullptr;
}

bool SimulatedActuators::command(uint8_t device, float value) {
    std::lock_guard<std::mutex> lock(mutex_);
    advance();
    State *s = find(device);
    if (s) {
        s->target = std::min(std::max(value, s->joint.min), s->joint.max);
    }
    return s != nullptr;
}

bool SimulatedActuators::place(uint8_t device, float value) {
    std::lock_guard<std::mutex> lock(mutex_);
    advance();
    State *s = find(device);
    if (s) {
        s->position = s->target = std::min(std::max(value, s->joint.min), s->joint.max);
    }
    return s != nullptr;
}

bool SimulatedActuators::standby(uint8_t device) {
    std::lock_guard<std::mutex> lock(mutex_);
    advance();
    State *s = find(device);
    if (s) {
        s->target = s->position;
    }
    return s != nullptr;
}

}  // namespace sep23
