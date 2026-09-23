// Copyright by BeeX [2026]
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace sep23 {

// Reach BPL packet ids and modes, fixed by the vendor protocol.
namespace bpl {
constexpr uint8_t MODE = 0x01, VELOCITY = 0x02, POSITION = 0x03, REQUEST = 0x60;
constexpr uint8_t STANDBY_MODE = 0x00, VELOCITY_MODE = 0x03;

struct Packet {
    uint8_t              device = 0, id = 0;
    std::vector<uint8_t> data;
};

// COBS-encoded data, id, device, length and CRC8, then a zero byte.
std::vector<uint8_t> encode(uint8_t device, uint8_t id, const std::vector<uint8_t> &data);
std::vector<uint8_t> encode(uint8_t device, uint8_t id, float value);

// Buffers incoming bytes and returns every complete, valid packet.
class Reader {
public:
    std::vector<Packet> feed(const uint8_t *bytes, size_t count);

private:
    std::vector<uint8_t> buffer_;
};
}  // namespace bpl

// One actuator per device id, in wire units: radians for rotary joints, millimetres for the jaw.
class Actuators {
public:
    virtual ~Actuators() = default;
    virtual bool position(uint8_t device, float &value) = 0;
    virtual bool command(uint8_t device, float value)   = 0;
    virtual bool standby(uint8_t device)                = 0;
};

// The real arm over a serial port.
class SerialActuators : public Actuators {
public:
    SerialActuators(const std::string &port, int baud, double reply_timeout_s);
    ~SerialActuators() override;
    const std::string &error() const { return error_; }
    // Nudges one joint and waits for velocity mode; the arm sleeps until it sees traffic.
    bool wake(uint8_t device, float speed);

    bool position(uint8_t device, float &value) override;
    bool command(uint8_t device, float value) override;
    bool standby(uint8_t device) override;

private:
    bool send(const std::vector<uint8_t> &frame);
    bool request(uint8_t device, uint8_t field, bpl::Packet &reply);

    int         fd_ = -1;
    double      reply_timeout_s_;
    std::string error_;
    bpl::Reader reader_;
    std::mutex  mutex_;
};

// Joints that walk towards their targets at a fixed speed.
class SimulatedActuators : public Actuators {
public:
    struct Joint {
        uint8_t device;
        float   min, max, start, speed;
    };
    explicit SimulatedActuators(const std::vector<Joint> &joints);

    bool position(uint8_t device, float &value) override;
    bool command(uint8_t device, float value) override;
    bool standby(uint8_t device) override;

private:
    struct State {
        Joint joint;
        float position, target;
    };
    State *find(uint8_t device);
    void   advance();

    std::vector<State> joints_;
    double             last_s_;
    std::mutex         mutex_;
};

}  // namespace sep23
