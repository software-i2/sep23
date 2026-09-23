// Copyright by BeeX [2026]
#include <sep23/fsm.h>

namespace sep23 {

const char *stateName(State s) {
    static const char *const kNames[] = {"READY",     "STREAM",    "COLLECT",  "PROCESS", "PICKSPOT",
                                         "GOTOSPOT",  "RESURVEY",  "PICKGRASP", "GOTOGRASP", "CLOSEJAW",
                                         "RETARGET",  "REPARK",    "SUCCESS",  "FAIL",    "ESTOP"};
    return kNames[static_cast<int>(s)];
}

bool isWorking(State s) {
    return s != State::READY && s != State::SUCCESS && s != State::FAIL && s != State::ESTOP;
}

State nextState(State s, Event e) {
    // Contact and a stop both release the arm; leaving takes another start.
    if (e == Event::COLLIDED || e == Event::STOP) {
        return isWorking(s) ? State::ESTOP : s;
    }
    if (e == Event::FAILURE) {
        return isWorking(s) ? State::FAIL : s;
    }
    if (e == Event::START) {
        return isWorking(s) ? s : State::STREAM;
    }
    switch (s) {
    case State::STREAM:
        return e == Event::STREAMING ? State::COLLECT : e == Event::NO_STREAM ? State::FAIL : s;
    case State::COLLECT:
        return e == Event::FRAMES_IN ? State::PROCESS : e == Event::NO_CANDIDATES_SPOT ? State::RESURVEY
                                                       : e == Event::NO_CANDIDATES_GRASP ? State::RETARGET : s;
    case State::PROCESS:
        switch (e) {
        case Event::CANDIDATES_SPOT: return State::PICKSPOT;
        case Event::CANDIDATES_GRASP: return State::PICKGRASP;
        case Event::NO_CANDIDATES_SPOT: return State::RESURVEY;  // nothing moved or was decided, so this loops freely
        case Event::NO_CANDIDATES_GRASP: return State::RETARGET;
        default: return s;
        }
    case State::RESURVEY:
        return e == Event::SURVEY_AGAIN ? State::COLLECT : e == Event::OUT_OF_TIME ? State::FAIL : s;
    case State::PICKSPOT:
        // Staying still still gets a fresh look: the grasp is planned from frames taken after the decision.
        return e == Event::SPOT_CHOSEN ? State::GOTOSPOT : e == Event::SPOT_UNCHANGED ? State::COLLECT
                                                         : e == Event::NO_SPOT ? State::REPARK : s;
    case State::GOTOSPOT:
        return e == Event::ARRIVED ? State::COLLECT : s;
    case State::PICKGRASP:
        return e == Event::PLAN_FOUND ? State::GOTOGRASP : e == Event::NO_PLAN ? State::RETARGET : s;
    case State::GOTOGRASP:
        return e == Event::REACHED ? State::CLOSEJAW : e == Event::STALLED ? State::RETARGET : s;
    case State::CLOSEJAW:
        return e == Event::JAW_SETTLED ? State::SUCCESS : s;
    case State::RETARGET:
        return e == Event::LOOK_AGAIN ? State::COLLECT : e == Event::OUT_OF_LOOKS ? State::REPARK : s;
    case State::REPARK:
        return e == Event::PARK_AGAIN ? State::COLLECT : e == Event::OUT_OF_PARKS ? State::FAIL : s;
    default:
        return s;
    }
}

}  // namespace sep23
