// Copyright by BeeX [2026]
#pragma once

namespace sep23 {

// COLLECT and PROCESS run twice: once to choose where to park, once to choose what to grab from there.
enum class State {
    READY,      // waiting for start
    STREAM,     // waiting for the camera to produce
    COLLECT,    // gathering a window of frames
    PROCESS,    // turning them into an obstacle map and candidates
    PICKSPOT,   // searching for somewhere to park
    GOTOSPOT,   // driving there
    RESURVEY,   // nothing in view to park for; survey again inside the budget
    PICKGRASP,  // planning a path to a handle
    GOTOGRASP,  // following it
    CLOSEJAW,   // closing on the handle and reading what was caught
    RETARGET,   // nothing to grab from this spot; look again
    REPARK,     // out of looks; park somewhere else
    SUCCESS,
    FAIL,
    ESTOP       // contact or a stop request; the arm is released
};

enum class Event {
    NONE,
    START,
    STOP,
    STREAMING,
    NO_STREAM,
    FRAMES_IN,
    CANDIDATES_SPOT,
    CANDIDATES_GRASP,
    NO_CANDIDATES_SPOT,
    NO_CANDIDATES_GRASP,
    SURVEY_AGAIN,
    OUT_OF_TIME,
    SPOT_CHOSEN,
    SPOT_UNCHANGED,  // staying is best, so nothing drives
    NO_SPOT,
    ARRIVED,
    PLAN_FOUND,
    NO_PLAN,
    REACHED,
    STALLED,         // out of arrival time without contact
    JAW_SETTLED,     // empty is recorded, not a failure
    LOOK_AGAIN,
    OUT_OF_LOOKS,
    PARK_AGAIN,
    OUT_OF_PARKS,
    COLLIDED,
    FAILURE
};

const char *stateName(State s);
bool        isWorking(State s);  // part way through a pick
State       nextState(State s, Event e);  // `s` itself when the event does not apply

}  // namespace sep23
