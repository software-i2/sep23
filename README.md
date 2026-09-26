# sep23

Handle grasping for a Reach Alpha 5 on a parked vehicle, ROS Noetic. One `/pick/start` surveys, parks, plans, drives the arm blind while tracking the target in the image (steering the goal with it), and closes the jaw.

## Setup

```bash
git clone --recursive https://github.com/software-i2/sep23.git && cd sep23
scripts/build.sh sep23 --no-deps   # always build here, not inside the container
```

Put `captures_grasp_poses.bag` in `data/`.

## Run

```bash
scripts/shell.sh                   # shell 1
roslaunch sep23 pick.launch sim:=true vision:=true bag:="$PWD/data/captures_grasp_poses.bag"   # simulated arm, bag on a loop
scripts/shell.sh                   # shell 2
scripts/soak.sh                    # 600 s of picks with steering; scripts/soak.sh 600 false without
```

Add-ons from `src/sep23/exp/`, off unless asked for:
- `occluded:=true cover:=100`: an obstacle painted over the camera while the arm moves blind, covering up to `cover` percent
- `synthetic:=true amplitude_m:=0.02`: SYNTHETIC camera, the bag's first frame swaying along a known path instead of the replay; `YAW_DEG=180 scripts/soak.sh` turns it a random yaw each attempt

Real arm: `roslaunch sep23 pick.launch vision:=true`.

Foxglove: `ws://localhost:8765`, display frame `world_locked`.
- `/tracker/debug_image`: the tracker's labels (what each frame decided, where the arm aims, what steering did); `rostopic pub /tracker/view std_msgs/String klt` (or `ransac`, `mask`) switches the view
- `/occluder/image`: the pick's raw camera, unlabelled on purpose (text would give the tracker corners to follow)
- `/non_occluded/pointcloud`: what the camera still sees (the pick's cloud); `/occluded/pointcloud`: what the obstacle hides

`soak.sh` prints one line per attempt; `on the handle` is the pass/fail for the blind motion.

## Config

`src/sep23/config/robot.yaml` is the hardware, `pick.yaml` the pick; `track:` holds the tracker and steering. Both are read strictly: a missing or unused key stops the node and lists the problem.

## Tests

```bash
scripts/build.sh sep23 --no-deps --catkin-make-args run_tests
```
