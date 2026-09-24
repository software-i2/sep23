#!/usr/bin/env python3
"""Synthetic arm occlusion for Stage 3: republishes /image with an obstacle painted in, and /pointcloud with it cut out.

Run inside the sep23 container, next to a pick or a bag:  python3 exp/occluder.py _mode:=2
Outputs, headers kept:
  /occluder/image, /occluder/pointcloud  the camera with the arm in the way, the feed pick.launch occluded:=true reads;
                                         the cloud stays organised, hidden points NaN
  /non_occluded/pointcloud               only the points still in view, packed: shrinks as the arm covers the mine
  /occluded/pointcloud                   only the points behind the arm, packed: grows as it covers the mine
  /occluder/coverage                     the covered fraction of the frame
Control: /occluder/enable (std_msgs/Bool), /occluder/mode (std_msgs/Int32, 0-4). Starts OFF: clean pass-through.
With _follow_pick:=true it switches itself ON when /pick/state reaches GOTOGRASP (the arm comes into view) and holds it
after the pick ends; it clears when the pick looks again (a new start, as soak.sh does, or a re-survey) or the mode changes.

Modes: 0 clear | 1 flat, linear 0-50% | 2 textured, jittered 0-50% | 3 flat, linear 0-100% | 4 textured, jittered 0-100%.
Coverage is of the whole frame; the obstacle enters from the right with a tilted leading edge.
"""
import numpy as np

RANGES = {1: (0.0, 0.5), 2: (0.0, 0.5), 3: (0.0, 1.0), 4: (0.0, 1.0)}  # each ramps up over sweep_s, then holds
TEXTURED = {2, 4}
FLAT_BGR = (70, 70, 70)


def coverage(mode, t, sweep_s):
    """Target fraction of the frame covered t seconds after the obstacle was switched on."""
    if mode not in RANGES:
        return 0.0
    lo, hi = RANGES[mode]
    s = min(t / sweep_s, 1.0)
    if mode in TEXTURED:  # a slow sway plus a choppy step, like an arm under load
        s += 0.08 * np.sin(2 * np.pi * 1.3 * t) + 0.04 * np.sign(np.sin(2 * np.pi * 3.1 * t))
    return lo + (hi - lo) * float(np.clip(s, 0.0, 1.0))


def obstacle_mask(shape, cover, tilt_deg):
    """Pixels right of a tilted edge, placed by bisection so the covered fraction of the frame is `cover`."""
    h, w = shape
    tilt = np.tan(np.radians(tilt_deg)) * (np.arange(h) - h / 2)  # the edge's offset on each row
    lo, hi = -np.abs(tilt).max() - 1, w + np.abs(tilt).max() + 1   # edge positions covering everything, nothing
    for _ in range(40):
        mid = (lo + hi) / 2
        if np.clip(w - mid - tilt, 0, w).sum() > cover * h * w:
            lo = mid
        else:
            hi = mid
    return np.arange(w)[None, :] >= (hi + tilt)[:, None]


def texture(shape, seed=7):
    """High-contrast stripes, checks and speckle: plenty of strong corners for KLT to latch onto wrongly."""
    h, w = shape
    v, u = np.mgrid[0:h, 0:w]
    stripes = ((u + v) // 12) % 2
    checks = ((u // 24) + (v // 24)) % 2
    speckle = np.random.default_rng(seed).random((h, w)) > 0.8
    gray = (255 * (stripes ^ checks ^ speckle)).astype(np.uint8)
    return np.dstack([gray, 255 - gray, gray])


def paint(bgr, mask, pattern, shift):
    """Fills the mask with the flat arm colour, or with the pattern slid by `shift` px so it moves with the obstacle."""
    out = bgr.copy()
    out[mask] = FLAT_BGR if pattern is None else np.roll(pattern, -int(shift), axis=1)[mask]
    return out


def hide(xyz, mask):
    """Drops the points behind the obstacle: the cloud keeps only what the camera can still see."""
    out = xyz.copy()
    out[mask] = np.nan
    return out


def pack(cloud_data, point_step, keep):
    """The rows of a PointCloud2 buffer where `keep` is set, packed densely: something to look at, with no NaN holes."""
    rows = np.frombuffer(cloud_data, np.uint8).reshape(-1, point_step)
    return rows[keep.ravel()].tobytes(), int(keep.sum())


def xyz_view(cloud_data, fields, point_step, count):
    """A writable (count, 3) float32 view onto x, y, z of a PointCloud2 buffer, whatever else each point carries."""
    offsets = {f.name: f.offset for f in fields}
    dtype = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3,
                      "offsets": [offsets["x"], offsets["y"], offsets["z"]], "itemsize": point_step})
    return np.frombuffer(cloud_data, dtype=dtype, count=count)


def run():
    import copy
    import time
    import message_filters
    import rosgraph
    import rospy
    from sensor_msgs.msg import Image, PointCloud2
    from std_msgs.msg import Bool, Float32, Int32

    channels = {"bgr8": 3, "rgb8": 3, "mono8": 1}
    while not rosgraph.is_master_online():  # started alongside roslaunch: wait rather than crash on the first param read
        time.sleep(0.5)
    rospy.init_node("occluder")
    state = {"mode": int(rospy.get_param("~mode", 1)), "on": bool(rospy.get_param("~enabled", False)),
             "since": rospy.get_time(), "pattern": None, "paired": rospy.get_time(), "seen": 0.0}
    sweep_s, tilt = rospy.get_param("~sweep_s", 3.0), rospy.get_param("~tilt_deg", 20.0)
    pub_image = rospy.Publisher("/occluder/image", Image, queue_size=2)
    pub_cloud = rospy.Publisher("/occluder/pointcloud", PointCloud2, queue_size=2)
    pub_cover = rospy.Publisher("/occluder/coverage", Float32, queue_size=2)
    pub_visible = rospy.Publisher("/non_occluded/pointcloud", PointCloud2, queue_size=2)
    pub_hidden = rospy.Publisher("/occluded/pointcloud", PointCloud2, queue_size=2)

    def subset(cloud, keep):
        out = copy.copy(cloud)
        out.data, out.width = pack(cloud.data, cloud.point_step, keep)
        out.height, out.row_step, out.is_dense = 1, out.width * cloud.point_step, True
        return out

    def rearm(on=None, mode=None):
        """Any switch restarts the sweep from the start; OFF is clean pass-through."""
        state["on"] = state["on"] if on is None else on
        state["mode"] = state["mode"] if mode is None else mode
        state["since"] = rospy.get_time()
        rospy.loginfo("[occluder] %s, mode %d", "ON" if state["on"] else "OFF", state["mode"])

    follow_pick = rospy.get_param("~follow_pick", False)

    def on_mode(msg):
        if msg.data in (0, 1, 2, 3, 4):
            rearm(on=False if follow_pick else None, mode=msg.data)  # following the pick, only the arm's motion turns it on
        else:
            rospy.logwarn("[occluder] mode %d is not 0-4", msg.data)

    def occlude(image, cloud):
        h, w, ch = image.height, image.width, channels[image.encoding]  # KeyError: an encoding we do not handle
        if cloud.width * cloud.height != h * w:
            raise ValueError(f"cloud has {cloud.width * cloud.height} points for a {w}x{h} image")
        cover = coverage(state["mode"], rospy.get_time() - state["since"], sweep_s)
        mask = obstacle_mask((h, w), cover, tilt)
        pixels = np.frombuffer(image.data, np.uint8).reshape(h, image.step)[:, :w * ch].reshape(h, w, ch)
        bgr = pixels[..., ::-1] if image.encoding == "rgb8" else np.repeat(pixels, 3, 2) if ch == 1 else pixels
        if state["mode"] in TEXTURED and (state["pattern"] is None or state["pattern"].shape[:2] != (h, w)):
            state["pattern"] = texture((h, w))
        painted = paint(bgr, mask, state["pattern"] if state["mode"] in TEXTURED else None, (1.0 - cover) * w)
        painted = painted[..., ::-1] if image.encoding == "rgb8" else painted[..., :1] if ch == 1 else painted
        out_image = copy.copy(image)
        out_image.step, out_image.data = w * ch, np.ascontiguousarray(painted).tobytes()

        buffer = bytearray(cloud.data)
        points = xyz_view(buffer, cloud.fields, cloud.point_step, h * w)
        xyz = np.stack([points["x"], points["y"], points["z"]], -1).reshape(h, w, 3)
        seen = hide(xyz, mask).reshape(-1, 3)
        points["x"], points["y"], points["z"] = seen[:, 0], seen[:, 1], seen[:, 2]
        out_cloud = copy.copy(cloud)
        out_cloud.data = bytes(buffer)
        return out_image, out_cloud, subset(cloud, ~mask), subset(cloud, mask), float(mask.mean())

    def on_pair(image, cloud):
        state["paired"] = rospy.get_time()
        nothing = np.zeros(cloud.width * cloud.height, bool)
        out_image, out_cloud, visible, hidden, covered = image, cloud, cloud, subset(cloud, nothing), 0.0
        if state["on"] and state["mode"] != 0:
            try:
                out_image, out_cloud, visible, hidden, covered = occlude(image, cloud)
            except (KeyError, ValueError) as e:  # never stall the pipeline: pass the frame through and say why
                rospy.logwarn_throttle(5.0, f"[occluder] passing through untouched: {type(e).__name__} {e}")
        pub_image.publish(out_image)
        pub_cloud.publish(out_cloud)
        pub_cover.publish(covered)
        pub_visible.publish(visible)
        pub_hidden.publish(hidden)

    def watchdog(_):
        now = rospy.get_time()
        if now - state["seen"] < 1.0 and now - state["paired"] > 2.0:
            rospy.logwarn_throttle(5.0, "[occluder] image and cloud arrive but never share a stamp; nothing is published")

    subs = [message_filters.Subscriber("/image", Image, queue_size=2, buff_size=2 ** 24),
            message_filters.Subscriber("/pointcloud", PointCloud2, queue_size=2, buff_size=2 ** 24)]
    for s in subs:
        s.registerCallback(lambda _: state.__setitem__("seen", rospy.get_time()))
    message_filters.TimeSynchronizer(subs, 10).registerCallback(on_pair)
    rospy.Subscriber("/occluder/enable", Bool, lambda m: rearm(on=m.data))
    rospy.Subscriber("/occluder/mode", Int32, on_mode)
    rospy.Timer(rospy.Duration(1.0), watchdog)
    if follow_pick:
        from sep23.msg import PickState  # needs the sep23 workspace sourced

        def on_pick(msg):
            if msg.state == "GOTOGRASP" and not state["on"]:
                rearm(on=True)
            elif msg.state in ("STREAM", "COLLECT") and state["on"]:  # the pick is looking again: give it clean frames
                rearm(on=False)

        rospy.Subscriber("/pick/state", PickState, on_pick)
    rearm()
    rospy.spin()


if __name__ == "__main__":
    run()
