#!/usr/bin/env python3
"""Paints a textured obstacle into /synthetic/image while the pick is in GOTOGRASP and splits /synthetic/pointcloud by it.
Publishes /occluder/image (the pick's camera), /non_occluded/pointcloud (what the camera still sees: the pick's cloud) and
/occluded/pointcloud (what the obstacle hides). Both clouds stay organised, NaN where the other has a point.
_cover:=100 is the most of the frame it covers, in percent."""
import copy

import numpy as np

SWEEP_S = 8.0  # time to ramp up to full cover


def roll(rng):
    """One switch-on's dice: the side the obstacle comes from and how it wanders. Waves are (amplitude, Hz, phase)."""
    terms = lambda n, amp, freq: [(rng.uniform(*amp), rng.uniform(*freq), rng.uniform(0, 2 * np.pi)) for _ in range(n)]
    return {"angle": rng.uniform(0.0, 360.0), "pace": rng.uniform(0.6, 1.4), "sway": terms(3, (0.02, 0.08), (0.2, 2.5)),
            "drift": terms(2, (10, 45), (0.05, 0.3)), "edge": terms(2, (0.01, 0.05), (0.5, 3))}


def waves(terms, x):
    return sum(a * np.sin(2 * np.pi * f * x + p) for a, f, p in terms)


def coverage(cover, t, dice):
    """Fraction of the frame covered t s after switch-on: ramps up to `cover`, then holds, wobbling."""
    return cover * float(np.clip(min(t / (SWEEP_S * dice["pace"]), 1.0) + waves(dice["sway"], t), 0.0, 1.0))


def obstacle_mask(shape, cover, angle_deg, edge=()):
    """The `cover` fraction of the frame lying furthest towards angle_deg, behind a ragged edge.
    Also returns (dy, dx), how far the edge has come in, for sliding the texture along with it."""
    h, w = shape
    step = 4  # ponytail: scored on 4 px blocks to keep up with the camera (~1 ms, not ~20); per pixel if the steps show
    v, u = np.mgrid[0:h:step, 0:w:step] + (step - 1) / 2 - np.array([(h - 1) / 2, (w - 1) / 2])[:, None, None]
    c, s, diag = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg)), np.hypot(h, w)
    score = u * c + v * s + diag * waves(edge, (v * c - u * s) / diag)
    k = int(round(cover * score.size))
    if k == 0:
        return np.zeros(shape, bool), (0, 0)
    edge_at = np.partition(score.ravel(), score.size - k)[score.size - k]
    mask = np.repeat(np.repeat(score >= edge_at, step, 0), step, 1)[:h, :w]
    return mask, (int(round(edge_at * s)), int(round(edge_at * c)))


def texture(shape, seed=7):
    """Stripes, checks and speckle: strong corners for KLT to latch onto wrongly."""
    h, w = shape
    v, u = np.mgrid[0:h, 0:w]
    speckle = np.random.default_rng(seed).random((h, w)) > 0.8
    gray = (255 * ((((u + v) // 12) % 2) ^ (((u // 24) + (v // 24)) % 2) ^ speckle)).astype(np.uint8)
    return np.dstack([gray, 255 - gray, gray])


def paint(bgr, mask, pattern, shift):
    out = bgr.copy()
    out[mask] = np.roll(pattern, shift, axis=(0, 1))[mask]
    return out


def xyz_view(cloud_data, fields, point_step, count):
    """A writable (count,) view onto x, y, z of a PointCloud2 buffer, whatever else each point carries."""
    offsets = {f.name: f.offset for f in fields}
    dtype = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3,
                      "offsets": [offsets["x"], offsets["y"], offsets["z"]], "itemsize": point_step})
    return np.frombuffer(cloud_data, dtype=dtype, count=count)


def split(cloud, hidden):
    """The cloud twice: what the camera still sees (hidden pixels NaN) and what is hidden (every other pixel NaN)."""
    out = []
    for drop in (hidden.ravel(), ~hidden.ravel()):
        buffer = bytearray(cloud.data)
        xyz = xyz_view(buffer, cloud.fields, cloud.point_step, cloud.width * cloud.height)
        for axis in "xyz":
            xyz[axis][drop] = np.nan
        part = copy.copy(cloud)
        part.data, part.is_dense = bytes(buffer), False
        out.append(part)
    return out


def run():
    import time
    import message_filters
    import rosgraph
    import rospy
    from sensor_msgs.msg import Image, PointCloud2
    from sep23.msg import PickState

    while not rosgraph.is_master_online():
        time.sleep(0.5)
    rospy.init_node("occluder")
    cover = min(max(rospy.get_param("~cover", 100), 0), 100) / 100.0
    rng = np.random.default_rng()
    state = {"on": False, "since": 0.0, "frozen": None, "dice": roll(rng), "pattern": None}
    pub_image = rospy.Publisher("/occluder/image", Image, queue_size=2)
    pub_seen = rospy.Publisher("/non_occluded/pointcloud", PointCloud2, queue_size=2)
    pub_hidden = rospy.Publisher("/occluded/pointcloud", PointCloud2, queue_size=2)

    def on_pick(msg):
        if msg.state == "GOTOGRASP" and not state["on"]:
            state.update(on=True, since=rospy.get_time(), frozen=None, dice=roll(rng))
            rospy.loginfo("[occluder] ON, from %.0f deg", state["dice"]["angle"])
        elif msg.state == "CLOSEJAW" and state["on"] and state["frozen"] is None:  # the arm stops: so does the sweep
            state["frozen"] = rospy.get_time() - state["since"]
        elif msg.state in ("STREAM", "COLLECT"):  # the pick looks again: clean frames
            state["on"] = False

    def publish(image, cloud, hidden):
        seen, behind = split(cloud, hidden)
        pub_image.publish(image)
        pub_seen.publish(seen)
        pub_hidden.publish(behind)

    def on_pair(image, cloud):
        if not state["on"]:
            publish(image, cloud, np.zeros(cloud.width * cloud.height, bool))
            return
        h, w, dice = image.height, image.width, state["dice"]
        t = rospy.get_time() - state["since"] if state["frozen"] is None else state["frozen"]
        mask, shift = obstacle_mask((h, w), coverage(cover, t, dice), dice["angle"] + waves(dice["drift"], t), dice["edge"])
        if state["pattern"] is None or state["pattern"].shape[:2] != (h, w):
            state["pattern"] = texture((h, w))
        bgr = np.frombuffer(image.data, np.uint8).reshape(h, image.step)[:, :w * 3].reshape(h, w, 3)
        out_image = copy.copy(image)
        out_image.step, out_image.data = w * 3, paint(bgr, mask, state["pattern"], shift).tobytes()
        publish(out_image, cloud, mask)

    subs = [message_filters.Subscriber("/synthetic/image", Image, queue_size=2, buff_size=2 ** 24),
            message_filters.Subscriber("/synthetic/pointcloud", PointCloud2, queue_size=2, buff_size=2 ** 24)]
    message_filters.TimeSynchronizer(subs, 10).registerCallback(on_pair)
    rospy.Subscriber("/pick/state", PickState, on_pick)
    rospy.spin()


if __name__ == "__main__":
    run()
