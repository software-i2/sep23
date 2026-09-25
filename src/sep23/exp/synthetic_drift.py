#!/usr/bin/env python3
"""SYNTHETIC camera, not real data: one bag frame swaying rigidly along a known path in camera_link.
Publishes /synthetic/image, pointcloud and grasp_poses. /synthetic/yaw_deg (read every frame) turns the scene about the
camera's axis through the handle: the camera looks straight down, so that is the mine turned on the seabed."""
import numpy as np


def offset(t, amplitude, period, phases):
    """Where the scene has swayed to at t s, camera_link metres from the bag frame: x right, y up, z towards the camera at half."""
    periods = period * np.array([1.0, 1.3, 1.7])
    return amplitude * np.array([1.0, 1.0, 0.5]) * np.sin(2 * np.pi * t / periods + phases)


def place(points, yaw, pivot, shift):
    """Points turned `yaw` rad about camera z through `pivot`, then moved by `shift`. Depth only changes by the shift."""
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = points[:, 0] - pivot[0], points[:, 1] - pivot[1]
    return np.stack([pivot[0] + c * x - s * y, pivot[1] + s * x + c * y, points[:, 2]], -1) + shift


def render(placed, nearest_first, camera):
    """The frame's placed points seen again: per pixel, the source pixel now nearest the camera there, -1 where nothing
    lands. `nearest_first` lists the seen source pixels by depth; a shift and a turn about camera z keep that order."""
    h, w, fx, fy, cx, cy = camera
    p = placed[nearest_first]
    u, v = np.rint(cx + fx * p[:, 0] / -p[:, 2]), np.rint(cy - fy * p[:, 1] / -p[:, 2])
    inside = (p[:, 2] < 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    pixel = (v[inside] * w + u[inside]).astype(np.int64)
    landed, first = np.unique(pixel, return_index=True)  # the nearest point wins each pixel
    src = np.full(h * w, -1, np.int64)
    src[landed] = nearest_first[inside][first]
    # A surface that came closer covers more pixels than it has points: fill those one-pixel cracks from a neighbour.
    src = src.reshape(h, w)
    for _ in range(2):
        for axis in (0, 1):
            for step in (1, -1):
                near = np.full_like(src, -1)
                if axis == 0:
                    near[max(step, 0):h + min(step, 0)] = src[max(-step, 0):h + min(-step, 0)]
                else:
                    near[:, max(step, 0):w + min(step, 0)] = src[:, max(-step, 0):w + min(-step, 0)]
                hole = src < 0
                src[hole] = near[hole]
    return src.ravel()


def run():
    import copy

    import rosbag
    import rospy
    from occluder import xyz_view

    rospy.init_node("synthetic_drift")
    from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion
    from tf.transformations import quaternion_about_axis, quaternion_multiply
    from sensor_msgs.msg import Image, PointCloud2

    k, rate_hz, period = 0, 10.0, 20.0  # bag frame, publish rate, sway period
    amplitude = rospy.get_param("~amplitude_m", 0.02)
    phases = np.random.default_rng(1).uniform(0, 2 * np.pi, 3)
    i = rospy.get_param("/robot/camera/intrinsics")
    camera = (i["height_px"], i["width_px"], i["fx_px"], i["fy_px"], i["cx_px"], i["cy_px"])

    seen = {}
    path = rospy.get_param("~bag", "")
    if not path:
        raise SystemExit("[synthetic] synthetic:=true needs bag:=<bag to take the frame from>")
    with rosbag.Bag(path) as bag:
        for topic, msg, _ in bag.read_messages(topics=["/image", "/pointcloud", "/grasp_poses"]):
            seen.setdefault(topic, []).append(msg)
    image, cloud, poses = (seen[t][k] for t in ("/image", "/pointcloud", "/grasp_poses"))
    if image.encoding != "bgr8" or not image.header.stamp == cloud.header.stamp == poses.header.stamp:
        raise SystemExit(f"[synthetic] frame {k} is not a bgr8 image, cloud and poses sharing one stamp")
    n = cloud.width * cloud.height
    rows = np.frombuffer(cloud.data, np.uint8).reshape(n, cloud.point_step)
    xyz = xyz_view(bytearray(cloud.data), cloud.fields, cloud.point_step, n)
    points = np.stack([xyz["x"], xyz["y"], xyz["z"]], -1).astype(np.float64)
    depth = -points[:, 2]
    seen_px = np.flatnonzero(np.isfinite(points).all(1) & (depth > 0))
    nearest_first = seen_px[np.argsort(depth[seen_px], kind="stable")]
    handle = np.array([[q.position.x, q.position.y, q.position.z] for q in poses.poses])
    pivot = 0.5 * (handle.min(0) + handle.max(0))  # the handle ring's middle, over the mine: it stays put, only turned
    pixels = np.frombuffer(image.data, np.uint8).reshape(image.height, image.step)[:, :image.width * 3].reshape(-1, 3)

    pub_image = rospy.Publisher("/synthetic/image", Image, queue_size=2)
    pub_cloud = rospy.Publisher("/synthetic/pointcloud", PointCloud2, queue_size=2)
    pub_poses = rospy.Publisher("/synthetic/grasp_poses", PoseArray, queue_size=2)
    rospy.logwarn("[synthetic] SYNTHETIC camera, not real data: bag frame %d swaying up to %.0f mm either side, period %.0f s, at %.0f Hz",
                  k, 1000 * amplitude, period, rate_hz)
    start, rate, yaw_deg = rospy.get_time(), rospy.Rate(rate_hz), None
    while not rospy.is_shutdown():
        stamp = rospy.Time.now()
        turned = float(rospy.get_param("/synthetic/yaw_deg", 0.0))
        if turned != yaw_deg:
            yaw_deg = turned
            rospy.logwarn("[synthetic] mine turned %+.0f deg", yaw_deg)
        yaw = np.radians(yaw_deg)
        shift = offset(stamp.to_sec() - start, amplitude, period, phases)
        placed = place(points, yaw, pivot, shift)
        src = render(placed, nearest_first, camera)
        empty = src < 0
        src[empty] = 0

        out_image = Image(header=copy.copy(image.header), height=image.height, width=image.width, encoding="bgr8", step=image.width * 3)
        bgr = pixels[src].copy()
        bgr[empty] = 0
        out_image.data = bgr.tobytes()

        buffer = bytearray(rows[src].tobytes())
        moved = xyz_view(buffer, cloud.fields, cloud.point_step, n)
        p = placed[src]
        p[empty] = np.nan
        moved["x"], moved["y"], moved["z"] = p[:, 0], p[:, 1], p[:, 2]
        out_cloud = PointCloud2(header=copy.copy(cloud.header), height=cloud.height, width=cloud.width, fields=cloud.fields,
                                is_bigendian=cloud.is_bigendian, point_step=cloud.point_step, row_step=cloud.row_step,
                                data=bytes(buffer), is_dense=False)

        out_poses = PoseArray(header=copy.copy(poses.header))
        turn = quaternion_about_axis(yaw, (0, 0, 1))
        out_poses.poses = [Pose(Point(*p), Quaternion(*quaternion_multiply(turn, (q.orientation.x, q.orientation.y, q.orientation.z, q.orientation.w))))
                           for p, q in zip(place(handle, yaw, pivot, shift), poses.poses)]

        for msg in (out_image, out_cloud, out_poses):
            msg.header.stamp = stamp
        pub_image.publish(out_image)
        pub_cloud.publish(out_cloud)
        pub_poses.publish(out_poses)
        rospy.loginfo_throttle(5.0, "[synthetic] SYNTHETIC drift now %+.1f %+.1f %+.1f mm" % tuple(1000 * shift))
        rate.sleep()


if __name__ == "__main__":
    run()
