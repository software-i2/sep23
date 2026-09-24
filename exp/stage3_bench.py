#!/usr/bin/env python3
"""Stage 3 tracking benchmark: inject a synthetic arm occlusion into a real bag, compare
A) KLT + 2D RANSAC, B) CSRT, C) depth-gated 3D RANSAC plane/centroid.

Bag layout used (captures_grasp_poses.bag): /image bgr8, /pointcloud organized xyz (metric Z),
/label mono8 target mask. Deps: pip install rosbags opencv-contrib-python-headless numpy
"""
import argparse, csv, time
from pathlib import Path
import numpy as np, cv2
from rosbags.highlevel import AnyReader

GATE = 0.15  # m, half-width of the depth gate around Z_mine (spec value; --gate overrides)
ARM_Z = 0.15  # m, depth written into occluded pixels; must sit outside the gate or the arm counts as target


def load(bag, rgb_t, cloud_t, label_t, max_frames):
    frames = {}
    with AnyReader([Path(bag)]) as r:
        cons = [c for c in r.connections if c.topic in (rgb_t, cloud_t, label_t)]
        for c, ts, raw in r.messages(connections=cons):
            m = r.deserialize(raw, c.msgtype)
            key = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
            f = frames.setdefault(key, {})
            if c.topic == rgb_t:
                f['rgb'] = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3).copy()
            elif c.topic == label_t:
                f['label'] = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width).copy()
            else:
                f['xyz'] = np.frombuffer(m.data, np.float32).reshape(-1, 3).copy()  # unorganized; reshaped to image below
    out = [(k * 1e-9, f) for k, f in sorted(frames.items()) if 'rgb' in f and 'xyz' in f]
    for _, f in out:  # 480000 pts == 800x600: row-major organized cloud published with height=1
        f['xyz'] = f['xyz'].reshape(*f['rgb'].shape[:2], 3)
        if np.nanmedian(f['xyz'][..., 2]) < 0:  # this bag's cloud has Z pointing away from the scene
            f['xyz'][..., 2] *= -1
    return out[:max_frames] if max_frames else out


def intrinsics(xyz):
    """Least-squares fx,fy,cx,cy from the organized cloud (bag has no CameraInfo)."""
    h, w = xyz.shape[:2]
    v, u = np.mgrid[:h, :w]
    ok = np.isfinite(xyz).all(-1) & (xyz[..., 2] > 0.05)
    x, y, z = (xyz[ok][:, i] for i in range(3))
    fx, cx = np.linalg.lstsq(np.c_[x / z, np.ones_like(z)], u[ok], rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.c_[y / z, np.ones_like(z)], v[ok], rcond=None)[0]
    return fx, fy, cx, cy


def init_target(frames):
    """Circle + Z_mine from the first 5 frames' label mask (fallback: gated centre blob)."""
    zs, lab = [], None
    for _, f in frames[:5]:
        m = f.get('label')
        if m is not None and (m > 0).sum() > 500:
            lab = m > 0 if lab is None else lab
            z = f['xyz'][..., 2][m > 0]
            zs.append(np.nanmedian(z[np.isfinite(z) & (z > 0.05)]))
    if lab is None:  # no label: mine face = nearest surface; Z_mine = median of the nearest 5% of points
        for _, f in frames[:5]:
            z = f['xyz'][..., 2]; z = z[np.isfinite(z) & (z > 0.05)]
            zs.append(np.median(z[z <= np.percentile(z, 5)]))
        z0 = frames[0][1]['xyz'][..., 2]
        lab = np.isfinite(z0) & (np.abs(z0 - np.median(zs)) <= GATE)
        lab = cv2.morphologyEx(lab.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)) > 0  # drop rope/handles
    cnts, _ = cv2.findContours(lab.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (u, v), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    return (u, v, r), float(np.median(zs))


# ---------- synthetic arm ----------
def arm_poly(circle, depth, shape):
    """Arm = thick link entering from the bottom-right corner, tip `depth` px past the disc edge."""
    u, v, r = circle
    h, w = shape
    base = np.array([w * 1.1, h * 1.2])
    d = np.array([u, v]) - base; d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    tip = np.array([u, v]) - d * r + d * depth
    half = 0.95 * r  # link width 1.9r: narrower cannot cover 80% of the disc
    return np.array([base + n * half * 1.5, base - n * half * 1.5, tip - n * half, tip + n * half * 0.6,
                     tip + d * 0.3 * r]).astype(np.int32)  # slightly asymmetric gripper tip


def arm_mask(circle, frac, shape):
    """Bisect arm penetration so it covers `frac` of the mine disc."""
    u, v, r = circle
    disc = np.zeros(shape, np.uint8); cv2.circle(disc, (int(u), int(v)), int(r), 1, -1)
    area = disc.sum()
    lo, hi = 0.0, 3 * r
    for _ in range(20):
        mid = (lo + hi) / 2
        m = np.zeros(shape, np.uint8); cv2.fillPoly(m, [arm_poly(circle, mid, shape)], 1)
        (lo, hi) = (mid, hi) if (m & disc).sum() / area < frac else (lo, mid)
    m = np.zeros(shape, np.uint8); cv2.fillPoly(m, [arm_poly(circle, hi, shape)], 1)
    return m.astype(bool), (m.astype(bool) & disc.astype(bool)).sum() / area


def occlusion_frac(t, t0, sweep_s, lo=0.3, hi=0.7):
    """Triangle wave lo->hi over sweep_s, then back."""
    p = ((t - t0) / sweep_s) % 2.0
    return lo + (hi - lo) * (p if p < 1 else 2 - p)


def occlude(f, mask):
    rgb, xyz = f['rgb'].copy(), f['xyz'].copy()
    rng = np.random.default_rng(0)
    rgb[mask] = np.clip(55 + rng.normal(0, 6, (mask.sum(), 1)), 0, 255).astype(np.uint8)  # dull grey arm
    xyz[mask] = xyz[mask] * (ARM_Z / np.maximum(xyz[mask][:, 2:3], 1e-3))  # same ray, pulled to ARM_Z
    return rgb, xyz


# ---------- trackers: step(rgb, xyz, gate_mask) -> (u, v, retention, extras) ----------
class KLT:
    name = 'A_klt_ransac2d'

    def __init__(self, circle):
        self.u, self.v, self.r = circle; self.prev = None; self.n0 = None

    def roi(self, shape, gate):
        m = np.zeros(shape, np.uint8); cv2.circle(m, (int(self.u), int(self.v)), int(self.r), 255, -1)
        return m & (gate.astype(np.uint8) * 255)

    def step(self, rgb, xyz, gate):
        g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        if self.prev is None:
            self.prev = g; self.pts = cv2.goodFeaturesToTrack(g, 300, 0.01, 5, mask=self.roi(g.shape, gate))
            self.n0 = 0 if self.pts is None else len(self.pts)
            return self.u, self.v, 1.0, {'kp': self.pts}
        ret, kp = 0.0, None
        if self.pts is not None and len(self.pts) >= 3:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, self.pts, None, winSize=(31, 31), maxLevel=4)
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g, self.prev, p1, None, winSize=(31, 31), maxLevel=4)
            ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(self.pts - p0r, axis=2)[:, 0] < 1.0)
            if ok.sum() >= 3:
                M, inl = cv2.estimateAffinePartial2D(self.pts[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if M is not None:
                    self.u, self.v = M @ [self.u, self.v, 1]
                    ret = inl.sum() / max(self.n0, 1); kp = p1[ok][inl[:, 0] == 1]
        # re-detect each frame: at 1 Hz long KLT tracks do not survive; arm+off-gate pixels excluded via `gate`
        self.prev = g; self.pts = cv2.goodFeaturesToTrack(g, 300, 0.01, 5, mask=self.roi(g.shape, gate))
        return self.u, self.v, ret, {'kp': kp}


class CSRT:
    name = 'B_csrt'

    def __init__(self, circle):
        self.u, self.v, self.r = circle; self.t = None

    def step(self, rgb, xyz, gate):
        if self.t is None:
            p = cv2.TrackerCSRT_Params(); p.use_segmentation = True
            self.t = cv2.TrackerCSRT_create(p)
            x, y, s = int(self.u - self.r), int(self.v - self.r), int(2 * self.r)
            self.box = (x, y, s, s)
            # spatial reliability prior = gated, un-occluded foreground inside the box
            self.t.setInitialMask(gate[max(y, 0):y + s, max(x, 0):x + s].astype(np.uint8))
            self.t.init(rgb, self.box)
            return self.u, self.v, 1.0, {'box': self.box}
        ok, box = self.t.update(rgb)
        if ok:
            self.box = box; self.u, self.v = box[0] + box[2] / 2, box[1] + box[3] / 2
        x, y, w, h = map(int, self.box)
        vis = gate[max(y, 0):y + h, max(x, 0):x + w].mean() if w > 0 and h > 0 else 0.0  # visible gated fraction
        return self.u, self.v, float(vis) if ok else 0.0, {'box': self.box}


class Gated3D:
    name = 'C_gated3d_ransac'

    def __init__(self, circle, K):
        self.u, self.v, self.r = circle; self.K = K; self.n0 = None; self.X = None

    def step(self, rgb, xyz, gate):
        h, w = gate.shape
        m = np.zeros((h, w), np.uint8); cv2.circle(m, (int(self.u), int(self.v)), int(1.3 * self.r), 1, -1)
        P = xyz[gate & m.astype(bool)]
        if len(P) < 50:
            return self.u, self.v, 0.0, {}
        n_all = len(P)
        P = P[np.random.default_rng(0).choice(len(P), min(len(P), 4000), replace=False)]
        best = None
        rng = np.random.default_rng(1)
        for _ in range(100):  # plane RANSAC
            a, b, c = P[rng.choice(len(P), 3, replace=False)]
            nrm = np.cross(b - a, c - a); nn = np.linalg.norm(nrm)
            if nn < 1e-9: continue
            inl = np.abs((P - a) @ (nrm / nn)) < 0.01
            if best is None or inl.sum() > best.sum(): best = inl
        Q = P[best]
        n_inl = best.mean() * n_all  # scale back up so the subsample doesn't hide lost points
        if self.n0 is None: self.n0 = n_inl
        X, Y, Z = Q.mean(0)  # ponytail: centroid of visible face is biased away from the arm; circle fit if that matters
        fx, fy, cx, cy = self.K
        self.u, self.v = fx * X / Z + cx, fy * Y / Z + cy
        self.X = (X, Y, Z)
        return self.u, self.v, n_inl / self.n0, {'xyz': self.X}


def to_metric(u, v, xyz, K, z_mine):
    """Pixel -> camera XY (m): median gated cloud around the pixel, else intrinsics at Z_mine."""
    h, w = xyz.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if 0 <= ui < w and 0 <= vi < h:
        p = xyz[max(vi - 3, 0):vi + 4, max(ui - 3, 0):ui + 4].reshape(-1, 3)
        p = p[np.isfinite(p).all(1) & (np.abs(p[:, 2] - z_mine) < GATE)]
        if len(p): return float(np.median(p[:, 0])), float(np.median(p[:, 1]))
    fx, fy, cx, cy = K
    return (u - cx) * z_mine / fx, (v - cy) * z_mine / fy


def run(frames, circle, z_mine, K, sweep_s, occluded, spot=0, occ=(0.3, 0.7)):
    trackers = [KLT(circle), CSRT(circle), Gated3D(circle, K)]
    rows, vis = [], []
    t0 = frames[0][0]
    for i, (t, f) in enumerate(frames):
        frac, real = 0.0, 0.0
        if occluded and i > 0:  # frame 0 = clean lock-on
            frac = occlusion_frac(t, t0, sweep_s, *occ)
            am, real = arm_mask(circle, frac, f['rgb'].shape[:2])
            rgb, xyz = occlude(f, am)
        else:
            rgb, xyz = f['rgb'], f['xyz']
        z = xyz[..., 2]
        gate = np.isfinite(z) & (np.abs(z - z_mine) <= GATE)
        out = {}
        for tr in trackers:
            s = time.perf_counter()
            u, v, ret, ex = tr.step(rgb, xyz, gate)
            ms = (time.perf_counter() - s) * 1e3
            X, Y = ex['xyz'][:2] if ex.get('xyz') else to_metric(u, v, xyz, K, z_mine)
            rows.append(dict(stream='occluded' if occluded else 'clean', method=tr.name, spot=spot, frame=i, t=t - t0,
                             occ_target=frac, occ_actual=real, u=u, v=v, X=X, Y=Y, retention=ret, latency_ms=ms))
            out[tr.name] = (u, v, ex)
        vis.append((rgb, gate, out, real, z_mine))
    return rows, vis


def draw(rgb, gate, out, occ, z_mine):
    img = rgb.copy()
    cnts, _ = cv2.findContours(gate.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, [c for c in cnts if cv2.contourArea(c) > 200], -1, (0, 255, 255), 1)  # depth-gate boundary
    u, v, ex = out[KLT.name]
    if ex.get('kp') is not None:
        for p in ex['kp'].reshape(-1, 2): cv2.circle(img, tuple(map(int, p)), 2, (0, 255, 0), -1)
    cv2.drawMarker(img, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    u, v, ex = out[CSRT.name]
    x, y, w, h = map(int, ex['box']); cv2.rectangle(img, (x, y), (x + w, y + h), (255, 128, 0), 2)
    cv2.drawMarker(img, (int(u), int(v)), (255, 128, 0), cv2.MARKER_TILTED_CROSS, 24, 2)
    u, v, _ = out[Gated3D.name]
    cv2.drawMarker(img, (int(u), int(v)), (0, 0, 255), cv2.MARKER_DIAMOND, 24, 2)
    for k, (txt, col) in enumerate([(f'occ {occ*100:.0f}%  gate {z_mine-GATE:.2f}-{z_mine+GATE:.2f} m', (0, 255, 255)),
                                    ('A KLT+RANSAC2D', (0, 255, 0)), ('B CSRT', (255, 128, 0)), ('C gated 3D', (0, 0, 255))]):
        cv2.putText(img, txt, (10, 25 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return img


def write_bag(path, frames, imgs):
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC)
    Img, Hdr, Time = ts.types['sensor_msgs/msg/Image'], ts.types['std_msgs/msg/Header'], ts.types['builtin_interfaces/msg/Time']
    Path(path).unlink(missing_ok=True)
    with Writer(path) as w:
        c = w.add_connection('/stage3_bench/overlay', Img.__msgtype__, typestore=ts)
        for i, ((t, _), img) in enumerate(zip(frames, imgs)):
            ns = int(t * 1e9)
            msg = Img(header=Hdr(seq=i, stamp=Time(sec=ns // 10**9, nanosec=ns % 10**9), frame_id='camera_link'),
                      height=img.shape[0], width=img.shape[1], encoding='bgr8', is_bigendian=0,
                      step=img.shape[1] * 3, data=np.ascontiguousarray(img).reshape(-1))
            w.write(c, ns, ts.serialize_ros1(msg, Img.__msgtype__))


def summarize(rows):
    """Metrics per parking spot (each spot re-locks, so offsets are relative to that spot's lock), then averaged."""
    out = []
    spots = sorted({x['spot'] for x in rows})
    for stream in ('clean', 'occluded'):
        for m in (KLT.name, CSRT.name, Gated3D.name):
            sig, err, ret = [], [], []
            for sp in spots:
                sel = lambda st: [x for x in rows if x['stream'] == st and x['method'] == m and x['spot'] == sp]
                r, clean = sel(stream), sel('clean')
                d = np.array([[x['X'], x['Y']] for x in r]); d -= d[0]
                e = np.array([[a['X'] - b['X'], a['Y'] - b['Y']] for a, b in zip(r, clean)])
                sig.append(np.sqrt(d[:, 0].var() + d[:, 1].var())); err.append(np.sqrt((e ** 2).sum(1).mean()))
                ret += [x['retention'] for x in r[1:]]
            r = [x for x in rows if x['stream'] == stream and x['method'] == m]
            hi = [x['retention'] for x in r if x['occ_actual'] >= 0.5]
            out.append(dict(stream=stream, method=m, spots=len(spots), frames=len(r),
                            sigma_xy_mm=1e3 * float(np.mean(sig)), sigma_xy_mm_worst_spot=1e3 * float(np.max(sig)),
                            err_vs_clean_rms_mm=1e3 * float(np.mean(err)), err_vs_clean_mm_worst_spot=1e3 * float(np.max(err)),
                            retention_mean=float(np.mean(ret)),
                            retention_at_occ50plus=float(np.mean(hi)) if hi else float('nan'),
                            latency_ms_mean=float(np.mean([x['latency_ms'] for x in r])),
                            latency_ms_p95=float(np.percentile([x['latency_ms'] for x in r], 95))))
    return out


def main():
    global GATE
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--rgb', default='/image')
    ap.add_argument('--cloud', default='/pointcloud')
    ap.add_argument('--label', default='/label')
    ap.add_argument('--sweep-s', type=float, default=3.0, help='seconds for arm to sweep occ-min -> occ-max of the face')
    ap.add_argument('--occ-min', type=float, default=0.3)
    ap.add_argument('--occ-max', type=float, default=0.7, help='peak face fraction hidden by the arm (<= ~0.8)')
    ap.add_argument('--gate', type=float, default=GATE, help='depth gate half-width, m')
    ap.add_argument('--spot-frames', type=int, default=10,
                    help='frames per parking spot; each spot re-locks target, gate and trackers (0 = whole bag is one spot)')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--out', default='stage3_out')
    a = ap.parse_args()
    GATE = a.gate
    out = Path(a.out); out.mkdir(exist_ok=True)

    frames = load(a.bag, a.rgb, a.cloud, a.label, a.max_frames)
    assert len(frames) >= 5, f'only {len(frames)} synced frames'
    K = intrinsics(frames[0][1]['xyz'])
    dt = np.diff([t for t, _ in frames])
    print(f'{len(frames)} frames @ {1/np.median(dt):.2f} Hz | K={np.round(K,1)}')

    # ponytail: the bag has no parking-spot markers, so spots are fixed-length windows; split on vehicle odom if recorded
    n = a.spot_frames or len(frames)
    rows, vis = [], []
    for sp, i in enumerate(range(0, len(frames), n)):
        seg = frames[i:i + n]
        if len(seg) < 3: break
        circle, z_mine = init_target(seg)  # Stage 2 lock: fresh Z_mine, gate and face circle for this spot
        print(f'spot {sp}: frames {i}-{i+len(seg)-1} | px={circle[0]:.0f},{circle[1]:.0f} r={circle[2]:.0f} '
              f'| Z_mine={z_mine:.3f} gate=[{z_mine-GATE:.3f},{z_mine+GATE:.3f}]')
        rc, _ = run(seg, circle, z_mine, K, a.sweep_s, occluded=False, spot=sp)
        ro, v = run(seg, circle, z_mine, K, a.sweep_s, occluded=True, spot=sp, occ=(a.occ_min, a.occ_max))
        rows += rc + ro; vis += v
    frames = frames[:len(vis)]
    with open(out / 'stage3_tracking_frames.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, rows[0].keys()); w.writeheader(); w.writerows(rows)
    summ = summarize(rows)
    with open(out / 'stage3_tracking_benchmark.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, summ[0].keys()); w.writeheader(); w.writerows(summ)
    for s in summ: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})

    imgs = [draw(*v) for v in vis]
    h, w = imgs[0].shape[:2]
    vw = cv2.VideoWriter(str(out / 'stage3_overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 2, (w, h))
    for im in imgs: vw.write(im)
    vw.release()
    write_bag(str(out / 'stage3_overlay.bag'), frames, imgs)
    print(f'wrote {out}/')


if __name__ == '__main__':
    main()
