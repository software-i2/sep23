"""Run: python3 test_occluder.py  (no ROS or bag needed)"""
from types import SimpleNamespace

import numpy as np
import occluder as o

# Coverage: each mode stays in its band, linear modes hit both ends.
assert o.coverage(0, 5.0, 3.0) == 0.0
assert o.coverage(1, 0.0, 3.0) == 0.0 and abs(o.coverage(1, 3.0, 3.0) - 0.5) < 1e-9
assert o.coverage(3, 0.0, 3.0) == 0.0 and abs(o.coverage(3, 9.0, 3.0) - 1.0) < 1e-9
ts = np.linspace(0, 6, 200)
for mode, (lo, hi) in o.RANGES.items():
    c = [o.coverage(mode, t, 3.0) for t in ts]
    assert lo - 1e-9 <= min(c) and max(c) <= hi + 1e-9, mode
assert np.all(np.diff([o.coverage(1, t, 3.0) for t in ts]) >= 0)  # constant-velocity sweep never backs off

# The mask covers what it claims: exactly none, exactly all, and close in between.
shape = (600, 800)
assert not o.obstacle_mask(shape, 0.0, 20).any() and o.obstacle_mask(shape, 1.0, 20).all()
for cover in (0.3, 0.5, 0.8):
    assert abs(o.obstacle_mask(shape, cover, 20).mean() - cover) < 0.005, cover

# Cloud: points behind the obstacle are gone, the rest are untouched.
xyz = np.dstack([np.full(shape, 0.1), np.full(shape, -0.05), np.full(shape, -0.40)]).astype(np.float32)
mask = o.obstacle_mask(shape, 0.5, 20)
seen = o.hide(xyz, mask)
assert np.isnan(seen[mask]).all() and np.array_equal(seen[~mask], xyz[~mask])

# Packing keeps exactly the chosen rows, whole and in order: visible and hidden split the frame between them.
rows = np.arange(4 * 16, dtype=np.uint8).reshape(4, 16)
hidden = np.array([[False, True], [True, False]])
seen, n_seen = o.pack(rows.tobytes(), 16, ~hidden)
gone, n_gone = o.pack(rows.tobytes(), 16, hidden)
assert (n_seen, n_gone) == (2, 2) and seen == rows[[0, 3]].tobytes() and gone == rows[[1, 2]].tobytes()

# The PointCloud2 view writes x, y, z in place and leaves an rgb field behind them alone.
fields = [SimpleNamespace(name=n, offset=4 * i) for i, n in enumerate("xyz")] + [SimpleNamespace(name="rgb", offset=12)]
raw = np.zeros((4, 4), np.float32)
raw[:, 3] = 7.0
buffer = bytearray(raw.tobytes())
points = o.xyz_view(buffer, fields, 16, 4)
points["z"] = -1.0
back = np.frombuffer(bytes(buffer), np.float32).reshape(4, 4)
assert (back[:, 2] == -1.0).all() and (back[:, 3] == 7.0).all()

# Paint: flat is the arm colour; textured is high contrast and slides with the obstacle.
bgr = np.zeros((*shape, 3), np.uint8)
assert (o.paint(bgr, mask, None, 0)[mask] == o.FLAT_BGR).all()
pattern = o.texture(shape)
a, b = o.paint(bgr, mask, pattern, 0)[mask], o.paint(bgr, mask, pattern, 5)[mask]
assert a.std() > 100 and not np.array_equal(a, b)
print("ok")
