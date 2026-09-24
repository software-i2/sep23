"""Run: python test_stage3.py  (no bag needed)"""
import numpy as np
import stage3_bench as b

circle, shape = (400, 300, 150), (600, 800)
for f in (0.3, 0.5, 0.7, 0.8):
    m, got = b.arm_mask(circle, f, shape)
    assert abs(got - f) < 0.02, (f, got)
assert abs(b.occlusion_frac(2.5, 0, 2.5) - 0.7) < 1e-9 and abs(b.occlusion_frac(5.0, 0, 2.5) - 0.3) < 1e-9

# arm pixels must fall outside the depth gate
xyz = np.dstack([np.zeros(shape), np.zeros(shape), np.full(shape, 0.33)]).astype(np.float32)
_, z = b.occlude({'rgb': np.zeros((*shape, 3), np.uint8), 'xyz': xyz}, m)
assert (np.abs(z[..., 2][m] - 0.33) > 0.15).all()
print('ok')
