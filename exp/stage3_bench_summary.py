"""Summarises stage3_bench CSVs.  python3 exp/stage3_bench_summary.py <dir holding */frames.csv>

err is the pick's tracker on the occluded camera against its clean twin; open loop is assuming the mine never moved
after tracking started. Both are measured against the clean twin, so neither is ground truth; mode 0 shows the noise floor.
"""
import csv, glob, os, statistics as st, sys
from collections import defaultdict


def q(v, f):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(f * (len(v) - 1))))]


def moved(a, b):  # mm the clean twin's target moved between two frames
    return 1000 * sum((a["clean_" + c] - b["clean_" + c]) ** 2 for c in "xyz") ** 0.5


runs = defaultdict(list)
for path in glob.glob(os.path.join(sys.argv[1], "*/frames.csv")):
    for r in csv.DictReader(open(path)):
        r = {k: float(v) for k, v in r.items()}
        runs[(int(r["mode"]), int(r["run"]))].append(r)
for v in runs.values():
    v.sort(key=lambda r: r["t_s"])

print(f"{'mode':>4} {'runs':>4} {'frames':>6} | tracker err mm: {'median':>6} {'p90':>6} {'max':>6} {'<10mm':>6} | "
      f"open loop {'median':>6} | end of motion: tracker {'median':>6} {'p90':>6} vs open loop {'median':>6} | tracker closer | frozen")
for mode in sorted({m for m, _ in runs}):
    rs = [v for (m, _), v in runs.items() if m == mode]
    err = [r["err_mm"] for v in rs for r in v]
    ol = [moved(r, v[0]) for v in rs for r in v]
    end, end_ol = [v[-1]["err_mm"] for v in rs], [moved(v[-1], v[0]) for v in rs]
    closer = sum(e < o for e, o in zip(err, ol)) / len(err)
    frozen = sum((a["occ_x"], a["occ_y"], a["occ_z"]) == (b["occ_x"], b["occ_y"], b["occ_z"]) for v in rs for a, b in zip(v, v[1:]))
    print(f"{mode:>4} {len(rs):>4} {len(err):>6} | {'':>15} {st.median(err):>6.1f} {q(err, .9):>6.1f} {max(err):>6.0f} "
          f"{sum(e < 10 for e in err) / len(err):>6.0%} | {'':>9} {st.median(ol):>6.1f} | {'':>22} {st.median(end):>6.1f} {q(end, .9):>6.1f} "
          f"{'':>12} {st.median(end_ol):>6.1f} | {closer:>14.0%} | {frozen / max(1, sum(len(v) - 1 for v in rs)):>6.0%}")

print("\ntracker error by coverage at that frame, modes 1-4 pooled: median / p90 mm (frames)")
for lo, hi in [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.999), (0.999, 1.01)]:
    e = [r["err_mm"] for (m, _), v in runs.items() if m > 0 for r in v if lo <= r["coverage"] < hi]
    if e:
        print(f"  {lo:>4.0%} to {min(hi, 1):>4.0%}: {st.median(e):6.1f} / {q(e, .9):6.1f}  ({len(e)})")
