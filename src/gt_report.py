"""Summarise ground-truth extraction: per-measurement table + error stats.

Usage: python gt_report.py
"""
import json
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
SUM = PROJ / "data" / "ground_truth" / "summary.json"

MEASURES = ["inlet", "outlet", "depth", "sacral", "t_w", "t_l",
            "mrf_w", "mrf_l", "t_cc", "junction"]

recs = json.loads(SUM.read_text(encoding="utf-8"))
by_pid = {}
flags = {}
for r in recs:
    if r.get("status") == "NO_MATCH":
        continue
    for k, v in r.get("measurements", {}).items():
        by_pid.setdefault(r["pid"], {})[k] = v
        flags.setdefault(r["pid"], {})[k] = (
            ("L" if v.get("confidence") == "low" else "") +
            ("!" if not r.get("slice_match_reliable", True) else ""))

hdr = f"{'pid':6s}" + "".join(f"{m:>10s}" for m in MEASURES)
print(hdr)
print("-" * len(hdr))
for pid in sorted(by_pid):
    row = f"{pid:6s}"
    for m in MEASURES:
        v = by_pid[pid].get(m)
        if v is None:
            row += f"{'--':>10s}"
        else:
            row += f"{v['delta']:>+8.1f}{flags[pid].get(m, ''):2s}"
    print(row)

print()
deltas_all, n_total, n_found, n_low, n_unrel = [], 0, 0, 0, 0
for pid, mm in by_pid.items():
    exp = 9 + (1 if "junction" in mm else 0)
    n_total += exp
    for m, v in mm.items():
        n_found += 1
        if v.get("confidence") == "low":
            n_low += 1
        if "!" in flags[pid].get(m, ""):
            n_unrel += 1
        else:
            deltas_all.append(abs(v["delta"]))
d = np.array(deltas_all)
print(f"found {n_found}/{n_total} measurements "
      f"({n_low} low-confidence, {n_unrel} on unreliable slice matches)")
print(f"abs error vs Excel (reliable slices): mean {d.mean():.2f}, "
      f"median {np.median(d):.2f}, p90 {np.percentile(d, 90):.2f}, "
      f"max {d.max():.2f}  (mm / degrees)")

# half-line check: in an axial pair the partner caliper must cross INSIDE
# the line, not at an endpoint (endpoint-crossing = only half the line found)
print("\nhalf-line suspects (partner crosses at an endpoint):")
any_suspect = False
for r in recs:
    ms = r.get("measurements", {})
    keys = [k for k in ms if k.endswith("_w") or k.endswith("_l")]
    if len(keys) != 2:
        continue
    (a1, b1), (a2, b2) = (np.array(ms[k]["screen_px"], float) for k in keys)
    u1, u2 = b1 - a1, b2 - a2
    den = u1[0] * u2[1] - u1[1] * u2[0]
    if abs(den) < 1e-9:
        continue
    w = a2 - a1
    t1 = (w[0] * u2[1] - w[1] * u2[0]) / den
    t2 = (w[0] * u1[1] - w[1] * u1[0]) / den
    for k, t, u in ((keys[0], t1, u1), (keys[1], t2, u2)):
        margin = min(t, 1 - t) * np.hypot(*u)
        if margin < 6:
            print(f"  {r['pid']} {r['type']} {k}: crossing {margin:.1f}px "
                  f"from an endpoint -> endpoint unreliable")
            any_suspect = True
if not any_suspect:
    print("  none")
