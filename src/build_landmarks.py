"""Collect pelvimetry landmarks per patient from the ground-truth JSONs.

Produces data/ground_truth/pelvimetry_landmarks.json with, per patient:
  slice (dicom path, spacing), 6 distance endpoints (canonical order) and the
  3 sacral-angle points, all in DICOM pixel coordinates of that slice.
Endpoints are canonicalised so the same anatomical point gets the same name
in every patient:
  inlet_prom (superior end) / inlet_pub;  outlet_pub (anterior end) /
  outlet_cocc;  depth_prom (superior) / depth_cocc;  sacral_vertex,
  sacral_up (upper ray end), sacral_down (lower ray end).
Self-check: distances recomputed from the landmarks must match the Excel.

Usage: python build_landmarks.py
"""
import json
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
GT = PROJ / "data" / "ground_truth"

# Visual-QC overrides (reference specific patients, so they live OUTSIDE
# the repository). Optional file data/qc_overrides.json, schema:
#   {"sacral_geom_unreliable": ["<pid>", ...]}
_QCP = GT.parent / "qc_overrides.json"
_QC = json.loads(_QCP.read_text(encoding="utf-8")) if _QCP.exists() else {}
SACRAL_GEOM_UNRELIABLE = set(_QC.get("sacral_geom_unreliable", []))

out = {}
for f in sorted(GT.glob("*_sagittal_landmarks.json")):
    r = json.loads(f.read_text(encoding="utf-8"))
    pid = r["pid"]
    ms = r["measurements"]
    need = ["inlet", "outlet", "depth", "sacral"]
    if any(k not in ms for k in need):
        print(f"{pid}: incomplete ({[k for k in need if k not in ms]}) - skipped")
        continue
    sp = r["px_spacing"]  # [row(dy), col(dx)]

    def canon(meas, rule):
        a, b = (np.array(p) for p in ms[meas]["dicom_px"])
        if rule == "y":   # first = smaller y (more superior)
            return (a, b) if a[1] <= b[1] else (b, a)
        return (a, b) if a[0] <= b[0] else (b, a)  # first = smaller x (anterior)

    i_prom, i_pub = canon("inlet", "y")
    o_pub, o_cocc = canon("outlet", "x")
    d_prom, d_cocc = canon("depth", "y")
    sv, s1, s2 = (np.array(p) for p in ms["sacral"]["dicom_px"])
    s_up, s_down = (s1, s2) if s1[1] <= s2[1] else (s2, s1)

    def mm(p, q):
        return float(np.hypot((p[0] - q[0]) * sp[1], (p[1] - q[1]) * sp[0]))

    def ang(v, e1, e2):
        # recorded sacral angulation = the OBTUSE angle between the two
        # Cobb segments (true for all 10 patients), so line-angle mod 180
        u1, u2 = np.array(e1) - v, np.array(e2) - v
        u1 = u1 * [sp[1], sp[0]]
        u2 = u2 * [sp[1], sp[0]]
        c = abs(np.dot(u1, u2)) / (np.linalg.norm(u1) * np.linalg.norm(u2))
        return 180.0 - float(np.degrees(np.arccos(np.clip(c, -1, 1))))

    checks = dict(inlet=mm(i_prom, i_pub), outlet=mm(o_pub, o_cocc),
                  depth=mm(d_prom, d_cocc), sacral=ang(sv, s_up, s_down))
    ok = all(abs(checks[k] - ms[k].get("mm", ms[k].get("deg"))) < 0.1
             for k in need)
    conf = {k: ms[k].get("confidence", "high") for k in need}
    if pid in SACRAL_GEOM_UNRELIABLE:
        conf["sacral"] = "low"
    out[pid] = dict(
        dicom_path=r["dicom_path"], sop=r["sop"], px_spacing=sp,
        ncc=r["ncc"], reliable=r["slice_match_reliable"],
        sacral_geom_ok=pid not in SACRAL_GEOM_UNRELIABLE,
        confidence=conf,
        landmarks={k: [float(v[0]), float(v[1])] for k, v in dict(
            inlet_prom=i_prom, inlet_pub=i_pub, outlet_pub=o_pub,
            outlet_cocc=o_cocc, depth_prom=d_prom, depth_cocc=d_cocc,
            sacral_vertex=sv, sacral_up=s_up, sacral_down=s_down).items()},
        excel={k: ms[k]["excel"] for k in need},
        self_check={k: round(v, 2) for k, v in checks.items()})
    print(f"{pid}: landmarks ok (self-check {'PASS' if ok else 'FAIL'}) "
          f"inlet={checks['inlet']:.1f} outlet={checks['outlet']:.1f} "
          f"depth={checks['depth']:.1f} sacral={checks['sacral']:.1f}")

with open(GT / "pelvimetry_landmarks.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print(f"\nwrote {len(out)} patients -> pelvimetry_landmarks.json")
