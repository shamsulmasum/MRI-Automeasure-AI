"""Collect tumour/MRF caliper endpoints per patient from the ground-truth
JSONs, for the click-prompted measurement tasks (O5).

Tasks and endpoint channels (canonical order):
  axial_tumour : t_w_a (left), t_w_b (right), t_l_a (upper), t_l_b (lower)
  axial_mrf    : mrf_w_a, mrf_w_b, mrf_l_a, mrf_l_b
  sag_tcc      : t_cc_a (upper), t_cc_b (lower)

Exclusions: patients whose slice match is unreliable (measured series
missing from the export). Channel masks: endpoints from low-confidence or
half-line-suspect measurements are excluded from TRAINING (values still
evaluated against the Excel at test time).

Usage: python build_soft_landmarks.py
"""
import json
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
GT = PROJ / "data" / "ground_truth"

TASKS = {
    "axial_tumour": ("axial_tumour", [("t_w", "x"), ("t_l", "y")]),
    "axial_mrf": ("axial_mrf", [("mrf_w", "x"), ("mrf_l", "y")]),
    "sag_tcc": ("sagittal_tumour", [("t_cc", "y")]),
}
# endpoints whose POSITION failed visual QC (half-line suspects); listed
# OUTSIDE the repo in data/qc_overrides.json:
#   {"halfline_suspects": [["<pid>", "<measure>"], ...]}
_QCP = GT.parent / "qc_overrides.json"
_QC = json.loads(_QCP.read_text(encoding="utf-8")) if _QCP.exists() else {}
HALFLINE = {tuple(x) for x in _QC.get("halfline_suspects", [])}

out = {t: {} for t in TASKS}
for task, (gt_type, measures) in TASKS.items():
    for f in sorted(GT.glob(f"*_{gt_type}.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        pid = r["pid"]
        if not r.get("slice_match_reliable", True):
            print(f"{task} {pid}: EXCLUDED (unreliable slice match)")
            continue
        ms = r["measurements"]
        if any(m not in ms for m, _ in measures):
            print(f"{task} {pid}: EXCLUDED (missing {[m for m,_ in measures if m not in ms]})")
            continue
        sp = r["px_spacing"]
        lms, mask, excel, checks = {}, {}, {}, {}
        for meas, rule in measures:
            a, b = (np.array(p, float) for p in ms[meas]["dicom_px"])
            if rule == "x":
                a, b = (a, b) if a[0] <= b[0] else (b, a)
            else:
                a, b = (a, b) if a[1] <= b[1] else (b, a)
            lms[f"{meas}_a"] = a.tolist()
            lms[f"{meas}_b"] = b.tolist()
            bad = (ms[meas].get("confidence", "high") == "low"
                   or (pid, meas) in HALFLINE)
            mask[f"{meas}_a"] = not bad
            mask[f"{meas}_b"] = not bad
            excel[meas] = ms[meas]["excel"]
            checks[meas] = float(np.hypot((a[0] - b[0]) * sp[1],
                                          (a[1] - b[1]) * sp[0]))
        ok = all(abs(checks[m] - (excel[m] + ms[m]["delta"])) < 0.15
                 for m, _ in measures)
        out[task][pid] = dict(
            dicom_path=r["dicom_path"], sop=r["sop"], px_spacing=sp,
            landmarks=lms, train_mask=mask, excel=excel,
            extracted={m: round(checks[m], 2) for m, _ in measures})
        flag = "" if all(mask.values()) else f" (masked: {[k for k,v in mask.items() if not v]})"
        print(f"{task} {pid}: ok (self-check {'PASS' if ok else 'FAIL'}) "
              f"{ {m: round(checks[m],1) for m,_ in measures} }{flag}")

with open(GT / "soft_landmarks.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
counts = {t: len(v) for t, v in out.items()}
print(f"\nwrote soft_landmarks.json  patients per task: {counts}")
