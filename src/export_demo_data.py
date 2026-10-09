"""Export demo-page data: slice images (base64 JPEG) + pelvimetry LOOCV
predictions + clinician values.

Writes data/demo/pelvimetry.json and data/demo/slices.json.
Usage: python export_demo_data.py
"""
import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pydicom

PROJ = Path(__file__).resolve().parents[1]
DICOM_ROOT = Path(os.environ.get("AUTOMEASURE_DICOM_ROOT", "./dicom"))
GT = PROJ / "data" / "ground_truth"
DEMO = PROJ / "data" / "demo"
DEMO.mkdir(parents=True, exist_ok=True)

LM = json.loads((GT / "pelvimetry_landmarks.json").read_text(encoding="utf-8"))
RES = json.loads((PROJ / "data" / "pelvimetry_loocv" / ".." / "pelvimetry_cnn" /
                  "results.json").resolve().read_text(encoding="utf-8"))
SOFT = json.loads((GT / "soft_landmarks.json").read_text(encoding="utf-8"))
MM = 0.3125
DISP = 512  # display size (px) for the demo page


def load_display(dicom_path, spacing):
    ds = pydicom.dcmread(str(DICOM_ROOT / dicom_path))
    arr = ds.pixel_array.astype(np.float32)
    lo, hi = np.percentile(arr, (1, 99.5))
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    fx, fy = spacing[1] / MM, spacing[0] / MM
    if abs(fx - 1) > 1e-3 or abs(fy - 1) > 1e-3:
        arr = cv2.resize(arr, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    h, w = arr.shape
    scale = DISP / max(h, w)
    disp = cv2.resize((arr * 255).astype(np.uint8),
                      (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 82])
    b64 = base64.b64encode(buf.tobytes()).decode()
    return b64, scale, disp.shape[1], disp.shape[0]


NAMES = ["inlet_prom", "inlet_pub", "outlet_pub", "outlet_cocc",
         "depth_prom", "depth_cocc", "sacral_vertex", "sacral_up", "sacral_down"]

pelv = {}
slices = {}
for pid in sorted(LM.keys()):
    b64, scale, w, h = load_display(LM[pid]["dicom_path"], LM[pid]["px_spacing"])
    slices[f"sag_{pid}"] = dict(jpg=b64, w=w, h=h)
    r = RES[pid]
    pelv[pid] = dict(
        img=f"sag_{pid}",
        scale=scale,  # full-res px -> display px
        mm_per_disp_px=MM / scale,
        pred_lm={n: [round(v * scale, 1) for v in r["pred_lm_px"][n]]
                 for n in NAMES},
        gt_lm={n: [round(v * scale, 1) for v in LM[pid]["landmarks"][n]]
               for n in NAMES},
        pred_values=r["pred"], excel=r["excel"], delta=r["delta"],
        sacral_gt_ok=LM[pid].get("sacral_geom_ok", True))

# axial/sagittal display slices for the click tasks
for task in ("axial_tumour", "axial_mrf", "sag_tcc"):
    for pid, rec in SOFT[task].items():
        key = f"{task}_{pid}"
        b64, scale, w, h = load_display(rec["dicom_path"], rec["px_spacing"])
        slices[key] = dict(jpg=b64, w=w, h=h, scale=scale)

(DEMO / "pelvimetry.json").write_text(json.dumps(pelv), encoding="utf-8")
(DEMO / "slices.json").write_text(json.dumps(slices), encoding="utf-8")
sz = sum(len(v["jpg"]) for v in slices.values()) / 1e6
print(f"pelvimetry.json: {len(pelv)} patients; slices.json: {len(slices)} "
      f"images, ~{sz:.1f} MB base64")
