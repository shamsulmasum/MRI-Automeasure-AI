"""Composite ground-truth QC images into review sheets (2 per sheet, cropped
to the anatomy region so annotations stay readable).

Usage: python qc_sheets.py <glob> <out_prefix>
e.g.:  python qc_sheets.py *_sagittal_landmarks.png sagland
"""
import sys
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
QC = PROJ / "data" / "ground_truth" / "qc"
OUT = PROJ / "data" / "ground_truth" / "qc_sheets"
OUT.mkdir(exist_ok=True)

pattern, prefix = sys.argv[1], sys.argv[2]
files = sorted(QC.glob(pattern))
CROP_X, CROP_W = 470, 530   # anatomy region of the 1469-wide screenshots
sheet = []
idx = 0
for i, f in enumerate(files):
    img = cv2.imread(str(f))
    h, w = img.shape[:2]
    x0 = min(CROP_X, w - CROP_W)
    crop = img[:, x0:x0 + CROP_W]
    crop = cv2.resize(crop, None, fx=1.35, fy=1.35,
                      interpolation=cv2.INTER_CUBIC)
    cv2.putText(crop, f.stem, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 255), 2)
    sheet.append(crop)
    if len(sheet) == 2 or i == len(files) - 1:
        H = max(s.shape[0] for s in sheet)
        row = np.hstack([cv2.copyMakeBorder(
            s, 0, H - s.shape[0], 0, 6, cv2.BORDER_CONSTANT) for s in sheet])
        cv2.imwrite(str(OUT / f"{prefix}_{idx:02d}.png"), row)
        idx += 1
        sheet = []
print(f"{idx} sheets -> {OUT}")
