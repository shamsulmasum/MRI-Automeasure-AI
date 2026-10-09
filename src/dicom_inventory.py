"""Inventory a Sectra CD-export DICOM tree.

Walks every patient folder, reads all DICOM headers (no pixel data),
writes a per-file CSV and prints a per-series summary. Also checks
anonymisation (PatientName / PatientID / BirthDate) per folder.

Usage: python dicom_inventory.py <root_dir> <out_csv>
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pydicom


def orientation_label(iop):
    if iop is None or len(iop) != 6:
        return ""
    r, c = np.array(iop[:3], float), np.array(iop[3:], float)
    n = np.cross(r, c)
    return {0: "SAG", 1: "COR", 2: "AX"}[int(np.argmax(np.abs(n)))]


def main(root: Path, out_csv: Path):
    rows = []
    for pat_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        dicom_root = pat_dir / "DICOM"
        if not dicom_root.exists():
            dicom_root = pat_dir
        for f in dicom_root.rglob("*"):
            if not f.is_file():
                continue
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            except Exception:
                continue
            px = getattr(ds, "PixelSpacing", None)
            rows.append({
                "folder": pat_dir.name,
                "patient_id": str(getattr(ds, "PatientID", "")),
                "patient_name": str(getattr(ds, "PatientName", "")),
                "birth_date": str(getattr(ds, "PatientBirthDate", "")),
                "sex": str(getattr(ds, "PatientSex", "")),
                "study_date": str(getattr(ds, "StudyDate", "")),
                "modality": str(getattr(ds, "Modality", "")),
                "series_num": str(getattr(ds, "SeriesNumber", "")),
                "series_desc": str(getattr(ds, "SeriesDescription", "")),
                "instance": str(getattr(ds, "InstanceNumber", "")),
                "rows": str(getattr(ds, "Rows", "")),
                "cols": str(getattr(ds, "Columns", "")),
                "px_spacing": "x".join(f"{float(v):.4g}" for v in px) if px else "",
                "slice_thk": str(getattr(ds, "SliceThickness", "")),
                "orient": orientation_label(getattr(ds, "ImageOrientationPatient", None)),
                "path": str(f.relative_to(root)),
            })
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"parsed {len(rows)} DICOM files -> {out_csv}\n")

    print("== anonymisation check (distinct identity fields per folder) ==")
    for folder in sorted({r["folder"] for r in rows}):
        sub = [r for r in rows if r["folder"] == folder]
        ids = {r["patient_id"] for r in sub}
        names = {r["patient_name"] for r in sub}
        bds = {r["birth_date"] for r in sub}
        sexes = {r["sex"] for r in sub}
        print(f"  {folder}: PatientID={sorted(ids)} Name={sorted(names)} "
              f"Birth={sorted(bds)} Sex={sorted(sexes)}")

    print("\n== per-series summary ==")
    keys = sorted({(r["folder"], r["study_date"], r["series_num"], r["series_desc"]) for r in rows},
                  key=lambda k: (k[0], k[1], int(k[2] or 0)))
    for folder, sdate, snum, sdesc in keys:
        sub = [r for r in rows if r["folder"] == folder and r["series_num"] == snum
               and r["series_desc"] == sdesc and r["study_date"] == sdate]
        orients = ",".join(sorted({r["orient"] for r in sub}))
        spacings = ",".join(sorted({r["px_spacing"] for r in sub}))
        thks = ",".join(sorted({r["slice_thk"] for r in sub}))
        dims = ",".join(sorted({r['rows'] + 'x' + r['cols'] for r in sub}))
        print(f"  {folder} | {sdate} | s{snum:>4} | {sdesc:<40} | n={len(sub):>3} "
              f"| {orients:<3} | {dims} | px={spacings} | thk={thks}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
