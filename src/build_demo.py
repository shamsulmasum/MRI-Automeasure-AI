"""Assemble the demo page: template + data -> demo/MRI_AutoMeasure_Demo.html.

Usage: python build_demo.py
"""
import json
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
DEMO = PROJ / "data" / "demo"

data = dict(
    slices=json.loads((DEMO / "slices.json").read_text(encoding="utf-8")),
    pelv=json.loads((DEMO / "pelvimetry.json").read_text(encoding="utf-8")),
    grid=json.loads((DEMO / "clickgrid.json").read_text(encoding="utf-8")),
)
tpl = (PROJ / "demo" / "demo_template.html").read_text(encoding="utf-8")
blob = json.dumps(data, separators=(",", ":"))
out = tpl.replace("/*__DATA__*/ null", blob, 1)
dst = PROJ / "demo" / "MRI_AutoMeasure_Demo.html"
dst.write_text(out, encoding="utf-8")
print(f"wrote {dst}  ({len(out) / 1e6:.1f} MB)")
