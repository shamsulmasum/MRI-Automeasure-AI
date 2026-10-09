# Interactive demo

`demo_template.html` is the complete single-file demo application (dark
PACS-style viewer, three interactive tabs, zero runtime dependencies). It is
a **template**: the published demo embeds patient images and precomputed
model outputs, which are patient-derived data and therefore not in this
repository.

## How the shipped demo is produced

1. `src/export_demo_data.py` — renders display slices and pelvimetry
   predictions (leave-one-out) to JSON.
2. `src/demo_grid.py` — retrains each one-click fold and records the model's
   real response to a 17×17 grid of click positions (±16 mm, 2 mm steps) per
   patient and task, so the page can answer any click without running a
   network in the browser.
3. `src/build_demo.py` — injects the JSON into the template, producing one
   self-contained HTML file that runs offline in any browser.

## Trying it

- **Collaborators / interested researchers:** the working demo on the pilot
  patients is shared privately — contact the team (see repository README).
- **With your own data:** run the pipeline in `src/` on your own DICOM +
  annotations, then the three steps above; the result is your own
  double-click-to-open demo file.
