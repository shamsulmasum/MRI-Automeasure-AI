# Methods (condensed, public version)

A fuller internal methods log exists with the clinical collaborators; this
version reports the methodology and aggregate results only — no patient-level
values or identifiers.

## Cohort and data

Ten anonymised rectal-cancer staging MRIs (one UK centre; a ~300-patient
UK–Dutch cohort is planned). Per patient: T2 TSE sagittal and axial series
(0.3125 mm in-plane, 3 mm slices typical), a clinician-completed measurement
spreadsheet (pelvic inlet/outlet/depth, sacral angulation, tumour width/
length/craniocaudal extent, mesorectal-package width/length), annotated and
unannotated viewer screenshots of the measured slices, and a narrated video
in which a clinician demonstrates every measurement. Five patients were
independently re-measured, providing an observer-variability benchmark:
bone measurements differed by ≤ ~6 mm / ~5°, soft tissue by up to 30 mm.

## Stage 1 — measurement protocol formalisation

The demonstration video was transcribed and analysed frame-by-frame (cursor
tracking) to produce a written protocol: slice-selection rules, exact bony
endpoints of the three pelvimetry distances, the Cobb-style two-segment
construction of the sacral angulation, and width/length conventions for the
axial tumour and mesorectal caliper crosses (`docs/measurement_protocol.md`).

## Stage 2 — ground-truth recovery from annotated screenshots

The spreadsheet stores only resulting values; training and honest evaluation
require the clicked positions. Pipeline: (1) match each screenshot to its
DICOM slice by normalized cross-correlation over all slices/series at a
sub-step-refined display scale; (2) subtract the aligned rendered slice to
isolate the burned-in overlay, extracting it at four threshold settings;
(3) Hough line detection with collinear merging; (4) endpoint refinement to
end-ticks / X-markers / line intersections; (5) label lines by matching
metric length to the recorded values **subject to structural constraints**
— the pelvimetry triangle is assigned jointly (inlet and depth must share
the promontory, outlet and depth the coccyx), axial width/length pairs must
perpendicularly cross each other, sacral segments must lie along the sacrum
with the bend inferior. (6) Every case receives a quality flag and a visual
QC overlay; all were reviewed by eye.

Validation: 90/91 values recovered; median absolute agreement with the
spreadsheet 0.34 mm (mean 0.83). Key methodological lesson: *a value check
cannot validate a position* — early versions occasionally labelled a bright
anatomy edge of coincidentally correct length as a measurement line; only
anatomical-configuration constraints plus per-case visual QC eliminated this.

## Stage 3 — automatic pelvimetry (negative result, then success)

**Multi-atlas registration (negative).** Coarse rotation/scale search on
gradient images + ECC affine + region-focused second pass + local patch
refinement, fusing nine transferred landmark sets. Leave-one-out result: no
better than predicting the cohort median (no-imaging baseline). Diagnosis:
inter-patient 2-D registration is dominated by soft-tissue and bladder/bowel
differences, and affine cannot span the observed 1.44× pelvis size range.

**Few-shot landmark network (success).** ~1 M-parameter U-Net, input
384×384 at 0.625 mm/px, one Gaussian heat-map per landmark (σ ≈ 4 mm).
Heavy augmentation (rotation ±12°, scale 0.85–1.2, shifts, elastic warps,
intensity/gamma/noise). Loss: MSE with 60× weight on positive regions —
without it the network collapses to the all-zero solution (blobs ≈ 0.05 % of
pixels). Prediction: sub-pixel heat-map peaks selected jointly for
consistency with a similarity transform of the training patients' mean
constellation (removes wrong-peak outliers). Training ≈ 7 min/fold on a
Quadro RTX 5000. Leave-one-out: inlet 2.3 mm, depth 3.4 mm, outlet 6.7 mm
mean absolute error; sacral angle 5.0° median with three large outliers
(short-ray formulation; fixes planned). Each fold fits its training images
to <1 mm, so residual test error is generalisation error.

## Stage 4 — one-click tumour / mesorectum measurement

Fully automatic tumour localisation is not learnable from eight examples
(demonstrated: a wide-view network confidently measured the wrong
structure). Design: the user clicks roughly on the tumour; the network sees
a fixed window centred on the click (80 mm tumour / 120 mm mesorectum and
craniocaudal tasks) and predicts heat-maps for the caliper endpoints.
Training simulates clicks ±8 mm off-centre; low-confidence ground-truth
endpoints are masked from the loss. Evaluation: leave-one-out, five random
clicks per test patient. Median errors fall within the expert re-reading
band (tumour 7–10 mm, mesorectum 8–9 mm); a minority of folds fail clearly
(degenerate heat-maps) and the craniocaudal extent is weakest — data-limited
rather than method-limited. Planned for the cohort phase: retraining at
n≈300 and automatic low-confidence flagging for human measurement.

## Honest-evaluation principles used throughout

Leave-one-out everywhere; a no-imaging baseline (cohort-median prediction)
reported alongside; observer variability as the success criterion rather
than zero error; negative results retained; every automated annotation
visually reviewed; uncertain values flagged, never silently included.
