"""O4 — automatic pelvimetry by multi-atlas landmark transfer, leave-one-out.

Per (atlas -> test) pair:
  1. GLOBAL: coarse exhaustive search (5 rotations x 11 scales 0.75-1.35 x
     translation) on gradient magnitude at 0.625 mm/px + ECC affine.
  2. ROI: second ECC affine restricted to the pelvic region (bounding box of
     all transferred landmarks + 30 mm), correcting residual global bias.
  3. LOCAL x2: per-landmark patch NCC refinement (31 mm patch); first pass
     +/-20 mm around each atlas's own transfer, second pass +/-12 mm around
     the fused estimate.
Fusion: trimmed coordinate-wise median (drop 3 worst atlases by local NCC).
Atlases with unreliable sacral ray geometry are excluded for sacral points.

Usage: python pelvimetry_loocv.py
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pydicom

PROJ = Path(__file__).resolve().parents[1]
DICOM_ROOT = Path(os.environ.get("AUTOMEASURE_DICOM_ROOT", "./dicom"))
LM = json.loads((PROJ / "data" / "ground_truth" / "pelvimetry_landmarks.json")
                .read_text(encoding="utf-8"))
OUT = PROJ / "data" / "pelvimetry_loocv"
QC = OUT / "qc"
NAMES = ["inlet_prom", "inlet_pub", "outlet_pub", "outlet_cocc",
         "depth_prom", "depth_cocc", "sacral_vertex", "sacral_up", "sacral_down"]
MM = 0.3125
DS = 2


def load_patient(pid):
    ds = pydicom.dcmread(str(DICOM_ROOT / LM[pid]["dicom_path"]))
    arr = ds.pixel_array.astype(np.float32)
    lo, hi = np.percentile(arr, (1, 99.5))
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    sp = LM[pid]["px_spacing"]
    lms = {n: np.array(LM[pid]["landmarks"][n], float) for n in NAMES}
    fx, fy = sp[1] / MM, sp[0] / MM
    if abs(fx - 1) > 1e-3 or abs(fy - 1) > 1e-3:
        arr = cv2.resize(arr, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        lms = {n: p * [fx, fy] for n, p in lms.items()}
    u8 = (arr * 255).astype(np.uint8)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8)
    eq = eq.astype(np.float32) / 255.0
    g = cv2.GaussianBlur(eq, (5, 5), 0)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    grad = np.clip(grad / max(np.percentile(grad, 99), 1e-6), 0, 1)
    return dict(img=arr, eq=eq, grad=grad.astype(np.float32), lms=lms)


def compose(A, B):
    """2x3 affines: return A o B (apply B first, then A)."""
    A3 = np.vstack([A, [0, 0, 1]])
    B3 = np.vstack([B, [0, 0, 1]])
    return (A3 @ B3)[:2]


def apply_aff(M, p):
    return np.array([M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2],
                     M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2]])


def global_affine(atlas, test):
    """2x3 affine mapping atlas full-res px -> test full-res px."""
    ga = cv2.resize(atlas["grad"], None, fx=1 / DS, fy=1 / DS,
                    interpolation=cv2.INTER_AREA)
    gt = cv2.resize(test["grad"], None, fx=1 / DS, fy=1 / DS,
                    interpolation=cv2.INTER_AREA)
    h, w = ga.shape
    canvas = int(1.5 * max(h, w))
    off = (canvas - w) // 2, (canvas - h) // 2
    pad = (canvas - min(h, w)) // 2 + 20
    gt_p = cv2.copyMakeBorder(gt, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    ctr = (w / 2, h / 2)
    best = None
    for th in (-12, -6, 0, 6, 12):
        for s in np.arange(0.75, 1.36, 0.06):
            M = cv2.getRotationMatrix2D(ctr, th, s)
            M[0, 2] += off[0]
            M[1, 2] += off[1]
            wa = cv2.warpAffine(ga, M, (canvas, canvas))
            res = cv2.matchTemplate(gt_p, wa, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if best is None or mx > best[0]:
                best = (mx, M.copy(), loc)
    score, M, loc = best
    M = M.copy()
    M[0, 2] += loc[0] - pad
    M[1, 2] += loc[1] - pad
    # ECC refinement at downsampled res (W maps test->atlas; invert for use)
    try:
        init = cv2.invertAffineTransform(M).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-6)
        _, W = cv2.findTransformECC(gt, ga, init, cv2.MOTION_AFFINE, crit,
                                    None, 5)
        M = cv2.invertAffineTransform(W)
    except cv2.error:
        pass
    S = np.array([[1 / DS, 0, 0], [0, 1 / DS, 0]], float)
    return compose(np.array([[DS, 0, 0], [0, DS, 0]], float),
                   compose(M, S)), float(score)


def roi_ecc(test_eq, warped_eq, pts, pad_px=96):
    """ECC affine restricted to the landmark region. Returns 2x3 correction
    mapping current test-frame points -> corrected test-frame points."""
    H, W_ = test_eq.shape
    arr = np.array(list(pts.values()))
    x0 = int(max(0, arr[:, 0].min() - pad_px))
    x1 = int(min(W_, arr[:, 0].max() + pad_px))
    y0 = int(max(0, arr[:, 1].min() - pad_px))
    y1 = int(min(H, arr[:, 1].max() + pad_px))
    if x1 - x0 < 80 or y1 - y0 < 80:
        return np.array([[1, 0, 0], [0, 1, 0]], float)
    t_roi = test_eq[y0:y1, x0:x1]
    a_roi = warped_eq[y0:y1, x0:x1]
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-6)
        init = np.eye(2, 3, dtype=np.float32)
        _, Wm = cv2.findTransformECC(t_roi, a_roi, init, cv2.MOTION_AFFINE,
                                     crit, None, 5)
        C = cv2.invertAffineTransform(Wm)
    except cv2.error:
        return np.array([[1, 0, 0], [0, 1, 0]], float)
    T0 = np.array([[1, 0, -x0], [0, 1, -y0]], float)
    T1 = np.array([[1, 0, x0], [0, 1, y0]], float)
    return compose(T1, compose(C, T0))


def local_refine(src_eq, dst_eq, p_src, p_dst0, patch_px=99, win_px=64):
    """Match a patch of src_eq around p_src into dst_eq near p_dst0."""
    r = patch_px // 2
    hs, ws = src_eq.shape
    hd, wd = dst_eq.shape
    xs, ys = int(round(p_src[0])), int(round(p_src[1]))
    xd, yd = int(round(p_dst0[0])), int(round(p_dst0[1]))
    if not (r < xs < ws - r and r < ys < hs - r):
        return p_dst0, 0.0
    if not (r + win_px < xd < wd - r - win_px and r + win_px < yd < hd - r - win_px):
        return p_dst0, 0.0
    tmpl = src_eq[ys - r:ys + r + 1, xs - r:xs + r + 1]
    if tmpl.std() < 1e-4:
        return p_dst0, 0.0
    sub = dst_eq[yd - r - win_px:yd + r + win_px + 1,
                 xd - r - win_px:xd + r + win_px + 1]
    res = cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    if mx < 0.25:
        return p_dst0, float(mx)
    return np.array([xd - win_px + loc[0], yd - win_px + loc[1]], float), float(mx)


def obtuse(v, e_up, e_down):
    u1, u2 = np.asarray(e_up) - v, np.asarray(e_down) - v
    c = abs(np.dot(u1, u2)) / (np.linalg.norm(u1) * np.linalg.norm(u2))
    return 180.0 - float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def values_from(pts):
    d = lambda a, b: float(np.linalg.norm(pts[a] - pts[b]) * MM)
    return dict(inlet=d("inlet_prom", "inlet_pub"),
                outlet=d("outlet_pub", "outlet_cocc"),
                depth=d("depth_prom", "depth_cocc"),
                sacral=obtuse(pts["sacral_vertex"],
                              pts["sacral_up"], pts["sacral_down"]))


def sacral_ok(pid):
    return LM[pid].get("sacral_geom_ok", True)


def fuse(transferred, weights, name, trim=3):
    """Trimmed coordinate-wise median, worst `trim` atlases (by local ncc)
    dropped; sacral points exclude geometry-unreliable atlases."""
    items = [(a, t[name]) for a, t in transferred.items()
             if not (name.startswith("sacral") and not sacral_ok(a))]
    if len(items) > trim + 2:
        items.sort(key=lambda kv: -weights[kv[0]])
        items = items[:len(items) - trim]
    return np.median(np.array([p for _, p in items]), axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)
    pids = sorted(LM.keys())
    pats = {p: load_patient(p) for p in pids}
    results = {}

    for test in pids:
        T = pats[test]
        transferred, local_w, warps, anchor = {}, {}, {}, {}
        for atlas in pids:
            if atlas == test:
                continue
            A = pats[atlas]
            Mf, g_ncc = global_affine(A, T)
            warped = cv2.warpAffine(A["eq"], Mf, T["eq"].shape[::-1])
            pts = {n: apply_aff(Mf, A["lms"][n]) for n in NAMES}
            C = roi_ecc(T["eq"], warped, pts)
            Mc = compose(C, Mf)
            warped = cv2.warpAffine(A["eq"], Mc, T["eq"].shape[::-1])
            anchor[atlas] = {n: apply_aff(Mc, A["lms"][n]) for n in NAMES}
            warps[atlas] = warped
            # local pass 1: search around each atlas's own transfer
            pts, nccs = {}, []
            for n in NAMES:
                p1, ncc = local_refine(warped, T["eq"], anchor[atlas][n],
                                       anchor[atlas][n], win_px=64)
                pts[n] = p1
                nccs.append(ncc)
            transferred[atlas] = pts
            local_w[atlas] = float(np.mean(nccs))
        fused = {n: fuse(transferred, local_w, n) for n in NAMES}
        # local pass 2: re-search each atlas patch around the FUSED estimate
        transferred2 = {}
        for atlas, pts in transferred.items():
            pts2 = {}
            for n in NAMES:
                p2, ncc = local_refine(warps[atlas], T["eq"],
                                       anchor[atlas][n], fused[n], win_px=40)
                pts2[n] = p2 if ncc >= 0.25 else pts[n]
            transferred2[atlas] = pts2
        fused2 = {n: fuse(transferred2, local_w, n) for n in NAMES}
        pred = values_from(fused2)

        excel = LM[test]["excel"]
        gt_pts = T["lms"]
        lm_err = {n: float(np.linalg.norm(fused2[n] - gt_pts[n]) * MM)
                  for n in NAMES}
        base = {k: float(np.median([LM[a]["excel"][k] for a in pids if a != test]))
                for k in excel}
        results[test] = dict(
            pred={k: round(v, 2) for k, v in pred.items()},
            excel=excel,
            delta={k: round(pred[k] - excel[k], 2) for k in excel},
            baseline_delta={k: round(base[k] - excel[k], 2) for k in excel},
            landmark_err_mm={k: round(v, 2) for k, v in lm_err.items()},
            atlas_local_ncc={a: round(w, 3) for a, w in local_w.items()},
            fused_landmarks_px={n: fused2[n].round(2).tolist() for n in NAMES})
        d = results[test]["delta"]
        print(f"{test}: inlet {d['inlet']:+6.1f}  outlet {d['outlet']:+6.1f}  "
              f"depth {d['depth']:+6.1f}  sacral {d['sacral']:+6.1f}   "
              f"(mean lm err {np.mean(list(lm_err.values())):.1f} mm)")

        vis = cv2.cvtColor((T["img"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for n in NAMES:
            g = gt_pts[n].astype(int)
            p = fused2[n].astype(int)
            cv2.drawMarker(vis, tuple(g), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
            cv2.circle(vis, tuple(p), 5, (0, 0, 255), 1)
            cv2.line(vis, tuple(g), tuple(p), (0, 200, 255), 1)
        for a, b in (("inlet_prom", "inlet_pub"), ("outlet_pub", "outlet_cocc"),
                     ("depth_prom", "depth_cocc"), ("sacral_vertex", "sacral_up"),
                     ("sacral_vertex", "sacral_down")):
            cv2.line(vis, tuple(fused2[a].astype(int)),
                     tuple(fused2[b].astype(int)), (0, 0, 255), 1)
        cv2.imwrite(str(QC / f"{test}.png"), vis)

    with open(OUT / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)

    print("\n== summary (|prediction - clinician|) ==")
    for k in ("inlet", "outlet", "depth", "sacral"):
        d = np.array([abs(results[p]["delta"][k]) for p in pids])
        b = np.array([abs(results[p]["baseline_delta"][k]) for p in pids])
        unit = "deg" if k == "sacral" else "mm"
        print(f"{k:>7s}: mean {d.mean():5.2f} median {np.median(d):5.2f} "
              f"max {d.max():5.2f} {unit:3s} | no-image baseline {b.mean():5.2f}")
    lm_all = np.array([[results[p]["landmark_err_mm"][n] for n in NAMES]
                       for p in pids])
    print(f"\nlandmark error (mm): mean {lm_all.mean():.2f}, "
          f"median {np.median(lm_all):.2f}, p90 {np.percentile(lm_all, 90):.2f}, "
          f"max {lm_all.max():.2f}")
    within = sum(all(abs(results[p]["delta"][k]) <= (5.1 if k == "sacral" else 6.3)
                     for k in ("inlet", "outlet", "depth", "sacral")) for p in pids)
    print(f"patients with ALL four values within observer variability: "
          f"{within}/{len(pids)}")


if __name__ == "__main__":
    main()
