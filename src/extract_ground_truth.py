"""Extract measurement ground truth from annotated PACS screenshots + DICOM.

For every <pid>_<type>_annotated.jpg:
  1. find the DICOM slice shown (NCC content match over all series/slices/scales)
  2. isolate the burned-in caliper overlay (screenshot minus aligned render)
  3. detect measurement segments + the 3-point angle
  4. label them by matching their mm length/degrees to the original Excel values
  5. save endpoints (screenshot px, DICOM px, patient mm), deltas, and a QC image

Usage: python extract_ground_truth.py [pid ...]
"""
import csv
import json
import os
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pydicom

PROJ = Path(__file__).resolve().parents[1]
DICOM_ROOT = Path(os.environ.get("AUTOMEASURE_DICOM_ROOT", "./dicom"))
INV_CSV = PROJ / "docs" / "dicom_inventory.csv"
OUT_DIR = PROJ / "data" / "ground_truth"
QC_DIR = OUT_DIR / "qc"

# Original Excel (verified against images earlier). Units mm / degrees.
# Clinician-recorded values (mm / degrees) per patient: patient-derived
# data, NOT included in this repository. Supply your own file
# (schema: example_data/clinician_values.example.json).
EXCEL = json.loads(
    Path(os.environ.get("AUTOMEASURE_VALUES",
                        PROJ / "data" / "clinician_values.json"))
    .read_text(encoding="utf-8"))
TYPE_MEASURES = {
    "sagittal_landmarks": ["inlet", "outlet", "depth"],       # + sacral angle
    "sagittal_tumour": ["t_cc", "junction"],                   # optional per patient
    "axial_tumour": ["t_w", "t_l"],
    "axial_mrf": ["mrf_w", "mrf_l"],
}
DIST_TOL_MM = 2.8
ANGLE_TOL_DEG = 3.0


def orientation_ok(meas, ra, rb, slope=0.70):
    """Axial widths are near-horizontal, lengths near-vertical (<=35 deg off
    by default). Rejects diagonal label-leader lines. Others: unconstrained."""
    dx, dy = abs(rb[0] - ra[0]), abs(rb[1] - ra[1])
    if meas.endswith("_w"):
        return dy <= slope * dx
    if meas.endswith("_l"):
        return dx <= slope * dy
    # t_cc/junction: no orientation gate — the tumour axis follows the local
    # bowel direction and can approach horizontal (seen in one pilot patient)
    return True


def line_intersection(pa, ua, pb, ub):
    d = cross2(ua, ub)
    if abs(d) < 1e-9:
        return None
    t = cross2(pb - pa, ub) / d
    return pa + ua * t


def load_inventory():
    series = defaultdict(list)  # (folder, series_num, series_desc) -> [(instance, path)]
    with open(INV_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["folder"], r["series_num"], r["series_desc"])
            series[key].append((int(r["instance"] or 0), r["path"]))
    for v in series.values():
        v.sort()
    return series


def render(ds):
    """DICOM -> uint8 grayscale with robust windowing."""
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + inter
    lo, hi = np.percentile(arr, (1, 99.5))
    if hi <= lo:
        hi = lo + 1
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


class SeriesCache:
    def __init__(self):
        self.cache = {}

    def get(self, path):
        if path not in self.cache:
            ds = pydicom.dcmread(str(DICOM_ROOT / path))
            self.cache[path] = (ds, render(ds))
        return self.cache[path]


PAD = 14


def coarse_then_fine_match(screen_gray, cands, cache):
    """cands: list of dicom rel-paths. Returns (path, scale, x0, y0, ncc)."""
    screen_gray = cv2.copyMakeBorder(screen_gray, PAD, PAD, PAD, PAD,
                                     cv2.BORDER_CONSTANT, value=0)
    H, W = screen_gray.shape
    q = 0.25
    screen_q = cv2.resize(screen_gray, (int(W * q), int(H * q)), interpolation=cv2.INTER_AREA)
    best = None  # (ncc, path, scale, x, y)
    for path in cands:
        _, img = cache.get(path)
        base = (H - 2) / img.shape[0]  # viewer displays roughly fit-to-height
        for f in (0.70, 0.76, 0.82, 0.88, 0.94, 1.0):
            s = base * f
            t = cv2.resize(img, None, fx=s * q, fy=s * q, interpolation=cv2.INTER_AREA)
            if t.shape[0] >= screen_q.shape[0] or t.shape[1] >= screen_q.shape[1]:
                continue
            res = cv2.matchTemplate(screen_q, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if best is None or mx > best[0]:
                best = (mx, path, s, loc[0] / q, loc[1] / q)
    if best is None:
        return None
    _, path, s0, x0, y0 = best
    # fine search at full res around the coarse optimum
    _, img = cache.get(path)
    s_cap = (H - 2) / img.shape[0]
    samples = []
    for s in np.arange(max(0.4, s0 - 0.04), min(s0 + 0.045, s_cap), 0.005):
        t = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if t.shape[0] >= H or t.shape[1] >= W:
            continue
        res = cv2.matchTemplate(screen_gray, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        samples.append((s, mx, loc))
    if not samples:
        return None
    k = int(np.argmax([m for _, m, _ in samples]))
    s_best, ncc, loc = samples[k]
    # sub-step scale: parabolic interpolation over the NCC(s) samples
    if 0 < k < len(samples) - 1:
        y0_, y1_, y2_ = samples[k - 1][1], samples[k][1], samples[k + 1][1]
        denom = (y0_ - 2 * y1_ + y2_)
        if abs(denom) > 1e-9:
            off = 0.5 * (y0_ - y2_) / denom
            s_ref = s_best + np.clip(off, -1, 1) * 0.005
            t = cv2.resize(img, None, fx=s_ref, fy=s_ref,
                           interpolation=cv2.INTER_AREA)
            if t.shape[0] < H and t.shape[1] < W:
                res = cv2.matchTemplate(screen_gray, t, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc2 = cv2.minMaxLoc(res)
                if mx >= ncc - 0.002:
                    s_best, ncc, loc = s_ref, mx, loc2
    return path, float(s_best), float(loc[0] - PAD), float(loc[1] - PAD), float(ncc)


def overlay_residual(screen_gray, img, s, x0, y0):
    """Residual (screenshot minus aligned render) + validity, full-canvas."""
    H, W = screen_gray.shape
    M = np.float32([[s, 0, x0], [0, s, y0]])
    warped = cv2.warpAffine(img, M, (W, H)).astype(np.float32)
    valid = cv2.warpAffine(np.full(img.shape, 255, np.uint8), M, (W, H)) > 128
    scr = screen_gray.astype(np.float32)
    a, b = np.polyfit(warped[valid].ravel(), scr[valid].ravel(), 1)
    resid = scr - (a * warped + b)
    return resid, scr, valid


# (residual threshold, min screen brightness) — several operating points so a
# line that is marginal in one mask is caught in another; the collinear merge
# dedupes candidates across variants
MASK_VARIANTS = [(45, 0), (60, 0), (30, 178), (24, 195)]


def overlay_masks(resid, scr, valid):
    kernel = np.ones((3, 3), np.uint8)
    out = []
    union = np.zeros(scr.shape, np.uint8)
    for rt, bt in MASK_VARIANTS:
        m = (valid & (resid > rt) & (scr > bt)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        out.append(m)
        union |= m
    return out, union


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def merge_segments(segs):
    """Cluster collinear HoughP segments, return per cluster the extreme endpoints."""
    clusters = []
    for x1, y1, x2, y2 in segs:
        p1, p2 = np.array([x1, y1], float), np.array([x2, y2], float)
        d = p2 - p1
        L = np.hypot(*d)
        if L < 1:
            continue
        u = d / L
        placed = False
        for c in clusters:
            if abs(cross2(u, c["u"])) < 0.06:  # ~3.5 deg
                mid = (p1 + p2) / 2
                off = abs(cross2(mid - c["p0"], c["u"]))
                if off < 4.0:
                    c["pts"] += [p1, p2]
                    placed = True
                    break
        if not placed:
            clusters.append({"u": u, "p0": p1.copy(), "pts": [p1, p2]})
    out = []
    for c in clusters:
        pts = np.array(c["pts"])
        center = pts.mean(axis=0)
        d = pts - center
        # principal direction of the cluster
        u = c["u"]
        t = d @ u
        pa = center + u * t.min()
        pb = center + u * t.max()
        out.append((pa, pb, float(t.max() - t.min())))
    return out


def refine_endpoint(mask, region, p, u_out):
    """Snap an endpoint to the caliper end-tick (perpendicular stroke).

    mask is region-local; p is in screenshot coords; u_out points outward
    along the line. Returns refined endpoint (screenshot coords)."""
    x0, y0 = region[0], region[1]
    n = np.array([-u_out[1], u_out[0]])
    best = None
    for t in range(-9, 10):
        c = p + u_out * t
        cnt = 0
        # off-line pixel density in a box around c: catches perpendicular
        # ticks AND the X-shaped click markers of the polyline tools
        for dt in range(-3, 4):
            for k in list(range(-5, -1)) + list(range(2, 6)):
                q = c + u_out * dt + n * k
                xi, yi = int(round(q[0] - x0)), int(round(q[1] - y0))
                if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
                    cnt += 1
        if cnt >= 8 and (best is None or cnt > best[0] or
                         (cnt == best[0] and abs(t) < abs(best[1]))):
            best = (cnt, t)
    if best is None:
        return p
    return p + u_out * best[1]


def x_marker_center(mask, region, P0, lineA, lineB):
    """Find the X click-marker near P0: overlay pixels off BOTH line axes."""
    x0, y0 = region[0], region[1]
    best_p, best_c = None, 3
    for dy in range(-8, 9):
        for dx in range(-8, 9):
            P = P0 + np.array([dx, dy], float)
            cnt = 0
            for ddy in range(-6, 7):
                for ddx in range(-6, 7):
                    if ddx * ddx + ddy * ddy > 36:
                        continue
                    q = P + [ddx, ddy]
                    xi, yi = int(round(q[0] - x0)), int(round(q[1] - y0))
                    if not (0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]) \
                            or not mask[yi, xi]:
                        continue
                    if abs(cross2(q - lineA[0], lineA[1])) < 1.7:
                        continue
                    if abs(cross2(q - lineB[0], lineB[1])) < 1.7:
                        continue
                    cnt += 1
            if cnt > best_c:
                best_p, best_c = P, cnt
    return best_p


def extract(pid, img_type, screen_path, series_inv, cache, results):
    screen_bgr = cv2.imread(str(screen_path))
    screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    pat_series = {k: v for k, v in series_inv.items() if k[0] == pid}
    cands = [p for (_, _, _), files in pat_series.items() for _, p in files]

    m = coarse_then_fine_match(screen_gray, cands, cache)
    if m is None:
        results.append(dict(pid=pid, type=img_type, status="NO_MATCH"))
        return
    path, s, x0, y0, ncc = m
    ds, img = cache.get(path)
    region = (0, 0, screen_gray.shape[1], screen_gray.shape[0])
    resid, scr, valid = overlay_residual(screen_gray, img, s, x0, y0)
    masks, mask = overlay_masks(resid, scr, valid)  # mask = union (for refine)

    all_segs = []
    for m_v in masks:
        lines = cv2.HoughLinesP(m_v, 1, np.pi / 360, threshold=30,
                                minLineLength=16, maxLineGap=4)
        if lines is not None:
            all_segs.append(lines.reshape(-1, 4).astype(np.float64))
    segs = np.vstack(all_segs) if all_segs else []
    clusters = merge_segments(segs)

    px_mm = float(ds.PixelSpacing[0])  # mm per DICOM pixel (in-plane, isotropic here)
    mm_per_screenpx = px_mm / s

    expected = [k for k in TYPE_MEASURES[img_type] if k in EXCEL[pid]]

    # refine every sufficiently long cluster to its end-ticks
    refined = []
    for pa, pb, L in clusters:
        if L < 15:
            continue
        u = (pb - pa) / max(np.hypot(*(pb - pa)), 1e-6)
        ra = refine_endpoint(mask, region, pa, -u)
        rb = refine_endpoint(mask, region, pb, u)
        refined.append([ra, rb])

    # snap shared polygon vertices (e.g. promontory = inlet ∩ depth) to the
    # exact intersection of the two lines
    for i in range(len(refined)):
        for j in range(i + 1, len(refined)):
            ui = refined[i][1] - refined[i][0]
            uj = refined[j][1] - refined[j][0]
            ui = ui / max(np.hypot(*ui), 1e-6)
            uj = uj / max(np.hypot(*uj), 1e-6)
            for ei in range(2):
                for ej in range(2):
                    # tight radius: clinicians sometimes click nearby but
                    # DIFFERENT points for adjacent measurements (e.g. pelvic
                    # depth vs outlet at the coccyx, ~9 px apart)
                    if np.hypot(*(refined[i][ei] - refined[j][ej])) < 6:
                        if abs(cross2(ui, uj)) > 0.42:  # >~25 deg: intersection is stable
                            X = line_intersection(refined[i][0], ui, refined[j][0], uj)
                        else:  # shallow angle (chained calipers): find the X marker
                            mid = (refined[i][ei] + refined[j][ej]) / 2
                            X = x_marker_center(
                                mask, region, mid,
                                (refined[i][1 - ei], ui), (refined[j][1 - ej], uj))
                            if X is None:
                                X = mid
                        if X is not None and \
                           np.hypot(*(X - refined[i][ei])) < 7 and \
                           np.hypot(*(X - refined[j][ej])) < 7:
                            refined[i][ei] = X.copy()
                            refined[j][ej] = X.copy()
    refined = [(ra, rb, float(np.hypot(*(rb - ra)))) for ra, rb in refined]

    assigned = {}
    used = set()

    # axial tumour/MRF: the two calipers form a perpendicular CROSSING pair —
    # assign jointly (kills diagonal label-leaders, admits oblique crosses)
    if img_type.startswith("axial") and len(expected) == 2:
        m_w, m_l = expected  # width first by construction
        tw, tl = EXCEL[pid][m_w], EXCEL[pid][m_l]
        best = None
        for ci, (ra, rb, La) in enumerate(refined):
            for cj, (rc, rd, Lb) in enumerate(refined):
                if ci == cj:
                    continue
                mm_i = La * mm_per_screenpx
                mm_j = Lb * mm_per_screenpx
                if abs(mm_i - tw) >= DIST_TOL_MM or abs(mm_j - tl) >= DIST_TOL_MM:
                    continue
                ui = (rb - ra) / max(np.hypot(*(rb - ra)), 1e-6)
                uj = (rd - rc) / max(np.hypot(*(rd - rc)), 1e-6)
                if abs(np.dot(ui, uj)) > 0.34:  # not within ~20 deg of perpendicular
                    continue
                X = line_intersection(ra, ui, rc, uj)
                if X is None:
                    continue
                ti = (X - ra) @ ui
                tj = (X - rc) @ uj
                if not (-6 <= ti <= La + 6 and -6 <= tj <= Lb + 6):
                    continue  # must actually cross each other
                # 'width' should be the more horizontal member of the pair
                pen = 0.5 if abs(ui[0]) < abs(uj[0]) else 0.0
                err = abs(mm_i - tw) + abs(mm_j - tl) + pen
                if best is None or err < best[0]:
                    best = (err, ci, cj, mm_i, mm_j)
        if best is not None:
            _, ci, cj, mm_i, mm_j = best
            assigned[m_w] = (refined[ci][0], refined[ci][1], mm_i, "high")
            assigned[m_l] = (refined[cj][0], refined[cj][1], mm_j, "high")
            used.update((ci, cj))

    PARTNER = {"t_w": "t_l", "t_l": "t_w", "mrf_w": "mrf_l", "mrf_l": "mrf_w"}

    def near_partner(meas, ra, rb):
        """A width/length caliper must sit near its assigned partner (the two
        form a cross) — rejects far-away coincidental lines (e.g. fat edges)."""
        p = PARTNER.get(meas)
        if p is None or p not in assigned:
            return True
        pa, pb, *_ = assigned[p]
        return np.hypot(*((ra + rb) / 2 - (pa + pb) / 2)) < 55

    # sagittal pelvimetry: inlet/outlet/depth form a connected polygon —
    # inlet & depth share the promontory, outlet & depth the coccyx. Assign
    # the TRIPLE jointly; a stray anatomy edge with a coincidentally right
    # length cannot join the polygon.
    if img_type == "sagittal_landmarks":
        # wider net than the greedy tolerance: connectivity is the real
        # filter here, and contaminated clusters can be a few mm off
        cand = {}
        for meas in ("inlet", "outlet", "depth"):
            target = EXCEL[pid][meas]
            cand[meas] = [(ci, L * mm_per_screenpx)
                          for ci, (ra, rb, L) in enumerate(refined)
                          if abs(L * mm_per_screenpx - target) < 6.0]
        best = None
        for ci, mi in cand["inlet"]:
            for cj, mj in cand["outlet"]:
                for ck, mk in cand["depth"]:
                    if len({ci, cj, ck}) < 3:
                        continue
                    for de in (0, 1):  # which depth endpoint is the promontory
                        gp = min(np.hypot(*(refined[ci][e] - refined[ck][de]))
                                 for e in (0, 1))
                        gc = min(np.hypot(*(refined[cj][e] - refined[ck][1 - de]))
                                 for e in (0, 1))
                        if gp > 15 or gc > 15:
                            continue
                        score = (abs(mi - EXCEL[pid]["inlet"])
                                 + abs(mj - EXCEL[pid]["outlet"])
                                 + abs(mk - EXCEL[pid]["depth"])
                                 + 0.2 * (gp + gc) * mm_per_screenpx)
                        if best is None or score < best[0]:
                            best = (score, ci, cj, ck, mi, mj, mk)
        if best is not None:
            _, ci, cj, ck, mi, mj, mk = best
            conf = ("high" if all(abs(m - EXCEL[pid][k]) < DIST_TOL_MM
                                  for m, k in ((mi, "inlet"), (mj, "outlet"),
                                               (mk, "depth"))) else "low")
            assigned["inlet"] = (refined[ci][0], refined[ci][1], mi, conf)
            assigned["outlet"] = (refined[cj][0], refined[cj][1], mj, conf)
            assigned["depth"] = (refined[ck][0], refined[ck][1], mk, conf)
            used.update((ci, cj, ck))

    # globally-greedy assignment for whatever remains. Pelvimetry distances
    # are TRIPLE-ONLY: an isolated length match proved untrustworthy (bright
    # anatomy edges can coincidentally measure right) — better missing than
    # silently wrong.
    triple_only = ({"inlet", "outlet", "depth"}
                   if img_type == "sagittal_landmarks" else set())
    pairs = []
    for mi, meas in enumerate(expected):
        if meas in assigned or meas in triple_only:
            continue
        target = EXCEL[pid][meas]
        for ci, (ra, rb, L) in enumerate(refined):
            if ci in used:
                continue
            mm = L * mm_per_screenpx
            err = abs(mm - target)
            if err < DIST_TOL_MM and orientation_ok(meas, ra, rb):
                pairs.append((err, mi, ci, mm))
    for _, mi, ci, mm in sorted(pairs):
        meas = expected[mi]
        if meas in assigned or ci in used:
            continue
        if not near_partner(meas, refined[ci][0], refined[ci][1]):
            continue
        assigned[meas] = (refined[ci][0], refined[ci][1], mm, "high")
        used.add(ci)
    # map used refined-cluster indices back for the angle search
    clusters = [(ra, rb, L) for ra, rb, L in refined]

    # fallback pass for still-missing measures: wider tolerance, plus
    # bridging of collinear cluster pairs broken by their text label
    still = [m for m in expected if m not in assigned]
    if still:
        pool = [c for i, c in enumerate(clusters) if i not in used and c[2] > 15]
        bridged = []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                ui = pool[i][1] - pool[i][0]
                uj = pool[j][1] - pool[j][0]
                ui = ui / max(np.hypot(*ui), 1e-6)
                uj = uj / max(np.hypot(*uj), 1e-6)
                if abs(cross2(ui, uj)) > 0.10:
                    continue
                if abs(cross2(pool[j][0] - pool[i][0], ui)) > 6:
                    continue
                pts = np.array([pool[i][0], pool[i][1], pool[j][0], pool[j][1]])
                ctr = pts.mean(axis=0)
                t = (pts - ctr) @ ui
                bridged.append((ctr + ui * t.min(), ctr + ui * t.max(),
                                float(t.max() - t.min())))
        fb_pairs = []
        cand_pool = pool + bridged
        for mi, meas in enumerate(expected):
            if meas in assigned or meas in triple_only:
                continue
            target = EXCEL[pid][meas]
            for ci, (ra, rb, L) in enumerate(cand_pool):
                mm = L * mm_per_screenpx
                err = abs(mm - target)
                if err < 4.5 and orientation_ok(meas, ra, rb, slope=1.0):
                    fb_pairs.append((err, mi, ci, mm))
        fb_used = set()
        for _, mi, ci, mm in sorted(fb_pairs):
            meas = expected[mi]
            if meas in assigned or ci in fb_used:
                continue
            if not near_partner(meas, cand_pool[ci][0], cand_pool[ci][1]):
                continue
            assigned[meas] = (cand_pool[ci][0], cand_pool[ci][1], mm, "low")
            fb_used.add(ci)

    # sacral angle: Cobb-style — TWO DISJOINT segments (upper sacrum, lower
    # sacrum/coccyx); the value is the angle between their directions.
    angle_info = None
    if img_type == "sagittal_landmarks":
        target = EXCEL[pid]["sacral"]

        def on_assigned_line(p):
            for pa, pb, *_ in assigned.values():
                u = (pb - pa) / max(np.hypot(*(pb - pa)), 1e-6)
                w = p - pa
                t = w @ u
                if -3 <= t <= np.hypot(*(pb - pa)) + 3 and abs(cross2(w, u)) < 3.5:
                    return True
            return False

        def near_depth(ra, rb):
            """Sacral rays run along the sacrum, close to the depth line."""
            if "depth" not in assigned:
                return True
            da, db, *_ = assigned["depth"]
            u = (db - da) / max(np.hypot(*(db - da)), 1e-6)
            mid = (ra + rb) / 2
            w = mid - da
            t = np.clip(w @ u, 0, np.hypot(*(db - da)))
            return np.hypot(*(w - t * u)) < 90

        rays = []
        for i, (ra, rb, L) in enumerate(clusters):
            if i in used or L < 22:
                continue
            if on_assigned_line(ra) and on_assigned_line(rb):
                continue
            if not near_depth(ra, rb):
                continue
            rays.append((ra, rb))
        best = None
        for i in range(len(rays)):
            for j in range(i + 1, len(rays)):
                mid_d = np.hypot(*((rays[i][0] + rays[i][1]) / 2
                                   - (rays[j][0] + rays[j][1]) / 2))
                if mid_d > 230:
                    continue
                ui = rays[i][1] - rays[i][0]
                uj = rays[j][1] - rays[j][0]
                ang = math.degrees(math.acos(np.clip(
                    abs(np.dot(ui, uj)) / (np.linalg.norm(ui) * np.linalg.norm(uj)), -1, 1)))
                for cand in (ang, 180 - ang):
                    err = abs(cand - target)
                    if err < ANGLE_TOL_DEG and (best is None or err < best[0]):
                        X = line_intersection(
                            rays[i][0], ui / np.linalg.norm(ui),
                            rays[j][0], uj / np.linalg.norm(uj))
                        vertex = X if X is not None else (rays[i][1] + rays[j][0]) / 2
                        # the sacral bend is at the LOWER sacrum: vertex must
                        # be nearer the coccyx end of the depth line than the
                        # promontory end (kills label-leader pairs up top)
                        if "depth" in assigned:
                            da, db, *_ = assigned["depth"]
                            prom, cocc = (da, db) if da[1] <= db[1] else (db, da)
                            if np.hypot(*(vertex - cocc)) > np.hypot(*(vertex - prom)):
                                continue
                        far_i = max(rays[i], key=lambda p: np.hypot(*(p - vertex)))
                        far_j = max(rays[j], key=lambda p: np.hypot(*(p - vertex)))
                        best = (err, vertex, far_i, far_j, cand)
        if best is not None:
            angle_info = best

    # coordinate transforms
    ipp = np.array(ds.ImagePositionPatient, float)
    iop = np.array(ds.ImageOrientationPatient, float)
    rowv, colv = iop[:3], iop[3:]
    sp = [float(v) for v in ds.PixelSpacing]  # [row(dy), col(dx)]

    def to_dicom_px(p):
        return (p[0] - x0) / s, (p[1] - y0) / s

    def to_patient(p):
        cx, cy = to_dicom_px(p)
        return (ipp + rowv * cx * sp[1] + colv * cy * sp[0]).tolist()

    rec = dict(pid=pid, type=img_type, status="OK",
               dicom_path=str(path), series=str(ds.SeriesNumber),
               series_desc=str(getattr(ds, "SeriesDescription", "")),
               instance=int(ds.InstanceNumber), sop=str(ds.SOPInstanceUID),
               ncc=round(ncc, 4), slice_match_reliable=bool(ncc >= 0.85),
               scale=round(s, 5),
               offset=[round(x0, 2), round(y0, 2)],
               px_spacing=sp, measurements={})
    qc = screen_bgr.copy()
    for meas, (pa, pb, mm, conf) in assigned.items():
        col = (0, 0, 255) if conf == "high" else (0, 165, 255)
        rec["measurements"][meas] = dict(
            mm=round(mm, 2), excel=EXCEL[pid][meas],
            delta=round(mm - EXCEL[pid][meas], 2), confidence=conf,
            screen_px=[pa.round(1).tolist(), pb.round(1).tolist()],
            dicom_px=[[round(v, 2) for v in to_dicom_px(pa)],
                      [round(v, 2) for v in to_dicom_px(pb)]],
            patient_mm=[to_patient(pa), to_patient(pb)])
        cv2.line(qc, tuple(pa.astype(int)), tuple(pb.astype(int)), col, 1)
        for p in (pa, pb):
            cv2.circle(qc, tuple(p.astype(int)), 4, (0, 255, 0), 1)
        cv2.putText(qc, f"{meas} {mm:.1f}", tuple((pa + [4, -6]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    if angle_info is not None:
        _, vx, e1, e2, ang = angle_info
        rec["measurements"]["sacral"] = dict(
            deg=round(ang, 2), excel=EXCEL[pid]["sacral"],
            delta=round(ang - EXCEL[pid]["sacral"], 2),
            screen_px=[vx.round(1).tolist(), e1.round(1).tolist(), e2.round(1).tolist()],
            dicom_px=[[round(v, 2) for v in to_dicom_px(p)] for p in (vx, e1, e2)],
            patient_mm=[to_patient(p) for p in (vx, e1, e2)])
        cv2.line(qc, tuple(vx.astype(int)), tuple(e1.astype(int)), (255, 0, 255), 1)
        cv2.line(qc, tuple(vx.astype(int)), tuple(e2.astype(int)), (255, 0, 255), 1)
        cv2.putText(qc, f"sacral {ang:.1f}", tuple((vx + [6, 18]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    missing = [m for m in expected if m not in assigned]
    if img_type == "sagittal_landmarks" and angle_info is None:
        missing.append("sacral")
    rec["missing"] = missing
    if missing:
        cl = sorted(round(L * mm_per_screenpx, 1) for _, _, L in clusters if L > 15)
        print(f"    DEBUG {pid} {img_type}: clusters(mm)={cl} "
              f"targets={ {m: EXCEL[pid].get(m) for m in missing} }")
        for i, (ra, rb, L) in enumerate(clusters):
            if L > 25:
                tag = "used" if i in used else "free"
                print(f"      {tag} {L * mm_per_screenpx:6.1f}mm "
                      f"({ra[0]:.1f},{ra[1]:.1f})-({rb[0]:.1f},{rb[1]:.1f})")
    if missing:
        rec["status"] = "PARTIAL"
    results.append(rec)
    cv2.rectangle(qc, (int(x0), int(y0)),
                  (int(x0 + img.shape[1] * s), int(y0 + img.shape[0] * s)),
                  (80, 80, 80), 1)
    cv2.imwrite(str(QC_DIR / f"{pid}_{img_type}.png"), qc)
    with open(OUT_DIR / f"{pid}_{img_type}.json", "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    series_inv = load_inventory()
    pids = sys.argv[1:] or sorted(EXCEL.keys())
    results = []
    for pid in pids:
        cache = SeriesCache()
        folder = PROJ / pid
        for f in sorted(folder.glob("*anno*.jpg")):
            name = f.stem.replace(pid + "_", "").replace("_annotated", "").replace("_annoted", "")
            if name not in TYPE_MEASURES:
                continue
            extract(pid, name, f, series_inv, cache, results)
            r = results[-1]
            if r["status"] == "NO_MATCH":
                print(f"{pid} {name:20s} NO_MATCH")
            else:
                deltas = ", ".join(
                    f"{k}:{v['delta']:+.1f}{'*' if v.get('confidence') == 'low' else ''}"
                    for k, v in r["measurements"].items())
                rel = "" if r["slice_match_reliable"] else " WRONG-SLICE?"
                print(f"{pid} {name:20s} {r['status']:7s} ncc={r['ncc']:.3f} "
                      f"s{r['series']}#{r['instance']:>3} [{deltas}] "
                      f"missing={r['missing']}{rel}")
    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n{ok}/{len(results)} screenshots fully extracted")
    # merge into existing summary so per-patient reruns don't clobber others
    sum_path = OUT_DIR / "summary.json"
    merged = {}
    if sum_path.exists():
        for r in json.loads(sum_path.read_text(encoding="utf-8")):
            merged[(r["pid"], r["type"])] = r
    for r in results:
        merged[(r["pid"], r["type"])] = r
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(sorted(merged.values(), key=lambda r: (r["pid"], r["type"])),
                  fh, indent=1)


if __name__ == "__main__":
    main()
