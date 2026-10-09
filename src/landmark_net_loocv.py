"""O4 (method 2) — pelvimetry landmark detection with a small U-Net,
leave-one-out over the 10 pilot patients.

Tiny data regime: 9 training images per fold, compensated by heavy on-the-fly
augmentation (similarity + elastic deformation + intensity/gamma/noise).
Input 384x384 at 0.625 mm/px; output = 9 Gaussian heatmaps; prediction =
sub-pixel centroid of each heatmap peak. Patients flagged with unreliable
sacral ray geometry have those channels masked out of training.

Usage: python landmark_net_loocv.py [--steps N]
"""
import argparse
import os
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
import torch.nn as nn

PROJ = Path(__file__).resolve().parents[1]
DICOM_ROOT = Path(os.environ.get("AUTOMEASURE_DICOM_ROOT", "./dicom"))
LM = json.loads((PROJ / "data" / "ground_truth" / "pelvimetry_landmarks.json")
                .read_text(encoding="utf-8"))
OUT = PROJ / "data" / "pelvimetry_cnn"
QC = OUT / "qc"
NAMES = ["inlet_prom", "inlet_pub", "outlet_pub", "outlet_cocc",
         "depth_prom", "depth_cocc", "sacral_vertex", "sacral_up", "sacral_down"]
MM_FULL = 0.3125
NET = 384                 # network input size (0.625 mm/px)
SIGMA = 7.0               # heatmap sigma in net px
POS_W = 60.0              # extra loss weight on positive heatmap regions
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)
rng = np.random.default_rng(7)


def load_patient(pid):
    ds = pydicom.dcmread(str(DICOM_ROOT / LM[pid]["dicom_path"]))
    arr = ds.pixel_array.astype(np.float32)
    lo, hi = np.percentile(arr, (1, 99.5))
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    sp = LM[pid]["px_spacing"]
    lms = {n: np.array(LM[pid]["landmarks"][n], float) for n in NAMES}
    fx, fy = sp[1] / MM_FULL, sp[0] / MM_FULL
    if abs(fx - 1) > 1e-3 or abs(fy - 1) > 1e-3:
        arr = cv2.resize(arr, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        lms = {n: p * [fx, fy] for n, p in lms.items()}
    u8 = (arr * 255).astype(np.uint8)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8)
    eq = eq.astype(np.float32) / 255.0
    h, w = eq.shape
    canvas = np.zeros((NET * 2, NET * 2), np.float32)
    canvas[:min(h, NET * 2), :min(w, NET * 2)] = eq[:NET * 2, :NET * 2]
    img = cv2.resize(canvas, (NET, NET), interpolation=cv2.INTER_AREA)
    pts = np.array([lms[n] for n in NAMES]) / 2.0  # full-res px -> net px
    return img, pts


def augment(img, pts):
    h, w = img.shape
    for _ in range(12):
        th = rng.uniform(-12, 12)
        s = rng.uniform(0.85, 1.20)
        tx, ty = rng.uniform(-24, 24, 2)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), th, s)
        M[0, 2] += tx
        M[1, 2] += ty
        p2 = (M[:, :2] @ pts.T).T + M[:, 2]
        if (p2 > 10).all() and (p2 < NET - 10).all():
            break
    else:
        M = np.eye(2, 3, dtype=np.float64)
        p2 = pts.copy()
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    # elastic
    grid = rng.uniform(-8, 8, (2, 5, 5)).astype(np.float32)
    dx = cv2.resize(grid[0], (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(grid[1], (w, h), interpolation=cv2.INTER_CUBIC)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    out = cv2.remap(out, xx + dx, yy + dy, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT)
    ip = p2.round().astype(int).clip(0, NET - 1)
    d_at = np.stack([dx[ip[:, 1], ip[:, 0]], dy[ip[:, 1], ip[:, 0]]], axis=1)
    p2 = p2 - d_at
    # intensity
    out = np.clip(out, 0, 1) ** rng.uniform(0.7, 1.4)
    out = out * rng.uniform(0.85, 1.15) + rng.uniform(-0.08, 0.08)
    out = out + rng.normal(0, 0.02, out.shape).astype(np.float32)
    return np.clip(out, 0, 1), p2


AMP = 10.0  # heatmap peak amplitude: keeps MSE gradients from being
            # swamped by the empty background


def heatmaps(pts):
    """Gaussian blobs, computed only inside a 6-sigma window (fast)."""
    hm = np.zeros((len(NAMES), NET, NET), np.float32)
    r = int(6 * SIGMA)
    for i, (x, y) in enumerate(pts):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - r), min(NET, xi + r + 1)
        y0, y1 = max(0, yi - r), min(NET, yi + r + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1)[None, :] - x
        ys = np.arange(y0, y1)[:, None] - y
        hm[i, y0:y1, x0:x1] = AMP * np.exp(-(xs ** 2 + ys ** 2) / (2 * SIGMA ** 2))
    return hm


def top_peaks(hm, k=3, suppress=12):
    """Top-k local maxima of one heatmap channel."""
    h = hm.copy()
    out = []
    for _ in range(k):
        y, x = np.unravel_index(np.argmax(h), h.shape)
        if h[y, x] <= 0:
            break
        out.append(np.array([x, y], float))
        y0, y1 = max(0, y - suppress), min(h.shape[0], y + suppress + 1)
        x0, x1 = max(0, x - suppress), min(h.shape[1], x + suppress + 1)
        h[y0:y1, x0:x1] = -1
    return out


def umeyama(src, dst):
    """Least-squares similarity transform src->dst. Returns (s, R, t)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    cov = b.T @ a / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / max((a ** 2).sum() / len(src), 1e-9)
    t = md - s * (R @ ms)
    return s, R, t


def shape_select(hm, mean_shape):
    """Choose, per landmark, the top-3 peak most consistent with a similarity
    transform of the training mean shape; iterate fit->reselect."""
    cands = [top_peaks(hm[i]) for i in range(len(NAMES))]
    picks = np.array([c[0] if c else np.array([NET / 2, NET / 2])
                      for c in cands])
    for _ in range(3):
        # robust similarity fit: drop the 2 worst-fitting landmarks
        s, R, t = umeyama(mean_shape, picks)
        proj = (mean_shape @ (s * R).T) + t
        resid = np.linalg.norm(picks - proj, axis=1)
        keep = np.argsort(resid)[:len(NAMES) - 2]
        s, R, t = umeyama(mean_shape[keep], picks[keep])
        proj = (mean_shape @ (s * R).T) + t
        new = []
        for i, c in enumerate(cands):
            if not c:
                new.append(proj[i])
                continue
            d = [np.linalg.norm(p - proj[i]) for p in c]
            j = int(np.argmin(d))
            # accept a peak only within 40 px of the shape prediction
            new.append(c[j] if d[j] < 40 else proj[i])
        picks = np.array(new)
    # sub-pixel: centroid refine around each selected peak
    ref = []
    for i, p in enumerate(picks):
        x, y = int(round(p[0])), int(round(p[1]))
        r = 6
        y0, y1 = max(0, y - r), min(NET, y + r + 1)
        x0, x1 = max(0, x - r), min(NET, x + r + 1)
        w = np.clip(hm[i, y0:y1, x0:x1], 0, None) ** 2
        if w.sum() > 1e-9:
            ys, xs = np.mgrid[y0:y1, x0:x1]
            ref.append(np.array([(xs * w).sum() / w.sum(),
                                 (ys * w).sum() / w.sum()]))
        else:
            ref.append(p)
    return np.array(ref)


class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.ReLU(True),
            nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.ReLU(True))

    def forward(self, x):
        return self.f(x)


class UNet(nn.Module):
    def __init__(self, ch=(16, 32, 64, 128)):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = (Block(1, ch[0]), Block(ch[0], ch[1]),
                                              Block(ch[1], ch[2]), Block(ch[2], ch[3]))
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(ch[3], ch[2], 2, 2)
        self.d3 = Block(ch[2] * 2, ch[2])
        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2)
        self.d2 = Block(ch[1] * 2, ch[1])
        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2)
        self.d1 = Block(ch[0] * 2, ch[0])
        self.out = nn.Conv2d(ch[0], len(NAMES), 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat([self.up3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


def peak(hm):
    """sub-pixel centroid around the max of one heatmap channel."""
    y, x = np.unravel_index(np.argmax(hm), hm.shape)
    r = 6
    y0, y1 = max(0, y - r), min(hm.shape[0], y + r + 1)
    x0, x1 = max(0, x - r), min(hm.shape[1], x + r + 1)
    w = np.clip(hm[y0:y1, x0:x1], 0, None) ** 2
    if w.sum() < 1e-9:
        return np.array([x, y], float)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return np.array([(xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()])


def obtuse(v, e_up, e_down):
    u1, u2 = np.asarray(e_up) - v, np.asarray(e_down) - v
    c = abs(np.dot(u1, u2)) / (np.linalg.norm(u1) * np.linalg.norm(u2))
    return 180.0 - float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def values_from(pts):
    i = {n: k for k, n in enumerate(NAMES)}
    d = lambda a, b: float(np.linalg.norm(pts[i[a]] - pts[i[b]]) * 2 * MM_FULL)
    return dict(inlet=d("inlet_prom", "inlet_pub"),
                outlet=d("outlet_pub", "outlet_cocc"),
                depth=d("depth_prom", "depth_cocc"),
                sacral=obtuse(pts[i["sacral_vertex"]],
                              pts[i["sacral_up"]], pts[i["sacral_down"]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=700)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)
    pids = sorted(LM.keys())
    data = {p: load_patient(p) for p in pids}
    # resume: reload folds already completed by a previous (killed) run
    results = {}
    res_path = OUT / "results.json"
    if res_path.exists():
        results = json.loads(res_path.read_text(encoding="utf-8"))
        print(f"resuming: {sorted(results)} already done", flush=True)

    for test in pids:
        if test in results:
            continue
        t0 = time.time()
        train_pids = [p for p in pids if p != test]
        net = UNet().to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.steps, eta_min=1e-4)
        net.train()
        for step in range(args.steps):
            bx, by, bw = [], [], []
            for _ in range(4):
                p = train_pids[rng.integers(len(train_pids))]
                img, pts = augment(*data[p])
                bx.append(img[None])
                by.append(heatmaps(pts))
                w = np.ones(len(NAMES), np.float32)
                if not LM[p].get("sacral_geom_ok", True):
                    w[NAMES.index("sacral_vertex")] = 0
                    w[NAMES.index("sacral_up")] = 0
                    w[NAMES.index("sacral_down")] = 0
                bw.append(w)
            x = torch.from_numpy(np.stack(bx)).to(DEV)
            y = torch.from_numpy(np.stack(by)).to(DEV)
            w = torch.from_numpy(np.stack(bw))[:, :, None, None].to(DEV)
            opt.zero_grad()
            out = net(x)
            # positive-weighted MSE: without this the net collapses to the
            # all-zeros solution (blobs are ~0.05% of pixels)
            pw = 1.0 + POS_W * (y / AMP)
            loss = (((out - y) ** 2) * pw * w).mean()
            loss.backward()
            opt.step()
            sched.step()
            if step % 50 == 0 or step == args.steps - 1:
                print(f"    fold {test} step {step:4d} loss {loss.item():.5f}",
                      flush=True)
        net.eval()
        # sanity: how well does the net fit a TRAINING image (no augmentation)?
        tr = train_pids[0]
        tr_img, tr_pts = data[tr]
        with torch.no_grad():
            hm_tr = net(torch.from_numpy(tr_img[None, None]).to(DEV))[0].cpu().numpy()
        tr_err = np.mean([np.linalg.norm(peak(hm_tr[i]) - tr_pts[i]) * 2 * MM_FULL
                          for i in range(len(NAMES))])
        print(f"    fold {test}: train-image ({tr}) mean landmark err "
              f"{tr_err:.1f} mm", flush=True)
        img, gt_pts = data[test]
        with torch.no_grad():
            hm = net(torch.from_numpy(img[None, None]).to(DEV))[0].cpu().numpy()
        # mean training shape (geometry-flagged patients excluded for sacral)
        shapes = []
        for p in train_pids:
            shapes.append(data[p][1])
        shapes = np.array(shapes)
        mean_shape = shapes.mean(axis=0)
        for i, n in enumerate(NAMES):
            if n.startswith("sacral"):
                ok = [k for k, p in enumerate(train_pids)
                      if LM[p].get("sacral_geom_ok", True)]
                mean_shape[i] = shapes[ok, i].mean(axis=0)
        pred_pts = shape_select(hm, mean_shape)
        pred = values_from(pred_pts)
        excel = LM[test]["excel"]
        lm_err = {n: float(np.linalg.norm(pred_pts[i] - gt_pts[i]) * 2 * MM_FULL)
                  for i, n in enumerate(NAMES)}
        base = {k: float(np.median([LM[a]["excel"][k] for a in pids if a != test]))
                for k in excel}
        results[test] = dict(
            pred={k: round(v, 2) for k, v in pred.items()},
            excel=excel,
            delta={k: round(pred[k] - excel[k], 2) for k in excel},
            baseline_delta={k: round(base[k] - excel[k], 2) for k in excel},
            landmark_err_mm={k: round(v, 2) for k, v in lm_err.items()},
            pred_lm_px={n: (pred_pts[i] * 2).round(2).tolist()
                        for i, n in enumerate(NAMES)},
            gt_lm_px={n: (gt_pts[i] * 2).round(2).tolist()
                      for i, n in enumerate(NAMES)},
            sacral_gt_ok=LM[test].get("sacral_geom_ok", True),
            train_seconds=round(time.time() - t0, 1))
        d = results[test]["delta"]
        print(f"{test}: inlet {d['inlet']:+6.1f}  outlet {d['outlet']:+6.1f}  "
              f"depth {d['depth']:+6.1f}  sacral {d['sacral']:+6.1f}   "
              f"(mean lm err {np.mean(list(lm_err.values())):.1f} mm, "
              f"{results[test]['train_seconds']:.0f}s)", flush=True)
        with open(OUT / "results.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)  # incremental: survive kills

        vis = cv2.cvtColor((cv2.resize(img, (NET * 2, NET * 2)) * 255
                            ).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for i, n in enumerate(NAMES):
            g = (gt_pts[i] * 2).astype(int)
            p = (pred_pts[i] * 2).astype(int)
            cv2.drawMarker(vis, tuple(g), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
            cv2.circle(vis, tuple(p), 5, (0, 0, 255), 1)
            cv2.line(vis, tuple(g), tuple(p), (0, 200, 255), 1)
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
    lm_all = []
    for p in pids:
        for n in NAMES:
            if n.startswith("sacral") and not results[p]["sacral_gt_ok"]:
                continue
            lm_all.append(results[p]["landmark_err_mm"][n])
    lm_all = np.array(lm_all)
    print(f"\nlandmark error vs GT (mm): mean {lm_all.mean():.2f}, "
          f"median {np.median(lm_all):.2f}, p90 {np.percentile(lm_all, 90):.2f}, "
          f"max {lm_all.max():.2f}")
    within = sum(all(abs(results[p]["delta"][k]) <= (5.1 if k == "sacral" else 6.3)
                     for k in ("inlet", "outlet", "depth", "sacral")) for p in pids)
    print(f"patients with ALL four values within observer variability: "
          f"{within}/{len(pids)}")


if __name__ == "__main__":
    main()
