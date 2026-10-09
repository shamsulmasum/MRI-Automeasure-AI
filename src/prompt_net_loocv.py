"""O5 — click-prompted tumour/MRF measurement, leave-one-out.

The "user" clicks roughly on the tumour/rectum; the network sees a 160 mm
window centred on the click (256x256 at 0.625 mm/px) and predicts heatmaps
for the caliper endpoints of that task. Training clicks are jittered +/-8 mm
around the true caliper-cross centre; evaluation repeats each test patient
with 5 random clicks to measure click-robustness.

Tasks: axial_tumour (t_w, t_l), axial_mrf (mrf_w, mrf_l), sag_tcc (t_cc).

Usage: python prompt_net_loocv.py [--task all] [--steps 900]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch

sys.path.insert(0, str(Path(__file__).parent))
from landmark_net_loocv import UNet, peak  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
DICOM_ROOT = Path(os.environ.get("AUTOMEASURE_DICOM_ROOT", "./dicom"))
SOFT = json.loads((PROJ / "data" / "ground_truth" / "soft_landmarks.json")
                  .read_text(encoding="utf-8"))
OUT = PROJ / "data" / "prompt_cnn"
MMPX = 0.625              # working resolution
CROP = 256                # set per task in run_task
# tight windows: force locality — "measure the structure AT the click",
# excluding lookalike distractors elsewhere in the pelvis
TASK_CROP = {"axial_tumour": 128, "axial_mrf": 192, "sag_tcc": 192}
SIGMA = 6.0
AMP, POS_W = 10.0, 60.0
JIT = 13                  # click jitter in px (~8 mm)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(11)
rng = np.random.default_rng(11)

TASK_MEAS = {"axial_tumour": ["t_w", "t_l"], "axial_mrf": ["mrf_w", "mrf_l"],
             "sag_tcc": ["t_cc"]}


def chan_names(task):
    return [f"{m}_{e}" for m in TASK_MEAS[task] for e in ("a", "b")]


def load_patient(task, pid):
    r = SOFT[task][pid]
    ds = pydicom.dcmread(str(DICOM_ROOT / r["dicom_path"]))
    arr = ds.pixel_array.astype(np.float32)
    lo, hi = np.percentile(arr, (1, 99.5))
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    sp = r["px_spacing"]
    fx, fy = sp[1] / MMPX, sp[0] / MMPX
    arr = cv2.resize(arr, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
        (arr * 255).astype(np.uint8)).astype(np.float32) / 255.0
    pts = np.array([np.array(r["landmarks"][n]) * [fx, fy]
                    for n in chan_names(task)])
    mask = np.array([r["train_mask"][n] for n in chan_names(task)], np.float32)
    return eq, pts, mask, r["excel"]


def crop_at(eq, pts, centre):
    """256x256 crop centred on `centre` (padded if outside)."""
    h, w = eq.shape
    x0 = int(round(centre[0])) - CROP // 2
    y0 = int(round(centre[1])) - CROP // 2
    canvas = np.zeros((CROP, CROP), np.float32)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + CROP), min(h, y0 + CROP)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = eq[sy0:sy1, sx0:sx1]
    return canvas, pts - [x0, y0]


def heatmaps(pts, k):
    hm = np.zeros((k, CROP, CROP), np.float32)
    r = int(6 * SIGMA)
    for i, (x, y) in enumerate(pts):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - r), min(CROP, xi + r + 1)
        y0, y1 = max(0, yi - r), min(CROP, yi + r + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1)[None, :] - x
        ys = np.arange(y0, y1)[:, None] - y
        hm[i, y0:y1, x0:x1] = AMP * np.exp(-(xs ** 2 + ys ** 2) / (2 * SIGMA ** 2))
    return hm


def augment(img, pts):
    h, w = img.shape
    for _ in range(12):
        th = rng.uniform(-15, 15)
        s = rng.uniform(0.85, 1.20)
        tx, ty = rng.uniform(-8, 8, 2)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), th, s)
        M[0, 2] += tx
        M[1, 2] += ty
        p2 = (M[:, :2] @ pts.T).T + M[:, 2]
        if (p2 > 8).all() and (p2 < CROP - 8).all():
            break
    else:
        M = np.eye(2, 3)
        p2 = pts.copy()
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    grid = rng.uniform(-7, 7, (2, 5, 5)).astype(np.float32)
    dx = cv2.resize(grid[0], (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(grid[1], (w, h), interpolation=cv2.INTER_CUBIC)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    out = cv2.remap(out, xx + dx, yy + dy, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT)
    ip = p2.round().astype(int).clip(0, CROP - 1)
    p2 = p2 - np.stack([dx[ip[:, 1], ip[:, 0]], dy[ip[:, 1], ip[:, 0]]], axis=1)
    out = np.clip(out, 0, 1) ** rng.uniform(0.7, 1.4)
    out = out * rng.uniform(0.85, 1.15) + rng.uniform(-0.08, 0.08)
    out = out + rng.normal(0, 0.02, out.shape).astype(np.float32)
    return np.clip(out, 0, 1), p2


def values_from(pts, task):
    v = {}
    for j, m in enumerate(TASK_MEAS[task]):
        a, b = pts[2 * j], pts[2 * j + 1]
        v[m] = float(np.linalg.norm(a - b) * MMPX)
    return v


def run_task(task, steps):
    global CROP
    CROP = TASK_CROP[task]
    pids = sorted(SOFT[task].keys())
    data = {p: load_patient(task, p) for p in pids}
    K = len(chan_names(task))
    qc_dir = OUT / "qc" / task
    qc_dir.mkdir(parents=True, exist_ok=True)
    res_path = OUT / f"results_{task}.json"
    results = json.loads(res_path.read_text(encoding="utf-8")) \
        if res_path.exists() else {}

    for test in pids:
        if test in results:
            continue
        t0 = time.time()
        train = [p for p in pids if p != test]
        net = UNet().to(DEV)
        net.out = torch.nn.Conv2d(16, K, 1).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=steps, eta_min=1e-4)
        net.train()
        for step in range(steps):
            bx, by, bw = [], [], []
            for _ in range(6):
                p = train[rng.integers(len(train))]
                eq, pts, mask, _ = data[p]
                centre = pts.mean(axis=0) + rng.uniform(-JIT, JIT, 2)
                img, cpts = crop_at(eq, pts, centre)
                img, cpts = augment(img, cpts)
                bx.append(img[None])
                by.append(heatmaps(cpts, K))
                bw.append(mask)
            x = torch.from_numpy(np.stack(bx)).to(DEV)
            y = torch.from_numpy(np.stack(by)).to(DEV)
            w = torch.from_numpy(np.stack(bw))[:, :, None, None].to(DEV)
            opt.zero_grad()
            out = net(x)
            pw = 1.0 + POS_W * (y / AMP)
            loss = (((out - y) ** 2) * pw * w).mean()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        eq, gt_pts, gt_mask, excel = data[test]
        clicks, click_res = [], []
        for ci in range(5):
            centre = gt_pts.mean(axis=0) + rng.uniform(-JIT, JIT, 2)
            img, cpts = crop_at(eq, gt_pts, centre)
            with torch.no_grad():
                hm = net(torch.from_numpy(img[None, None]).to(DEV))[0].cpu().numpy()
            pred = np.array([peak(hm[i]) for i in range(K)])
            click_res.append(dict(
                values=values_from(pred, task),
                lm_err=[float(np.linalg.norm(pred[i] - cpts[i]) * MMPX)
                        for i in range(K)]))
            clicks.append(centre.tolist())
            if ci == 0:
                vis = cv2.cvtColor((img * 255).astype(np.uint8),
                                   cv2.COLOR_GRAY2BGR)
                vis = cv2.resize(vis, (CROP * 2, CROP * 2),
                                 interpolation=cv2.INTER_CUBIC)
                cv2.drawMarker(vis, (CROP, CROP), (255, 128, 0),
                               cv2.MARKER_TILTED_CROSS, 14, 2)
                for i in range(K):
                    g = (cpts[i] * 2).astype(int)
                    q = (pred[i] * 2).astype(int)
                    cv2.drawMarker(vis, tuple(g), (0, 255, 0),
                                   cv2.MARKER_CROSS, 12, 1)
                    cv2.circle(vis, tuple(q), 5, (0, 0, 255), 1)
                cv2.drawMarker(vis, (CROP, CROP), (255, 128, 0),
                               cv2.MARKER_TILTED_CROSS, 14, 2)
                cv2.imwrite(str(qc_dir / f"{test}.png"), vis)
        mean_vals = {m: float(np.mean([c["values"][m] for c in click_res]))
                     for m in TASK_MEAS[task]}
        spread = {m: float(np.std([c["values"][m] for c in click_res]))
                  for m in TASK_MEAS[task]}
        results[test] = dict(
            excel=excel,
            pred_mean={m: round(v, 2) for m, v in mean_vals.items()},
            delta={m: round(mean_vals[m] - excel[m], 2) for m in mean_vals},
            click_spread_sd={m: round(v, 2) for m, v in spread.items()},
            per_click=click_res, clicks=clicks,
            gt_mask=[bool(v) for v in gt_mask],
            train_seconds=round(time.time() - t0, 1))
        d = results[test]["delta"]
        print(f"{task} {test}: " + "  ".join(
            f"{m} {d[m]:+6.1f} (sd {spread[m]:.1f})" for m in TASK_MEAS[task])
            + f"   [{results[test]['train_seconds']:.0f}s]", flush=True)
        with open(res_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)

    print(f"\n== {task} summary (|mean-of-5-clicks - clinician|, mm) ==")
    for m in TASK_MEAS[task]:
        d = np.array([abs(results[p]["delta"][m]) for p in pids])
        sd = np.array([results[p]["click_spread_sd"][m] for p in pids])
        print(f"{m:>6s}: mean {d.mean():5.2f} median {np.median(d):5.2f} "
              f"max {d.max():5.2f} | click-sd {sd.mean():4.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all")
    ap.add_argument("--steps", type=int, default=900)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = list(TASK_MEAS) if a.task == "all" else [a.task]
    print(f"device: {DEV}", flush=True)
    for t in tasks:
        run_task(t, a.steps)


if __name__ == "__main__":
    main()
