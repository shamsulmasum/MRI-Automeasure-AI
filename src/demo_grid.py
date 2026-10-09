"""Precompute click-tool responses for the demo page.

For every task and held-out patient: train the leave-one-out fold (same
recipe as prompt_net_loocv), then run inference for a 17x17 grid of click
positions (+/-16 mm, 2 mm steps) around the true caliper-cross centre.
Endpoints are stored in DISPLAY pixels (matching export_demo_data's 512-px
slice images). Writes data/demo/clickgrid.json (incremental, resumable).

Usage: python demo_grid.py [--steps 900]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import prompt_net_loocv as P

PROJ = Path(__file__).resolve().parents[1]
DEMO = PROJ / "data" / "demo"
DEMO.mkdir(parents=True, exist_ok=True)
OUT = DEMO / "clickgrid.json"
DISP = 512
GRID_MM = list(range(-16, 17, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=900)
    a = ap.parse_args()
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    print(f"device: {P.DEV}", flush=True)
    for task in P.TASK_MEAS:
        P.CROP = P.TASK_CROP[task]
        # keep module-level CROP in sync for crop_at/heatmaps/augment
        setattr(P, "CROP", P.TASK_CROP[task])
        pids = sorted(P.SOFT[task].keys())
        data = {p: P.load_patient(task, p) for p in pids}
        K = len(P.chan_names(task))
        for test in pids:
            key = f"{task}|{test}"
            if key in out:
                continue
            t0 = time.time()
            train = [p for p in pids if p != test]
            net = P.UNet().to(P.DEV)
            net.out = torch.nn.Conv2d(16, K, 1).to(P.DEV)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=a.steps, eta_min=1e-4)
            net.train()
            for step in range(a.steps):
                bx, by, bw = [], [], []
                for _ in range(6):
                    p = train[P.rng.integers(len(train))]
                    eq, pts, mask, _ = data[p]
                    centre = pts.mean(axis=0) + P.rng.uniform(-P.JIT, P.JIT, 2)
                    img, cpts = P.crop_at(eq, pts, centre)
                    img, cpts = P.augment(img, cpts)
                    bx.append(img[None])
                    by.append(P.heatmaps(cpts, K))
                    bw.append(mask)
                x = torch.from_numpy(np.stack(bx)).to(P.DEV)
                y = torch.from_numpy(np.stack(by)).to(P.DEV)
                w = torch.from_numpy(np.stack(bw))[:, :, None, None].to(P.DEV)
                opt.zero_grad()
                o = net(x)
                pw = 1.0 + P.POS_W * (y / P.AMP)
                loss = (((o - y) ** 2) * pw * w).mean()
                loss.backward()
                opt.step()
                sched.step()
            net.eval()
            eq, gt_pts, _, excel = data[test]
            h, w_ = eq.shape
            to_disp = DISP / max(h, w_)
            centre0 = gt_pts.mean(axis=0)
            entries = []
            with torch.no_grad():
                for dym in GRID_MM:
                    for dxm in GRID_MM:
                        centre = centre0 + [dxm / P.MMPX, dym / P.MMPX]
                        img, _ = P.crop_at(eq, gt_pts, centre)
                        hm = net(torch.from_numpy(img[None, None]).to(P.DEV)
                                 )[0].cpu().numpy()
                        pred = np.array([P.peak(hm[i]) for i in range(K)])
                        x0 = int(round(centre[0])) - P.CROP // 2
                        y0 = int(round(centre[1])) - P.CROP // 2
                        slice_px = pred + [x0, y0]
                        vals = P.values_from(pred, task)
                        entries.append(dict(
                            c=[dxm, dym],
                            e=(slice_px * to_disp).round(1).tolist(),
                            v={m: round(v, 1) for m, v in vals.items()}))
            out[key] = dict(
                centre_disp=(centre0 * to_disp).round(1).tolist(),
                mm_per_disp_px=round(P.MMPX / to_disp, 5),
                gt_e=(gt_pts * to_disp).round(1).tolist(),
                excel=excel, grid_mm=GRID_MM, entries=entries,
                train_seconds=round(time.time() - t0, 1))
            OUT.write_text(json.dumps(out), encoding="utf-8")
            print(f"{key}: {len(entries)} grid clicks "
                  f"[{out[key]['train_seconds']:.0f}s]", flush=True)
    print("done")


if __name__ == "__main__":
    main()
