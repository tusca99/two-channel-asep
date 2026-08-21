"""
Targeted L1000 phase diagram: cheap everywhere, heavy only in the HD/LD window.

The brute-force 31x31 grid at 100k steps/site took >7h (real ~71 Mstep/s).
The full SSB equilibration (100k steps/site) is only needed in the HD/LD
broken-symmetry band (high alpha, low-mid beta). Elsewhere LD/MC equilibrates
fast at 5k steps/site.

Two-tier:
  - coarse pass (5k steps/site) over the whole grid -> LD/MC
  - heavy pass (100k steps/site) only in HD window (alpha>0.45, beta<0.4)

Run:  python scripts/redo_fig2_targeted.py --Ls 1000
"""
import os
import sys
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from asep.parallel import scan_grid_gpu
from asep.observables import classify_phase
from scripts.params_record import params_fig2
from scripts.remake_L import TeeLog

COARSE_SPS = 5000
HEAVY_SPS = 100000
HD_WINDOW = dict(alpha=(0.45, 1.0), beta=(0.05, 0.4))


def in_hd_window(a, b):
    return (HD_WINDOW["alpha"][0] <= a <= HD_WINDOW["alpha"][1]
            and HD_WINDOW["beta"][0] <= b <= HD_WINDOW["beta"][1])


def classify_batch(alphas, betas, pts, L, sps, n_reps, seed):
    """Run a set of (alpha,beta) grid points at one budget in ONE ensemble
    launch (run_ensemble_cuda with per-replica alpha/beta arrays, so no
    cross-product blowup). Returns {(i,j): label}."""
    grid = {}
    if not pts:
        return grid
    from asep.cuda_ensemble import run_ensemble_cuda
    unique = sorted(set(pts))
    npts = len(unique)
    steps = int(L * sps); warmup = int(L * (sps // 5))
    # per-replica alpha/beta: n_reps replicas per point, point-major
    aa = np.repeat(np.array([alphas[i] for i, j in unique]), n_reps)
    bb = np.repeat(np.array([betas[j] for i, j in unique]), n_reps)
    out = run_ensemble_cuda(0.0, 0.0, L, steps, npts * n_reps, seed=seed,
                            sample_every=400, warmup=warmup, alphas=aa, betas=bb)
    for k, (i, j) in enumerate(unique):
        sl = slice(k * n_reps, (k + 1) * n_reps)
        J1 = np.mean(out["cur1"][sl] / out["ttime"][sl])
        J2 = np.mean(out["cur2"][sl] / out["ttime"][sl])
        # per-replica samples: stack (rho1,rho2) -> (n_reps,2) for classify
        samples = np.column_stack([out["rho1"][sl], out["rho2"][sl]])
        lab, _ = classify_phase(J1, J2, out["rho1"][sl].mean(),
                                out["rho2"][sl].mean(), alphas[i], betas[j],
                                L, samples=samples, j_current=0.25)
        grid[i, j] = lab
    return grid


def run_grid(alphas, betas, L, n_reps, seed):
    heavy = [(i, j) for i in range(len(alphas)) for j in range(len(betas))
             if in_hd_window(alphas[i], betas[j])]
    light = [(i, j) for i in range(len(alphas)) for j in range(len(betas))
             if not in_hd_window(alphas[i], betas[j])]
    # merge: light (coarse) then heavy (SSB-equilibrated) fills HD window
    merged = {}
    merged.update(classify_batch(alphas, betas, light, L, COARSE_SPS, n_reps, seed))
    merged.update(classify_batch(alphas, betas, heavy, L, HEAVY_SPS, n_reps, seed + 100))
    grid = np.empty((len(alphas), len(betas)), dtype=object)
    for i in range(len(alphas)):
        for j in range(len(betas)):
            grid[i, j] = merged.get((i, j), "LD")
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", nargs="+", type=int, default=[1000])
    ap.add_argument("--n_reps", type=int, default=16)
    ap.add_argument("--tag", default="redo_fig2_targeted")
    args = ap.parse_args()
    sys.stdout = TeeLog(os.path.join(ROOT, "results", f"{args.tag}.log"))
    print(f"TARGETED fig2 for {args.Ls}", flush=True)
    for L in args.Ls:
        out = os.path.join(ROOT, "results", f"L{L}", "fig2")
        os.makedirs(out, exist_ok=True)
        alphas = np.linspace(0.05, 0.95, 31)
        betas = np.linspace(0.05, 0.95, 31)
        grid = run_grid(alphas, betas, L, args.n_reps, seed=0)
        np.save(f"{out}/grid_full.npy", grid, allow_pickle=True)
        np.save(f"{out}/alphas_full.npy", alphas)
        np.save(f"{out}/betas_full.npy", betas)
        za = np.linspace(0.05, 0.95, 31)
        zb = np.linspace(0.2, 0.4, 21)
        zoom = run_grid(za, zb, L, args.n_reps, seed=1)
        np.save(f"{out}/grid_zoom.npy", zoom, allow_pickle=True)
        np.save(f"{out}/alphas_zoom.npy", za)
        np.save(f"{out}/betas_zoom.npy", zb)
        print(f"[fig2] L={L} targeted saved", flush=True)
        labels, counts = np.unique(grid, return_counts=True)
        print(f"  L={L}: {dict(zip(labels, counts))}", flush=True)
    print("ALL TARGETED FIG2 DONE", flush=True)


if __name__ == "__main__":
    main()
