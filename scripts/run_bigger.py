"""
Bigger L run on i7-13700F + RTX 4070.

Two physical limits decide the scope (checked on this machine):
  * GPU aggregate ~120-180 Mstep/s (4070), but SSB equilibration is the
    binding cost, not throughput. Equilibration ~ L^3 (see CLAUDE.md), so
    L=8000 for SSB is a wall-clock sink (hours-tens of hours). The feasible,
    equilibration-limited sweet spot is L~800-2000 (todo line 113 asks L=800).
  * fig5 boundary scaling is exactly where big L is CHEAP (few replicas, CPU
    4.4M step/s/core) -> run L=8000 here, not for SSB.

Deliverables:
  1. SSB continuous scan (run_ensemble_cuda_continue, continuous trajectories)
     at L=800 and L=2000, into results/L800/ and results/L2000/.
  2. fig5 phase-boundary vs L extended with L=800 and L=8000 (CPU, cheap),
     recovered for L=200/500/1000 from existing data + new big-L points.
  3. fig6 refit at L=1000 with an L-SCALED equilibration budget (fixes the
     L=500 dense~0.75 vs L=200 ~0.92 gap that comes from a fixed 3000 steps/site).
  4. L1000 figure folder materialized from existing results/gpu data (it is
     already L=1000) so we have a self-consistent results/L1000/.

Run:  python scripts/run_bigger.py [--tag NAME] [--no-ssb] [--no-fig5] [--no-fig6] [--no-l1000]
Progress is streamed to results/<tag>.log (tqdm -> file, flushed per update)
so a hang is visible in the log instead of a silent stall.
"""
import os
import sys
import time
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from numba import cuda
from tqdm import tqdm

from asep.cuda_ensemble import run_ensemble_cuda_continue

ALPHA = 0.9

# SSB scan betas (paper-sensitive region)
SSB_BETAS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.2, 0.25, 0.3]


class TeeLog:
    """Redirect stdout to both a .log file and the console, auto-flushed."""

    def __init__(self, path):
        self.f = open(path, "a", buffering=1)
        self._orig = sys.stdout

    def write(self, s):
        self.f.write(s)
        self.f.flush()
        self._orig.write(s)

    def flush(self):
        self.f.flush()
        self._orig.flush()


def ssb_scan(L, nrep_per_beta, chunks, steps_per_chunk, sample_every, out_dir,
             warmup_chunks=0, beta_range=None):
    """Continuous SSB scan for several betas advanced together.

    All betas share one persistent GPU ensemble (one block per beta), so the
    GPU stays occupied while each beta keeps a full replica block. Trajectories
    are CONTINUOUS across chunks (the fix that reaches the HD/LD basin).
    """
    os.makedirs(out_dir, exist_ok=True)
    betas = beta_range or SSB_BETAS
    nb = len(betas)
    n_total = nb * nrep_per_beta
    print(f"[ssb] L={L} {nb} betas x {nrep_per_beta} reps "
          f"({n_total} threads), {chunks}x{steps_per_chunk} steps/site",
          flush=True)

    state = run_ensemble_cuda_continue(L, n_total, seed=0)
    state["alpha_d"] = cuda.to_device(np.full(n_total, ALPHA, np.float64))
    state["beta_d"] = cuda.to_device(np.repeat(np.array(betas, np.float64),
                                               nrep_per_beta))

    offs = {b: i * nrep_per_beta for i, b in enumerate(betas)}
    dens = {b: np.zeros((nrep_per_beta, chunks)) for b in betas}
    dilu = {b: np.zeros((nrep_per_beta, chunks)) for b in betas}
    stdb = {b: np.zeros((nrep_per_beta, chunks)) for b in betas}

    # continuous warmup (not sampled)
    if warmup_chunks:
        print(f"  warmup {warmup_chunks} chunks", flush=True)
        for _ in range(warmup_chunks):
            state["advance"](steps_per_chunk, warmup=0, sample_every=0)

    for ci in tqdm(range(chunks), desc=f"L={L} ssb", total=chunks):
        res = state["advance"](steps_per_chunk, warmup=0, sample_every=sample_every)
        for i, b in enumerate(betas):
            o = offs[b]
            dens[b][:, ci] = res["dense"][o:o + nrep_per_beta]
            dilu[b][:, ci] = res["dilute"][o:o + nrep_per_beta]
            stdb[b][:, ci] = res["std_diff"][o:o + nrep_per_beta]

    for b in betas:
        np.savez(f"{out_dir}/ssb_beta{b}.npz",
                 dense=dens[b], dilute=dilu[b], std=stdb[b],
                 alpha=ALPHA, L=L, chunks=chunks,
                 steps_per_chunk=steps_per_chunk, nrep=nrep_per_beta)
    print(f"[ssb] L={L} saved -> {out_dir}", flush=True)


def fig5_large_L(out_dir, Ls=(800, 8000), n_reps=8, warmup=1_000_000,
                 n_workers=None):
    """Extend the fig5 phase-boundary-vs-L curve to large L (CPU).

    Reuses the classifier in scripts/fig5_boundaries.py. Appends new Ls to the
    existing results/fig5_boundaries.npz if present, and re-plots.
    `n_workers` caps the ProcessPool (memory-aware default in fig5_boundaries).
    """
    from scripts.fig5_boundaries import (phase_boundary_in_beta,
                                         phase_boundary_in_alpha, classify_point,
                                         _smooth_labels)
    os.makedirs(out_dir, exist_ok=True)
    # steps scale ~ L for bulk current measurement (equilibration not the
    # binding cost here; few reps)
    steps_by_L = {L: int(L * 1e4) for L in Ls}

    beta_fixed = 1.0
    alpha_fixed = 0.9
    a_theory = alpha_fixed / (1 + alpha_fixed + alpha_fixed**2)
    b_theory_mc = 2 * beta_fixed / (4 * beta_fixed - 1)

    new_asym = []
    new_mcld = []
    for L in Ls:
        # (a) beta boundary asym->LD at alpha=0.9
        betas = np.linspace(0.05, 0.6, 24)
        grid = phase_boundary_in_beta([alpha_fixed], betas, L,
                                      steps_by_L[L], 1_000_000, 200, n_reps,
                                      n_workers=n_workers)
        labels = []
        for j in range(len(betas)):
            J1s, J2s, r1s, r2s = grid[0][j]
            lab, _ = classify_point(r1s, r2s, J1s, J2s, alpha_fixed, betas[j], L)
            labels.append(lab)
        labels = _smooth_labels(labels)
        tr = None
        for j in range(1, len(betas)):
            if labels[j - 1] != "LD" and labels[j] == "LD":
                tr = 0.5 * (betas[j - 1] + betas[j])
        new_asym.append(tr)
        print(f"  L={L}: asym->LD at beta={tr}", flush=True)

        # (b) alpha boundary (LD->MC) at beta=1.0
        alphas = np.linspace(0.2, 0.95, 24)
        grid = phase_boundary_in_alpha(alphas, [beta_fixed], L,
                                       steps_by_L[L], 1_000_000, 50, n_reps,
                                       n_workers=n_workers)
        labels = []
        for i in range(len(alphas)):
            J1s, J2s, r1s, r2s = grid[i][0]
            lab, _ = classify_point(r1s, r2s, J1s, J2s, alphas[i], beta_fixed, L)
            labels.append(lab)
        labels = _smooth_labels(labels)
        tr = None
        for i in range(1, len(alphas)):
            if labels[i - 1] == "LD" and labels[i] == "MC":
                tr = 0.5 * (alphas[i - 1] + alphas[i])
        new_mcld.append(tr)
        print(f"  L={L}: LD->alpha at alpha={tr}", flush=True)

    # merge with existing tracked file (read-only; do NOT overwrite it).
    # Write the merged result into this run's out_dir instead.
    src = os.path.join(ROOT, "results", "gpu", "fig5_boundaries.npz")
    Ls_all = np.array(list(Ls), dtype=float)
    ba = np.array(new_asym, dtype=float)
    bm = np.array(new_mcld, dtype=float)
    if os.path.exists(src):
        old = np.load(src, allow_pickle=True)
        Ls_all = np.concatenate([old["Ls"], Ls_all])
        ba = np.concatenate([old["beta_asym"], ba])
        bm = np.concatenate([old["alpha_mcld"], bm])
    dst = os.path.join(out_dir, "fig5_boundaries_merged.npz")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(dst, Ls=Ls_all, beta_asym=ba, alpha_mcld=bm,
             theory_hdld=a_theory, theory_mc=b_theory_mc)
    print(f"[fig5] saved merged boundaries -> {dst}", flush=True)
    print("  (re-plot with scripts/fig5_boundaries.py or plot_all)", flush=True)


def fig6_equilibrated(L, out_dir, alphas=(0.1, 0.8, 0.9), n_reps=32,
                      steps_per_site=None):
    """fig6 currents/densities vs beta with an L-scaled equilibration budget.

    The old fig6 used a FIXED 3000 steps/site for every L, which under-
    equilibrates large L (SSB basin needs ~100k steps/site at L=1000; the
    L=500 dense~0.75 vs L=200 dense~0.92 gap is that bias, not noise). Here
    steps/site ~ c*L with c tuned so c*L = 100k at L=1000 (c=100). Total steps
    per replica ~ c*L^2. On the 4070 (~150 Mstep/s aggregate) L=1000 fig6 is
    ~20-30 min.
    """
    from asep.parallel import scan_beta_gpu
    os.makedirs(out_dir, exist_ok=True)
    if steps_per_site is None:
        steps_per_site = int(100 * L)      # =100k steps/site at L=1000
    warmup_per_site = steps_per_site // 5
    betas = np.linspace(0.05, 0.95, 30)
    steps = L * steps_per_site
    warmup = L * warmup_per_site
    print(f"[fig6] L={L} {steps_per_site}k steps/site, {steps} steps/rep, "
          f"{n_reps} reps (MFT vs MC + SSB, equilibrated)", flush=True)
    for alpha in alphas:
        res = scan_beta_gpu(alpha, betas, L, steps, warmup, 400,
                            n_reps=n_reps, seed=0)
        J1, J2, r1, r2, eJ1, eJ2, er1, er2, dense, dilute, edense, edilute = res
        np.savez(f"{out_dir}/alpha{alpha}.npz",
                 J1=J1, J2=J2, rho1=r1, rho2=r2, eJ1=eJ1, eJ2=eJ2,
                 erho1=er1, erho2=er2, dense=dense, dilute=dilute,
                 edense=edense, edilute=edilute, betas=betas)
        print(f"  alpha={alpha} saved", flush=True)
    print(f"[fig6] L={L} saved -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="big")
    ap.add_argument("--no-ssb", action="store_true")
    ap.add_argument("--no-fig5", action="store_true")
    ap.add_argument("--no-fig6", action="store_true")
    ap.add_argument("--no-l1000", action="store_true")
    args = ap.parse_args()

    # tee stdout -> results/<tag>.log (also keeps console)
    sys.stdout = TeeLog(os.path.join(ROOT, "results", f"{args.tag}.log"))

    if not args.no_ssb:
        # L=800 (sweet spot) and L=2000 (aggressive but feasible)
        ssb_scan(800, nrep_per_beta=1024, chunks=15,
                 steps_per_chunk=800 * 20000, sample_every=5000,
                 out_dir=os.path.join(ROOT, "results", f"{args.tag}_L800"))
        ssb_scan(2000, nrep_per_beta=512, chunks=10,
                 steps_per_chunk=2000 * 20000, sample_every=5000,
                 out_dir=os.path.join(ROOT, "results", f"{args.tag}_L2000"))

    if not args.no_fig5:
        fig5_large_L(os.path.join(ROOT, "results", f"{args.tag}_fig5"), Ls=(800, 8000))

    if not args.no_fig6:
        # refit fig6 with an L-scaled budget so large L actually equilibrates
        # into the dense basin (fixes the L=500 dense~0.75 vs L=200 ~0.92 gap).
        fig6_equilibrated(1000, os.path.join(ROOT, "results", f"{args.tag}_L1000_fig6"))

    if not args.no_l1000:
        # materialize L1000 from existing results/gpu data (it is L=1000)
        dst = os.path.join(ROOT, "results", f"{args.tag}_L1000")
        os.makedirs(dst, exist_ok=True)
        import shutil
        for sub in ["fig2", "fig6"]:
            src = os.path.join(ROOT, "results", "gpu", sub)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(dst, sub), dirs_exist_ok=True)
        for f in ["fig3_points.npz", "ssb_order_vs_beta.png"]:
            s = os.path.join(ROOT, "results", "gpu", f)
            if os.path.exists(s):
                shutil.copy(s, os.path.join(dst, f))
        print(f"[l1000] materialised L1000 -> {dst}", flush=True)

    print("ALL BIGGER RUN DONE", flush=True)


if __name__ == "__main__":
    main()
