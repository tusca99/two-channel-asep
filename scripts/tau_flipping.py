#!/usr/bin/env python3
"""tau(L): flipping time between SSB basins, single long trajectories.

Zhu et al. PRE 85, 041132 (2012) Fig 3: tau grows exponentially with L ->
SSB genuine. We measure first-passage / dwell statistics of d = rho1 - rho2
along single long trajectories (BKL Fenwick, CPU-friendly).

Method: sample (rho1, rho2) every `sample_every` steps, smooth d over a
window, extract dwell times between sign changes; tau = median dwell
(in MC steps). One trajectory per (L, beta) at fixed alpha=0.9, 12 points
in the broken band + controls.

Usage: python scripts/tau_flipping.py
Saves incrementally to results/tau/tau_L<L>_b<BETA>.npz and logs to
results/tau/tau.log. One process per point, launched with staggering.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from asep.model import TwoChannelASEP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "tau")
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.9


def run_point(L, beta, n_steps, seed, sample_every, warmup_frac=0.05):
    t0 = time.time()
    m = TwoChannelASEP(L=L, alpha=ALPHA, beta=beta, seed=seed)
    warmup = int(n_steps * warmup_frac)
    m.run(n_steps, sample_every=sample_every, warmup=warmup)
    j = np.array(m._joint_samples)
    d = j[:, 0] - j[:, 1]
    t = np.array(m._time_samples)
    return d, t, m.total_time, n_steps, time.time() - t0


def dwells_basin(d, t, dmin):
    """Basin dwell times: trajectory is in a basin when |d| >= dmin (dense or
    dilute); a flip = leaving one basin and entering the other. Avoids the
    noise-crossing artifact of smoothed-sign detection near the band edge."""
    if len(d) < 100:
        return np.array([])
    basin = np.where(np.abs(d) >= dmin, np.sign(d), 0)  # +1/-1 in-basin, 0 = interface
    # forward-fill interface samples with the previous basin state
    filled = basin.copy()
    last = 0.0
    for i in range(len(filled)):
        if filled[i] != 0:
            last = filled[i]
        else:
            filled[i] = last
    cross = np.where(filled[1:] * filled[:-1] < 0)[0]
    if len(cross) < 2:
        return np.array([])
    return np.diff(t[cross])


def main(L, beta, n_steps, seed, sample_every, dmin):
    tag = f"tau_L{L}_b{str(beta).replace('.', '')}"
    npz = os.path.join(OUT, f"{tag}.npz")
    if os.path.exists(npz):
        print(f"[skip] {npz} exists", flush=True)
        return
    print(f"[start] L={L} beta={beta} steps={n_steps} seed={seed} "
          f"sample_every={sample_every} dmin={dmin}", flush=True)
    d, t, tt, nst, wall = run_point(L, beta, n_steps, seed, sample_every)
    dw = dwells_basin(d, t, dmin)
    tau = float(np.median(dw)) if len(dw) else float("nan")
    np.savez_compressed(
        npz, d=d, t=t, L=L, beta=beta, alpha=ALPHA, n_steps=n_steps,
        seed=seed, total_time=tt, wall_s=wall, dwells=dw, tau=tau,
        mean_d=float(d.mean()), std_d=float(d.std()), dmin=dmin,
    )
    print(f"[done] {tag}: tau={tau:.4g} dwells={len(dw)} "
          f"mean_d={d.mean():+.3f} wall={wall/60:.1f}min", flush=True)


if __name__ == "__main__":
    L = int(sys.argv[1])
    beta = float(sys.argv[2])
    n_steps = int(float(sys.argv[3]))
    seed = int(sys.argv[4])
    sample_every = int(sys.argv[5])
    dmin = float(sys.argv[6])
    main(L, beta, n_steps, seed, sample_every, dmin)