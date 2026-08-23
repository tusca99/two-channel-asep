"""
fig5 (GPU) — phase boundaries vs L, paper Fig 5, both panels.

Runs on the H100 via the replica-parallel ensemble in asep.cuda_ensemble
(one CUDA thread per independent (alpha, beta, L) replica). This is the
right backend for large ensembles: ~500-900 Mstep/s aggregate at 8k-32k
replicas, vs ~90 Mstep/s from 31 CPU cores (6-10x on this box).

Panels
------
(a) asym -> LD boundary in beta vs L,   alpha=0.9
(b) LD -> MC boundary in alpha vs L,    beta=1.0

Protocol: paper-mid 2e8 steps/rep, 5% warmup (first 5% discarded). Every
(alpha,beta,L) point is a batch of nrep independent replicas; classification
uses the replica ensemble's broken-vs-symmetric fraction (std(rho1-rho2)
vs an L-adaptive noise floor). ALL raw per-replica observables are saved
so a later L extension / reclassification needs no re-simulation.

Env
---
  CUDA_HOME must be set to a CUDA 12 toolkit (see scripts/enable_gpu.sh
  in this repo, or run with the env below to build one in /tmp/opencode).

Usage:
    # build env (once per session)
    source scripts/enable_gpu.sh
    uv run python scripts/fig5_Lscan_gpu.py \\
        --L 1000,4000,10000 --nrep 500
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asep.cuda_ensemble import run_ensemble_cuda
from fig5_boundaries import _smooth_labels  # noqa  (label smoothing)
from tqdm import tqdm

OUT = os.path.join(ROOT, "results", "fig5_Lscan_gpu")
os.makedirs(OUT, exist_ok=True)

STEPS = int(2.0e8)   # set from --steps in main()


def _budget(L):
    """(total_steps, warmup) per L.

    STEPS total/replica (default 2e8, paper band 2e7-5e8/site is for tight
    mean error bars; for a boundary scan we resolve the phase by the ensemble
    fraction of broken replicas, which only needs steady state). warmup = 5%.
    """
    total = int(STEPS)
    warmup = int(0.05 * total)
    sample_every = max(1, (total - warmup) // 100)
    return total, warmup, sample_every


def _one_point_L(L, alpha, beta, nrep, seed):
    """One (alpha, beta) at size L: run nrep replicas on GPU; return per-replica arrays."""
    total, warmup, sample_every = _budget(L)
    out = run_ensemble_cuda(alpha, beta, int(L), int(total), int(nrep),
                            seed=int(seed), block=256,
                            warmup=warmup, sample_every=sample_every)
    # per-replica broken/symmetric indicator
    return out


def _classify_point(out, alpha, beta, L, mc_rho=0.45):
    """Label one point from an ensemble of replica (rho1, rho2, J1, J2).

    Mirrors fig5_boundaries.classify_point but operates on per-replica arrays
    returned by run_ensemble_cuda. Uses std(rho1-rho2) across the broken
    replicas as the SSB order parameter; an L-adaptive threshold
    (0.04*sqrt(1000/L)) calibrates the noise floor (paper/CLAUDE notes).
    """
    r1 = np.array(out["rho1"])
    r2 = np.array(out["rho2"])
    J1 = np.array(out["cur1"]) / np.maximum(out["ttime"], 1e-12)
    J2 = np.array(out["cur2"]) / np.maximum(out["ttime"], 1e-12)
    d = r1 - r2
    asym = d.std()
    asym_threshold = 0.04 * np.sqrt(1000.0 / L)
    avg_J = 0.5 * (J1.mean() + J2.mean())
    avg_rho = 0.5 * (r1.mean() + r2.mean())
    if asym < asym_threshold:
        if avg_J >= 0.25 - 0.02 and avg_rho > mc_rho:
            return "MC", asym, avg_rho, avg_J
        return "LD", asym, avg_rho, avg_J
    return "asym", asym, avg_rho, avg_J


def _scan_dimension(points, fixed, is_beta, L, nrep, seed0=0):
    fn = f"{'a_beta' if is_beta else 'b_alpha'}_L{L}_s{STEPS}_n{nrep}_p{len(points)}.npz"
    fpath = os.path.join(OUT, fn)
    if os.path.exists(fpath):
        with np.load(fpath) as d:
            print(f"  [cached] L={L} {('beta' if is_beta else 'alpha')}-boundary ="
                  f" {float(d['transition'])} (skipping)", flush=True)
        return float(d["transition"])
    labels, asyms, avg_rhos, avg_Js = [], [], [], []
    rep_data = {}
    bar = tqdm(points, desc=f"{'(a)' if is_beta else '(b)'} L={L}", unit="pt",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} pts "
                          "[{elapsed} < {remaining}]  {postfix}")
    for j, p in enumerate(points):
        a = fixed if is_beta else p
        b = p if is_beta else fixed
        out = _one_point_L(L, a, b, nrep, seed0 + j * 7919)
        lab, asym, avg_rho, avg_J = _classify_point(out, a, b, L)
        bar.set_postfix_str(f"{p} -> {lab} (asym={asym:.4f}, rho={avg_rho:.3f})")
        labels.append(lab)
        asyms.append(asym)
        avg_rhos.append(avg_rho)
        avg_Js.append(avg_J)
        rep_data[f"rho1_{j}"] = np.array(out["rho1"], dtype=np.float32)
        rep_data[f"rho2_{j}"] = np.array(out["rho2"], dtype=np.float32)
        rep_data[f"cur1_{j}"] = np.array(out["cur1"], dtype=np.float64)
        rep_data[f"cur2_{j}"] = np.array(out["cur2"], dtype=np.float64)
    bar.close()
    labels = _smooth_labels(labels)
    # transition = first (or last consistent) boundary between broken / symmetric
    if is_beta:
        trans = None
        for j in range(1, len(labels)):
            if labels[j - 1] != "LD" and labels[j] == "LD":
                trans = 0.5 * (points[j - 1] + points[j])
    else:
        trans = None
        for j in range(1, len(labels)):
            if labels[j - 1] == "LD" and labels[j] == "MC":
                trans = 0.5 * (points[j - 1] + points[j])
    np.savez(os.path.join(OUT, fn),
             axis=np.array(points), labels=np.array(labels),
             asym=np.array(asyms, dtype=float),
             avg_rho=np.array(avg_rhos, dtype=float),
             avg_J=np.array(avg_Js, dtype=float),
             transition=np.float64(trans) if trans is not None else np.float64(np.nan),
             L=L, nrep=int(nrep), fixed=float(fixed),
             is_beta=np.bool_(is_beta), **rep_data)
    print(f"  L={L} {('beta' if is_beta else 'alpha')}-boundary = {trans}  "
          f"labels={labels}", flush=True)
    return trans


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", default="1000,4000,10000")
    ap.add_argument("--nrep", type=int, default=8192,
                    help="independent replicas per (a,b,L) point on GPU (default 8192)")
    ap.add_argument("--steps", type=int, default=200_000_000,
                    help="MC steps per replica (default 2e8)")
    ap.add_argument("--npts", type=int, default=16,
                    help="grid points per panel (default 16)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    global STEPS
    STEPS = int(args.steps)
    npts = max(3, int(args.npts))
    Ls = [int(x) for x in args.L.split(",")]
    print(f"GPU fig5: L={Ls} nrep={args.nrep} steps/rep={STEPS} npts={npts}",
          flush=True)

    alpha_fixed = 0.9
    betas = np.linspace(0.20, 0.40, npts)
    beta_boundary = []
    for L in Ls:
        beta_boundary.append(_scan_dimension(betas, alpha_fixed, True, L,
                                             args.nrep, seed0=args.seed + 1000 + L))

    beta_fixed = 1.0
    alphas = np.linspace(0.55, 0.95, npts)
    alpha_boundary = []
    for L in Ls:
        alpha_boundary.append(_scan_dimension(alphas, beta_fixed, False, L,
                                              args.nrep, seed0=args.seed + 2000 + L))

    a_theory = alpha_fixed / (1 + alpha_fixed + alpha_fixed ** 2)
    b_theory = 2 * beta_fixed / (4 * beta_fixed - 1)

    a_boundary = np.array(beta_boundary, dtype=float)
    b_boundary = np.array(alpha_boundary, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, Ls, bnd, theo, panel, ylab in [
            (axes[0], Ls, a_boundary, a_theory,
             rf"(a) asym$-$LD boundary, $\alpha={alpha_fixed}$", r"$\beta$ boundary"),
            (axes[1], Ls, b_boundary, b_theory,
             rf"(b) MC$/$LD boundary, $\beta={beta_fixed}$", r"$\alpha$ boundary")]:
        valid = np.isfinite(bnd)
        if valid.any():
            ax.plot(Ls, bnd, "o-", ms=6, capsize=4, label="MC boundary")
        ax.axhline(theo, color="k", ls=":", lw=1.2,
                   label=f"MFT = {theo:.3f}")
        ax.set_xscale("log"); ax.grid(alpha=0.2, which="both")
        ax.set_xlabel(r"$L$"); ax.set_ylabel(ylab)
        ax.set_title(panel)
        ax.legend(fontsize=9)
    for ax, yrange in zip([axes[0], axes[1]], [(0.2, 0.4), (0.5, 1.0)]):
        ax.set_ylim(*yrange)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_phase_boundary_vs_L.png"), dpi=150)
    np.savez(os.path.join(OUT, "fig5_boundaries_summary.npz"),
             Ls=np.array(Ls), beta_boundary=a_boundary,
             alpha_boundary=b_boundary, theory_beta=a_theory,
             theory_alpha=b_theory, nrep=args.nrep)
    print(f"\nBeta boundary (asym->LD): {beta_boundary}", flush=True)
    print(f"Alpha boundary (LD->MC) : {alpha_boundary}", flush=True)
    print(f"saved {OUT}/fig5_phase_boundary_vs_L.png", flush=True)


if __name__ == "__main__":
    main()
