"""
fig5 phase boundaries vs L, REGENERATED with the current fixed classifier
(MC-by-current-saturation + L-adaptive SSB threshold) AND bootstrap error bars.

Fixes the stale-data problem: results/gpu/fig5_boundaries.npz was produced
before the classifier fix, so its alpha_mcld~0.477 is an artifact, not physics.
This recomputes both boundaries for all L using the current classify_point and
adds bootstrap error bars over replicas.

Boundaries:
  (a) asym->LD : beta where labels leave 'asym', alpha=0.9, vs L
  (b) LD->MC   : alpha where labels turn 'MC', beta=1.0, vs L

Error bars: resample the n_reps replicas with replacement, re-smooth,
re-find the transition; std over 200 resamples.

Usage: python scripts/fig5_corrected.py [--Ls 200 500 1000 2000 4000 8000]
Output: results/fig5_corrected/fig5_boundaries_corrected.npz + .png
"""
import os
import sys
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from scripts.fig5_boundaries import (classify_point, phase_boundary_in_beta,
                                     phase_boundary_in_alpha, _n_workers,
                                     _smooth_labels)
from scripts.remake_L import TeeLog

ALPHA = 0.9
BETA_MC = 1.0
OUT = os.path.join(ROOT, "results", "fig5_corrected")


def find_asym(r1s, r2s, J1s, J2s, betas, L):
    """Return the beta where the HD/LD (or LD/HD) broken state ends -> LD/LD/LD.

    This is the paper Fig5(a) LEFT boundary: the HD/LD phase is bounded at
    beta by the transition to LD/LD (or LD). Tracks the last beta whose label
    is HD/LD or LD/HD, then the boundary is just past it.
    """
    labels = []
    for j in range(len(betas)):
        lab, _ = classify_point(r1s[j], r2s[j], J1s[j], J2s[j], ALPHA, betas[j], L)
        labels.append(lab)
    labels = _smooth_labels(labels)
    # last index still in a broken HD/LD-like state
    last_broken = None
    for j in range(len(betas)):
        if labels[j] in ("HD/LD", "LD/HD"):
            last_broken = j
    tr = None
    if last_broken is not None and last_broken < len(betas) - 1:
        tr = 0.5 * (betas[last_broken] + betas[last_broken + 1])
    return tr, labels


def find_mcld(r1s, r2s, J1s, J2s, alphas, L):
    """Return the alpha of the LD->MC transition (beta=1.0), plus labels."""
    labels = []
    for i in range(len(alphas)):
        lab, _ = classify_point(r1s[i], r2s[i], J1s[i], J2s[i], alphas[i],
                                BETA_MC, L)
        labels.append(lab)
    labels = _smooth_labels(labels)
    tr = None
    for i in range(1, len(alphas)):
        if labels[i - 1] == "LD" and labels[i] == "MC":
            tr = 0.5 * (alphas[i - 1] + alphas[i])
    return tr, labels


def run_one_L(L, n_reps, n_boot=200):
    """Compute both boundaries + bootstrap errors for one L.

    The beta boundary (HD/LD->LD/LD) needs DEEP SSB equilibration to reach the
    dense basin at large L: a fixed steps=L*1e4 under-equilibrates L>=2000
    (dense stuck <0.5 -> no HD/LD found -> NaN). Use an L-scaled budget
    (steps/site ~ 20*L, c=20 -> 100k at L=5000) for the beta scan. The alpha
    (LD->MC) scan is current-saturation based and equilibrates fast, so it can
    keep a lighter budget.
    """
    nw = _n_workers(L, int(L*1e4))
    betas = np.linspace(0.05, 0.6, 24)
    alphas = np.linspace(0.2, 0.95, 24)

    # (a) asym->LD, alpha=0.9 : L-scaled budget to reach HD/LD basin
    beta_steps = int(L * max(20 * L, 1e4))     # ~20*L steps/site, c=20
    beta_warmup = int(beta_steps // 5)
    grid = phase_boundary_in_beta([ALPHA], betas, L, beta_steps, beta_warmup, 200,
                                  n_reps, n_workers=_n_workers(L, beta_steps))
    r1s = np.array([grid[0][j][2] for j in range(len(betas))])
    r2s = np.array([grid[0][j][3] for j in range(len(betas))])
    J1s = np.array([grid[0][j][0] for j in range(len(betas))])
    J2s = np.array([grid[0][j][1] for j in range(len(betas))])
    asym, labels_a = find_asym(r1s, r2s, J1s, J2s, betas, L)

    # (b) LD->MC, beta=1.0 : lighter budget (current-based, fast equilibration)
    alpha_steps = int(L * 2e4)
    grid2 = phase_boundary_in_alpha(alphas, [BETA_MC], L, alpha_steps, int(alpha_steps//5), 50,
                                    n_reps, n_workers=_n_workers(L, alpha_steps))
    r1a = np.array([grid2[i][0][2] for i in range(len(alphas))])
    r2a = np.array([grid2[i][0][3] for i in range(len(alphas))])
    J1a = np.array([grid2[i][0][0] for i in range(len(alphas))])
    J2a = np.array([grid2[i][0][1] for i in range(len(alphas))])
    mcld, labels_b = find_mcld(r1a, r2a, J1a, J2a, alphas, L)

    # bootstrap error bars
    rng_a = np.random.default_rng(L)
    ea = []
    for _ in range(n_boot):
        idx = rng_a.integers(0, n_reps, n_reps)
        t, _ = find_asym(r1s[:, idx], r2s[:, idx], J1s[:, idx], J2s[:, idx], betas, L)
        if t is not None:
            ea.append(t)
    e_asym = np.std(ea) if ea else np.nan

    rng_b = np.random.default_rng(100 + L)
    eb = []
    for _ in range(n_boot):
        idx = rng_b.integers(0, n_reps, n_reps)
        t, _ = find_mcld(r1a[:, idx], r2a[:, idx], J1a[:, idx], J2a[:, idx], alphas, L)
        if t is not None:
            eb.append(t)
    e_mcld = np.std(eb) if eb else np.nan

    print(f"  L={L}: asym->LD={asym}±{e_asym:.4f} (labels={labels_a})", flush=True)
    print(f"        LD->MC={mcld}±{e_mcld:.4f} (labels={labels_b})", flush=True)
    return asym, e_asym, mcld, e_mcld


def _save_progress(OUT, Ls, asym, e_asym, mcld, e_mcld):
    theory_hdld = ALPHA / (1 + ALPHA + ALPHA**2)
    theory_mc = 2 * BETA_MC / (4 * BETA_MC - 1)
    np.savez(os.path.join(OUT, "fig5_boundaries_corrected.npz"),
             Ls=np.array(Ls), beta_asym=asym, e_beta_asym=e_asym,
             alpha_mcld=mcld, e_alpha_mcld=e_mcld,
             theory_hdld=theory_hdld, theory_mc=theory_mc)
    print(f"  [saved] progress -> fig5_boundaries_corrected.npz "
          f"({len(np.asarray(Ls))} Ls done)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", nargs="+", type=int, default=[200, 500, 1000, 2000, 4000, 8000])
    ap.add_argument("--n_reps", type=int, default=8)
    ap.add_argument("--n_boot", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    # tee stdout+stderr -> results/fig5_corrected.log (tqdm visible)
    sys.stdout = TeeLog(os.path.join(ROOT, "results", "fig5_corrected.log"))

    Ls = np.array(args.Ls)
    asym = np.full(len(Ls), np.nan)
    e_asym = np.full(len(Ls), np.nan)
    mcld = np.full(len(Ls), np.nan)
    e_mcld = np.full(len(Ls), np.nan)
    theory_hdld = ALPHA / (1 + ALPHA + ALPHA**2)
    theory_mc = 2 * BETA_MC / (4 * BETA_MC - 1)

    # incremental: save after each L so an early failure keeps completed work
    for i, L in enumerate(tqdm(Ls, desc="fig5 Ls")):
        a, ea, m, em = run_one_L(L, args.n_reps, args.n_boot)
        asym[i], e_asym[i], mcld[i], e_mcld[i] = a, ea, m, em
        _save_progress(OUT, Ls[:i + 1], asym[:i + 1], e_asym[:i + 1],
                       mcld[:i + 1], e_mcld[:i + 1])

    # final plot (uses full arrays)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.errorbar(Ls, asym, yerr=e_asym, fmt="o-", ms=6, capsize=4,
                label="MC asym$\\to$LD boundary")
    ax.axhline(theory_hdld, color="k", ls=":", lw=1.2,
               label=f"MFT (eq23)={theory_hdld:.3f}")
    ax.set_xscale("log"); ax.set_xlabel(r"$L$"); ax.set_ylabel(r"$\beta$ boundary")
    ax.set_title(rf"(a) asym$-$LD boundary, $\alpha=0.9$")
    ax.set_ylim(0.2, 0.45)
    ax.legend(fontsize=8)

    ax = axes[1]
    m = ~np.isnan(mcld)
    ax.errorbar(Ls[m], mcld[m], yerr=e_mcld[m], fmt="s-", ms=6, capsize=4,
                label="LD$\\to$MC boundary")
    ax.axhline(theory_mc, color="k", ls=":", lw=1.2, label=f"MFT MC/LD (eq10)={theory_mc:.3f}")
    ax.set_xscale("log"); ax.set_xlabel(r"$L$"); ax.set_ylabel(r"$\alpha$ boundary")
    ax.set_title(rf"(b) LD/MC boundary, $\beta=1.0$")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_phase_boundary_vs_L.png"), dpi=150)
    print("saved", os.path.join(OUT, "fig5_phase_boundary_vs_L.png"))


if __name__ == "__main__":
    main()
