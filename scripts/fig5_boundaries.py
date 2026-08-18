"""
Figure 5 (GPU): phase boundaries vs channel size L.

Reproduces paper Fig 5:
  (a) Phase boundaries between HD/LD, LD/LD and LD phases vs L, for alpha=0.9
      (locate the beta where the phase changes).
  (b) Phase boundary between MC and LD phases vs L, for beta=1.0
      (locate the alpha of the MC/LD transition).

Each grid point is classified from an ensemble of replicas (per-replica
rho1,rho2). Runs on the GPU via scan_grid_gpu (one thread per (alpha,beta,rep)).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "gpu")
os.makedirs(OUT, exist_ok=True)

from asep.parallel import scan_grid_gpu


def classify_point(rhos1, rhos2, alpha, beta, asym_threshold=0.04,
                   mc_rho=0.35):
    """Classify phase from per-replica densities (ensemble method).

    Uses std(rho1-rho2) as the robust SSB order parameter. Our model shows a
    WEAK asymmetry (dense~0.27, not a true HD with rho>0.5), so we classify
    by symmetry rather than by the paper's HD/LD criterion:
      - asymmetric (std_diff > threshold): LD/LD-type broken state
      - symmetric low density: LD
      - symmetric high density: MC
    """
    d = np.array(rhos1) - np.array(rhos2)
    asym = d.std()
    rho_avg = (np.mean(rhos1) + np.mean(rhos2)) / 2
    if asym < asym_threshold:
        return "MC" if rho_avg > mc_rho else "LD", asym
    return "asym", asym


def phase_boundary_in_beta(alphas, betas, L, n_steps, warmup, sample_every,
                           n_reps, seed=0):
    """For each alpha, classify phases vs beta; return list of transition
    beta positions and the phase labels per beta."""
    res = scan_grid_gpu(alphas, betas, L, n_steps, warmup, sample_every,
                        n_reps=n_reps, seed=seed, desc=f"L={L} beta-scan")
    na, nb = len(alphas), len(betas)
    # collect per-replica rho per (alpha,beta)
    grid = np.empty((na, nb), dtype=object)
    k = 0
    for i in range(na):
        for j in range(nb):
            reps = res[k * n_reps:(k + 1) * n_reps]
            rhos1 = [r[2] for r in reps]
            rhos2 = [r[3] for r in reps]
            grid[i, j] = (rhos1, rhos2)
            k += 1
    return grid


def phase_boundary_in_alpha(alphas, betas, L, n_steps, warmup, sample_every,
                            n_reps, seed=0):
    return phase_boundary_in_beta(alphas, betas, L, n_steps, warmup,
                                  sample_every, n_reps, seed)


def find_transition(values, labels, target_order):
    """Find positions where the label changes (transitions)."""
    # values: the scan axis (betas or alphas)
    transitions = []
    prev = labels[0]
    for j in range(1, len(labels)):
        if labels[j] != prev:
            transitions.append(0.5 * (values[j - 1] + values[j]))
            prev = labels[j]
    return transitions


def main():
    n_reps = 8
    n_steps = 2_000_000
    warmup = 300_000
    sample_every = 200
    Ls = np.array([200, 500, 1000, 2000, 4000])

    # (a) beta boundary (asym -> LD) vs L, alpha = 0.9
    alpha_fixed = 0.9
    betas = np.linspace(0.05, 0.6, 24)
    a_theory = alpha_fixed / (1 + alpha_fixed + alpha_fixed**2)
    print("(a) beta boundary (asym->LD) vs L, alpha=0.9", flush=True)
    a_asym = []
    for L in Ls:
        grid = phase_boundary_in_beta([alpha_fixed], betas, L, n_steps,
                                      warmup, sample_every, n_reps)
        labels = []
        for j in range(len(betas)):
            r1s, r2s = grid[0][j]
            lab, _ = classify_point(r1s, r2s, alpha_fixed, betas[j])
            labels.append(lab)
        # transition: last asym -> LD
        trans = None
        for j in range(1, len(betas)):
            if labels[j - 1] != "LD" and labels[j] == "LD":
                trans = 0.5 * (betas[j - 1] + betas[j])
        a_asym.append(trans)
        print(f"  L={L}: asym->LD at beta={trans}  (labels={labels})", flush=True)
    a_asym = np.array(a_asym, dtype=float)

    # (b) alpha boundary (LD -> MC) vs L, beta = 1.0
    beta_fixed = 1.0
    alphas = np.linspace(0.2, 0.95, 24)
    b_theory_mc = 2 * beta_fixed / (4 * beta_fixed - 1)
    print("(b) alpha boundary (LD->MC) vs L, beta=1.0", flush=True)
    b_mcld = []
    for L in Ls:
        grid = phase_boundary_in_alpha(alphas, [beta_fixed], L, n_steps,
                                       warmup, sample_every, n_reps)
        labels = []
        for i in range(len(alphas)):
            r1s, r2s = grid[i][0]
            lab, _ = classify_point(r1s, r2s, alphas[i], beta_fixed)
            labels.append(lab)
        trans = None
        for i in range(1, len(alphas)):
            if labels[i - 1] == "LD" and labels[i] == "MC":
                trans = 0.5 * (alphas[i - 1] + alphas[i])
        b_mcld.append(trans)
        print(f"  L={L}: LD->MC at alpha={trans}  (labels={labels})", flush=True)
    b_mcld = np.array(b_mcld, dtype=float)

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.errorbar(Ls, a_asym, fmt="o-", ms=6, capsize=4,
                label="MC asym$\\to$LD boundary")
    ax.axhline(a_theory, color="k", ls=":", lw=1.2,
               label=f"MFT (eq23)={a_theory:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\beta$ boundary")
    ax.set_title(rf"(a) asym$-$LD boundary, $\alpha={alpha_fixed}$")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.errorbar(Ls, b_mcld, fmt="o-", ms=6, capsize=4,
                label="MC LD/MC boundary")
    ax.axhline(b_theory_mc, color="k", ls=":", lw=1.2,
               label=f"MFT MC/LD (eq10)={b_theory_mc:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\alpha$ boundary")
    ax.set_title(rf"(b) MC/LD boundary, $\beta={beta_fixed}$")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_phase_boundary_vs_L.png", dpi=150)
    print("saved fig5_phase_boundary_vs_L.png", flush=True)
    np.savez(f"{OUT}/fig5_boundaries.npz", Ls=Ls, beta_asym=a_asym,
             alpha_mcld=b_mcld, theory_hdld=a_theory, theory_mc=b_theory_mc)


if __name__ == "__main__":
    main()
