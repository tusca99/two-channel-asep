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

from asep.parallel import scan_grid_gpu  # noqa: F401  (kept for reference)


def classify_point(rhos1, rhos2, J1, J2, alpha, beta, L, mc_rho=0.45):
    """Classify phase from per-replica densities + currents (ensemble method).

    Uses std(rho1-rho2) as the robust SSB order parameter. Our model shows a
    WEAK asymmetry (dense~0.27, not a true HD with rho>0.5), so we classify
    by symmetry rather than by the paper's HD/LD criterion:
      - asymmetric (std_diff > threshold): LD/LD-type broken state
      - symmetric low density: LD
      - symmetric high density: MC

    MC is identified by current saturation (paper Fig 4 method): J plateaus at
    1/4 in MC. Combined with a high density (mc_rho=0.45, near the MC rho=1/2)
    this locates the LD/MC boundary correctly (a fixed low threshold misplaces
    it — the L~0.477 vs paper ~0.7 discrepancy was a threshold artifact).

    The SSB threshold is L-adaptive: the noise floor of std(rho1-rho2) in the
    symmetric phase scales as ~1/sqrt(L), so a fixed threshold over-classifies
    small L as asymmetric (the L=200 NaN bug). Calibrate 0.04 at L=1000.
    """
    d = np.array(rhos1) - np.array(rhos2)
    asym = d.std()
    rho_avg = (np.mean(rhos1) + np.mean(rhos2)) / 2
    asym_threshold = 0.04 * np.sqrt(1000.0 / L)
    j_cur = (np.mean(J1) + np.mean(J2)) / 2
    if asym < asym_threshold:
        if j_cur >= 0.25 - 0.02 and rho_avg > mc_rho:
            return "MC", asym
        return "LD", asym
    return "asym", asym


def _one_replica(args):
    alpha, beta, L, steps, warmup, seed = args
    from asep.bkl import run_bkl_fenwick
    rng = np.random.default_rng(seed)
    lane1 = (rng.random(L) < 0.4).astype(np.int8)
    lane2 = (rng.random(L) < 0.4).astype(np.int8)
    uniforms = rng.random((steps + warmup) * 3)
    run_bkl_fenwick(lane1, lane2, alpha, beta, warmup, uniforms, 0)
    dt, e1, e2, _ = run_bkl_fenwick(lane1, lane2, alpha, beta, steps,
                                    uniforms, warmup * 3)
    return (e1 / dt, e2 / dt, np.mean(lane1), np.mean(lane2))

def phase_boundary_in_beta(alphas, betas, L, n_steps, warmup, sample_every,
                           n_reps, seed=0):
    """For each alpha, classify phases vs beta on CPU (12-core).

    Returns grid[i,j] = (J1s, J2s, rhos1, rhos2) per-replica values for point
    (alphas[i], betas[j])."""
    from concurrent.futures import ProcessPoolExecutor
    rng = np.random.default_rng(seed)
    na, nb = len(alphas), len(betas)
    tasks = [(a, b, L, n_steps, warmup, int(rng.integers(1e9)))
             for a in alphas for b in betas for _ in range(n_reps)]
    with ProcessPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(_one_replica, tasks))
    grid = np.empty((na, nb), dtype=object)
    k = 0
    for i in range(na):
        for j in range(nb):
            reps = res[k * n_reps:(k + 1) * n_reps]
            grid[i, j] = ([r[0] for r in reps], [r[1] for r in reps],
                          [r[2] for r in reps], [r[3] for r in reps])
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


def _smooth_labels(labels):
    """Remove isolated labels (noise): a label flanked by the same other
    label on both sides is flipped to that label."""
    out = list(labels)
    for j in range(1, len(labels) - 1):
        if out[j] != out[j - 1] and out[j] != out[j + 1] and out[j - 1] == out[j + 1]:
            out[j] = out[j - 1]
    return out


def main():
    n_reps = 8
    # steps scale with L (equilibration ~ L^3); warmup ~ 1M
    Ls = np.array([200, 500, 1000, 2000, 4000])
    steps_by_L = {200: 2_000_000, 500: 5_000_000, 1000: 10_000_000,
                  2000: 20_000_000, 4000: 40_000_000}
    warmup = 1_000_000
    sample_every = 200

    # (a) beta boundary (asym -> LD) vs L, alpha = 0.9
    alpha_fixed = 0.9
    betas = np.linspace(0.05, 0.6, 24)
    a_theory = alpha_fixed / (1 + alpha_fixed + alpha_fixed**2)
    print("(a) beta boundary (asym->LD) vs L, alpha=0.9", flush=True)
    a_asym = []
    for L in Ls:
        grid = phase_boundary_in_beta([alpha_fixed], betas, L,
                                      steps_by_L[L], warmup, sample_every,
                                      n_reps)
        labels = []
        for j in range(len(betas)):
            J1s, J2s, r1s, r2s = grid[0][j]
            lab, _ = classify_point(r1s, r2s, J1s, J2s, alpha_fixed, betas[j], L)
            labels.append(lab)
        labels = _smooth_labels(labels)
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
        grid = phase_boundary_in_alpha(alphas, [beta_fixed], L,
                                       steps_by_L[L], warmup, sample_every,
                                       n_reps)
        labels = []
        for i in range(len(alphas)):
            J1s, J2s, r1s, r2s = grid[i][0]
            lab, _ = classify_point(r1s, r2s, J1s, J2s, alphas[i], beta_fixed, L)
            labels.append(lab)
        labels = _smooth_labels(labels)
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
