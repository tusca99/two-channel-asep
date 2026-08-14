"""
Reproduce Figure 5 of Pronina & Kolomeisky (2007): phase boundaries vs L.

Two panels:
(a) beta_LD/HD <-> LD boundary vs L (for fixed alpha): where the system
    transitions from asymmetric SSB to symmetric LD, located via the order
    parameter <|rho1-rho2|>.
(b) alpha boundary vs L: the alpha where a transition occurs, for fixed beta.

Each point is averaged over n_runs independent seeds (mean +/- std error).
L is capped at 4000: larger L cannot equilibrate within a reasonable run
(the paper used 2e7-5e8 steps/site to reach L=12000).
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from asep.parallel import make_tasks, scan_points


def scan_boundary_curves(alphas, betas, L, n_steps, warmup, sample_every,
                         n_runs, seed=0):
    """
    Scan (alphas x betas) grid with n_runs statistics.

    Returns (mean_order, err_order) each shape (n_alphas, n_betas), where
    order = <|rho1-rho2|>.
    """
    rng = np.random.default_rng(seed)
    na, nb = len(alphas), len(betas)
    order_sum = np.zeros((na, nb))
    order_sq = np.zeros((na, nb))
    # one task per (alpha, beta, run)
    tasks = [(a, b, L, n_steps, warmup, sample_every,
              int(rng.integers(1e9)))
             for a in alphas for b in betas for _ in range(n_runs)]
    res = scan_points(tasks, desc=f"L={L} ({n_runs} runs/pt)")
    k = 0
    for i in range(na):
        for j in range(nb):
            vals = [np.abs(r[2] - r[3]) for r in res[k:k + n_runs]]
            k += n_runs
            order_sum[i, j] = np.mean(vals)
            order_sq[i, j] = np.std(vals)
    return order_sum, order_sq


def locate_boundary(betas, order, err, threshold=0.1):
    """First beta where order drops below threshold; return (beta, err_beta)."""
    for j in range(len(betas)):
        if order[j] < threshold:
            return betas[j], err[j]
    return None, None


def main():
    n_runs = 10
    n_steps = 10_000_000
    warmup = 1_000_000
    sample_every = 400
    Ls = np.array([200, 800, 2000, 4000])

    # (a) beta-boundary vs L for fixed alpha=0.9
    alpha = 0.9
    betas = np.linspace(0.02, 0.30, 15)
    print(f"(a) beta-boundary vs L, alpha={alpha}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    L_boundaries = []
    L_errs = []
    for L in Ls:
        order, err = scan_boundary_curves([alpha], betas, L, n_steps, warmup,
                                          sample_every, n_runs)
        bc, bc_err = locate_boundary(betas, order[0], err[0])
        L_boundaries.append(bc)
        L_errs.append(bc_err)
        print(f"  L={L}: beta_boundary={bc}, err={bc_err}")
    L_boundaries = np.array(L_boundaries)
    L_errs = np.array(L_errs)
    ax.errorbar(Ls, L_boundaries, yerr=L_errs, fmt="o-", ms=6, capsize=4,
                label="MC boundary")
    ax.axhline(alpha / (1 + alpha + alpha**2), color="k", ls="--", lw=1.2,
               label="MFT (eq 23)")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\beta_{HD/LD \leftrightarrow LD}$")
    ax.set_title(rf"(a) Beta boundary vs L, $\alpha={alpha}$")
    ax.set_xscale("log")
    ax.legend(fontsize=8)

    # (b) alpha-boundary vs L for fixed beta: find alpha where order drops
    beta_fixed = 0.1
    alphas = np.linspace(0.02, 0.30, 15)
    print(f"(b) alpha-boundary vs L, beta={beta_fixed}")
    ax = axes[1]
    L_a_boundaries = []
    L_a_errs = []
    for L in Ls:
        order, err = scan_boundary_curves(alphas, [beta_fixed], L, n_steps,
                                          warmup, sample_every, n_runs)
        # order vs alpha: SSB when one channel is high -> |drho| large at small alpha
        ac, ac_err = locate_boundary(alphas, order[:, 0], err[:, 0])
        L_a_boundaries.append(ac)
        L_a_errs.append(ac_err)
        print(f"  L={L}: alpha_boundary={ac}, err={ac_err}")
    L_a_boundaries = np.array(L_a_boundaries)
    L_a_errs = np.array(L_a_errs)
    ax.errorbar(Ls, L_a_boundaries, yerr=L_a_errs, fmt="o-", ms=6, capsize=4,
                label="MC boundary")
    ax.axhline(beta_fixed / (1 + beta_fixed + beta_fixed**2), color="k",
               ls="--", lw=1.2, label="MFT")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\alpha_{HD/LD \leftrightarrow LD}$")
    ax.set_title(rf"(b) Alpha boundary vs L, $\beta={beta_fixed}$")
    ax.set_xscale("log")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("results/fig5_phase_boundary_vs_L.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
