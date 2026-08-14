"""
Reproduce the ORIGINAL Figure 5 of Pronina & Kolomeisky (2007), fast (no stats).

(a) Phase boundary between asymmetric HD/LD and LD/LD phases vs L, for alpha=0.9.
    The paper finds this boundary does NOT depend on L.
(b) Phase boundary between symmetric LD and MC phases vs L, for beta=1.
    The paper finds this boundary shifts toward the MFT prediction with L.

Fast version: single run per point (no error bars), to check the qualitative
behavior matches the paper.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from asep.parallel import make_tasks, scan_points


def scan_curves(alphas, betas, L, n_steps, warmup, sample_every, seed=0):
    """Scan (alphas x betas) grid, single run per point."""
    rng = np.random.default_rng(seed)
    tasks = [(a, b, L, n_steps, warmup, sample_every, int(rng.integers(1e9)))
             for a in alphas for b in betas]
    res = scan_points(tasks, desc=f"L={L}")
    na, nb = len(alphas), len(betas)
    J1 = np.zeros((na, nb)); J2 = np.zeros((na, nb))
    r1 = np.zeros((na, nb)); r2 = np.zeros((na, nb))
    k = 0
    for i in range(na):
        for j in range(nb):
            J1[i, j], J2[i, j], r1[i, j], r2[i, j] = res[k]
            k += 1
    return J1, J2, r1, r2


def main():
    Ls = np.array([200, 400, 800, 1600])
    n_steps = 2_000_000
    warmup = 200_000
    sample_every = 400

    # (a) HD/LD <-> LD/LD boundary vs L for alpha=0.9
    # Locate the beta where the order parameter |rho1-rho2| transitions
    # between the two asymmetric phases. We track the beta where the
    # asymmetry magnitude changes character.
    alpha = 0.9
    betas = np.linspace(0.02, 0.30, 20)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for L in Ls:
        J1, J2, r1, r2 = scan_curves([alpha], betas, L, n_steps, warmup,
                                     sample_every)
        order = np.abs(r1[0] - r2[0])
        ax.plot(betas, order, "o-", ms=4, label=rf"L={L}")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\langle|\rho_1-\rho_2|\rangle$")
    ax.set_title(rf"(a) Order parameter vs beta, $\alpha={alpha}$")
    ax.legend(fontsize=8)

    # (b) LD <-> MC boundary vs L for beta=1: locate where Jtot saturates
    beta = 1.0
    alphas = np.linspace(0.2, 1.0, 20)
    ax = axes[1]
    for L in Ls:
        J1, J2, r1, r2 = scan_curves(alphas, [beta], L, n_steps, warmup,
                                     sample_every)
        Jtot = J1[:, 0] + J2[:, 0]
        ax.plot(alphas, Jtot, "o-", ms=4, label=rf"L={L}")
    # MFT MC boundary: alpha = 2*beta/(4*beta-1) = 2/3 for beta=1
    ax.axvline(2.0 / 3.0, color="k", ls="--", lw=1.2, label="MFT (eq 10)")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$J_1+J_2$")
    ax.set_title(rf"(b) Current vs alpha, $\beta={beta}$")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("results/fig5_original_fast.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
