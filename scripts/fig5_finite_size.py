"""
Reproduce Figure 5 of Pronina & Kolomeisky (2007): phase boundaries vs L.

(a) For alpha = 0.9, locate the boundary beta between the asymmetric
    (HD/LD, SSB) and symmetric (LD) phases, and plot how this boundary
    beta moves with the channel size L. This uses the density-distribution /
    order-parameter method: the system is asymmetric when the mean
    |rho1 - rho2| is large, symmetric when it is near 0.

The mean-field boundary is beta = alpha/(1+alpha+alpha^2) (eq 23).
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from asep.parallel import make_tasks, scan_points


def scan_order(alpha, betas, L, n_steps, warmup, sample_every, seed=0):
    """Return mean |rho1-rho2| over beta for fixed alpha, L (parallel)."""
    rng = np.random.default_rng(seed)
    tasks = [(alpha, b, L, n_steps, warmup, sample_every,
              int(rng.integers(1e9))) for b in betas]
    res = scan_points(tasks, desc=f"L={L}")
    return np.array([np.abs(r[2] - r[3]) for r in res])


def locate_boundary(betas, order, threshold=0.1):
    """
    First beta where the order parameter drops below threshold (asym->sym).
    Returns None if never symmetric.
    """
    for j in range(len(betas)):
        if order[j] < threshold:
            return betas[j]
    return None


def main():
    alpha = 0.9
    # L capped at 800: at larger L the system flips between broken states,
    # washing out the time-averaged order parameter (the paper's Fig 5 uses
    # the density-distribution method to handle this).
    Ls = np.array([200, 400, 800])
    betas = np.linspace(0.02, 0.30, 40)
    n_steps = 4_000_000
    warmup = 400_000
    sample_every = 400

    mft_boundary = alpha / (1 + alpha + alpha**2)
    print(f"alpha={alpha}: MFT asym/sym boundary beta = {mft_boundary:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    boundaries = []
    for L in tqdm(Ls, desc="L scan"):
        order = scan_order(alpha, betas, L, n_steps, warmup, sample_every)
        b = locate_boundary(betas, order)
        boundaries.append(b)
        print(f"  L={L}: asym/sym boundary beta={b}")

    ax.plot(Ls, boundaries, "o-", ms=6, label="MC boundary")
    ax.axhline(mft_boundary, color="k", ls="--", lw=1.2, label="MFT (eq 23)")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\beta_{HD/LD \leftrightarrow LD}$")
    ax.set_title(rf"Phase boundary vs L, $\alpha={alpha}$")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("results/fig5_phase_boundary_vs_L.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
