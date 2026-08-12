"""
Spontaneous symmetry breaking (SSB) analysis via the joint density
distribution P(rho1, rho2).

Reproduces the method of Pronina & Kolomeisky (2007, Sec 3): measure the
simultaneous bulk densities (rho1, rho2) after every L/10 MC steps and build
a 2D histogram. In symmetric phases P is unimodal (single peak near the
diagonal rho1=rho2); in SSB phases it is bimodal (two peaks off the diagonal,
related by the Z2 channel-swap symmetry).
"""
import numpy as np
import matplotlib.pyplot as plt

from asep import TwoChannelASEP


def collect_joint_samples(alpha, beta, L, n_steps, warmup, seed=0):
    """
    Run MC and return the (rho1, rho2) joint density samples.

    Sampling interval follows the paper: every L/10 MC steps.
    """
    sample_every = max(1, L // 10)
    sim = TwoChannelASEP(L=L, alpha=alpha, beta=beta, seed=seed)
    sim.run(n_steps=n_steps, sample_every=sample_every, warmup=warmup)
    return sim.get_joint_density_samples()


def bimodality(samples, bins=40):
    """
    Detect SSB from the joint density distribution.

    Returns (is_ssb, asymmetry) where asymmetry is the mean |rho1 - rho2|.
    In an SSB phase the distribution is bimodal and |rho1 - rho2| is large;
    in a symmetric phase it is unimodal and |rho1 - rho2| ~ 0.
    """
    r1, r2 = samples[:, 0], samples[:, 1]
    asymmetry = np.mean(np.abs(r1 - r2))
    # SSB: the two channels spend time at different densities
    is_ssb = asymmetry > 0.1
    return is_ssb, asymmetry


def plot_joint_distribution(samples, alpha, beta, L, ax=None):
    """2D histogram of P(rho1, rho2)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    r1, r2 = samples[:, 0], samples[:, 1]
    h = ax.hist2d(r1, r2, bins=40, range=[[0, 1], [0, 1]], cmap="viridis")
    ax.plot([0, 1], [0, 1], "r--", lw=1, label="rho1 = rho2")
    ax.set_xlabel(r"$\rho_1$")
    ax.set_ylabel(r"$\rho_2$")
    ax.set_title(rf"P($\rho_1,\rho_2$), $\alpha={alpha}$, $\beta={beta}$, L={L}")
    ax.legend(fontsize=7)
    return h


def main():
    L = 200
    n_steps = 1_000_000
    warmup = 50_000

    # Points: symmetric LD, SSB HD/LD, and near the LD/LD region
    points = [
        (0.9, 0.9, "symmetric (MC)"),
        (0.9, 0.1, "SSB (HD/LD)"),
        (0.5, 0.1, "SSB (HD/LD)"),
        (0.3, 0.2, "near LD/LD"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (a, b, label) in zip(axes.ravel(), points):
        samples = collect_joint_samples(a, b, L, n_steps, warmup)
        is_ssb, asym = bimodality(samples)
        plot_joint_distribution(samples, a, b, L, ax)
        ax.set_title(ax.get_title() + f"\n[{label}] SSB={is_ssb}, |drho|={asym:.2f}",
                     fontsize=9)

    fig.tight_layout()
    fig.savefig("results/ssb_joint_density.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
