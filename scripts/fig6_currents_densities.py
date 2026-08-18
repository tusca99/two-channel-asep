"""
Reproduce Figure 6 of Pronina & Kolomeisky (2007): stationary properties
vs beta, showing the finite-size approach to L=1000.

Layout: one row per alpha. For each alpha, J1/J2 (currents) or rho1/rho2
(bulk densities) are plotted vs beta with MULTIPLE lattice sizes overlaid
as lines, so the finite-size approach is visible. MFT line shown dashed.

  alpha=0.1 : currents J1, J2
  alpha=0.8 : currents J1, J2
  alpha=0.9 : bulk densities rho1, rho2
"""
import numpy as np
import matplotlib.pyplot as plt

from asep.parallel import scan_points


def scan_alpha_beta(alphas, betas, L, n_steps, warmup, sample_every, seed=0):
    """Scan the full (alphas x betas) grid; return (J1,J2,r1,r2) arrays."""
    rng = np.random.default_rng(seed)
    tasks = []
    for a in alphas:
        for b in betas:
            tasks.append((a, b, L, n_steps, warmup, sample_every,
                          int(rng.integers(1e9))))
    res = scan_points(tasks, desc=f"L={L}", adaptive=True)
    na, nb = len(alphas), len(betas)
    J1 = np.zeros((na, nb)); J2 = np.zeros((na, nb))
    r1 = np.zeros((na, nb)); r2 = np.zeros((na, nb))
    k = 0
    for i in range(na):
        for j in range(nb):
            J1[i, j], J2[i, j], r1[i, j], r2[i, j] = res[k]
            k += 1
    return J1, J2, r1, r2


def mft_currents(alpha, beta):
    """MFT current prediction (single-lane LD: a1(1-a1))."""
    disc = (alpha + beta) ** 2 - 4 * alpha**2 * beta
    if disc < 0:
        return np.nan
    a1 = (alpha + beta - np.sqrt(disc)) / (2 * alpha)
    return a1 * (1 - a1)


def main():
    Ls = np.array([200, 400, 800, 1600])
    n_steps = 2_000_000
    warmup = 200_000
    sample_every = 400
    betas = np.linspace(0.05, 0.95, 40)
    alphas = np.array([0.1, 0.8, 0.9])
    colors = ["C0", "C1", "C2", "C3"]

    fig, axes = plt.subplots(3, 1, figsize=(7, 11))
    for ai, a in enumerate(alphas):
        ax = axes[ai]
        # collect over L
        for li, L in enumerate(Ls):
            J1, J2, r1, r2 = scan_alpha_beta([a], betas, L, n_steps, warmup,
                                             sample_every)
            if a < 0.5:
                ax.plot(betas, J1[0], "-", color=colors[li], lw=1.5,
                        label=rf"$J_1$, L={L}")
                ax.plot(betas, J2[0], "--", color=colors[li], lw=1.2,
                        label=rf"$J_2$, L={L}")
            else:
                ax.plot(betas, r1[0], "-", color=colors[li], lw=1.5,
                        label=rf"$\rho_1$, L={L}")
                ax.plot(betas, r2[0], "--", color=colors[li], lw=1.2,
                        label=rf"$\rho_2$, L={L}")
        # MFT line
        if a < 0.5:
            mft = [mft_currents(a, b) for b in betas]
            ax.plot(betas, mft, "k:", lw=1.5, label="MFT")
            ax.set_ylabel(r"$J$")
        else:
            ax.set_ylabel(r"$\rho$")
        ax.set_xlabel(r"$\beta$")
        ax.set_title(rf"$\alpha={a}$")
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Figure 6: currents (alpha=0.1,0.8) and densities (alpha=0.9)")
    fig.tight_layout()
    fig.savefig("results/fig6_curves_vs_L.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
