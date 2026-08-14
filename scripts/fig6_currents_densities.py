"""
Reproduce Figure 6 of Pronina & Kolomeisky (2007): stationary properties
vs beta.

The paper shows: (a) currents in both channels for alpha=0.1, (b) currents
for alpha=0.8, (c) bulk densities for alpha=0.9. MFT lines overlaid.

We plot J1, J2 (currents) and rho1, rho2 (bulk densities) vs beta for several
lattice sizes L, so the finite-size approach to the L=1000 (paper) limit is
visible. Also provide heatmap panels (alpha vs beta) for a global view.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from asep.parallel import scan_points


def scan_point_results(tasks, desc="scan"):
    """Run tasks and return list of (J1,J2,rho1,rho2)."""
    return scan_points(tasks, desc=desc)


def scan_alpha_beta(alphas, betas, L, n_steps, warmup, sample_every, seed=0):
    """Scan the full (alphas x betas) grid; return dict keyed by alpha with arrays."""
    rng = np.random.default_rng(seed)
    tasks = []
    for a in alphas:
        for b in betas:
            tasks.append((a, b, L, n_steps, warmup, sample_every,
                          int(rng.integers(1e9))))
    res = scan_points(tasks, desc=f"L={L}")
    # reshape
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


def plot_curves_panel(ax, alphas, betas, J1, J2, L, what="J"):
    """Plot J1,J2 vs beta for one alpha across several L (overlaid)."""
    for a in alphas:
        for label, data in [("ch1", J1), ("ch2", J2)]:
            ax.plot(betas, data, "o-", ms=3, alpha=0.7,
                    label=rf"$\alpha={a}$ {label}")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$J$" if what == "J" else r"$\rho$")
    ax.set_title(f"L={L}")


def plot_heatmap(ax, alphas, betas, mat, title, cmap="viridis"):
    im = ax.imshow(mat, aspect="auto", origin="lower",
                   extent=[betas[0], betas[-1], alphas[0], alphas[-1]],
                   cmap=cmap, interpolation="nearest")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(title)
    return im


def main():
    Ls = np.array([200, 1000])
    n_steps = 2_000_000
    warmup = 200_000
    sample_every = 400
    betas = np.linspace(0.05, 0.95, 40)
    alphas = np.array([0.1, 0.8, 0.9])

    # Per-alpha: J1, J2, rho1, rho2 vs beta for each L
    fig, axes = plt.subplots(3, len(Ls), figsize=(4 * len(Ls), 10))
    for li, L in enumerate(Ls):
        J1, J2, r1, r2 = scan_alpha_beta(alphas, betas, L, n_steps, warmup,
                                         sample_every)
        for ai, a in enumerate(alphas):
            ax = axes[ai, li]
            if a < 0.5:
                # currents
                ax.plot(betas, J1[ai], "o-", ms=3, label="ch1")
                ax.plot(betas, J2[ai], "s-", ms=3, label="ch2")
                mft = [mft_currents(a, b) for b in betas]
                ax.plot(betas, mft, "k--", lw=1, label="MFT")
                ax.set_ylabel(r"$J$")
                ax.set_title(rf"$\alpha={a}$, L={L}")
            else:
                # densities
                ax.plot(betas, r1[ai], "o-", ms=3, label="ch1")
                ax.plot(betas, r2[ai], "s-", ms=3, label="ch2")
                ax.set_ylabel(r"$\rho$")
                ax.set_title(rf"$\alpha={a}$, L={L}")
            ax.set_xlabel(r"$\beta$")
            ax.legend(fontsize=6)
    fig.suptitle("Figure 6: currents (alpha=0.1,0.8) and densities (alpha=0.9)")
    fig.tight_layout()
    fig.savefig("results/fig6_curves_vs_L.png", dpi=150)
    plt.show()

    # Heatmap: J1 over alpha x beta for one L
    L = 1000
    J1, J2, r1, r2 = scan_alpha_beta(np.linspace(0.1, 1.0, 10), betas, L,
                                     n_steps, warmup, sample_every)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_heatmap(axes[0], np.linspace(0.1, 1.0, 10), betas, J1,
                 rf"$J_1(\alpha,\beta)$, L={L}")
    plot_heatmap(axes[1], np.linspace(0.1, 1.0, 10), betas,
                 (r1 + r2) / 2, rf"$\bar\rho(\alpha,\beta)$, L={L}")
    fig.tight_layout()
    fig.savefig("results/fig6_heatmaps.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
