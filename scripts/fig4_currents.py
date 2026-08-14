"""
Reproduce Figure 4 of Pronina & Kolomeisky (2007): current and its derivative.

The paper shows (a) current J vs beta and (b) dJ/dbeta vs beta for a fixed
alpha. The LD/MC phase boundary is where dJ/dbeta reaches zero (the current
saturates to its maximum in the MC phase).

We split into two clean panels: left = J vs beta, right = dJ/dbeta vs beta.
For multiple alphas we use a heatmap (alpha on y, beta on x, color = J)
instead of overlapping curves, which is clearer.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from asep.parallel import scan_points


def scan_currents(alpha, betas, L, n_steps, warmup, sample_every, seed=0):
    """Return Jtot = J1+J2 over beta for fixed alpha (parallel)."""
    rng = np.random.default_rng(seed)
    tasks = [(alpha, b, L, n_steps, warmup, sample_every,
              int(rng.integers(1e9))) for b in betas]
    res = scan_points(tasks, desc=f"alpha={alpha}")
    return np.array([r[0] + r[1] for r in res])


def plot_j_vs_beta(ax, alphas, betas, J_by_alpha, mft_lines=False):
    """Current vs beta for several alphas (heatmap if many)."""
    if len(alphas) > 3:
        # Heatmap: y=alpha, x=beta, color=J
        Jmat = np.vstack(J_by_alpha)
        im = ax.imshow(Jmat, aspect="auto", origin="lower",
                       extent=[betas[0], betas[-1], alphas[0], alphas[-1]],
                       cmap="viridis", interpolation="nearest")
        ax.set_ylabel(r"$\alpha$")
        cb = plt.colorbar(im, ax=ax)
        cb.set_label(r"$J_1+J_2$")
    else:
        for alpha, J in zip(alphas, J_by_alpha):
            ax.plot(betas, J, "o-", ms=4, label=rf"$\alpha={alpha}$")
        ax.legend(fontsize=8)
        ax.set_ylabel(r"$J_1+J_2$")
    ax.set_xlabel(r"$\beta$")
    ax.axhline(0.5, color="r", ls=":", lw=1, label="MC max")


def plot_dJ_vs_beta(ax, alphas, betas, J_by_alpha):
    """dJ/dbeta vs beta for several alphas (heatmap if many)."""
    if len(alphas) > 3:
        dJmat = np.vstack([np.gradient(J, betas) for J in J_by_alpha])
        im = ax.imshow(dJmat, aspect="auto", origin="lower",
                       extent=[betas[0], betas[-1], alphas[0], alphas[-1]],
                       cmap="coolwarm", interpolation="nearest")
        ax.set_ylabel(r"$\alpha$")
        cb = plt.colorbar(im, ax=ax)
        cb.set_label(r"$dJ/d\beta$")
    else:
        for alpha, J in zip(alphas, J_by_alpha):
            dJ = np.gradient(J, betas)
            ax.plot(betas, dJ, "o-", ms=4, label=rf"$\alpha={alpha}$")
        ax.legend(fontsize=8)
        ax.set_ylabel(r"$dJ/d\beta$")
    ax.set_xlabel(r"$\beta$")
    ax.axhline(0, color="k", ls=":", lw=1)


def main():
    L = 1000
    n_steps = 2_000_000
    warmup = 200_000
    sample_every = 400
    betas = np.linspace(0.05, 0.95, 40)

    # Single alpha, two clean panels (like the paper): J | dJ/dbeta
    alpha = 0.9
    J = scan_currents(alpha, betas, L, n_steps, warmup, sample_every)
    dJ = np.gradient(J, betas)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(betas, J, "o-", ms=4)
    axes[0].set_xlabel(r"$\beta$"); axes[0].set_ylabel(r"$J_1+J_2$")
    axes[0].set_title(rf"Current, $\alpha={alpha}$, $L={L}$")
    axes[1].plot(betas, dJ, "o-", ms=4, color="C1")
    axes[1].axhline(0, color="k", ls=":", lw=1)
    axes[1].set_xlabel(r"$\beta$"); axes[1].set_ylabel(r"$dJ/d\beta$")
    axes[1].set_title(rf"Current derivative, $\alpha={alpha}$, $L={L}$")
    fig.suptitle("Figure 4: LD/MC boundary via current saturation")
    fig.tight_layout()
    fig.savefig("results/fig4_current_derivative.png", dpi=150)
    plt.show()

    # Optional: heatmap over many alphas
    alphas = np.linspace(0.3, 1.0, 8)
    J_by_alpha = []
    for a in tqdm(alphas, desc="alpha sweep"):
        J_by_alpha.append(scan_currents(a, betas, L, n_steps, warmup, sample_every))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_j_vs_beta(axes[0], alphas, betas, J_by_alpha)
    axes[0].set_title(rf"Heatmap: $J(\alpha,\beta)$, $L={L}$")
    plot_dJ_vs_beta(axes[1], alphas, betas, J_by_alpha)
    axes[1].set_title(rf"Heatmap: $dJ/d\beta(\alpha,\beta)$, $L={L}$")
    fig.tight_layout()
    fig.savefig("results/fig4_heatmap.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
