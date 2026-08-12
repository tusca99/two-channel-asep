"""
Reproduce Figure 2 of Pronina & Kolomeisky (2007): phase diagram.

Plots the mean-field phase boundaries (eqs 10, 13, 23, 33) and overlays
Monte Carlo phase classifications from the numba kernel. Axes follow the
paper: alpha on x, beta on y. Two panels: (a) full parameter space,
(b) zoom on 0.2 < beta < 0.4 (as in the paper's Fig 2b).
"""
import numpy as np
import matplotlib.pyplot as plt

from asep import TwoChannelASEP
from asep.observables import classify_phase

# Colors for the MC phase classifications (used in the legend)
PHASE_COLORS = {
    "LD": "tab:blue",
    "MC": "tab:green",
    "HD/LD": "tab:red",
    "LD/HD": "tab:red",
    "LD/LD": "tab:orange",
}


# --- Mean-field boundaries ------------------------------------------------

def mc_boundary(beta):
    """eq 10: alpha > 2*beta/(4*beta-1) for MC (beta > 1/2)."""
    return 2 * beta / (4 * beta - 1)


def ld_boundary(beta):
    """eq 13: alpha < 2*beta/(4*beta-1) for LD."""
    return 2 * beta / (4 * beta - 1)


def hdld_boundary(alpha):
    """eq 23: beta < alpha/(1+alpha+alpha^2) for HD/LD."""
    return alpha / (1 + alpha + alpha**2)


def ldld_upper(alpha):
    """eq 33 upper: beta < [alpha(1-2a)+2a sqrt(a^2-a+1)]/3."""
    a = alpha
    return (a * (1 - 2 * a) + 2 * a * np.sqrt(a**2 - a + 1)) / 3


def plot_mft_boundaries(ax):
    """Draw the four MFT phase boundaries on axis `ax` (alpha on x, beta on y)."""
    # MC / LD boundary (eq 10/13), only for beta > 1/2
    b = np.linspace(0.5, 1.0, 200)
    a = mc_boundary(b)
    ax.plot(a, b, "k-", lw=1.5, label="LD/MC (eq 10)")

    # HD/LD boundary (eq 23)
    a = np.linspace(0.0, 1.0, 200)
    b = hdld_boundary(a)
    ax.plot(a, b, "k--", lw=1.5, label="HD/LD (eq 23)")

    # LD/LD upper boundary (eq 33) - nearly coincides with eq 23
    a = np.linspace(0.0, 1.0, 200)
    b = ldld_upper(a)
    ax.plot(a, b, "k:", lw=1.5, label="LD/LD (eq 33)")


# --- Monte Carlo scan ----------------------------------------------------

def scan_phase_diagram(alphas, betas, L, n_steps, warmup, sample_every, seed=0):
    """
    Classify the phase at each (alpha, beta) grid point via MC.

    Returns a 2D array of phase labels (strings).
    """
    rng = np.random.default_rng(seed)
    grid = np.empty((len(alphas), len(betas)), dtype=object)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            sim = TwoChannelASEP(L=L, alpha=a, beta=b, seed=int(rng.integers(1e9)))
            sim.run(n_steps=n_steps, sample_every=sample_every, warmup=warmup)
            J1, J2 = sim.get_currents()
            rho1, rho2 = sim.get_bulk_densities()
            label, _ = classify_phase(J1, J2, rho1, rho2, a, b, L)
            grid[i, j] = label
    return grid


def plot_mc_grid(ax, alphas, betas, grid):
    """Scatter the MC phase classifications on `ax` (alpha on x, beta on y)."""
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            label = grid[i, j]
            ax.scatter(a, b, s=40, c=PHASE_COLORS.get(label, "gray"),
                       marker="o", edgecolors="k", linewidths=0.3)


def add_phase_legend(ax):
    """Legend for the colored MC phase points."""
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markeredgecolor="k", markersize=7, label=label)
        for label, c in PHASE_COLORS.items()
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, title="MC phase")


def main():
    L = 200
    n_steps = 300_000
    warmup = 30_000
    sample_every = 200

    alphas = np.linspace(0.05, 0.95, 31)
    betas = np.linspace(0.05, 0.95, 31)

    grid = scan_phase_diagram(alphas, betas, L, n_steps, warmup, sample_every)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) full parameter space
    ax = axes[0]
    plot_mft_boundaries(ax)
    plot_mc_grid(ax, alphas, betas, grid)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    add_phase_legend(ax)
    ax.set_title("(a) Full parameter space")

    # (b) zoom on 0.2 < beta < 0.4 (as in paper Fig 2b)
    ax = axes[1]
    plot_mft_boundaries(ax)
    plot_mc_grid(ax, alphas, betas, grid)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.4)
    add_phase_legend(ax)
    ax.set_title(r"(b) Zoom: $0.2<\beta<0.4$")

    fig.suptitle("Two-channel ASEP phase diagram (MFT lines + MC points)")
    fig.tight_layout()
    fig.savefig("results/phase_diagram.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
