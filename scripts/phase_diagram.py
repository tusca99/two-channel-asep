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

def scan_phase_diagram(alphas, betas, L, n_steps, warmup, sample_every, seed=0,
                       use_gpu=True, n_reps=1):
    """
    Classify the phase at each (alpha, beta) grid point via MC.

    Uses the density-distribution method: collects joint (rho1, rho2) samples
    and detects symmetry breaking via std(rho1-rho2), which is robust to the
    state-flipping that washes out the time-averaged |rho1-rho2| at large L.

    With use_gpu=True each grid point is replicated n_reps times and all
    replicas run in a single GPU launch (~3x faster than CPU cores).
    Falls back to the CPU parallel scan if CUDA is unavailable.

    Returns a 2D array of phase labels (strings).
    """
    if use_gpu:
        from asep.parallel import _cuda_available, scan_phase_diagram_gpu
        if _cuda_available():
            return scan_phase_diagram_gpu(alphas, betas, L, n_steps, warmup,
                                          sample_every, n_reps=n_reps, seed=seed)
    from asep.parallel import make_tasks, scan_points_samples
    tasks = make_tasks(alphas, betas, L, n_steps, warmup, sample_every, seed)
    res = scan_points_samples(tasks, desc="phase grid")
    na, nb = len(alphas), len(betas)
    grid = np.empty((na, nb), dtype=object)
    k = 0
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            J1, J2, r1, r2, samples = res[k]
            grid[i, j] = classify_phase(J1, J2, r1, r2, a, b, L,
                                        samples=samples)[0]
            k += 1
    return grid


def plot_mc_grid(ax, alphas, betas, grid):
    """Scatter the MC phase classifications on `ax` (alpha on x, beta on y)."""
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            label = grid[i, j]
            ax.scatter(a, b, s=40, c=PHASE_COLORS.get(label, "gray"),
                       marker="o", edgecolors="k", linewidths=0.3)


def add_phase_legend(ax):
    """Legend for the colored MC phase points and the MFT boundary lines."""
    phase_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markeredgecolor="k", markersize=7, label=label)
        for label, c in PHASE_COLORS.items()
    ]
    line_handles = [
        plt.Line2D([0], [0], color="k", ls="-", lw=1.5, label="LD/MC (eq 10)"),
        plt.Line2D([0], [0], color="k", ls="--", lw=1.5, label="HD/LD (eq 23)"),
        plt.Line2D([0], [0], color="k", ls=":", lw=1.5, label="LD/LD (eq 33)"),
    ]
    ax.legend(handles=phase_handles + line_handles, loc="upper left",
              fontsize=7, title="MC phase / MFT lines")


def main():
    import os
    L = 1000
    n_steps = 2_000_000
    warmup = 200_000
    sample_every = 400
    n_reps = 16
    OUT = "results/fig2"
    os.makedirs(OUT, exist_ok=True)

    # (a) full parameter space: coarse grid
    alphas = np.linspace(0.05, 0.95, 31)
    betas = np.linspace(0.05, 0.95, 31)
    grid = scan_phase_diagram(alphas, betas, L, n_steps, warmup, sample_every,
                              n_reps=n_reps)
    np.save(f"{OUT}/grid_full.npy", grid, allow_pickle=True)
    np.save(f"{OUT}/alphas_full.npy", alphas)
    np.save(f"{OUT}/betas_full.npy", betas)

    # (b) zoom on 0.2 < beta < 0.4: finer grid to resolve the thin LD/LD band
    zoom_alphas = np.linspace(0.05, 0.95, 31)
    zoom_betas = np.linspace(0.2, 0.4, 21)
    zoom_grid = scan_phase_diagram(zoom_alphas, zoom_betas, L, n_steps, warmup,
                                   sample_every, seed=1, n_reps=n_reps)
    np.save(f"{OUT}/grid_zoom.npy", zoom_grid, allow_pickle=True)
    np.save(f"{OUT}/alphas_zoom.npy", zoom_alphas)
    np.save(f"{OUT}/betas_zoom.npy", zoom_betas)

    # (a) full parameter space - square
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_mft_boundaries(ax)
    plot_mc_grid(ax, alphas, betas, grid)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    add_phase_legend(ax)
    ax.set_title("(a) Full parameter space")
    fig.tight_layout()
    fig.savefig(f"{OUT}/phase_diagram_full.png", dpi=150)
    plt.close(fig)

    # (b) zoom on 0.2 < beta < 0.4 - square
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_mft_boundaries(ax)
    plot_mc_grid(ax, zoom_alphas, zoom_betas, zoom_grid)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.4)
    add_phase_legend(ax)
    ax.set_title(r"(b) Zoom: $0.2<\beta<0.4$")
    fig.tight_layout()
    fig.savefig(f"{OUT}/phase_diagram_zoom.png", dpi=150)
    plt.close(fig)

    print(f"Figures and data saved in {OUT}/")


if __name__ == "__main__":
    main()
