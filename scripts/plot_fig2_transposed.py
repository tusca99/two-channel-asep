"""Transposed phase diagrams (beta on x, alpha on y) to match the paper's
Fig 2 axis convention -- for the "Ours vs original" comparison slide only.

Source: the SAME L=2000 high-budget (50k/site, 30-worker CPU) grids behind
results/L2000/fig2/*.png. The raw grid_*_recovered.npy files were digitized
back from those PNGs (the original .npy was lost); classification values are
clean discrete labels, so recovery is exact up to dot-pixel snapping (all
961 cells recovered, majority-voted).

Writes: results/L2000/fig2/phase_diagram_full_T.png (+ _zoom_T, + _proj 200dpi)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.phase_diagram import (mc_boundary, hdld_boundary, ldld_upper,
                                   PHASE_COLORS)

D = os.path.join(ROOT, "results", "L2000", "fig2")


def scatter_T(ax, alphas, betas, grid):
    """Grid was indexed [alpha_i, beta_j]; transposed axes: beta on x."""
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            label = grid[i, j]
            ax.scatter(b, a, s=40, c=PHASE_COLORS.get(label, "gray"),
                       marker="o", edgecolors="k", linewidths=0.3)


def mft_boundaries_T(ax):
    """Same three MFT lines, drawn with alpha as the vertical coordinate."""
    # MC/LD boundary (eq 10/13), beta > 1/2:  alpha vs beta
    b = np.linspace(0.5, 1.0, 200)
    ax.plot(b, mc_boundary(b), "k-", lw=1.5)
    # HD/LD (eq 23): beta = f(alpha)  -> alpha on y
    a = np.linspace(0.0, 1.0, 200)
    ax.plot(hdld_boundary(a), a, "k--", lw=1.5)
    # LD/LD (eq 33)
    ax.plot(ldld_upper(a), a, "k:", lw=1.5)
    ax.legend(["LD/MC (eq 10)", "HD/LD (eq 23)", "LD/LD (eq 33)"],
              fontsize=7, loc="upper right")


def main():
    a_full = np.linspace(0.05, 0.95, 31)
    b_full = np.linspace(0.05, 0.95, 31)
    g_full = np.load(f"{D}/grid_full_recovered.npy", allow_pickle=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    mft_boundaries_T(ax)
    scatter_T(ax, a_full, b_full, g_full)
    ax.set_xlabel(r"$\beta$")          # paper convention: beta on x
    ax.set_ylabel(r"$\alpha$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"{D}/phase_diagram_full_T.png", dpi=150)
    fig.savefig(f"{D}/phase_diagram_full_T_proj.png", dpi=200)
    plt.close(fig)

    az = np.linspace(0.05, 0.95, 31)
    bz = np.linspace(0.2, 0.4, 21)
    gz = np.load(f"{D}/grid_zoom_recovered.npy", allow_pickle=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    mft_boundaries_T(ax)
    scatter_T(ax, az, bz, gz)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_xlim(0.2, 0.4); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{D}/phase_diagram_zoom_T.png", dpi=150)
    fig.savefig(f"{D}/phase_diagram_zoom_T_proj.png", dpi=200)
    plt.close(fig)
    print("saved", f"{D}/phase_diagram_full_T_proj.png",
          f"{D}/phase_diagram_zoom_T_proj.png")


if __name__ == "__main__":
    main()