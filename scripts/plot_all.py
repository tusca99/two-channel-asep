"""
Plot all figures from the saved GPU data (results/gpu/*.npz).

This script ONLY reads data produced by run_all_gpu.py — it never runs MC.
It reproduces:
  fig2  phase diagram (full + zoom)
  fig3  P(rho1,rho2) snapshots + 2D/3D animations
  fig4  current derivative & heatmap
  fig5  phase boundary vs L (original + fast)
  fig6  currents/densities vs beta
  mf_mc LD/MC boundary deviation
  ssb   order parameter vs beta, finite-size, joint density

Usage:
  python scripts/plot_all.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(ROOT, "results", "gpu")
if "--out" in sys.argv:
    _OUT = sys.argv[sys.argv.index("--out") + 1]
OUT = os.path.abspath(_OUT)
ALPHA = 0.9
L = 1000


def _load(nm):
    return np.load(os.path.join(OUT, nm), allow_pickle=True)


# ---- fig2: phase diagram -------------------------------------------------

def plot_fig2():
    import matplotlib.pyplot as plt
    from scripts.phase_diagram import plot_mft_boundaries, plot_mc_grid, \
        add_phase_legend  # noqa: F401

    d = os.path.join(OUT, "fig2")
    alphas = np.load(f"{d}/alphas_full.npy")
    betas = np.load(f"{d}/betas_full.npy")
    grid = np.load(f"{d}/grid_full.npy", allow_pickle=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_mft_boundaries(ax)
    plot_mc_grid(ax, alphas, betas, grid)
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$\beta$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    add_phase_legend(ax)
    ax.set_title("(a) Full parameter space")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2/phase_diagram_full.png", dpi=150)
    plt.close(fig)

    za = np.load(f"{d}/alphas_zoom.npy")
    zb = np.load(f"{d}/betas_zoom.npy")
    zgrid = np.load(f"{d}/grid_zoom.npy", allow_pickle=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_mft_boundaries(ax)
    plot_mc_grid(ax, za, zb, zgrid)
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$\beta$")
    ax.set_xlim(0, 1); ax.set_ylim(0.2, 0.4)
    add_phase_legend(ax)
    ax.set_title(r"(b) Zoom: $0.2<\beta<0.4$")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2/phase_diagram_zoom.png", dpi=150)
    plt.close(fig)
    print("fig2 plotted")


def plot_fig6():
    """fig6: currents (alpha=0.1) and densities (alpha=0.8,0.9) vs beta."""
    import matplotlib.pyplot as plt
    from asep.theory import mft_currents, mft_dense_dilute

    for alpha in [0.1, 0.8, 0.9]:
        d = np.load(f"{OUT}/fig6/alpha{alpha}.npz", allow_pickle=True)
        betas = d["betas"]
        if alpha < 0.5:
            # currents
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.errorbar(betas, d["J1"], yerr=d["eJ1"], fmt="o-", ms=4,
                        capsize=2, label=r"$J_1$ (MC)")
            ax.errorbar(betas, d["J2"], yerr=d["eJ2"], fmt="s-", ms=4,
                        capsize=2, label=r"$J_2$ (MC)")
            mft1 = [mft_currents(alpha, b)[0] for b in betas]
            mft2 = [mft_currents(alpha, b)[1] for b in betas]
            ax.plot(betas, mft1, "k--", lw=1.2, label=r"$J_1$ (MFT)")
            ax.plot(betas, mft2, "k:", lw=1.2, label=r"$J_2$ (MFT)")
            ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$J$")
            ax.set_title(rf"Currents, $\alpha={alpha}$")
            ymax = max(np.nanmax(d["J1"]), np.nanmax(d["J2"]),
                       np.nanmax(mft1), np.nanmax(mft2))
            ax.set_ylim(0, ymax * 1.15)
            ax.legend(fontsize=7, loc="lower right")
            fig.tight_layout()
            fig.savefig(f"{OUT}/fig6/currents_alpha{alpha}.png", dpi=150)
            plt.close(fig)
        else:
            # densities (dense/dilute)
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.errorbar(betas, d["dense"], yerr=d["edense"], fmt="o-", ms=4,
                        capsize=2, label=r"$\rho_{dense}$ (MC)")
            ax.errorbar(betas, d["dilute"], yerr=d["edilute"], fmt="s-", ms=4,
                        capsize=2, label=r"$\rho_{dilute}$ (MC)")
            mft1 = [mft_dense_dilute(alpha, b)[0] for b in betas]
            mft2 = [mft_dense_dilute(alpha, b)[1] for b in betas]
            ax.plot(betas, mft1, "k--", lw=1.2, label=r"$\rho_{dense}$ (MFT)")
            ax.plot(betas, mft2, "k:", lw=1.2, label=r"$\rho_{dilute}$ (MFT)")
            ax.axhline(0.5, color="r", ls=":", lw=1, label="MC $\\rho=1/2$")
            ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$\rho$")
            ax.set_title(rf"Bulk densities, $\alpha={alpha}$")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.legend(fontsize=7, loc="lower right")
            fig.tight_layout()
            fig.savefig(f"{OUT}/fig6/densities_alpha{alpha}.png", dpi=150)
            plt.close(fig)
    print("fig6 plotted")


def plot_ssb():
    """SSB order parameter (diff & std) vs beta, from long-run data."""
    betas = [0.05, 0.08, 0.10, 0.12, 0.15, 0.2, 0.25, 0.3]
    diff = []
    std = []
    for b in betas:
        d = np.load(f"{OUT}/ssb_beta{b}.npz", allow_pickle=True)
        dense = d["dense"]; dil = d["dilute"]
        diff.append(np.mean(dense[:, -1] - dil[:, -1]))
        std.append(np.mean(d["std"][:, -1]))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(betas, diff, "o-", ms=5,
            label=r"$\langle\rho_{dense}-\rho_{dilute}\rangle$")
    ax.plot(betas, std, "s--", ms=5, label=r"std$(\rho_1-\rho_2)$")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("SSB order parameter")
    ax.set_title(rf"SSB vs $\beta$, L={L}, $\alpha={ALPHA}$ (long runs)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/ssb_order_vs_beta.png", dpi=150)
    plt.close(fig)
    print("ssb plotted")


def plot_fig3():
    """fig3: P(rho1,rho2) snapshots + animations from fig3_points.npz."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import fig3_plot as f3
    f3.OUT = OUT  # point fig3_plot at the same output dir
    pts = f3.load_points("fig3_points.npz")
    f3.snapshots(pts)
    f3.animation_2d(pts)
    f3.animation_3d(pts)
    print("fig3 plotted")


def main():
    plot_fig2()
    plot_fig6()
    plot_ssb()
    plot_fig3()
    print("ALL FIGURES PLOTTED")


if __name__ == "__main__":
    main()
