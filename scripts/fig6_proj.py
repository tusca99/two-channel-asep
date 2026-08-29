#!/usr/bin/env python3
"""fig6 `_proj` figures: high-DPI SIM-vs-MFT currents/densities per (L, alpha).

Replicates the "final presentation bozza" (ee144ea) style:
  - big fonts (ticks 15, labels 21, title 19) for projector legibility
  - densities @ alpha=0.9: shaded phase bands (HD/LD red, LD/LD orange, LD
    blue, MC green) with tags + SIM 0.23 / MFT 0.332 jump annotations
  - densities @ other alpha: plain, MFT dense/dilute + MC rho=1/2 line
  - currents: J1/J2 MC (symbols+line) vs MFT (dashed/dotted)
Legend placement is chosen to sit in empty regions (never over the data).

Usage:  python scripts/fig6_proj.py [--base results/L2000 results/L200 ...]
Writes  <base>/fig6/{densities,currents}_alpha<a>_proj.png
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from asep.theory import mft_currents, mft_dense_dilute  # noqa: E402

# measured SIM / MFT jump positions (band divider for alpha=0.9 densities)
BAND_EDGES = [
    (0.00, 0.225, "#d62728", "HD/LD"),
    (0.225, 0.315, "#ff7f0e", "LD/LD"),
    (0.315, 0.55, "#1f77b4", "LD"),
    (0.55, 1.00, "#2ca02c", "MC"),
]
SIM_JUMP = 0.23    # measured HD/LD jump, alpha=0.9
MFT_JUMP = 0.332   # MFT eq 23, alpha=0.9


def plot_densities(base, alpha):
    d = np.load(f"{base}/fig6/alpha{alpha}.npz", allow_pickle=True)
    betas = d["betas"]
    fig, ax = plt.subplots(figsize=(8.4, 6.9))

    if abs(alpha - 0.9) < 1e-9:
        for x0, x1, c, tag in BAND_EDGES:
            ax.axvspan(x0, x1, color=c, alpha=0.08, lw=0, zorder=0)
            ax.text((x0 + x1) / 2, 0.955, tag, ha="center", va="top",
                    fontsize=15, color=c,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=c, lw=1.2))
    ax.errorbar(betas, d["dense"], yerr=d["edense"], fmt="o-", ms=9, lw=2.6,
                capsize=3, color="#d62728", label=r"$\rho_{\rm dense}$ (MC)",
                zorder=5)
    ax.errorbar(betas, d["dilute"], yerr=d["edilute"], fmt="s-", ms=9, lw=2.6,
                capsize=3, color="#1f77b4", label=r"$\rho_{\rm dilute}$ (MC)",
                zorder=5)
    mft1 = [mft_dense_dilute(alpha, b)[0] for b in betas]
    mft2 = [mft_dense_dilute(alpha, b)[1] for b in betas]
    ax.plot(betas, mft1, "k--", lw=2.2, label=r"$\rho_{\rm dense}$ (MFT)")
    ax.plot(betas, mft2, "k:", lw=2.2, label=r"$\rho_{\rm dilute}$ (MFT)")
    ax.axhline(0.5, color="0.45", ls=":", lw=1.6, zorder=1)

    if abs(alpha - 0.9) < 1e-9:
        ax.annotate(rf"SIM {SIM_JUMP:.2f}", xy=(SIM_JUMP, 0.47),
                    xytext=(0.09, 0.70), fontsize=15, color="#d62728",
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=2))
        ax.annotate(rf"MFT {MFT_JUMP:.3f}", xy=(MFT_JUMP, 0.50),
                    xytext=(0.42, 0.79), fontsize=15, color="k",
                    arrowprops=dict(arrowstyle="->", lw=2))

    ax.set_xlabel(r"$\beta$", fontsize=21)
    ax.set_ylabel(r"$\rho$", fontsize=21)
    suffix = r" — SIM vs MFT" if abs(alpha - 0.9) < 1e-9 else ""
    try:
        L = int(np.load(f"{base}/fig6/alpha{alpha}.npz", allow_pickle=True)["L"])
    except Exception:
        L = int(base.rstrip("/").split("L")[-1])
    ax.set_title(rf"$\alpha={alpha}$, $L={L}$" + suffix, fontsize=19)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=15)
    # legend in empty space: densities -> lower right (free for every alpha)
    ax.legend(fontsize=13, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{base}/fig6/densities_alpha{alpha}_proj.png", dpi=200)
    plt.close(fig)


def plot_currents(base, alpha):
    d = np.load(f"{base}/fig6/alpha{alpha}.npz", allow_pickle=True)
    betas = d["betas"]
    try:
        L = int(d["L"])
    except Exception:
        L = int(base.rstrip("/").split("L")[-1])
    fig, ax = plt.subplots(figsize=(8.4, 6.9))
    ax.errorbar(betas, d["J1"], yerr=d["eJ1"], fmt="o-", ms=9, lw=2.6,
                capsize=3, color="#1f77b4", label=r"$J_1$ (MC)", zorder=5)
    ax.errorbar(betas, d["J2"], yerr=d["eJ2"], fmt="s-", ms=9, lw=2.6,
                capsize=3, color="#ff7f0e", label=r"$J_2$ (MC)", zorder=5)
    mft1 = [mft_currents(alpha, b)[0] for b in betas]
    mft2 = [mft_currents(alpha, b)[1] for b in betas]
    ax.plot(betas, mft1, "k--", lw=2.2, label=r"$J_1$ (MFT)")
    ax.plot(betas, mft2, "k:", lw=2.4, label=r"$J_2$ (MFT)")
    ax.set_xlabel(r"$\beta$", fontsize=21)
    ax.set_ylabel(r"$J$", fontsize=21)
    if alpha < 0.5:
        ax.set_title(rf"Currents, $\alpha={alpha}$, $L={L}$ --- SIM vs MFT (LD)",
                     fontsize=17)
    else:
        ax.set_title(rf"Currents, $\alpha={alpha}$", fontsize=19)
    ymax = max(np.nanmax(d["J1"]), np.nanmax(d["J2"]),
               np.nanmax(mft1), np.nanmax(mft2))
    ax.set_ylim(0, ymax * 1.15)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=15)
    # empty corner: J collapses by beta~0.6 -> legend lower right is always free
    ax.legend(fontsize=13, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{base}/fig6/currents_alpha{alpha}_proj.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+",
                    default=["results/L2000", "results/L500",
                             "results/L1000", "results/L200"])
    ap.add_argument("--alphas", nargs="+", default=[0.1, 0.8, 0.9])
    args = ap.parse_args()
    for base in args.base:
        if not os.path.isdir(f"{base}/fig6"):
            print("skip (no fig6):", base)
            continue
        L = int(base.split("L")[-1])
        for a in args.alphas:
            if not os.path.exists(f"{base}/fig6/alpha{a}.npz"):
                print("skip (no npz):", base, a)
                continue
            plot_currents(base, a)
            plot_densities(base, a)
        print("done:", base)


if __name__ == "__main__":
    main()