#!/usr/bin/env python3
"""fig5a `_proj` figure: phase boundaries vs L (paper Fig 5a, alpha=0.9).

Replicates the "final presentation bozza" (ee144ea) style with fixes:
  - scatter only (no connecting lines — the L-series is not a trend)
  - colors swapped: HD/LD->LD/LD boundary = ORANGE, LD/LD->LD = BLUE
  - grey shade for L >= 4000 (c=50 under-equilibration caveat) with note
  - legend OUTSIDE top-right so it never covers the data
  - k-notation ticks (200 500 1k 2k 4k 8k 10k 12k), no overlap

Usage:  python scripts/fig5a_proj.py
Writes  results/fig5_variant_left/fig5a_left_variant{,_linear,_log}.png (+ _proj)
from    results/fig5_variant_left/fig5a_variant.npz
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "results", "fig5_variant_left", "fig5a_variant.npz")
OUT = os.path.join(ROOT, "results", "fig5_variant_left")

# colors swapped per request: HD/LD boundary orange, LD/LD->LD blue
C_HDLD = "#ff7f0e"
C_LDL = "#1f77b4"


def lab(x):
    return f"{int(x)}" if x < 1000 else f"{int(x)//1000}k"


def main():
    d = np.load(NPZ)
    Ls, hdld, ldl = d["Ls"], d["hdld"], d["ldl"]
    e_hdld, e_ldl = d["e_hdld"], d["e_ldl"]
    th_hdld, th_ldld = float(d["theory_hdld"]), float(d["theory_ldld"])
    L_shade = 4000

    fig, ax = plt.subplots(figsize=(10.6, 6.9))
    # grey band: under-equilibration region (budget, not physics)
    ax.axvspan(L_shade, Ls.max() * 1.15, color="0.85", zorder=0)
    ax.text(0.985, 0.97, "c=50 under-equil.\nneed c≈100",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=13, color="0.35", style="italic", zorder=6)

    ax.axhline(th_hdld, color="k", ls="--", lw=2.4,
               label=f"MFT eq 23 ({th_hdld:.3f})", zorder=2)

    ax.errorbar(Ls, hdld, yerr=e_hdld, fmt="o", ms=11, capsize=5, lw=2.6,
                ls="none", color=C_HDLD,
                label=r"HD/LD $\to$ LD/LD", zorder=5)
    ax.errorbar(Ls, ldl, yerr=e_ldl, fmt="s", ms=11, capsize=5, lw=2.6,
                ls="none", color=C_LDL,
                label=r"LD/LD $\to$ LD", zorder=5)

    ax.set_xscale("log")
    ax.set_xticks(Ls)
    tick_labs = [lab(x) if x not in (10000,) else "" for x in Ls]
    ax.set_xticklabels(tick_labs, fontsize=14, rotation=0)
    ax.get_xaxis().set_minor_locator(plt.NullLocator())
    ax.set_xlabel(r"$L$", fontsize=21)
    ax.set_ylabel(r"$\beta$ boundary ($\alpha=0.9$)", fontsize=21)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_title("Boundaries vs $L$ — flat to $L=2000$", fontsize=19)
    # legend inside, bottom-left corner (empty region of the plot)
    ax.legend(fontsize=14, loc="lower left", frameon=True, framealpha=0.9)
    fig.tight_layout()

    fig.savefig(os.path.join(OUT, "fig5a_left_variant_linear.png"), dpi=150,
                bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig5a_left_variant_linear_proj.png"),
                dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig5a_left_variant_log.png"), dpi=150,
                bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig5a_left_variant_linear2_proj.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved fig5a variant (scatter, colors swapped, legend outside)")


if __name__ == "__main__":
    main()