"""Correct SSB order parameter vs beta (fixes the mislabeled orange curve).

Two curves, both computed at the ENSEMBLE level from per-replica (rho1, rho2)
time-averages (fig3_points.npz, 512 replicas x 49 betas, L=1000, alpha=0.9):

  * <|rho1 - rho2|>          : broken-symmetry magnitude ("magnetization")
  * std(rho1 - rho2) ens-wide: split between the two basins (+,-) and (-,+);
                               -> 0 in a symmetric phase, O(1) when broken.

The previously plotted orange curve was the per-replica *time* std (~0.125
flat), which is a temperature-like fluctuation, not the SSB diagnostic.

Usage: python scripts/plot_ssb_order.py
Writes: results/L1000/ssb_order_vs_beta_proj.png (and .png without _proj)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "L1000", "fig3_points.npz")
OUT = os.path.join(ROOT, "results", "L1000")


def main():
    d = np.load(SRC)
    betas = d["betas"]
    m_abs, s_ens, e_abs = [], [], []
    for b in betas:
        r = d[f"b{int(round(b * 10000))}"]
        gap = r[:, 0] - r[:, 1]
        m_abs.append(np.abs(gap).mean())
        e_abs.append(np.abs(gap).std() / np.sqrt(gap.size))
        s_ens.append(gap.std())

    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.errorbar(betas, m_abs, yerr=e_abs, color="tab:blue", marker="o", ms=4,
                lw=1.6, capsize=2, label=r"$\langle|\rho_1-\rho_2|\rangle$")
    ax.plot(betas, s_ens, color="tab:red", ls="--", marker="s", ms=4, lw=1.4,
            label=r"$\mathrm{std}_{\rm ens}(\rho_1-\rho_2)$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("SSB order parameter")
    ax.set_title(r"SSB order parameter, $L=1000$, $\alpha=0.9$ (512 replicas)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for name in ("ssb_order_vs_beta_proj.png", "ssb_order_vs_beta_new.png"):
        fig.savefig(os.path.join(OUT, name), dpi=200)
        print(f"saved {os.path.join(OUT, name)}")
    # numeric summary for the slide text
    for b, m, s in zip(betas, m_abs, s_ens):
        print(f"b={b:.3f}  <|gap|>={m:.3f}  std_ens={s:.3f}")


if __name__ == "__main__":
    main()