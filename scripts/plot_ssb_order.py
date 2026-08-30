"""Final SSB order-parameter figure for slide 34 (v2).

Two ensemble-level diagnostics vs beta, from per-replica steady-state gaps
g_i = <rho1>_i - <rho2>_i (results/ssb_rerun, 128 indep. replicas/beta,
L=1000, alpha=0.9, 120k steps/site):

  * broken moment   m(beta)    = <|g|>            (left axis, blue)
  * broken fraction f_br(beta)= frac |g| > 0.1   (right axis, red)

The first is the "magnetization" magnitude; the second shows the coexistence
wedge at the band edge and collapses to 0 in the symmetric phase. The
single-replica time-average, per-replica time std, and ensemble std(g) are
all degenerate or washed-out (see notes_ssb_discrepancy.md) and are dropped.

Usage: python scripts/plot_ssb_order.py [--tag ssb_rerun] [--chunks 6]
Writes: results/ssb_rerun/ssb_order_vs_beta_v2.png (+ _proj)
"""
import os
import sys
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESH = 0.1          # |g| above this = broken replica (basin width ~0.12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ssb_rerun")
    ap.add_argument("--chunks", type=int, default=6,
                    help="average over the last N chunks (time-average)")
    args = ap.parse_args()
    out = os.path.join(ROOT, "results", args.tag)

    files = sorted(glob.glob(os.path.join(out, "chunk*_beta*.npz"))
                   + glob.glob(os.path.join(out, "chunks", "chunk*_beta*.npz")),
                   key=lambda f: os.path.basename(f))
    by_beta = {}
    for f in files:
        b = int(os.path.basename(f).split("_beta")[-1][:-4]) / 10000.0
        by_beta.setdefault(b, []).append(f)

    betas, mom, mom_err, fbr = [], [], [], []
    for b in sorted(by_beta):
        take = by_beta[b][-args.chunks:]
        gaps = np.mean([np.load(f)["rho1"] - np.load(f)["rho2"]
                        for f in take], axis=0)   # avg last N chunks per replica
        g = np.abs(gaps)
        betas.append(b)
        mom.append(g.mean())
        mom_err.append(g.std() / np.sqrt(g.size))
        fbr.append((g > THRESH).mean())
    betas, mom, mom_err, fbr = map(np.array, (betas, mom, mom_err, fbr))

    band = (betas >= 0.24) & (betas <= 0.30)      # band-edge shade
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    ax.axvspan(betas[band].min(), betas[band].max(), color="0.85", lw=0,
               zorder=0)
    ax.errorbar(betas, mom, yerr=mom_err, color="tab:blue", marker="o",
                ms=3.5, lw=1.6, capsize=0,
                label=r"broken moment $\langle|\rho_1-\rho_2|\rangle$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\langle|\rho_1-\rho_2|\rangle$", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_ylim(-0.03, 1.0)
    ax.set_xlim(0.0, 0.36)
    ax2 = ax.twinx()
    ax2.plot(betas, fbr, "s--", ms=3.5, lw=1.4, color="tab:red",
             label=r"broken fraction $f_{\rm br}$")
    ax2.set_ylabel("fraction of broken replicas", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(-0.03, 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="center right")
    ax.text(0.272, 0.55, "SSB band\n(phase diagram)",
            fontsize=8, color="0.35", ha="left")
    ax.set_title(r"SSB order parameters, $L=1000$, $\alpha=0.9$ "
                 rf"({len(by_beta[betas[0]]) if False else 128} replicas)")
    fig.tight_layout()
    for name in (f"{args.tag}_ssb_order_v2.png", f"{args.tag}_ssb_order_v2_proj.png"):
        p = os.path.join(ROOT, "results", args.tag, name)
        fig.savefig(p, dpi=200)
        print("saved", p)

    # summary for slide text
    for b, m, f in zip(betas, mom, fbr):
        print(f"b={b:.3f}  m={m:.3f}  f_br={f:.2f}")


if __name__ == "__main__":
    main()