"""
Plot fig5 phase boundaries vs L from results/fig5_big/fig5_boundaries_merged.npz.

Panel (a): asym->LD beta boundary vs L, alpha=0.9, with MFT eq23 line.
Panel (b): LD->MC alpha boundary vs L, beta=1.0, with MFT eq10 line.
Log-x axis, merged across L=200..8000.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
npz = os.path.join(ROOT, "results", "fig5_big", "fig5_boundaries_merged.npz")
OUT = os.path.join(ROOT, "results", "fig5_big", "fig5_phase_boundary_vs_L.png")

d = np.load(npz)
Ls = np.array(d["Ls"])
asym = np.array(d["beta_asym"])
mcld = np.array(d["alpha_mcld"])

# sort by L for a clean monotone curve
idx = np.argsort(Ls)
Ls, asym, mcld = Ls[idx], asym[idx], mcld[idx]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(Ls, asym, "o-", ms=6, label="MC asym$\\to$LD boundary")
ax.axhline(d["theory_hdld"], color="k", ls=":", lw=1.2,
           label=f"MFT (eq23)={d['theory_hdld']:.3f}")
ax.set_xscale("log")
ax.set_xlabel(r"$L$"); ax.set_ylabel(r"$\beta$ boundary")
ax.set_title(rf"(a) asym$-$LD boundary, $\alpha=0.9$")
ax.set_ylim(0.2, 0.42)
ax.legend(fontsize=8)

ax = axes[1]
mask = ~np.isnan(mcld)
ax.plot(Ls[mask], mcld[mask], "s-", ms=6, label="MC LD/MC boundary")
ax.axhline(d["theory_mc"], color="k", ls=":", lw=1.2,
           label=f"MFT MC/LD (eq10)={d['theory_mc']:.3f}")
ax.set_xscale("log")
ax.set_xlabel(r"$L$"); ax.set_ylabel(r"$\alpha$ boundary")
ax.set_title(r"(b) LD/MC boundary, $\beta=1.0$")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved", OUT)
