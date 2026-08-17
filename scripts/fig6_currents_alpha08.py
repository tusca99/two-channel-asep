"""One-off: currents figure for alpha=0.8 (paper Fig 6b) from saved fig6 data."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "gpu", "fig6")

import sys
sys.path.insert(0, os.path.dirname(HERE))
from asep.theory import mft_currents  # noqa: E402

d = np.load(os.path.join(OUT, "alpha0.8.npz"), allow_pickle=True)
betas = d["betas"]

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.errorbar(betas, d["J1"], yerr=d["eJ1"], fmt="o-", ms=4, capsize=2,
            label=r"$J_1$ (MC)")
ax.errorbar(betas, d["J2"], yerr=d["eJ2"], fmt="s-", ms=4, capsize=2,
            label=r"$J_2$ (MC)")
mft1 = [mft_currents(0.8, b)[0] for b in betas]
mft2 = [mft_currents(0.8, b)[1] for b in betas]
ax.plot(betas, mft1, "k--", lw=1.2, label=r"$J_1$ (MFT)")
ax.plot(betas, mft2, "k:", lw=1.2, label=r"$J_2$ (MFT)")
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$J$")
ax.set_title(rf"Currents, $\alpha=0.8$")
ymax = max(np.nanmax(d["J1"]), np.nanmax(d["J2"]),
           np.nanmax(mft1), np.nanmax(mft2))
ax.set_ylim(0, ymax * 1.15)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "currents_alpha0.8.png"), dpi=150)
print("wrote", os.path.join(OUT, "currents_alpha0.8.png"))
