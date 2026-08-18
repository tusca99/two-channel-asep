"""
MFT vs MC current comparison from existing L200 fig6 data (no re-run).

Shows J1_MC vs J1_MFT (and J2) vs beta for alpha in {0.1, 0.8, 0.9},
highlighting the MFT-vs-MC deviation (the paper's central message).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "L200")
os.makedirs(OUT, exist_ok=True)

from asep.theory import mft_currents

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, alpha in zip(axes, [0.1, 0.8, 0.9]):
    d = np.load(f"{OUT}/fig6/alpha{alpha}.npz")
    betas = d["betas"]
    J1, J2 = d["J1"], d["J2"]
    mft = [mft_currents(alpha, b) for b in betas]
    m1 = np.array([m[0] for m in mft])
    m2 = np.array([m[1] for m in mft])
    ax.plot(betas, J1, "o-", ms=4, label=r"$J_1$ (MC)")
    ax.plot(betas, J2, "s-", ms=4, label=r"$J_2$ (MC)")
    ax.plot(betas, m1, "k--", lw=1.2, label=r"$J_1$ (MFT)")
    ax.plot(betas, m2, "k:", lw=1.2, label=r"$J_2$ (MFT)")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$J$")
    ax.set_title(rf"$\alpha={alpha}$")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1)
fig.suptitle("MFT vs MC currents (L=200)")
fig.tight_layout()
fig.savefig(f"{OUT}/mft_vs_mc_currents.png", dpi=150)
print("saved mft_vs_mc_currents.png", flush=True)

# deviation plot
fig2, ax2 = plt.subplots(figsize=(5, 4))
for alpha, c in zip([0.1, 0.8, 0.9], ["C0", "C1", "C2"]):
    d = np.load(f"{OUT}/fig6/alpha{alpha}.npz")
    betas = d["betas"]
    J1 = d["J1"]
    m1 = np.array([mft_currents(alpha, b)[0] for b in betas])
    dev = J1 - m1
    ax2.plot(betas, dev, "o-", ms=4, color=c, label=rf"$\alpha={alpha}$")
ax2.axhline(0, color="k", lw=0.8)
ax2.set_xlabel(r"$\beta$")
ax2.set_ylabel(r"$J_1^{MC} - J_1^{MFT}$")
ax2.set_title("MFT-vs-MC current deviation (L=200)")
ax2.legend(fontsize=8)
ax2.set_xlim(0, 1)
fig2.tight_layout()
fig2.savefig(f"{OUT}/mft_mc_deviation.png", dpi=150)
print("saved mft_mc_deviation.png", flush=True)
