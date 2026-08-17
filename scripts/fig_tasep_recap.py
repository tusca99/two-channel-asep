"""Generate the single-lane TASEP recap figures for the presentation.

Outputs (into results/gpu/figures/):
  tasep_phases.png     LD/HD/MC regions in the (alpha, beta) plane
  tasep_current.png    steady-state current J vs beta for fixed alphas
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "gpu", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 12})


def j_alpha_beta(a, b):
    """Single-lane TASEP current, exact in MFT."""
    if a < 1 / 2 and a < b:
        return a * (1 - a)
    if b < 1 / 2 and b < a:
        return b * (1 - b)
    return 1 / 4


def rho_alpha_beta(a, b):
    """Single-lane TASEP bulk density."""
    if a < 1 / 2 and a < b:
        return a
    if b < 1 / 2 and b < a:
        return 1 - b
    return 1 / 2


# ---- Phase diagram in (alpha, beta) ----
fig, ax = plt.subplots(figsize=(6, 5))
a = np.linspace(0, 1, 400)
b = np.linspace(0, 1, 400)
A, B = np.meshgrid(a, b)
J = np.vectorize(j_alpha_beta)(A, B)
ph = np.zeros_like(J, dtype=int)
ph[(A < 1 / 2) & (A < B)] = 1          # LD
ph[(B < 1 / 2) & (B < A)] = 2          # HD
ph[(A > 1 / 2) & (B > 1 / 2)] = 3      # MC
cmap = matplotlib.colors.ListedColormap(["white", "#aad4f5", "#f5d0a8", "#c8e6c9"])
ax.pcolormesh(A, B, ph, cmap=cmap, shading="auto", alpha=0.9)

# Exact phase-boundary segments (the three meeting at (1/2,1/2)):
ax.plot([0, 1 / 2], [0, 1 / 2], "k-", lw=1.5)          # LD | HD
ax.plot([1 / 2, 1], [1 / 2, 1 / 2], "k-", lw=1.5)      # HD | MC
ax.plot([1 / 2, 1 / 2], [1 / 2, 1], "k-", lw=1.5)      # LD | MC

ax.text(0.25, 0.78, "LD", ha="center", va="center", fontsize=14)
ax.text(0.78, 0.22, "HD", ha="center", va="center", fontsize=14)
ax.text(0.78, 0.78, "MC", ha="center", va="center", fontsize=14)
ax.text(0.62, 0.60, "J = 1/4", ha="center", fontsize=9)
ax.text(0.25, 0.67, r"$J=\alpha(1-\alpha)$", ha="center", fontsize=9)
ax.text(0.75, 0.08, r"$J=\beta(1-\beta)$", ha="center", fontsize=9)

ax.set_xlabel(r"entrance rate $\alpha$")
ax.set_ylabel(r"exit rate $\beta$")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Single-lane TASEP: mean-field phases")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tasep_phases.png"), dpi=150)
plt.close(fig)

# ---- Current and density vs beta for fixed alphas ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
bvec = np.linspace(0.001, 1, 300)
for al in (0.3, 0.7, 0.9):
    Js = [j_alpha_beta(al, bb) for bb in bvec]
    rs = [rho_alpha_beta(al, bb) for bb in bvec]
    ax1.plot(bvec, Js, lw=2, label=rf"$\alpha={al}$")
    ax2.plot(bvec, rs, lw=2, label=rf"$\alpha={al}$")
ax1.axhline(1 / 4, color="0.6", ls=":", lw=1)
ax1.set_xlabel(r"exit rate $\beta$")
ax1.set_ylabel(r"current $J$")
ax1.legend()
ax2.set_xlabel(r"exit rate $\beta$")
ax2.set_ylabel(r"bulk density $\rho$")
ax2.legend()
fig.suptitle("Single-lane TASEP steady state (MFT is exact)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tasep_current.png"), dpi=150)
plt.close(fig)

print("wrote", OUT)
