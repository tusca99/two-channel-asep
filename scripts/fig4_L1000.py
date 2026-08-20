"""
Figure 4 for L=1000 with a proper run length: current J and dJ/dbeta vs beta.

Fixes the short-run issue: the original fig4 used n_steps=2M (only 2000 steps/
site at L=1000) which under-equilibrates. Here we use the L-scaled budget
(c=100: 100k steps/site at L=1000) so the current saturates cleanly and
dJ/dbeta locates the LD/MC boundary at the MFT value.

Runs on CPU (scan_points, ProcessPool). Logs to results/fig4_L1000.log and
saves params.json.

Usage: python scripts/fig4_L1000.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.params_record import write_params
from scripts.remake_L import TeeLog


def main():
    sys.stdout = TeeLog(os.path.join(ROOT, "results", "fig4_L1000.log"))
    from asep.parallel import scan_points
    L = 1000
    alpha = 0.9
    sps = 100 * L                 # 100k steps/site (L-scaled, c=100)
    n_steps = L * sps
    warmup = L * (sps // 5)       # 20k steps/site warmup
    sample_every = 400
    betas = np.linspace(0.05, 0.95, 40)

    print(f"fig4 L={L}: {n_steps} steps (100k steps/site), alpha={alpha}",
          flush=True)
    rng = np.random.default_rng(0)
    tasks = [(alpha, b, L, n_steps, warmup, sample_every,
              int(rng.integers(1e9))) for b in betas]
    res = scan_points(tasks, desc=f"alpha={alpha}")
    J = np.array([r[0] + r[1] for r in res])
    dJ = np.gradient(J, betas)

    out = os.path.join(ROOT, "results", "L1000")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "fig4_currents.npz"),
             betas=betas, J=J, dJ=dJ, alpha=alpha, L=L,
             n_steps=n_steps, warmup=warmup)
    write_params(os.path.join(out, "fig4"), L=L, alpha=alpha,
                 n_steps=n_steps, steps_per_site=sps, warmup=warmup,
                 sample_every=sample_every, betas=betas, backend="CPU",
                 paper="Pronina & Kolomeisky 2007, Fig 4")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(betas, J, "o-", ms=4)
    axes[0].set_xlabel(r"$\beta$"); axes[0].set_ylabel(r"$J_1+J_2$")
    axes[0].set_title(rf"Current, $\alpha={alpha}$, $L={L}$")
    axes[0].axhline(0.5, color="r", ls=":", lw=1, label="MC max")
    axes[1].plot(betas, dJ, "o-", ms=4, color="C1")
    axes[1].axhline(0, color="k", ls=":", lw=1)
    axes[1].set_xlabel(r"$\beta$"); axes[1].set_ylabel(r"$dJ/d\beta$")
    axes[1].set_title(rf"Current derivative, $\alpha={alpha}$, $L={L}$")
    fig.suptitle("Figure 4: LD/MC boundary via current saturation")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_current_derivative.png"), dpi=150)
    print(f"saved {out}/fig4_current_derivative.png", flush=True)
    print("fig4_L1000 DONE", flush=True)


if __name__ == "__main__":
    main()
