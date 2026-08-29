"""
Figure 6: stationary-state properties vs beta with statistics and MFT lines.

Produces separate square plots (saved as PNG + data as .npy in results/fig6/):
  J1, J2 vs beta  (with MFT lines)   - currents, one row per alpha
  rho1, rho2 vs beta (with MFT lines) - bulk densities
  dJ/dbeta vs beta                    - current derivative

Data is cached in .npy so re-plotting does not re-run the MC.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from asep.theory import mft_currents, mft_dense_dilute
from asep.parallel import make_tasks, scan_points


OUT = "results/fig6"
os.makedirs(OUT, exist_ok=True)


def scan_beta_stats(alpha, betas, L, n_steps, warmup, sample_every, n_reps,
                    seed=0, use_gpu=True):
    """Mean +/- sem over beta for fixed alpha, across n_reps seeds.

    Also returns dense/dilute channel densities (max/min of rho1,rho2 per
    sample, then averaged) which are invariant to state-flipping.

    With use_gpu=True all (beta, rep) replicas run in a single GPU launch.
    """
    if use_gpu:
        from asep.parallel import _cuda_available, scan_beta_gpu
        if _cuda_available():
            return scan_beta_gpu(alpha, betas, L, n_steps, warmup, sample_every,
                                 n_reps, seed=seed)
    rng = np.random.default_rng(seed)
    tasks = [(alpha, b, L, n_steps, warmup, sample_every,
              int(rng.integers(1e9)))
             for b in betas for _ in range(n_reps)]
    res = scan_points(tasks, desc=f"alpha={alpha} ({n_reps} reps)",
                      adaptive=True)
    nb = len(betas)
    J1 = np.zeros(nb); J2 = np.zeros(nb); r1 = np.zeros(nb); r2 = np.zeros(nb)
    eJ1 = np.zeros(nb); eJ2 = np.zeros(nb); er1 = np.zeros(nb); er2 = np.zeros(nb)
    dense = np.zeros(nb); dilute = np.zeros(nb)
    edense = np.zeros(nb); edilute = np.zeros(nb)
    for j in range(nb):
        vals = res[j * n_reps:(j + 1) * n_reps]
        Jv = np.array([v[0] for v in vals]); J2v = np.array([v[1] for v in vals])
        r1v = np.array([v[2] for v in vals]); r2v = np.array([v[3] for v in vals])
        J1[j] = Jv.mean(); eJ1[j] = Jv.std() / np.sqrt(n_reps)
        J2[j] = J2v.mean(); eJ2[j] = J2v.std() / np.sqrt(n_reps)
        r1[j] = r1v.mean(); er1[j] = r1v.std() / np.sqrt(n_reps)
        r2[j] = r2v.mean(); er2[j] = r2v.std() / np.sqrt(n_reps)
        # per-seed dense/dilute (max/min of the two channels)
        d = np.maximum(r1v, r2v); dil = np.minimum(r1v, r2v)
        dense[j] = d.mean(); edense[j] = d.std() / np.sqrt(n_reps)
        dilute[j] = dil.mean(); edilute[j] = dil.std() / np.sqrt(n_reps)
    return (J1, J2, r1, r2, eJ1, eJ2, er1, er2, dense, dilute, edense, edilute)


def load_or_scan(alpha, betas, L, n_steps, warmup, sample_every, n_reps,
                 use_gpu=True):
    """Load cached .npy or scan; save result."""
    fname = f"{OUT}/alpha{alpha}.npz"
    if os.path.exists(fname):
        d = np.load(fname)
        return (d["J1"], d["J2"], d["rho1"], d["rho2"],
                d["eJ1"], d["eJ2"], d["erho1"], d["erho2"],
                d["dense"], d["dilute"], d["edense"], d["edilute"])
    res = scan_beta_stats(alpha, betas, L, n_steps, warmup, sample_every,
                          n_reps, use_gpu=use_gpu)
    np.savez(fname, J1=res[0], J2=res[1], rho1=res[2], rho2=res[3],
             eJ1=res[4], eJ2=res[5], erho1=res[6], erho2=res[7],
             dense=res[8], dilute=res[9], edense=res[10], edilute=res[11],
             betas=betas)
    return res


def plot_currents(alpha, betas, J1, J2, eJ1, eJ2, fname):
    """J1, J2 vs beta with MFT lines, square."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.errorbar(betas, J1, yerr=eJ1, fmt="o-", ms=4, capsize=2, label=r"$J_1$ (MC)")
    ax.errorbar(betas, J2, yerr=eJ2, fmt="s-", ms=4, capsize=2, label=r"$J_2$ (MC)")
    mft1 = [mft_currents(alpha, b)[0] for b in betas]
    mft2 = [mft_currents(alpha, b)[1] for b in betas]
    ax.plot(betas, mft1, "k--", lw=1.2, label=r"$J_1$ (MFT)")
    ax.plot(betas, mft2, "k:", lw=1.2, label=r"$J_2$ (MFT)")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$J$")
    ax.set_title(rf"Currents, $\alpha={alpha}$")
    ax.set_xlim(0, 1)
    # y-limit: max of MC and MFT currents, with headroom
    ymax = max(np.nanmax(J1), np.nanmax(J2), np.nanmax(mft1), np.nanmax(mft2))
    ax.set_ylim(0, ymax * 1.15)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


def plot_densities(alpha, betas, dense, dilute, edense, edilute, fname):
    """Dense/dilute channel densities vs beta with MFT lines, square.

    Uses per-sample max/min of (rho1,rho2) so the HD/LD phase is visible even
    when the system flips between broken states (which washes out the
    time-averaged rho1, rho2 separately).
    """
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.errorbar(betas, dense, yerr=edense, fmt="o-", ms=4, capsize=2,
                label=r"$\rho_{dense}$ (MC)")
    ax.errorbar(betas, dilute, yerr=edilute, fmt="s-", ms=4, capsize=2,
                label=r"$\rho_{dilute}$ (MC)")
    mft1 = [mft_dense_dilute(alpha, b)[0] for b in betas]
    mft2 = [mft_dense_dilute(alpha, b)[1] for b in betas]
    ax.plot(betas, mft1, "k--", lw=1.2, label=r"$\rho_{dense}$ (MFT)")
    ax.plot(betas, mft2, "k:", lw=1.2, label=r"$\rho_{dilute}$ (MFT)")
    ax.axhline(0.5, color="r", ls=":", lw=1, label="MC $\\rho=1/2$")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$\rho$")
    ax.set_title(rf"Bulk densities, $\alpha={alpha}$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


def main():
    L = 1000
    n_steps = 3_000_000
    warmup = 300_000
    sample_every = 400
    n_reps = 32
    betas = np.linspace(0.05, 0.95, 30)

    alphas = [0.1, 0.8, 0.9]
    for alpha in alphas:
        (J1, J2, r1, r2, eJ1, eJ2, er1, er2,
         dense, dilute, edense, edilute) = load_or_scan(
            alpha, betas, L, n_steps, warmup, sample_every, n_reps)
        if alpha < 0.5:
            plot_currents(alpha, betas, J1, J2, eJ1, eJ2,
                          f"{OUT}/currents_alpha{alpha}.png")
        else:
            plot_densities(alpha, betas, dense, dilute, edense, edilute,
                           f"{OUT}/densities_alpha{alpha}.png")

    print(f"Figures and data saved in {OUT}/")


if __name__ == "__main__":
    main()
