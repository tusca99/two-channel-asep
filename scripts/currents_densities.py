"""
Figure 6: stationary-state properties vs beta, with statistics, at L=1000.

Layout (one row per alpha, as the paper's Fig 6):
  alpha = 0.1 : currents J1, J2 vs beta  (+ dJ/dbeta panel)
  alpha = 0.8 : currents J1, J2 vs beta
  alpha = 0.9 : bulk densities rho1, rho2 vs beta
Each point is the mean +/- standard error over n_reps independent seeds.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from asep.parallel import make_tasks, scan_points


def scan_beta_stats(alpha, betas, L, n_steps, warmup, sample_every, n_reps,
                    seed=0):
    """Mean +/- sem over beta for fixed alpha, across n_reps seeds."""
    rng = np.random.default_rng(seed)
    tasks = [(alpha, b, L, n_steps, warmup, sample_every,
              int(rng.integers(1e9)))
             for b in betas for _ in range(n_reps)]
    res = scan_points(tasks, desc=f"alpha={alpha} ({n_reps} reps)")
    nb = len(betas)
    J1 = np.zeros(nb); J2 = np.zeros(nb); r1 = np.zeros(nb); r2 = np.zeros(nb)
    eJ1 = np.zeros(nb); eJ2 = np.zeros(nb); er1 = np.zeros(nb); er2 = np.zeros(nb)
    for j in range(nb):
        vals = res[j * n_reps:(j + 1) * n_reps]
        Jv = np.array([v[0] for v in vals]); J2v = np.array([v[1] for v in vals])
        r1v = np.array([v[2] for v in vals]); r2v = np.array([v[3] for v in vals])
        J1[j] = Jv.mean(); eJ1[j] = Jv.std() / np.sqrt(n_reps)
        J2[j] = J2v.mean(); eJ2[j] = J2v.std() / np.sqrt(n_reps)
        r1[j] = r1v.mean(); er1[j] = r1v.std() / np.sqrt(n_reps)
        r2[j] = r2v.mean(); er2[j] = r2v.std() / np.sqrt(n_reps)
    return J1, J2, r1, r2, eJ1, eJ2, er1, er2


def mft_currents(alpha, beta):
    """MFT single-lane LD current a1(1-a1)."""
    disc = (alpha + beta) ** 2 - 4 * alpha**2 * beta
    if disc < 0:
        return np.nan
    a1 = (alpha + beta - np.sqrt(disc)) / (2 * alpha)
    return a1 * (1 - a1)


def main():
    L = 1000
    n_steps = 3_000_000
    warmup = 300_000
    sample_every = 400
    n_reps = 8
    betas = np.linspace(0.05, 0.95, 30)

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.5, 1], hspace=0.4,
                          wspace=0.3)

    # alpha=0.1: currents (left) + dJ/dbeta (right)
    axJ = fig.add_subplot(gs[0, 0])
    axdJ = fig.add_subplot(gs[0, 1])
    J1, J2, r1, r2, eJ1, eJ2, er1, er2 = scan_beta_stats(
        0.1, betas, L, n_steps, warmup, sample_every, n_reps)
    axJ.errorbar(betas, J1, yerr=eJ1, fmt="o-", ms=3, capsize=2, label="$J_1$")
    axJ.errorbar(betas, J2, yerr=eJ2, fmt="s-", ms=3, capsize=2, label="$J_2$")
    mft = [mft_currents(0.1, b) for b in betas]
    axJ.plot(betas, mft, "k--", lw=1.2, label="MFT")
    axJ.set_ylabel(r"$J$"); axJ.set_title(r"Currents, $\alpha=0.1$")
    axJ.legend(fontsize=7)
    Jtot = J1 + J2
    axdJ.plot(betas, np.gradient(Jtot, betas), "o-", ms=3, color="C3")
    axdJ.axhline(0, color="k", ls=":", lw=1)
    axdJ.set_title(r"$dJ/d\beta$, $\alpha=0.1$")

    # alpha=0.8: currents (left) + dJ/dbeta (right)
    axJ = fig.add_subplot(gs[1, 0])
    axdJ = fig.add_subplot(gs[1, 1])
    J1, J2, r1, r2, eJ1, eJ2, er1, er2 = scan_beta_stats(
        0.8, betas, L, n_steps, warmup, sample_every, n_reps)
    axJ.errorbar(betas, J1, yerr=eJ1, fmt="o-", ms=3, capsize=2, label="$J_1$")
    axJ.errorbar(betas, J2, yerr=eJ2, fmt="s-", ms=3, capsize=2, label="$J_2$")
    mft = [mft_currents(0.8, b) for b in betas]
    axJ.plot(betas, mft, "k--", lw=1.2, label="MFT")
    axJ.set_ylabel(r"$J$"); axJ.set_title(r"Currents, $\alpha=0.8$")
    axJ.legend(fontsize=7)
    Jtot = J1 + J2
    axdJ.plot(betas, np.gradient(Jtot, betas), "o-", ms=3, color="C3")
    axdJ.axhline(0, color="k", ls=":", lw=1)
    axdJ.set_title(r"$dJ/d\beta$, $\alpha=0.8$")

    # alpha=0.9: bulk densities (span both columns)
    axr = fig.add_subplot(gs[2, :])
    J1, J2, r1, r2, eJ1, eJ2, er1, er2 = scan_beta_stats(
        0.9, betas, L, n_steps, warmup, sample_every, n_reps)
    axr.errorbar(betas, r1, yerr=er1, fmt="o-", ms=3, capsize=2,
                 label=r"$\rho_1$")
    axr.errorbar(betas, r2, yerr=er2, fmt="s-", ms=3, capsize=2,
                 label=r"$\rho_2$")
    axr.axhline(0.5, color="r", ls=":", lw=1, label="MC $\\rho=1/2$")
    axr.set_xlabel(r"$\beta$"); axr.set_ylabel(r"$\rho$")
    axr.set_title(r"Bulk densities, $\alpha=0.9$")
    axr.legend(fontsize=7)

    for ax in fig.axes:
        ax.set_xlabel(r"$\beta$")
        ax.set_xlim(0, 1)

    fig.suptitle(f"Figure 6: stationary properties (L={L}, {n_reps} seeds)")
    fig.savefig("results/currents_densities.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
