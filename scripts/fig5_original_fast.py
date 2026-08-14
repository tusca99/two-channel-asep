"""
Reproduce the ORIGINAL Figure 5 of Pronina & Kolomeisky (2007) - FAST version.

The figure plots PHASE BOUNDARY POSITIONS vs L, NOT raw order/current curves:
(a) beta position of the phase boundaries (HD/LD <-> LD/LD <-> LD) vs L,
    for alpha = 0.9.
(b) alpha position of the symmetric LD <-> MC boundary vs L, for beta = 1.

Fast: single run per point (no stats), to validate the logic before the
long statistically-averaged run.
"""
import numpy as np
import matplotlib.pyplot as plt

from asep.parallel import make_tasks, scan_points


def scan_curves(alphas, betas, L, n_steps, warmup, sample_every, seed=0):
    """Scan (alphas x betas) grid, single run per point."""
    rng = np.random.default_rng(seed)
    tasks = [(a, b, L, n_steps, warmup, sample_every, int(rng.integers(1e9)))
             for a in alphas for b in betas]
    res = scan_points(tasks, desc=f"L={L}")
    na, nb = len(alphas), len(betas)
    J1 = np.zeros((na, nb)); J2 = np.zeros((na, nb))
    r1 = np.zeros((na, nb)); r2 = np.zeros((na, nb))
    k = 0
    for i in range(na):
        for j in range(nb):
            J1[i, j], J2[i, j], r1[i, j], r2[i, j] = res[k]
            k += 1
    return J1, J2, r1, r2


def find_beta_boundaries(order, betas, order_thresh=0.1):
    """
    Find the beta position(s) of the phase boundaries for fixed alpha,
    where the order parameter <|rho1-rho2|> crosses below order_thresh
    (asymmetric -> symmetric). The LAST crossing is the robust asymmetric->LD
    boundary (earlier crossings are noise from non-monotonic order at finite L).
    Returns list of boundary betas.
    """
    # Last index where order is still above threshold (asymmetric)
    last_asym = None
    for j in range(len(betas)):
        if order[j] > order_thresh:
            last_asym = j
    if last_asym is None:
        return []
    # boundary is where it drops after the last asymmetric point
    if last_asym < len(betas) - 1:
        return [betas[last_asym + 1]]
    return []


def find_alpha_mc_boundary(Jtot, alphas):
    """
    Find the alpha position of the LD <-> MC boundary for fixed beta:
    the alpha where Jtot saturates (dJ/dalpha -> 0, current reaches plateau).
    Returns alpha_boundary (None if not reached).
    """
    dJ = np.gradient(Jtot, alphas)
    # MC phase: J constant -> dJ ~ 0. Find first alpha where |dJ| small.
    for j in range(len(alphas)):
        if abs(dJ[j]) < 0.01:
            return alphas[j]
    return None


def main():
    Ls = np.array([200, 400, 800])
    n_steps = 5_000_000
    warmup = 500_000
    sample_every = 400

    # (a) beta-boundary (HD/LD <-> LD) vs L for alpha=0.9
    alpha = 0.9
    betas = np.linspace(0.02, 0.30, 20)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    beta_boundaries = []
    for L in Ls:
        J1, J2, r1, r2 = scan_curves([alpha], betas, L, n_steps, warmup,
                                     sample_every)
        order = np.abs(r1[0] - r2[0])
        bounds = find_beta_boundaries(order, betas)
        beta_boundaries.append(bounds)
        print(f"L={L}: beta boundaries at {bounds}")
    # plot boundary position vs L
    for bi in range(max(len(b) for b in beta_boundaries)):
        vals = [b[bi] if bi < len(b) else np.nan for b in beta_boundaries]
        ax.plot(Ls, vals, "o-", ms=6, label=f"boundary {bi+1}")
    ax.axhline(alpha / (1 + alpha + alpha**2), color="k", ls="--", lw=1.2,
               label="MFT (eq 23)")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\beta_{boundary}$")
    ax.set_title(rf"(a) Beta boundaries vs L, $\alpha={alpha}$")
    ax.set_xscale("log")
    ax.legend(fontsize=8)

    # (b) alpha LD<->MC boundary vs L for beta=1
    beta = 1.0
    alphas = np.linspace(0.2, 1.0, 20)
    ax = axes[1]
    alpha_boundaries = []
    for L in Ls:
        J1, J2, r1, r2 = scan_curves(alphas, [beta], L, n_steps, warmup,
                                     sample_every)
        Jtot = J1[:, 0] + J2[:, 0]
        ab = find_alpha_mc_boundary(Jtot, alphas)
        alpha_boundaries.append(ab)
        print(f"L={L}: alpha LD/MC boundary at {ab}")
    ax.plot(Ls, alpha_boundaries, "o-", ms=6, label="MC boundary")
    ax.axhline(2.0 / 3.0, color="k", ls="--", lw=1.2, label="MFT (eq 10)")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\alpha_{LD \leftrightarrow MC}$")
    ax.set_title(rf"(b) Alpha LD/MC boundary vs L, $\beta={beta}$")
    ax.set_xscale("log")
    ax.legend(fontsize=8)

    fig.suptitle("Figure 5 (original paper) - FAST version")
    fig.tight_layout()
    fig.savefig("results/fig5_original_fast.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
