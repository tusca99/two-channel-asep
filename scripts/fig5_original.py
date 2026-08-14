"""
Reproduce the ORIGINAL Figure 5 of Pronina & Kolomeisky (2007) - with stats.

Plots PHASE BOUNDARY POSITIONS vs L (as the paper):
(a) beta position of the HD/LD <-> LD boundary vs L, for alpha = 0.9.
(b) alpha position of the symmetric LD <-> MC boundary vs L, for beta = 1.
Each boundary position is a mean +/- std over n_runs independent seeds.
The solid line is the MFT prediction.

Boundary location: (a) via the order parameter <|rho1-rho2|> dropping below
threshold (last crossing); (b) via the current saturation (dJ/dalpha -> 0).
"""
import numpy as np
import matplotlib.pyplot as plt

from asep.parallel import make_tasks, scan_points


def scan_curves(alphas, betas, L, n_steps, warmup, sample_every, n_runs,
                seed=0):
    """Scan (alphas x betas) grid with n_runs stats. Returns (mean, err)."""
    rng = np.random.default_rng(seed)
    tasks = [(a, b, L, n_steps, warmup, sample_every, int(rng.integers(1e9)))
             for a in alphas for b in betas for _ in range(n_runs)]
    res = scan_points(tasks, desc=f"L={L} ({n_runs} runs)")
    na, nb = len(alphas), len(betas)
    J1m = np.zeros((na, nb)); J2m = np.zeros((na, nb))
    r1m = np.zeros((na, nb)); r2m = np.zeros((na, nb))
    J1e = np.zeros((na, nb)); J2e = np.zeros((na, nb))
    r1e = np.zeros((na, nb)); r2e = np.zeros((na, nb))
    k = 0
    for i in range(na):
        for j in range(nb):
            vals = res[k:k + n_runs]
            k += n_runs
            J1 = np.array([v[0] for v in vals]); J2 = np.array([v[1] for v in vals])
            r1 = np.array([v[2] for v in vals]); r2 = np.array([v[3] for v in vals])
            J1m[i, j] = J1.mean(); J1e[i, j] = J1.std()
            J2m[i, j] = J2.mean(); J2e[i, j] = J2.std()
            r1m[i, j] = r1.mean(); r1e[i, j] = r1.std()
            r2m[i, j] = r2.mean(); r2e[i, j] = r2.std()
    return (J1m, J2m, r1m, r2m), (J1e, J2e, r1e, r2e)


def find_beta_boundary(order, betas, order_thresh=0.1):
    """Last beta where order > thresh; boundary is the next beta. None if never asym."""
    last_asym = None
    for j in range(len(betas)):
        if order[j] > order_thresh:
            last_asym = j
    if last_asym is None or last_asym >= len(betas) - 1:
        return None
    return betas[last_asym + 1]


def find_alpha_mc_boundary(Jtot, alphas):
    """Alpha where Jtot saturates (dJ/dalpha -> 0). None if not reached."""
    dJ = np.gradient(Jtot, alphas)
    for j in range(len(alphas)):
        if abs(dJ[j]) < 0.01:
            return alphas[j]
    return None


def main():
    Ls = np.array([200, 400, 800])
    n_steps = 5_000_000
    warmup = 500_000
    sample_every = 400
    n_runs = 10

    # (a) beta HD/LD<->LD boundary vs L for alpha=0.9
    alpha = 0.9
    betas = np.linspace(0.02, 0.30, 20)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    beta_b = []
    beta_b_err = []
    for L in Ls:
        (J1, J2, r1, r2), (e1, e2, er1, er2) = scan_curves(
            [alpha], betas, L, n_steps, warmup, sample_every, n_runs)
        order = np.abs(r1[0] - r2[0])
        b = find_beta_boundary(order, betas)
        # boundary uncertainty: how much the order fluctuates around threshold
        err = np.mean(er1[0] + er2[0]) if b is not None else np.nan
        beta_b.append(b); beta_b_err.append(err)
        print(f"(a) L={L}: beta boundary={b}")
    beta_b = np.array(beta_b); beta_b_err = np.array(beta_b_err)
    ax.errorbar(Ls, beta_b, yerr=beta_b_err, fmt="o-", ms=6, capsize=4,
                label="MC boundary")
    ax.axhline(alpha / (1 + alpha + alpha**2), color="k", ls="--", lw=1.2,
               label="MFT (eq 23)")
    ax.set_xlabel(r"$L$"); ax.set_ylabel(r"$\beta_{HD/LD \leftrightarrow LD}$")
    ax.set_title(rf"(a) Beta boundary vs L, $\alpha={alpha}$")
    ax.set_xscale("log"); ax.legend(fontsize=8)

    # (b) alpha LD<->MC boundary vs L for beta=1
    beta = 1.0
    alphas = np.linspace(0.2, 1.0, 20)
    ax = axes[1]
    alpha_b = []
    alpha_b_err = []
    for L in Ls:
        (J1, J2, r1, r2), (e1, e2, er1, er2) = scan_curves(
            alphas, [beta], L, n_steps, warmup, sample_every, n_runs)
        Jtot = J1[:, 0] + J2[:, 0]
        ab = find_alpha_mc_boundary(Jtot, alphas)
        err = np.mean(e1[:, 0] + e2[:, 0]) if ab is not None else np.nan
        alpha_b.append(ab); alpha_b_err.append(err)
        print(f"(b) L={L}: alpha boundary={ab}")
    alpha_b = np.array(alpha_b); alpha_b_err = np.array(alpha_b_err)
    ax.errorbar(Ls, alpha_b, yerr=alpha_b_err, fmt="o-", ms=6, capsize=4,
                label="MC boundary")
    ax.axhline(2.0 / 3.0, color="k", ls="--", lw=1.2, label="MFT (eq 10)")
    ax.set_xlabel(r"$L$"); ax.set_ylabel(r"$\alpha_{LD \leftrightarrow MC}$")
    ax.set_title(rf"(b) Alpha boundary vs L, $\beta={beta}$")
    ax.set_xscale("log"); ax.legend(fontsize=8)

    fig.suptitle(f"Figure 5 (original paper), {n_runs} runs/pt")
    fig.tight_layout()
    fig.savefig("results/fig5_original.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
