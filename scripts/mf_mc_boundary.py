"""
Phase 2.6: Quantify the deviation of the LD/MC transition position from MFT.

The paper's central message: unlike standard single-lane TASEP (where MFT
gives exact phase boundaries), in the two-channel model with narrow entrances
the MFT transition positions deviate from MC.

We measure the LD/MC boundary for fixed alpha and compare the located beta to
the MFT prediction beta = alpha/(4*alpha-2). Because the transition is smoothed
at finite L, we locate it via a fixed criterion (bulk density crossing rho*);
we also check the current plateau. By scanning L we test whether the deviation
is a finite-size effect or a genuine MF failure.
"""
import numpy as np
import matplotlib.pyplot as plt

from asep.parallel import make_tasks, scan_points, scan_points_gpu


def beta_mc_mft(alpha):
    """MFT LD/MC boundary (eq 10): alpha = 2*beta/(4*beta-1)."""
    return alpha / (4 * alpha - 2)


def locate_boundary_density(betas, rho_avg, rho_star=0.40):
    """First beta where mean bulk density crosses rho_star (MC onset).

    Note: at finite L and with limited equilibration the MC density does not
    reach 1/2; use a moderate threshold (default 0.40) to track the onset.
    """
    for j in range(len(betas)):
        if rho_avg[j] >= rho_star:
            return betas[j]
    return None


def _scan(tasks, n_reps=1, use_gpu=True):
    """CPU scan_points or GPU scan_points_gpu depending on availability.

    Returns a per-point averaged list (one element per task), averaging across
    the n_reps replicas that scan_points_gpu emits per task.
    """
    if use_gpu:
        from asep.parallel import _cuda_available
        if _cuda_available():
            res = scan_points_gpu(tasks, n_reps=n_reps, seed=0)
            nb = len(tasks)
            out = []
            for j in range(nb):
                vals = res[j * n_reps:(j + 1) * n_reps]
                out.append((np.mean([v[0] for v in vals]),
                            np.mean([v[1] for v in vals]),
                            np.mean([v[2] for v in vals]),
                            np.mean([v[3] for v in vals])))
            return out
    return scan_points(tasks, adaptive=True)


def main():
    alphas = np.array([0.8, 0.9, 1.0])
    Ls = np.array([200, 400, 800])
    betas = np.linspace(0.2, 0.95, 31)
    n_steps = 1_000_000
    warmup = 100_000
    sample_every = 400
    n_reps = 4

    # (a) density and current curves for one alpha, several L
    alpha = 0.9
    mft_beta = beta_mc_mft(alpha)
    print(f"alpha={alpha}: MFT LD/MC boundary beta = {mft_beta:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    for L in Ls:
        tasks = make_tasks([alpha], betas, L, n_steps, warmup, sample_every)
        res = _scan(tasks, n_reps=n_reps)
        Jtot = np.array([r[0] + r[1] for r in res])
        ax.plot(betas, Jtot, "o-", ms=4, label=rf"L={L}")
    ax.axvline(mft_beta, color="k", ls="--", lw=1.2, label="MFT boundary")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$J_1+J_2$")
    ax.set_title(rf"Total current, $\alpha={alpha}$"); ax.legend(fontsize=8)

    ax = axes[1]
    for L in Ls:
        tasks = make_tasks([alpha], betas, L, n_steps, warmup, sample_every)
        res = _scan(tasks, n_reps=n_reps)
        rho = np.array([(r[2] + r[3]) / 2 for r in res])
        ax.plot(betas, rho, "o-", ms=4, label=rf"L={L}")
    ax.axvline(mft_beta, color="k", ls="--", lw=1.2, label="MFT boundary")
    ax.axhline(0.5, color="r", ls=":", lw=1, label="MC rho=1/2")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$\bar\rho$")
    ax.set_title(rf"Mean bulk density, $\alpha={alpha}$"); ax.legend(fontsize=8)

    # (b) MC boundary position from density criterion vs L, for several alpha
    ax = axes[2]
    for alpha in alphas:
        mft = beta_mc_mft(alpha)
        mc_betas = []
        for L in Ls:
            tasks = make_tasks([alpha], betas, L, n_steps, warmup, sample_every)
            res = _scan(tasks, n_reps=n_reps)
            rho = np.array([(r[2] + r[3]) / 2 for r in res])
            mc = locate_boundary_density(betas, rho)
            mc_betas.append(mc)
            dev = f"{mc-mft:+.3f}" if mc is not None else "None"
            print(f"  alpha={alpha} L={L}: MC beta={mc}, dev={dev}")
        ax.plot(Ls, mc_betas, "o-", ms=5, label=rf"$\alpha={alpha}$ (MC)")
        ax.axhline(mft, color="k", ls="--", lw=1.0,
                   label=rf"$\alpha={alpha}$ (MFT)")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"LD/MC transition $\beta$")
    ax.set_title("LD/MC boundary vs L")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig("results/mf_mc_ldmc_boundary.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
