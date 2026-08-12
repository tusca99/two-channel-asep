"""
Reproduce Figure 6 of Pronina & Kolomeisky (2007): currents & densities.

Scans beta for fixed alpha and plots MC currents/densities against the
mean-field predictions (LD, MC, HD/LD phases).
"""
import numpy as np
import matplotlib.pyplot as plt

from asep import TwoChannelASEP


# --- Mean-field predictions ----------------------------------------------

def mft_ld(alpha, beta):
    """LD phase: effective entrance rate (eq 12), J = a1(1-a1), rho = a1."""
    disc = (alpha + beta) ** 2 - 4 * alpha**2 * beta
    if disc < 0:
        return np.nan, np.nan
    a1 = (alpha + beta - np.sqrt(disc)) / (2 * alpha)
    return a1 * (1 - a1), a1


def mft_mc(alpha, beta):
    """MC phase: J = 1/4, rho = 1/2."""
    return 0.25, 0.5


def mft_hdld(alpha, beta):
    """HD/LD: J1 = beta(1-beta), rho1 = 1-beta; J2 = a2(1-a2), rho2 = a2, a2 = alpha*beta."""
    a2 = alpha * beta
    J1, rho1 = beta * (1 - beta), 1 - beta
    J2, rho2 = a2 * (1 - a2), a2
    return (J1, J2), (rho1, rho2)


def mft_currents(alpha, beta):
    """Best MFT current prediction for (alpha, beta)."""
    # MC boundary (eq 10)
    if beta > 0.5 and alpha > 2 * beta / (4 * beta - 1):
        return mft_mc(alpha, beta)[0], mft_mc(alpha, beta)[0]
    # HD/LD boundary (eq 23)
    if beta < alpha / (1 + alpha + alpha**2):
        (J1, J2), _ = mft_hdld(alpha, beta)
        return J1, J2
    # LD
    J, _ = mft_ld(alpha, beta)
    return J, J


def mft_densities(alpha, beta):
    """Best MFT density prediction for (alpha, beta)."""
    if beta > 0.5 and alpha > 2 * beta / (4 * beta - 1):
        return mft_mc(alpha, beta)[1], mft_mc(alpha, beta)[1]
    if beta < alpha / (1 + alpha + alpha**2):
        _, (rho1, rho2) = mft_hdld(alpha, beta)
        return rho1, rho2
    _, rho = mft_ld(alpha, beta)
    return rho, rho


# --- MC scan --------------------------------------------------------------

def scan_beta(alpha, betas, L, n_steps, warmup, sample_every, seed=0):
    """Return (J1, J2, rho1, rho2) arrays over beta for fixed alpha."""
    rng = np.random.default_rng(seed)
    J1s, J2s, r1s, r2s = [], [], [], []
    for b in betas:
        sim = TwoChannelASEP(L=L, alpha=alpha, beta=b, seed=int(rng.integers(1e9)))
        sim.run(n_steps=n_steps, sample_every=sample_every, warmup=warmup)
        J1, J2 = sim.get_currents()
        r1, r2 = sim.get_bulk_densities()
        J1s.append(J1); J2s.append(J2); r1s.append(r1); r2s.append(r2)
    return np.array(J1s), np.array(J2s), np.array(r1s), np.array(r2s)


def plot_currents(ax, alpha, betas, J1, J2):
    """Currents vs beta for fixed alpha, with MFT lines."""
    mft1 = [mft_currents(alpha, b)[0] for b in betas]
    mft2 = [mft_currents(alpha, b)[1] for b in betas]
    ax.plot(betas, mft1, "k-", lw=1.2, label="MFT ch1")
    ax.plot(betas, mft2, "k--", lw=1.2, label="MFT ch2")
    ax.plot(betas, J1, "o", ms=4, label="MC ch1")
    ax.plot(betas, J2, "s", ms=4, label="MC ch2")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$J$")
    ax.set_title(rf"Currents, $\alpha={alpha}$")


def plot_densities(ax, alpha, betas, r1, r2):
    """Bulk densities vs beta for fixed alpha, with MFT lines."""
    mft1 = [mft_densities(alpha, b)[0] for b in betas]
    mft2 = [mft_densities(alpha, b)[1] for b in betas]
    ax.plot(betas, mft1, "k-", lw=1.2, label="MFT ch1")
    ax.plot(betas, mft2, "k--", lw=1.2, label="MFT ch2")
    ax.plot(betas, r1, "o", ms=4, label="MC ch1")
    ax.plot(betas, r2, "s", ms=4, label="MC ch2")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title(rf"Bulk densities, $\alpha={alpha}$")


def plot_dJdbeta(ax, alpha, betas, J1, J2):
    """
    Derivative of the total current w.r.t. beta vs beta.

    Reproduces the method of Pronina & Kolomeisky (2007, Sec 3, Fig 4): the
    LD/MC boundary is located where dJ/dbeta reaches zero (or fluctuates
    around zero), since in the MC phase the current is constant at its
    maximum value.
    """
    Jtot = J1 + J2
    dJ = np.gradient(Jtot, betas)
    ax.plot(betas, dJ, "o-", ms=4, label=r"$dJ/d\beta$")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$dJ/d\beta$")
    ax.set_title(rf"Current derivative, $\alpha={alpha}$")


def main():
    L = 200
    n_steps = 300_000
    warmup = 30_000
    sample_every = 200
    betas = np.linspace(0.05, 0.95, 19)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, alpha in zip(axes[0], [0.1, 0.8]):
        J1, J2, r1, r2 = scan_beta(alpha, betas, L, n_steps, warmup, sample_every)
        if alpha < 0.5:
            plot_currents(ax, alpha, betas, J1, J2)
        else:
            plot_densities(ax, alpha, betas, r1, r2)
        ax.legend(fontsize=7)

    # dJ/dbeta for alpha=0.8 (crosses into MC) and alpha=0.9
    for ax, alpha in zip(axes[1], [0.8, 0.9]):
        J1, J2, _, _ = scan_beta(alpha, betas, L, n_steps, warmup, sample_every)
        plot_dJdbeta(ax, alpha, betas, J1, J2)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig("results/currents_densities.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
