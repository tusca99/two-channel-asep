"""
Finite-size scaling of the spontaneous symmetry breaking (SSB) phases.

Reproduces the size-scaling analysis of Pronina & Kolomeisky (2007, Sec 3,
Fig 5): the asymmetric HD/LD phase is stable with L, while the LD/LD phase
shrinks with increasing L and likely vanishes in the thermodynamic limit.
"""
import numpy as np
import matplotlib.pyplot as plt

from scripts.ssb_analysis import collect_joint_samples, bimodality


def scan_L(alpha, beta, Ls, n_steps, warmup, seed=0):
    """Return SSB order parameter (mean |rho1-rho2|) vs L for fixed (alpha, beta)."""
    orders = []
    for L in Ls:
        samples = collect_joint_samples(alpha, beta, L, n_steps, warmup, seed)
        _, order = bimodality(samples)
        orders.append(order)
    return np.array(orders)


def main():
    # Moderate L range: at very large L the system flips between the two
    # broken states, washing out the time-averaged order parameter (the
    # paper's Fig 5 uses the density-distribution method to handle this).
    Ls = np.array([100, 200, 400, 800])
    n_steps = 4_000_000
    warmup = 100_000

    # HD/LD (stable SSB) vs LD/LD (finite-size) points
    cases = [
        (0.9, 0.1, "HD/LD (stable SSB)"),
        (0.3, 0.2, "LD/LD (finite-size)"),
    ]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for a, b, label in cases:
        orders = scan_L(a, b, Ls, n_steps, warmup)
        ax.plot(Ls, orders, "o-", label=rf"{label}, $\alpha={a}$, $\beta={b}$")
        print(f"{label}: |drho| = {np.round(orders, 3)}")

    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\langle |\rho_1-\rho_2| \rangle$")
    ax.set_title("Finite-size scaling of SSB order parameter")
    ax.legend(fontsize=8)
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig("results/ssb_finite_size.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
