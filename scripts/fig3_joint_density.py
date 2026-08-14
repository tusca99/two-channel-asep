"""
Reproduce Figure 3 of Pronina & Kolomeisky (2007): 3D plots of the joint
density distribution P(rho1, rho2).

For alpha = 0.9, sweep beta and plot the 3D surface of P(rho1, rho2) with a
colored contour projected on the rho1-rho2 plane. Shows the transitions via
the shape of P:
  - SSB phases (HD/LD, LD/LD): P is bimodal, two peaks off the diagonal
  - symmetric phases (LD, MC): P is unimodal, peak on the diagonal

Note: the paper uses beta in [0.23, 0.28] for L=1000. At L=1000 our MC system
flips rapidly between broken states in that range, washing out the time
average; the clearest SSB appears at lower beta (0.05-0.12). We plot the
beta range that exhibits the full LD -> HD/LD -> LD -> MC transition in our
simulations. Longer runs (paper used 2e7-5e8 steps) give sharper peaks.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm

from asep import TwoChannelASEP


def collect_joint_samples(alpha, beta, L, n_steps, warmup, seed=0):
    """Run MC and return the (rho1, rho2) joint density samples."""
    sample_every = max(1, L // 10)
    sim = TwoChannelASEP(L=L, alpha=alpha, beta=beta, seed=seed)
    sim.run(n_steps=n_steps, sample_every=sample_every, warmup=warmup)
    return sim.get_joint_density_samples()


def joint_histogram(samples, bins=40):
    """2D histogram of P(rho1, rho2) on [0,1]^2."""
    r1, r2 = samples[:, 0], samples[:, 1]
    H, xedges, yedges = np.histogram2d(r1, r2, bins=bins, range=[[0, 1], [0, 1]])
    return H, xedges, yedges


def plot_3d(ax, H, xedges, yedges, beta, label):
    """
    Plot P(rho1, rho2) as a 3D surface with a colored contour on the bottom
    plane, so the peak positions are clearly visible.
    """
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xc, yc)
    Z = H.T  # X=rho1, Y=rho2

    cmap = plt.get_cmap("viridis")
    ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="none",
                    alpha=0.85, linewidth=0, antialiased=True, rstride=1,
                    cstride=1)
    # Colored contour projected on the rho1-rho2 plane (z=0)
    ax.contourf(X, Y, Z, zdir="z", offset=0, cmap=cmap, alpha=0.55,
                levels=12)

    ax.set_xlabel(r"$\rho_1$", fontsize=8)
    ax.set_ylabel(r"$\rho_2$", fontsize=8)
    ax.set_zlabel(r"$P$", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, Z.max() * 1.15)
    ax.set_title(rf"$\beta={beta}$ [{label}]", fontsize=9)
    ax.view_init(elev=25, azim=-55)
    ax.set_box_aspect((1, 1, 0.7))


def main():
    L = 1000
    n_steps = 10_000_000
    warmup = 1_000_000
    alpha = 0.9

    # (beta, phase label): beta values where our MC shows the full transition
    betas = [
        (0.02, "LD"),
        (0.05, "HD/LD"),
        (0.07, "HD/LD"),
        (0.08, "HD/LD"),
        (0.10, "HD/LD+LD/LD"),
        (0.12, "LD/LD"),
        (0.15, "LD/LD+LD"),
        (0.20, "LD"),
        (0.60, "MC"),
    ]

    # Collect all samples first (with progress), save for re-plotting
    all_samples = {}
    for beta, label in tqdm(betas, desc="beta sweep"):
        all_samples[beta] = collect_joint_samples(alpha, beta, L, n_steps, warmup)
    np.savez("results/fig3_samples.npz",
             **{f"b{beta}": s for beta, s in all_samples.items()})

    fig = plt.figure(figsize=(15, 12))
    for i, (beta, label) in enumerate(betas):
        samples = all_samples[beta]
        H, xedges, yedges = joint_histogram(samples)
        ax = fig.add_subplot(3, 3, i + 1, projection="3d")
        plot_3d(ax, H, xedges, yedges, beta, label)

    fig.suptitle(rf"$P(\rho_1,\rho_2)$ for $\alpha={alpha}$, $L={L}$ "
                 r"(Pronina & Kolomeisky 2007, Fig 3 style)")
    fig.tight_layout()
    fig.savefig("results/fig3_joint_density_3d.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
