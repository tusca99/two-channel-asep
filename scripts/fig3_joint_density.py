"""
Reproduce Figure 3 of Pronina & Kolomeisky (2007): 3D plots of the joint
density distribution P(rho1, rho2).

For alpha = 0.9, L = 1000, sweep beta through the phases and plot the 3D
surface of P(rho1, rho2) (with contour projected on the rho1-rho2 plane).
Shows the HD/LD -> LD/LD -> LD -> MC transitions via the shape of P:
- SSB phases (HD/LD, LD/LD): P is bimodal, peaks off the diagonal rho1=rho2
- symmetric phases (LD, MC): P is unimodal, peak on the diagonal

Beta values follow the paper's Fig 3 caption.
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


def plot_3d(ax, H, xedges, yedges, alpha, beta, label):
    """3D surface plot of P(rho1, rho2) with contour projected below."""
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xc, yc)
    Z = H.T  # transpose so X=rho1, Y=rho2

    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none",
                    alpha=0.9, linewidth=0, antialiased=True)
    # Contour projected on the rho1-rho2 plane
    ax.contourf(X, Y, Z, zdir="z", offset=0, cmap="viridis", alpha=0.4)

    ax.set_xlabel(r"$\rho_1$", fontsize=8)
    ax.set_ylabel(r"$\rho_2$", fontsize=8)
    ax.set_zlabel(r"$P$", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, Z.max() * 1.1)
    ax.set_title(rf"$\beta={beta}$ [{label}]", fontsize=9)
    ax.view_init(elev=30, azim=-60)


def main():
    L = 1000
    n_steps = 2_000_000
    warmup = 200_000
    alpha = 0.9

    # (beta, phase label) from the paper's Fig 3 caption
    betas = [
        (0.23, "HD/LD"),
        (0.245, "HD/LD"),
        (0.255, "HD/LD"),
        (0.258, "HD/LD"),
        (0.2595, "HD/LD+LD/LD"),
        (0.262, "LD/LD"),
        (0.2685, "LD/LD+LD"),
        (0.28, "LD"),
        (0.95, "MC"),
    ]

    # Collect all samples first (with progress), save for re-plotting
    all_samples = {}
    for beta, label in tqdm(betas, desc="beta sweep"):
        all_samples[beta] = collect_joint_samples(alpha, beta, L, n_steps, warmup)
    np.savez("results/fig3_samples.npz", **{f"b{beta}": s for beta, s in all_samples.items()})

    fig = plt.figure(figsize=(15, 12))
    for i, (beta, label) in enumerate(betas):
        samples = all_samples[beta]
        H, xedges, yedges = joint_histogram(samples)
        ax = fig.add_subplot(3, 3, i + 1, projection="3d")
        plot_3d(ax, H, xedges, yedges, alpha, beta, label)

    fig.suptitle(rf"$P(\rho_1,\rho_2)$ for $\alpha={alpha}$, $L={L}$ "
                 r"(Pronina & Kolomeisky 2007, Fig 3)")
    fig.tight_layout()
    fig.savefig("results/fig3_joint_density_3d.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
