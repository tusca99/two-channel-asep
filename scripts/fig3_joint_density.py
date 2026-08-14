"""
Reproduce Figure 3 of Pronina & Kolomeisky (2007): 3D plots of P(rho1, rho2).

Key fix: at L=1000 a single MC run gets STUCK in one broken symmetry-broken
state (one sharp peak in P). The paper's P shows TWO peaks because their
sampling visits both broken states. We reconstruct the two-peak structure by
averaging the joint density over an ENSEMBLE of seeds (each lands in a random
broken state), which is statistically equivalent to the paper's long-run
sampling.

Outputs:
- Snapshots at the paper's beta values (0.23..0.95), alpha=0.9, L=1000
- A 3D animation sweeping beta finely, showing peaks emerge/split
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm

from asep import TwoChannelASEP


def collect_ensemble(alpha, beta, L, n_steps, warmup, n_seeds=8, seed0=0):
    """Concatenate joint-density samples over an ensemble of seeds."""
    all_s = []
    for k in range(n_seeds):
        sim = TwoChannelASEP(L=L, alpha=alpha, beta=beta, seed=seed0 + k)
        sim.run(n_steps=n_steps, sample_every=max(1, L // 10), warmup=warmup)
        all_s.append(sim.get_joint_density_samples())
    return np.concatenate(all_s)


def joint_histogram(samples, bins=48):
    H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1],
                                       bins=bins, range=[[0, 1], [0, 1]])
    return H, xedges, yedges


def plot_3d(ax, H, xedges, yedges, beta, label):
    """3D surface with colored contour on the bottom plane; good contrast."""
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xc, yc)
    Z = H.T

    # Normalize so the max peak is prominent (high contrast)
    Z = Z / Z.max()

    # A perceptually-uniform colormap; lower the surface opacity so peaks
    # are not hidden, keep contour opaque for peak locations
    ax.plot_surface(X, Y, Z, cmap="turbo", edgecolor="none",
                    alpha=0.95, linewidth=0, antialiased=True,
                    rstride=1, cstride=1, shade=True)
    ax.contourf(X, Y, Z, zdir="z", offset=0, cmap="turbo", alpha=0.9,
                levels=14)

    ax.set_xlabel(r"$\rho_1$", fontsize=8)
    ax.set_ylabel(r"$\rho_2$", fontsize=8)
    ax.set_zlabel(r"$P/P_{max}$", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1.05)
    ax.set_title(rf"$\beta={beta}$ [{label}]", fontsize=9)
    ax.view_init(elev=25, azim=-55)
    ax.set_box_aspect((1, 1, 0.6))


def snapshots(alpha, L, n_steps, warmup, n_seeds):
    # Paper's exact beta values
    betas = [
        (0.23, "HD/LD"), (0.245, "HD/LD"), (0.255, "HD/LD"),
        (0.258, "HD/LD"), (0.2595, "HD/LD+LD/LD"), (0.262, "LD/LD"),
        (0.2685, "LD/LD+LD"), (0.28, "LD"), (0.95, "MC"),
    ]
    fig = plt.figure(figsize=(15, 12))
    for i, (beta, label) in enumerate(tqdm(betas, desc="snapshots")):
        samples = collect_ensemble(alpha, beta, L, n_steps, warmup, n_seeds)
        H, xe, ye = joint_histogram(samples)
        ax = fig.add_subplot(3, 3, i + 1, projection="3d")
        plot_3d(ax, H, xe, ye, beta, label)
    fig.suptitle(rf"$P(\rho_1,\rho_2)$, $\alpha={alpha}$, $L={L}$, "
                 rf"ensemble of {n_seeds} seeds (paper Fig 3)")
    fig.tight_layout()
    fig.savefig("results/fig3_joint_density_3d.png", dpi=150)
    plt.show()


def _run_frame(args):
    """Run one beta frame: ensemble over seeds, return concatenated samples."""
    alpha, L, n_steps, warmup, n_seeds, b = args
    all_s = []
    for k in range(n_seeds):
        sim = TwoChannelASEP(L=L, alpha=alpha, beta=float(b), seed=k)
        sim.run(n_steps=n_steps, sample_every=max(1, L // 10), warmup=warmup)
        all_s.append(sim.get_joint_density_samples())
    return np.concatenate(all_s)


def animation(alpha, L, n_steps, warmup, n_seeds,
              b_min=0.04, b_max=0.35, n_frames=80, out="results/fig3_anim.mp4"):
    """3D animation sweeping beta, showing peaks emerge and merge.

    Precomputation of the frames is parallelized across cores (each frame is
    an independent ensemble run).
    """
    from matplotlib import animation
    from concurrent.futures import ProcessPoolExecutor
    betas = np.linspace(b_min, b_max, n_frames)
    tasks = [(alpha, L, n_steps, warmup, n_seeds, float(b)) for b in betas]

    precompute = []
    with ProcessPoolExecutor() as ex:
        for samples in tqdm(ex.map(_run_frame, tasks), total=len(tasks),
                            desc="precompute beta sweep"):
            precompute.append(joint_histogram(samples))

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        H, xe, ye = precompute[frame]
        plot_3d(ax, H, xe, ye, betas[frame], "")
        ax.set_title(rf"$\beta={betas[frame]:.4f}$")
        return ax

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   blit=False, interval=150)
    anim.save(out, writer="ffmpeg", fps=8, dpi=90)
    plt.show()


if __name__ == "__main__":
    ALPHA = 0.9
    L = 1000
    N_STEPS = 3_000_000
    WARMUP = 300_000
    N_SEEDS = 8

    snapshots(ALPHA, L, N_STEPS, WARMUP, N_SEEDS)
    animation(ALPHA, L, N_STEPS, WARMUP, N_SEEDS)
