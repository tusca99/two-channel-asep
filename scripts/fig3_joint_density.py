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
    """3D surface with colored contour on the bottom plane; good contrast.

    rho1 axis is reversed so that, like the paper, both rho1 and rho2 start
    at 0 at the nearest corner of the plot.
    """
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
    # Invert x (rho1) so rho1=0 is at the near corner, matching the paper
    # (both axes start at 0 nearest the viewer).
    ax.set_xlim(1, 0); ax.set_ylim(0, 1); ax.set_zlim(0, 1.05)
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


def _run_single(args):
    """Run one (beta, seed) MC run; return that seed's joint samples."""
    alpha, L, n_steps, warmup, b, seed = args
    sim = TwoChannelASEP(L=L, alpha=alpha, beta=float(b), seed=seed)
    sim.run(n_steps=n_steps, sample_every=max(1, L // 10), warmup=warmup)
    return sim.get_joint_density_samples()


def precompute_frames(alpha, L, n_steps, warmup, n_seeds, betas, bins=48):
    """Precompute joint histograms for each beta (parallelized).

    Parallelizes at the individual (beta, seed) run level for fine-grained
    load balancing: n_frames * n_seeds independent MC runs are distributed
    across the pool, then grouped back per beta.

    Returns (mean_frames, std_frames) where std_frames is the per-cell std of
    the histogram across seeds (a measure of run-to-run uncertainty).
    """
    from concurrent.futures import ProcessPoolExecutor
    n_frames = len(betas)
    # Flatten to individual runs: (beta, seed) pairs
    tasks = [(alpha, L, n_steps, warmup, betas[fi], seed)
             for fi in range(n_frames) for seed in range(n_seeds)]

    results = []
    with ProcessPoolExecutor() as ex:
        for samples in tqdm(ex.map(_run_single, tasks), total=len(tasks),
                            desc="precompute MC runs"):
            results.append(samples)

    # Group by beta index
    mean_frames = []
    std_frames = []
    for fi in range(n_frames):
        seeds = [results[fi * n_seeds + seed] for seed in range(n_seeds)]
        # per-seed histogram for the std of the distribution
        H_per_seed = np.stack([joint_histogram(s, bins=bins)[0] for s in seeds])
        group = np.concatenate(seeds)
        mean_frames.append(joint_histogram(group, bins=bins))
        std_frames.append(H_per_seed.std(axis=0))
    return mean_frames, std_frames


def animation(alpha, L, n_steps, warmup, n_seeds,
              b_min=0.04, b_max=0.35, n_frames=80, bins=48,
              out="results/fig3_anim.mp4"):
    """3D animation sweeping beta, showing peaks emerge and merge.

    Precomputation of the frames is parallelized across cores (each frame is
    an independent ensemble run).
    """
    from matplotlib import animation
    betas = np.linspace(b_min, b_max, n_frames)
    mean_frames, _ = precompute_frames(alpha, L, n_steps, warmup, n_seeds,
                                       betas, bins=bins)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        H, xe, ye = mean_frames[frame]
        plot_3d(ax, H, xe, ye, betas[frame], "")
        ax.set_title(rf"$\beta={betas[frame]:.4f}$")
        return ax

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   blit=False, interval=150)
    anim.save(out, writer="ffmpeg", fps=8, dpi=90)
    plt.show()


def animation_2d(alpha, L, n_steps, warmup, n_seeds,
                 b_min=0.04, b_max=0.35, n_frames=80, bins=48,
                 out="results/fig3_anim_2d.mp4"):
    """2D heatmap animation of P(rho1,rho2) sweeping beta.

    Top-down heatmap makes the two peaks much easier to distinguish than a 3D
    surface (which can hide them behind the base/grey). Same precompute as the
    3D animation, so running both is cheap.

    Two panels: mean P on the left, run-to-run std on the right (shows where
    the ensemble is uncertain, e.g. the SSB region where seeds land in
    different broken states).
    """
    from matplotlib import animation
    betas = np.linspace(b_min, b_max, n_frames)
    mean_frames, std_frames = precompute_frames(alpha, L, n_steps, warmup,
                                                n_seeds, betas, bins=bins)

    H0, xe, ye = mean_frames[0]
    vmax = max(H.max() for H, _, _ in mean_frames)
    smax = max(S.max() for S in std_frames)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axm, axs = axes

    im = axm.imshow(H0.T, origin="lower", aspect="equal",
                    extent=[xe[0], xe[-1], ye[0], ye[-1]],
                    cmap="turbo", vmin=0, vmax=vmax)
    axm.plot([0, 1], [0, 1], "r--", lw=0.8)
    axm.set_xlabel(r"$\rho_1$"); axm.set_ylabel(r"$\rho_2$")
    axm.set_title(r"mean $P(\rho_1,\rho_2)$")
    cb = plt.colorbar(im, ax=axm, label=r"$P$")

    ims = axs.imshow(std_frames[0].T, origin="lower", aspect="equal",
                     extent=[xe[0], xe[-1], ye[0], ye[-1]],
                     cmap="inferno", vmin=0, vmax=smax)
    axs.plot([0, 1], [0, 1], "r--", lw=0.8)
    axs.set_xlabel(r"$\rho_1$"); axs.set_ylabel(r"$\rho_2$")
    axs.set_title(r"run-to-run std of $P$")
    cbs = plt.colorbar(ims, ax=axs, label=r"$\sigma_P$")
    title = fig.suptitle(rf"$\beta={betas[0]:.4f}$")

    def update(frame):
        im.set_data(mean_frames[frame][0].T)
        ims.set_data(std_frames[frame].T)
        title.set_text(rf"$\beta={betas[frame]:.4f}$")
        return im, ims, title

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   blit=True, interval=150)
    anim.save(out, writer="ffmpeg", fps=8, dpi=90)
    def update(frame):
        im.set_data(mean_frames[frame][0].T)
        ims.set_data(std_frames[frame].T)
        title.set_text(rf"$\beta={betas[frame]:.4f}$")
        return im, ims, title

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   blit=True, interval=150)
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
