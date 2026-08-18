"""
Extra fig3 plots from existing L200 data (no re-run).

Uses results/L200/fig3_points.npz (49 betas, 512 reps each) to produce:
  - snapshots at a denser set of beta values (incl. the SSB region)
  - a 2D animation over ALL available betas (0.04..0.95)
  - a 3D animation over ALL available betas
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "L200")
os.makedirs(OUT, exist_ok=True)


def load_points(fname="fig3_points.npz"):
    d = np.load(os.path.join(OUT, fname), allow_pickle=True)
    pts = {}
    for k in d.files:
        if k != "betas":
            pts[float(k[1:]) / 10000] = d[k]
    return pts


def joint_histogram(samples, bins=48):
    H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1],
                                       bins=bins, range=[[0, 1], [0, 1]])
    return H, xedges, yedges


def plot_3d(ax, H, xedges, yedges, beta, label=""):
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xc, yc)
    Z = H.T
    Z = Z / Z.max()
    ax.plot_surface(X, Y, Z, cmap="turbo", edgecolor="none", alpha=0.95,
                    linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.contourf(X, Y, Z, zdir="z", offset=0, cmap="turbo", alpha=0.9, levels=14)
    ax.set_xlabel(r"$\rho_1$", fontsize=8)
    ax.set_ylabel(r"$\rho_2$", fontsize=8)
    ax.set_zlabel(r"$P/P_{max}$", fontsize=8)
    ax.set_xlim(1, 0); ax.set_ylim(0, 1); ax.set_zlim(0, 1.05)
    ax.set_title(rf"$\beta={beta}$ [{label}]", fontsize=9)
    ax.view_init(elev=25, azim=-55)
    ax.set_box_aspect((1, 1, 0.6))


def snapshots_dense(pts, bins=48):
    """Snapshots at a denser beta set covering the SSB region."""
    betas = [0.05, 0.08, 0.10, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35,
             0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    fig = plt.figure(figsize=(16, 12))
    for i, b in enumerate(betas):
        # find nearest available beta
        avail = min(pts.keys(), key=lambda x: abs(x - b))
        H, xe, ye = joint_histogram(pts[avail], bins=bins)
        ax = fig.add_subplot(4, 4, i + 1, projection="3d")
        plot_3d(ax, H, xe, ye, avail)
    fig.suptitle(rf"$P(\rho_1,\rho_2)$, L=200, $\alpha=0.9$ (dense snapshots)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_snapshots_dense.png", dpi=150)
    print("saved fig3_snapshots_dense.png", flush=True)


def animation_all(pts, bins=48):
    """2D + 3D animation over ALL available betas (0.04..0.95)."""
    from matplotlib import animation
    betas = sorted(pts.keys())
    frames = [joint_histogram(pts[b], bins=bins) for b in betas]

    # 2D
    vmax = max(H.max() for H, _, _ in frames)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    H0, xe, ye = frames[0]
    im = ax.imshow(H0.T, origin="lower", aspect="equal",
                   extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   cmap="turbo", vmin=0, vmax=vmax)
    ax.plot([0, 1], [0, 1], "r--", lw=0.8)
    ax.set_xlabel(r"$\rho_1$"); ax.set_ylabel(r"$\rho_2$")
    plt.colorbar(im, ax=ax, label=r"$P$")
    title = fig.suptitle(rf"$\beta={betas[0]:.4f}$")

    def update(frame):
        im.set_data(frames[frame][0].T)
        title.set_text(rf"$\beta={betas[frame]:.4f}$")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=len(betas),
                                   blit=True, interval=200)
    anim.save(f"{OUT}/fig3_anim_all_2d.mp4", writer="ffmpeg", fps=6, dpi=90)
    print("saved fig3_anim_all_2d.mp4", flush=True)

    # 3D
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update3(frame):
        ax.clear()
        H, xe, ye = frames[frame]
        plot_3d(ax, H, xe, ye, betas[frame])
        ax.set_title(rf"$\beta={betas[frame]:.4f}$")
        return ax

    anim3 = animation.FuncAnimation(fig, update3, frames=len(betas),
                                     blit=False, interval=200)
    anim3.save(f"{OUT}/fig3_anim_all_3d.mp4", writer="ffmpeg", fps=6, dpi=90)
    print("saved fig3_anim_all_3d.mp4", flush=True)


if __name__ == "__main__":
    pts = load_points()
    snapshots_dense(pts)
    animation_all(pts)
    print("ALL DONE", flush=True)
