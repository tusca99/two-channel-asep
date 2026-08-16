"""
Figure 3 (GPU, single scan): one ensemble run covering all betas, then all
plots/animation derived offline from the saved points.

Rationale: do NOT re-run MC for each figure. Do ONE GPU ensemble scan over the
union of betas (paper snapshots + animation range), save the per-replica
(rho1,rho2) points to an .npz, then build the 3D snapshot figure, the 2D
heatmap animation and the 3D animation entirely from that one file.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from asep.cuda_ensemble import run_ensemble_cuda

OUT = "/home/alessio/Documenti/two-channel-asep/results/gpu"
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.9
L = 1000
STEPS = 2_000_000      # window per replica (enough to settle, not to flip)
WARMUP = 200_000
N_REPS = 2048          # replicas per beta (ensemble)
SAMPLE_EVERY = 50000

_pts_keys = []

# Union of betas: paper snapshots (9) + animation range 0.04..0.35 (40)
SNAPSHOT_BETAS = [0.23, 0.245, 0.255, 0.258, 0.2595, 0.262, 0.2685, 0.28, 0.95]
ANIM_BETAS = np.linspace(0.04, 0.35, 40).tolist()
ALL_BETAS = sorted(set(SNAPSHOT_BETAS) | set(round(b, 6) for b in ANIM_BETAS))


def scan_all(out="fig3_points.npz"):
    """One GPU scan over ALL_BETAS; save per-replica (rho1,rho2) points.

    Saves each chunk incrementally to {out}.chunk{i}.npz so partial progress
    survives a crash; merges at the end.
    """
    chunk_betas = 8
    pts = {}
    # resume: load any existing chunk file
    resume_f = f"{OUT}/{out}.chunks.npz"
    if os.path.exists(resume_f):
        d = dict(np.load(resume_f, allow_pickle=True))
        done = set(float(k[1:]) / 10000 for k in d if k != "betas")
        for k, v in d.items():
            if k != "betas":
                pts[float(k[1:]) / 10000] = v
        print(f"resuming: {len(done)} betas already done", flush=True)
    else:
        done = set()
    print(f"scanning {len(ALL_BETAS)} betas, {N_REPS} reps each, "
          f"{STEPS} steps", flush=True)
    for c0 in range(0, len(ALL_BETAS), chunk_betas):
        chunk = ALL_BETAS[c0:c0 + chunk_betas]
        todo = [b for b in chunk if b not in done]
        if not todo:
            continue
        nrep = len(todo) * N_REPS
        bb = np.repeat(np.array(todo), N_REPS)
        res = run_ensemble_cuda(0.0, 0.0, L, STEPS, nrep, seed=7,
                                sample_every=SAMPLE_EVERY, warmup=WARMUP,
                                alphas=np.full(nrep, ALPHA), betas=bb)
        for i, b in enumerate(todo):
            sl = slice(i * N_REPS, (i + 1) * N_REPS)
            pts[b] = np.stack([res["rho1"][sl], res["rho2"][sl]], axis=-1)
        save_chunk(f"{OUT}/{out}", pts, todo)
    # final merged save
    arr = np.stack([pts[b] for b in ALL_BETAS])
    np.savez(f"{OUT}/{out}", **{f"b{int(b*10000)}": pts[b] for b in ALL_BETAS},
             betas=np.array(ALL_BETAS))
    print(f"saved {out}", flush=True)
    _pts_keys[:] = list(pts.keys())
    return pts


def save_chunk(path_base, pts, chunk_betas):
    """Append this chunk's beta->points to a running file keyed by beta."""
    fname = f"{path_base}.chunks"
    d = {}
    if os.path.exists(fname + ".npz"):
        d = dict(np.load(fname + ".npz", allow_pickle=True))
    d.pop("betas", None)
    for b in chunk_betas:
        d[f"b{int(b*10000)}"] = pts[b]
    betas = np.array(sorted(float(k[1:]) / 10000 for k in d))
    np.savez(fname + ".npz", **d, betas=betas)


def key_of(b):
    """Match b to the saved key with tolerance (keys stored to 4 decimals)."""
    target = round(float(b), 6)
    for k in _pts_keys:
        if abs(k - target) < 1e-4:
            return k
    raise KeyError(target)


def load_points(fname="fig3_points.npz"):
    global _pts_keys
    d = np.load(f"{OUT}/{fname}", allow_pickle=True)
    pts = {}
    for k in d.files:
        if k != "betas":
            pts[float(k[1:]) / 10000] = d[k]
    _pts_keys = list(pts.keys())
    return pts


def joint_histogram(samples, bins=48):
    H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1],
                                       bins=bins, range=[[0, 1], [0, 1]])
    return H, xedges, yedges


def plot_3d(ax, H, xedges, yedges, beta, label):
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


def snapshots(pts):
    betas = [(0.23, "HD/LD"), (0.245, "HD/LD"), (0.255, "HD/LD"),
             (0.258, "HD/LD"), (0.2595, "HD/LD+LD/LD"), (0.262, "LD/LD"),
             (0.2685, "LD/LD+LD"), (0.28, "LD"), (0.95, "MC")]
    fig = plt.figure(figsize=(15, 12))
    for i, (beta, label) in enumerate(betas):
        H, xe, ye = joint_histogram(pts[beta])
        ax = fig.add_subplot(3, 3, i + 1, projection="3d")
        plot_3d(ax, H, xe, ye, beta, label)
    fig.suptitle(rf"$P(\rho_1,\rho_2)$, $\alpha={ALPHA}$, $L={L}$, "
                 rf"GPU ensemble of {N_REPS} replicas (paper Fig 3)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_joint_density_3d.png", dpi=150)
    print("saved fig3_joint_density_3d.png", flush=True)


def animation_2d(pts, out="fig3_anim_2d.mp4"):
    from matplotlib import animation
    betas = ANIM_BETAS
    frames = [joint_histogram(pts[key_of(b)]) for b in betas]
    vmax = max(H.max() for H, _, _ in frames)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    H0, xe, ye = frames[0]
    im = ax.imshow(H0.T, origin="lower", aspect="equal",
                   extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   cmap="turbo", vmin=0, vmax=vmax)
    ax.plot([0, 1], [0, 1], "r--", lw=0.8)
    ax.set_xlabel(r"$\rho_1$"); ax.set_ylabel(r"$\rho_2$")
    cb = plt.colorbar(im, ax=ax, label=r"$P$")
    title = fig.suptitle(rf"$\beta={betas[0]:.4f}$")

    def update(frame):
        im.set_data(frames[frame][0].T)
        title.set_text(rf"$\beta={betas[frame]:.4f}$")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=len(betas),
                                   blit=True, interval=150)
    anim.save(f"{OUT}/{out}", writer="ffmpeg", fps=8, dpi=90)
    print(f"saved {out}", flush=True)


def animation_3d(pts, out="fig3_anim.mp4"):
    from matplotlib import animation
    betas = ANIM_BETAS
    frames = [joint_histogram(pts[key_of(b)]) for b in betas]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        H, xe, ye = frames[frame]
        plot_3d(ax, H, xe, ye, betas[frame], "")
        ax.set_title(rf"$\beta={betas[frame]:.4f}$")
        return ax

    anim = animation.FuncAnimation(fig, update, frames=len(betas),
                                   blit=False, interval=150)
    anim.save(f"{OUT}/{out}", writer="ffmpeg", fps=8, dpi=90)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    import sys
    # --load: only build plots from saved points, do NOT re-scan
    if "--load" in sys.argv:
        pts = load_points()
    else:
        pts = scan_all()
    snapshots(pts)
    animation_2d(pts)
    animation_3d(pts)
    print("ALL DONE", flush=True)
