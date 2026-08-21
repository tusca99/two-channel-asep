"""
Reduce the per-chunk output of unified_L1000.py into high-res fig3 + ssb.

The LAST chunk's raw (rho1,rho2) time-samples per replica feed the fine
(96x96) P(rho1,rho2) histogram; ssb uses the summary dense/dilute/std from
the equilibrated tail.

Usage: python scripts/reduce_L1000.py --tag L1000 [--bins 96]
"""
import os
import sys
import glob
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.9
FIG3_SNAPSHOT = [0.23, 0.245, 0.255, 0.258, 0.2595, 0.262, 0.2685, 0.28, 0.95]
FIG3_ANIM = [round(b, 6) for b in np.linspace(0.04, 0.35, 40)]
SSB_BETAS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.2, 0.25, 0.3]


def beta_of(fname):
    return int(os.path.basename(fname).replace(".npz", "").split("_beta")[-1]) / 10000.0


def group_chunks(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "chunks", "chunk*_beta*.npz")) +
                   glob.glob(os.path.join(out_dir, "chunk*_beta*.npz")),
                   key=lambda f: os.path.basename(f))
    by_beta = {}
    for f in files:
        by_beta.setdefault(int(beta_of(f) * 10000) / 10000.0, []).append(f)
    return by_beta


def trunc4(b):
    return int(round(b, 6) * 10000) / 10000.0


def nearest(by_beta, b, tol=0.0005):
    keys = np.array(sorted(by_beta.keys()))
    i = np.argmin(np.abs(keys - b))
    if abs(keys[i] - b) <= tol:
        return keys[i]
    return None


def collect_raw(files, nfinal=1):
    """Concatenate the last nfinal chunks' raw samples -> (npts, 2)."""
    take = files[-nfinal:]
    pts = []
    for f in take:
        d = np.load(f)
        sc = d["sample_count"]
        raw = d["raw_samples"]
        for i in range(raw.shape[0]):
            n = int(sc[i])
            pts.append(raw[i, :n])
    if not pts:
        return np.empty((0, 2))
    return np.concatenate(pts, axis=0)


def reduce_fig3(by_beta, out_dir, bins, nfinal=1):
    allb = [trunc4(b) for b in FIG3_SNAPSHOT + FIG3_ANIM]
    d = {}
    for b in allb:
        cb = nearest(by_beta, b)
        if cb is None:
            print(f"  [fig3] missing beta={b}", flush=True)
            continue
        files = by_beta[cb]
        d[cb] = collect_raw(files, nfinal)
        print(f"  [fig3] beta={cb}: {d[cb].shape[0]} raw points", flush=True)
    np.savez(os.path.join(out_dir, "fig3_points.npz"),
             **{f"b{int(b*10000)}": d[b] for b in d},
             betas=np.array(sorted(d)))
    print(f"[fig3] saved {len(d)} betas -> fig3_points.npz (bins={bins})",
          flush=True)


def reduce_ssb(by_beta, out_dir, nfinal):
    for beta in [trunc4(b) for b in SSB_BETAS]:
        cb = nearest(by_beta, beta)
        if cb is None:
            print(f"  [ssb] missing beta {beta}", flush=True)
            continue
        files = by_beta[cb]
        take = files[-nfinal:]
        dens = np.concatenate([np.load(f)["dense"][:, None] for f in take], axis=1)
        dilu = np.concatenate([np.load(f)["dilute"][:, None] for f in take], axis=1)
        stdd = np.concatenate([np.load(f)["std"][:, None] for f in take], axis=1)
        np.savez(os.path.join(out_dir, f"ssb_beta{beta}.npz"),
                 dense=dens, dilute=dilu, std=stdd, alpha=ALPHA, nfinal=nfinal)
        print(f"  [ssb] beta={beta}", flush=True)


def plot_ssb(out_dir):
    betas = SSB_BETAS
    diff = []; std = []
    for b in betas:
        d = np.load(f"{out_dir}/ssb_beta{trunc4(b)}.npz")
        diff.append(np.mean(d["dense"][:, -1] - d["dilute"][:, -1]))
        std.append(np.mean(d["std"][:, -1]))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(betas, diff, "o-", ms=5, label=r"$\langle\rho_{dense}-\rho_{dilute}\rangle$")
    ax.plot(betas, std, "s--", ms=5, label=r"std$(\rho_1-\rho_2)$")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel("SSB order parameter")
    ax.set_title(r"SSB vs $\beta$, L=1000, $\alpha=0.9$")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ssb_order_vs_beta.png"), dpi=150)
    print(f"saved {out_dir}/ssb_order_vs_beta.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="L1000")
    ap.add_argument("--bins", type=int, default=96)
    ap.add_argument("--nfinal", type=int, default=1)
    args = ap.parse_args()
    out = os.path.join(ROOT, "results", args.tag)
    by_beta = group_chunks(out)
    print(f"L1000: {len(by_beta)} betas", flush=True)
    reduce_fig3(by_beta, out, args.bins, args.nfinal)
    reduce_ssb(by_beta, out, args.nfinal)
    plot_ssb(out)
    print("ALL L1000 REDUCE DONE", flush=True)


if __name__ == "__main__":
    main()
