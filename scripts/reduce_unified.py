"""
Reduce the per-chunk raw output of unified_alpha09.py into the fig3 and ssb
figure data files (these are the two figures that share the alpha=0.9 run).

Reads results/<tag>_L<L>/chunk*.npz (one per (beta, chunk), each holding
per-replica rho1/rho2/dense/dilute/std) and writes:
  fig3_points.npz  : per-replica (rho1,rho2) for fig3's beta set
  ssb_beta*.npz    : dense/dilute/std per replica for ssb

fig6 is NOT here — it is a separate short run (see remake_L.py / run_all_gpu.py).

Time-averaging knob: `--nfinal` averages the final N chunks for fig3/ssb
(equilibrated tail; nfinal=1 = last chunk only).

Usage: python scripts/reduce_unified.py --tag unified --Ls 500 1000
"""
import os
import sys
import glob
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ALPHA = 0.9
FIG3_BETAS = [0.23, 0.245, 0.255, 0.258, 0.2595, 0.262, 0.2685, 0.28, 0.95]
FIG3_ANIM = [round(b, 6) for b in np.linspace(0.04, 0.35, 40)]
SSB_BETAS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.2, 0.25, 0.3]


def beta_of(fname):
    return int(os.path.basename(fname).replace(".npz", "").split("_beta")[-1]) / 10000.0


def group_chunks(out_dir):
    """beta -> sorted list of chunk file paths (by chunk index).

    Chunk files store beta truncated to 4 decimals (int(b*10000)), so keys
    are matched by the same truncation.
    """
    files = sorted(glob.glob(os.path.join(out_dir, "chunks", "chunk*_beta*.npz")) +
                   glob.glob(os.path.join(out_dir, "chunk*_beta*.npz")),
                   key=lambda f: os.path.basename(f))
    by_beta = {}
    for f in files:
        by_beta.setdefault(int(beta_of(f) * 10000) / 10000.0, []).append(f)
    return by_beta


def _trunc4(b):
    return int(round(b, 6) * 10000) / 10000.0


def _nearest(by_beta, b, tol=0.0005):
    """Return the stored beta closest to b within tol (float-precision-safe)."""
    keys = np.array(sorted(by_beta.keys()))
    i = np.argmin(np.abs(keys - b))
    if abs(keys[i] - b) <= tol:
        return keys[i]
    return None


def reduce_fig3(by_beta, out_dir, nfinal=1):
    allb = [_trunc4(b) for b in FIG3_BETAS + FIG3_ANIM]
    d = {}
    for b in allb:
        cb = _nearest(by_beta, b)
        if cb is None:
            print(f"  [fig3] missing beta={b}", flush=True)
            continue
        files = by_beta[cb]
        last = np.load(files[-nfinal])
        d[cb] = np.stack([last["rho1"], last["rho2"]], axis=-1)
    np.savez(os.path.join(out_dir, "fig3_points.npz"),
             **{f"b{int(b*10000)}": d[b] for b in d},
             betas=np.array(sorted(d)))
    print(f"[fig3] saved {len(d)}/{len(allb)} betas -> fig3_points.npz", flush=True)


def reduce_ssb(by_beta, out_dir, nfinal):
    for beta in [_trunc4(b) for b in SSB_BETAS]:
        cb = _nearest(by_beta, beta)
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
        print(f"  [ssb] beta={beta} (last {nfinal} chunks)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="unified")
    ap.add_argument("--Ls", nargs="+", type=int, default=[500, 1000])
    ap.add_argument("--nfinal", type=int, default=1,
                    help="average last N chunks for fig3/ssb (time-averaging)")
    args = ap.parse_args()
    for L in args.Ls:
        out = os.path.join(ROOT, "results", f"{args.tag}_L{L}")
        by_beta = group_chunks(out)
        print(f"L={L}: {len(by_beta)} betas found in {out}", flush=True)
        reduce_fig3(by_beta, out, args.nfinal)
        reduce_ssb(by_beta, out, args.nfinal)
        print(f"=== L={L} reduced ===", flush=True)
    print("ALL REDUCE DONE", flush=True)


if __name__ == "__main__":
    main()
