"""
Unified alpha=0.9 run: ONE continuous GPU pass per L feeds fig3 AND ssb
together (their beta sets overlap, so deriving ssb from fig3's run is free —
the reuse principle). Saved incrementally per chunk, so nothing is re-run.

fig6 and fig2 are NOT bundled here: they are short, cheap runs with tiny
statistics (n=32, 20% equilibration) — bundling them into this long, high-rep
run would RAISE total wall time (a bad reuse tradeoff). They run separately
(remake_L.py, or run_all_gpu.py).

Budget: L-scaled (CLAUDE.md c=100): steps/site = 100*L, so 50k at L=500,
100k at L=1000, total = c*L^2.

Usage: python scripts/unified_alpha09.py [--Ls 500 1000] [--tag NAME]
Progress -> results/<tag>.log (auto-flushed).
"""
import os
import sys
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from numba import cuda
from tqdm import tqdm
from asep.cuda_ensemble import run_ensemble_cuda_continue

ALPHA = 0.9
# fig3 (paper snapshots + anim range) UNION ssb betas -> free ssb reuse
FIG3_SNAPSHOT = [0.23, 0.245, 0.255, 0.258, 0.2595, 0.262, 0.2685, 0.28, 0.95]
FIG3_ANIM = np.linspace(0.04, 0.35, 40).tolist()
SSB_BETAS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.2, 0.25, 0.3]


def union_betas():
    return sorted(set(round(b, 6) for b in (FIG3_SNAPSHOT + FIG3_ANIM + SSB_BETAS)))


def steps_per_site(L):
    return int(100 * L)   # c=100 -> 100k steps/site at L=1000


class TeeLog:
    """Tee BOTH stdout and stderr to a log file (tqdm writes to stderr)."""
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)
        self._out = sys.stdout
        self._err = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, s):
        self.f.write(s); self.f.flush()
        self._out.write(s); self._out.flush()

    def flush(self):
        self.f.flush(); self._out.flush(); self._err.flush()


def unified_alpha09(L, out_dir, nrep_per_beta=1024, sample_every=5000):
    """One continuous alpha=0.9 run over fig3+ssb union of betas. Saves
    per-chunk raw rho1/rho2/dense/dilute/std/currents incrementally.
    """
    os.makedirs(out_dir, exist_ok=True)
    betas = union_betas()
    nb = len(betas)
    n_total = nb * nrep_per_beta
    sps = steps_per_site(L)
    chunk_steps = L * 20000              # 20k steps/site per chunk
    nchunks = max(1, round(sps / 20000)) # enough chunks to reach the budget
    print(f"[u] L={L}: {nb} betas x {nrep_per_beta} reps = {n_total} threads; "
          f"{sps}k steps/site over {nchunks} chunks", flush=True)

    state = run_ensemble_cuda_continue(L, n_total, seed=0)
    state["alpha_d"] = cuda.to_device(np.full(n_total, ALPHA, np.float64))
    state["beta_d"] = cuda.to_device(np.repeat(np.array(betas, np.float64),
                                               nrep_per_beta))
    for ci in tqdm(range(nchunks), desc=f"L={L} alpha=0.9 unified"):
        res = state["advance"](chunk_steps, warmup=0, sample_every=sample_every)
        # incremental save after each chunk: per-replica rho1/rho2/currents
        for i, b in enumerate(betas):
            o = i * nrep_per_beta
            np.savez(f"{out_dir}/chunk{ci:03d}_beta{int(b*10000):05d}.npz",
                     rho1=res["rho1"][o:o + nrep_per_beta],
                     rho2=res["rho2"][o:o + nrep_per_beta],
                     dense=res["dense"][o:o + nrep_per_beta],
                     dilute=res["dilute"][o:o + nrep_per_beta],
                     std=res["std_diff"][o:o + nrep_per_beta],
                     cur1=res["cur1"][o:o + nrep_per_beta],
                     cur2=res["cur2"][o:o + nrep_per_beta],
                     ttime=res["ttime"][o:o + nrep_per_beta])
    print(f"[unify] L={L} raw per-chunk saved -> {out_dir} ({nchunks} chunks)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", nargs="+", type=int, default=[500, 1000])
    ap.add_argument("--tag", default="unified")
    ap.add_argument("--nrep", type=int, default=1024)
    args = ap.parse_args()

    sys.stdout = TeeLog(os.path.join(ROOT, "results", f"{args.tag}.log"))
    print(f"UNIFIED Ls={args.Ls} alpha={ALPHA}", flush=True)
    for L in args.Ls:
        out = os.path.join(ROOT, "results", f"{args.tag}_L{L}")
        unified_alpha09(L, out, nrep_per_beta=args.nrep)
        print(f"=== L={L} complete ===", flush=True)
    print("ALL UNIFIED RUN DONE", flush=True)


if __name__ == "__main__":
    main()
