"""
Unified L=1000 run: ONE continuous alpha=0.9 GPU pass feeds high-resolution
fig3 P(rho1,rho2) AND ssb together, saving per-chunk incrementally.

High-res fig3: each replica emits MANY raw (rho1,rho2) TIME-samples (via the
sampled kernel), so with only ~128 reps we still get ~128k points per beta —
far more than the 96x96 bins need. The GPU runs many betas at once (full
occupancy), keeping n_reps low as requested.

Budget: L-scaled, c=100 (100k steps/site at L=1000). raw_samples captured each
`sample_every` step.

Run:  python scripts/unified_L1000.py [--tag NAME] [--nrep 128]
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


def unified_L1000(L, out_dir, nrep_per_beta=128, sample_every=100000,
                  chunk_raw=400, chunks=None):
    """One continuous alpha=0.9 run over fig3+ssb union. Saves per-chunk raw
    samples + summary stats incrementally.

    Per chunk, samples are taken every `sample_every` steps, so the number of
    samples per replica in a chunk is chunk_steps/sample_every (buffer sized by
    `chunk_raw`). Only the LAST chunk's samples are used for high-res fig3
    (steady-state). ~200+ samples/rep x 128 reps >> 96x96 bins.
    """
    os.makedirs(out_dir, exist_ok=True)
    betas = union_betas()
    nb = len(betas)
    n_total = nb * nrep_per_beta
    sps = steps_per_site(L)
    chunk_steps = L * 20000              # 20k steps/site per chunk
    nchunks = max(1, round(sps / 20000)) if chunks is None else chunks
    print(f"[L1000] L={L}: {nb} betas x {nrep_per_beta} reps = {n_total} threads, "
          f"{sps}k steps/site over {nchunks} chunks, chunk_raw={chunk_raw}",
          flush=True)

    state = run_ensemble_cuda_continue(L, n_total, seed=0, n_raw_samples=chunk_raw)
    state["alpha_d"] = cuda.to_device(np.full(n_total, ALPHA, np.float64))
    state["beta_d"] = cuda.to_device(np.repeat(np.array(betas, np.float64),
                                               nrep_per_beta))
    # ensure sample_count is zeroed for this new state
    cuda.synchronize()

    for ci in tqdm(range(nchunks), desc=f"L={L} alpha=0.9"):
        # reset raw-sample counter so this chunk's buffer holds only THIS
        # chunk's time-samples (first chunks = equilibration; the LAST chunk
        # is the steady-state histogram source for high-res fig3).
        state["sample_count"][:] = 0
        cuda.synchronize()
        res = state["advance"](chunk_steps, warmup=0, sample_every=sample_every)
        # incremental save per chunk
        for i, b in enumerate(betas):
            o = i * nrep_per_beta
            sc = res["sample_count"][o:o + nrep_per_beta]
            raw = res["raw_samples"][o:o + nrep_per_beta]  # (nrep, nsamp, 2)
            np.savez(f"{out_dir}/chunk{ci:03d}_beta{int(b*10000):05d}.npz",
                     raw_samples=raw, sample_count=sc,
                     rho1=res["rho1"][o:o + nrep_per_beta],
                     rho2=res["rho2"][o:o + nrep_per_beta],
                     dense=res["dense"][o:o + nrep_per_beta],
                     dilute=res["dilute"][o:o + nrep_per_beta],
                     std=res["std_diff"][o:o + nrep_per_beta],
                     cur1=res["cur1"][o:o + nrep_per_beta],
                     cur2=res["cur2"][o:o + nrep_per_beta],
                     ttime=res["ttime"][o:o + nrep_per_beta])
    print(f"[L] L={L} done -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="L1000")
    ap.add_argument("--nrep", type=int, default=128)
    ap.add_argument("--sample_every", type=int, default=100000)
    ap.add_argument("--chunk_raw", type=int, default=400)
    args = ap.parse_args()
    sys.stdout = TeeLog(os.path.join(ROOT, "results", f"{args.tag}.log"))
    print("L1000 unified fig3+ssb (high-res)", flush=True)
    out = os.path.join(ROOT, "results", args.tag)
    unified_L1000(1000, out, nrep_per_beta=args.nrep,
                  sample_every=args.sample_every, chunk_raw=args.chunk_raw)
    print("ALL L1000 DONE", flush=True)


if __name__ == "__main__":
    main()
