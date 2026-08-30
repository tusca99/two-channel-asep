"""
Clean rerun of the SSB beta-scan at L=1000, alpha=0.9 (slide-34 order parameter).

Why: results/L1000 fig3's animation pass was generated with one shared seed
per whole-beta ensemble (fig3_plot.py, run_ensemble_cuda seed=7), and 11/49
betas collapsed to a SINGLE basin (frac(gap>0)=1.000, impossible for
independent SSB replicas). This rerun uses the persistent continue-kernel
(same path as unified_L1000.py, which produced correct 50/50 splits) into a
FRESH directory, saves per-replica rho1/rho2 per chunk, and checks the
50/50 basin split as an independence monitor.

Budget: c=120 (120k steps/site = 1.2e8 steps/replica) over 6 chunks of
2e7 steps; 49 betas x 128 replicas = 6272 GPU threads.

Run:   uv run python scripts/ssb_rerun.py
Log:   results/ssb_rerun.log (auto-flushed). Chunk files are standalone;
       an early stop keeps all completed chunks.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from numba import cuda
from tqdm import tqdm
from asep.cuda_ensemble import run_ensemble_cuda_continue

ALPHA = 0.9
L = 1000
NREP = 128
SNAPSHOT = [0.23, 0.245, 0.255, 0.258, 0.2595, 0.262, 0.2685, 0.28, 0.95]
ANIM = [round(b, 6) for b in np.linspace(0.04, 0.35, 40)]
BETAS = sorted(set(round(b, 6) for b in (SNAPSHOT + ANIM)))

CHUNK_STEPS = L * 20000          # 2e7 steps/replica per chunk
NCHUNKS = 6                      # -> 1.2e8 steps/replica = 120k steps/site
SAMPLE_EVERY = 100000
SEED = 20260830                  # do NOT reuse any earlier seed
TAG = "ssb_rerun"


def tee_log(path):
    class Tee:
        def __init__(self, f, out):
            self.f, self.out = f, out
        def write(self, s):
            self.f.write(s); self.f.flush(); self.out.write(s); self.out.flush()
        def flush(self):
            self.f.flush(); self.out.flush()
    f = open(path, "a", buffering=1)
    sys.stdout = Tee(f, sys.stdout)
    sys.stderr = Tee(f, sys.stderr)


def main():
    out = os.path.join(ROOT, "results", TAG)
    os.makedirs(out, exist_ok=True)
    tee_log(os.path.join(ROOT, "results", f"{TAG}.log"))
    nb = len(BETAS)
    n_total = nb * NREP
    print(f"[{TAG}] started {time.strftime('%F %T')}: L={L} alpha={ALPHA} "
          f"{nb} betas x {NREP} reps = {n_total} threads, "
          f"{NCHUNKS} chunks x {CHUNK_STEPS} steps = "
          f"{NCHUNKS*CHUNK_STEPS//L} steps/site, seed={SEED}", flush=True)

    state = run_ensemble_cuda_continue(L, n_total, seed=SEED)
    state["alpha_d"] = cuda.to_device(np.full(n_total, ALPHA, np.float64))
    state["beta_d"] = cuda.to_device(
        np.repeat(np.array(BETAS, np.float64), NREP))
    cuda.synchronize()

    for ci in range(NCHUNKS):
        t0 = time.time()
        res = state["advance"](CHUNK_STEPS, warmup=0, sample_every=SAMPLE_EVERY)
        dt = time.time() - t0
        for i, b in enumerate(BETAS):
            o = i * NREP
            np.savez(
                os.path.join(out, f"chunk{ci:03d}_beta{int(b*10000):05d}.npz"),
                rho1=res["rho1"][o:o + NREP],
                rho2=res["rho2"][o:o + NREP],
                dense=res["dense"][o:o + NREP],
                dilute=res["dilute"][o:o + NREP],
                std=res["std_diff"][o:o + NREP],
                cur1=res["cur1"][o:o + NREP],
                cur2=res["cur2"][o:o + NREP],
                ttime=res["ttime"][o:o + NREP])
        # independence + equilibration monitor: signed gap per replica
        fr, gm, sd = [], [], []
        for i in range(nb):
            g = (res["rho1"][i*NREP:(i+1)*NREP]
                 - res["rho2"][i*NREP:(i+1)*NREP])
            fr.append((np.abs(g) > 0.1).mean())
            gm.append(g.mean()); sd.append(g.std())
        fr, gm, sd = np.array(fr), np.array(gm), np.array(sd)
        print(f"[{TAG}] chunk {ci+1}/{NCHUNKS} done in {dt:.0f}s "
              f"(~{dt*(NCHUNKS-ci-1)/60:.0f} min left) | "
              f"broken-replica frac>0.1: min={fr.min():.2f} "
              f"med={np.median(fr):.2f} max={fr.max():.2f} | "
              f"signed-gap mean: [{gm.min():+.2f},{gm.max():+.2f}] "
              f"std med={np.median(sd):.2f}", flush=True)
    print(f"[{TAG}] finished {time.strftime('%F %T')}", flush=True)


if __name__ == "__main__":
    main()