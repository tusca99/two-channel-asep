"""
fig5 large-L beta boundary on the H100 server (80GB VRAM, 32 cores).

Saturates the GPU with high replica counts (n_reps in 1024..8192) so the
ensemble fills the H100. Each (alpha,beta) point gets n_reps replicas; the
beta sweep (alpha=0.9) runs in ONE run_ensemble_cuda launch with per-replica
(alpha,beta), RNG streamed on-device (no giant host buffer).

Budget: L-scaled (steps/site ~ 20*L) to reach the HD/LD dense basin at large L.

Run on the server (in tmux):
  tmux new -s fig5
  cd <repo> && uv run python scripts/fig5_server.py --Ls 4000 8000 --nrep 4096
  # detach: Ctrl-b d ; reattach: tmux attach -t fig5
  # progress: tail -f results/fig5_gpu.log

Writes results/fig5_gpu/fig5_beta_L<L>.npz incrementally + results/fig5_gpu.log.
"""
import os
import sys
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tqdm import tqdm
from asep.cuda_ensemble import run_ensemble_cuda
from scripts.fig5_boundaries import classify_point
from scripts.remake_L import TeeLog

ALPHA = 0.9
OUT = os.path.join(ROOT, "results", "fig5_gpu")


def gpu_beta_sweep(alpha, betas, L, steps_per_site, n_reps, seed=0):
    """Per-replica rho1/rho2/J for each beta, ONE GPU launch (RNG on-device)."""
    steps = int(L * steps_per_site)
    warmup = int(L * (steps_per_site // 5))
    nb = len(betas)
    aa = np.full(nb * n_reps, alpha)
    bb = np.repeat(betas, n_reps)
    out = run_ensemble_cuda(alpha, 0.0, L, steps, nb * n_reps, seed=seed,
                            sample_every=400, warmup=warmup, alphas=aa, betas=bb)
    r1s, r2s, J1s, J2s = [], [], [], []
    for j in range(nb):
        sl = slice(j * n_reps, (j + 1) * n_reps)
        r1s.append(out['rho1'][sl]); r2s.append(out['rho2'][sl])
        J1s.append(out['cur1'][sl] / out['ttime'][sl])
        J2s.append(out['cur2'][sl] / out['ttime'][sl])
    return r1s, r2s, J1s, J2s


def find_hdld_boundary(r1s, r2s, J1s, J2s, betas, L):
    """Beta where the last HD/LD (or LD/HD) point sits; boundary just past it."""
    labels = []
    for j in range(len(betas)):
        lab, _ = classify_point(r1s[j], r2s[j], J1s[j], J2s[j], ALPHA, betas[j], L)
        labels.append(lab)
    last = None
    for j in range(len(betas)):
        if labels[j] in ("HD/LD", "LD/HD"):
            last = j
    tr = None
    if last is not None and last < len(betas) - 1:
        tr = 0.5 * (betas[last] + betas[last + 1])
    return tr, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", nargs="+", type=int, default=[4000, 8000])
    ap.add_argument("--nrep", type=int, default=1024,
                    help="replicas per (alpha,beta) point; saturate H100")
    ap.add_argument("--sps", type=int, default=0,
                    help="steps/site (default L-scaled ~20*L; with high nrep "
                         "the ensemble averages, so a smaller budget suffices)")
    ap.add_argument("--betas", type=int, default=24,
                    help="number of beta points in the sweep")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sys.stdout = TeeLog(os.path.join(ROOT, "results", "fig5_gpu.log"))

    betas = np.linspace(0.05, 0.6, args.betas)
    print(f"fig5 GPU large-L beta boundary (alpha={ALPHA}, nrep={args.nrep})",
          flush=True)
    for L in tqdm(args.Ls, desc="fig5 large L"):
        sps = args.sps or int(20 * L)
        r1s, r2s, J1s, J2s = gpu_beta_sweep(ALPHA, betas, L, sps, args.nrep)
        tr, labels = find_hdld_boundary(r1s, r2s, J1s, J2s, betas, L)
        dense = [np.maximum(r1s[j], r2s[j]).mean() for j in range(len(betas))]
        print(f"  L={L}: HD/LD->LD/LD beta={tr} (dense[0]={dense[0]:.3f}, "
              f"dense[-1]={dense[-1]:.3f})", flush=True)
        print(f"    labels={[labels[j] for j in range(len(betas))]}", flush=True)
        np.savez(os.path.join(OUT, f"fig5_beta_L{L}.npz"),
                 beta_boundary=tr, betas=betas, dense=np.array(dense),
                 sps=sps, nrep=args.nrep)
    print("ALL FIG5 GPU LARGE DONE", flush=True)


if __name__ == "__main__":
    main()
