"""
Generate fig3 + ssb data for L=200 on CPU (12-core), into results/L200/,
reusing the already-done fig2/fig6 data. Then plot all figures via plot_all.
"""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "L200")
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.9
L = 200
STEPS = 2_000_000
WARMUP = 1_000_000
N_REPS = 512


def _one_replica(args):
    alpha, beta, L, steps, warmup, seed = args
    from asep.bkl import run_bkl_fenwick
    rng = np.random.default_rng(seed)
    lane1 = (rng.random(L) < 0.4).astype(np.int8)
    lane2 = (rng.random(L) < 0.4).astype(np.int8)
    uniforms = rng.random((steps + warmup) * 3)
    run_bkl_fenwick(lane1, lane2, alpha, beta, warmup, uniforms, 0)
    dt, e1, e2, _ = run_bkl_fenwick(lane1, lane2, alpha, beta, steps,
                                    uniforms, warmup * 3)
    return (e1 / dt, e2 / dt, np.mean(lane1), np.mean(lane2))


def ensemble_beta(alpha, betas, n_reps, seed0=0):
    """Return dict beta -> (n_reps,2) of (rho1,rho2) per replica."""
    rng = np.random.default_rng(seed0)
    tasks = [(alpha, b, L, STEPS, WARMUP, int(rng.integers(1e9)))
             for b in betas for _ in range(n_reps)]
    with ProcessPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(_one_replica, tasks))
    out = {}
    for i, b in enumerate(betas):
        reps = res[i * n_reps:(i + 1) * n_reps]
        out[b] = np.array([[r[2], r[3]] for r in reps])
    return out


def main():
    # ---- fig3: joint (rho1,rho2) points ----
    snapshot_betas = [0.23, 0.245, 0.255, 0.258, 0.2595, 0.262, 0.2685,
                      0.28, 0.95]
    anim_betas = np.linspace(0.04, 0.35, 40).tolist()
    all_betas = sorted(set(snapshot_betas) |
                       set(round(b, 6) for b in anim_betas))
    print(f"fig3: {len(all_betas)} betas x {N_REPS} reps on CPU", flush=True)
    pts = ensemble_beta(ALPHA, all_betas, N_REPS)
    np.savez(f"{OUT}/fig3_points.npz",
             **{f"b{int(b*10000)}": pts[b] for b in all_betas},
             betas=np.array(all_betas))
    print("fig3_points saved", flush=True)

    # ---- ssb: order param vs beta (from the same points) ----
    ssb_betas = [0.05, 0.08, 0.10, 0.12, 0.15, 0.2, 0.25, 0.3]
    print(f"ssb: {len(ssb_betas)} betas x {N_REPS} reps on CPU", flush=True)
    sp = ensemble_beta(ALPHA, ssb_betas, N_REPS)
    for b in ssb_betas:
        r1 = sp[b][:, 0]; r2 = sp[b][:, 1]
        d = r1 - r2
        dense = np.maximum(r1, r2); dilute = np.minimum(r1, r2)
        np.savez(f"{OUT}/ssb_beta{b}.npz",
                 dense=dense[:, None], dilute=dilute[:, None],
                 std=d[:, None], alpha=ALPHA, L=L)
        print(f"  ssb beta={b} saved", flush=True)
    print("ALL DATA READY")


if __name__ == "__main__":
    main()
