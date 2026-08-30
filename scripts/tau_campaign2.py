"""
tau(L) campaign v2: MULTI-TRAJECTORY flip statistics with real error bars.

What improved vs scripts/tau_flipping.py (12 single trajectories):
  * n_traj independent trajectories per (L, beta) point -> error bars on tau
    (bootstrap over trajectories) instead of one number.
  * tqdm progress per trajectory + full log (results/tau2.log), incremental
    save after EVERY trajectory (crash-safe).
  * dwell detection: basin projection (|rho1-rho2| >= dmin, sign of gap),
    tau converted to MC STEPS via n_steps/total_time (fixes the units bug
    found in the v1 audit: BKL time is rate-weighted physical time).
  * points: beta=0.26 (band edge) ladder L=200..2000; beta=0.22 (deep) only
    L=200,500 (bounds elsewhere waste budget); beta=0.18 control at L=200.

Budget (~3.5 h on 12 cores):
  L=200 : 12 traj x 1e8   (~2 min/traj/core)
  L=500 : 12 traj x 4e8   (~8 min)
  L=1000:  8 traj x 2e9   (~40 min)
  L=2000:  8 traj x 6e9   (~2 h)

Run:   uv run python scripts/tau_campaign2.py
Log:   results/tau2.log (tee, auto-flush) + tqdm progress bars.
Data:  results/tau2/tau2_L<L>_b<B>[_k<k>].npz
"""
import os
import sys
import time
import argparse
from multiprocessing import Process

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tqdm import tqdm
from asep.model import TwoChannelASEP

OUT = os.path.join(ROOT, "results", "tau2")
LOG = os.path.join(ROOT, "results", "tau2.log")
ALPHA = 0.9
DMIN = {0.26: 0.15, 0.22: 0.35, 0.18: 0.35}

# (L, beta, n_steps, n_traj)
PLAN = [
    (200, 0.26, int(1e8), 12),
    (500, 0.26, int(4e8), 12),
    (1000, 0.26, int(2e9), 8),
    (2000, 0.26, int(6e9), 8),
    (200, 0.22, int(1e8), 12),
    (500, 0.22, int(4e8), 12),
    (200, 0.18, int(1e8), 12),
]


def tee_log():
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s); st.flush()
        def flush(self):
            for st in self.streams:
                st.flush()
    logf = open(LOG, "a", buffering=1)
    sys.stdout = Tee(sys.stdout, logf)
    sys.stderr = Tee(sys.stderr, logf)


def one_trajectory(L, beta, n_steps, seed, sample_every, dmin, tag):
    """Run one trajectory; save dwells; return (tau_steps, n_flips, wall_s)."""
    t0 = time.time()
    m = TwoChannelASEP(L=L, alpha=ALPHA, beta=beta, seed=seed)
    warmup = int(0.02 * n_steps)
    m.run(n_steps, sample_every=sample_every, warmup=warmup)
    j = np.array(m._joint_samples)
    d = j[:, 0] - j[:, 1]
    t = np.array(m._time_samples)
    dw_phys = dwells_basin(d, t, dmin)
    conv = n_steps / m.total_time          # MC steps per phys-time unit
    dw_steps = dw_phys * conv
    tau = float(np.median(dw_steps)) if len(dw_steps) else float("nan")
    np.savez_compressed(
        os.path.join(OUT, f"{tag}.npz"),
        d=d, t=t, L=L, beta=beta, alpha=ALPHA, n_steps=n_steps, seed=seed,
        total_time=m.total_time, wall_s=time.time() - t0, dwells=dw_steps,
        tau=tau, mean_d=float(d.mean()), std_d=float(d.std()), dmin=dmin,
        conv=conv)
    return tau, len(dw_steps), time.time() - t0


def dwells_basin(d, t, dmin):
    """Basin dwell times in physical units; flip = sign switch of filled basin."""
    if len(d) < 100:
        return np.array([])
    basin = np.where(np.abs(d) >= dmin, np.sign(d), 0)
    filled = basin.copy()
    last = 0.0
    for i in range(len(filled)):
        if filled[i] != 0:
            last = filled[i]
        else:
            filled[i] = last
    cross = np.where(filled[1:] * filled[:-1] < 0)[0]
    if len(cross) < 2:
        return np.array([])
    return np.diff(t[cross])


def worker(L, beta, n_steps, seed, sample_every, dmin, tag, tqdm_lock):
    try:
        tau, nf, wall = one_trajectory(L, beta, n_steps, seed,
                                       sample_every, dmin, tag)
        tqdm_lock.acquire()
        try:
            print(f"[done] {tag}: tau={tau:.4g} steps flips={nf} "
                  f"wall={wall/60:.1f}min", flush=True)
        finally:
            tqdm_lock.release()
    except Exception as e:  # never kill the campaign
        with tqdm_lock:
            print(f"[FAIL] {tag}: {e!r}", flush=True)


def bootstrap_tau(all_dwells, n=2000, rng=None):
    """Bootstrap CI of tau=median over the concatenation of dwell lists."""
    rng = rng or np.random.default_rng(0)
    meds = []
    pooled = np.concatenate(all_dwells)
    for _ in range(n):
        pick = rng.integers(0, len(pooled), len(pooled))
        meds.append(np.median(pooled[pick]))
    return np.percentile(meds, [16, 84])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tee_log()
    print(f"[tau2] campaign started {time.strftime('%F %T')} "
          f"({args.parallel} workers)", flush=True)

    from threading import Lock
    tqdm_lock = Lock()
    for L, beta, n_steps, n_traj in PLAN:
        dmin = DMIN[beta]
        sample_every = max(int(n_steps / 60000), L)   # ~60k samples/traj
        # trajectory slots not yet saved -> todo list for this (L, beta)
        todo = []
        for k in range(n_traj):
            tag = f"tau2_L{L}_b{str(beta).replace('.','')}_k{k}"
            if not os.path.exists(os.path.join(OUT, tag + ".npz")):
                todo.append((k, tag))
        if not todo:
            print(f"[skip] L={L} beta={beta} already complete", flush=True)
            continue
        print(f"[stage] L={L} beta={beta}: {len(todo)} traj x "
              f"{n_steps:.1e} steps, dmin={dmin}, "
              f"samples every {sample_every}", flush=True)
        slots = [None] * args.parallel
        pbar = tqdm(total=len(todo), desc=f"L={L} b={beta}", unit="traj",
                    position=0, leave=True)
        it = iter(todo)
        # simple fixed-slot scheduler
        try:
            while True:
                for si in range(args.parallel):
                    if slots[si] is None or not slots[si].is_alive():
                        if slots[si] is not None:
                            pbar.update(1)
                        k, tag = next(it, (None, None))
                        if k is None:
                            # wait for the rest
                            for s2 in slots:
                                if s2 is not None:
                                    s2.join()
                                    pbar.update(1)
                            raise StopIteration
                        # stagger seeds deterministically
                        seed = 8675309 + 104729 * k + 7 * L + int(beta * 100)
                        slots[si] = Process(target=worker, args=(
                            L, beta, n_steps, seed, sample_every, dmin,
                            tag, tqdm_lock))
                        slots[si].start()
                time.sleep(2)
        except StopIteration:
            pass
        pbar.close()

        # aggregate + bootstrap for this stage
        import glob
        files = sorted(glob.glob(os.path.join(
            OUT, f"tau2_L{L}_b{str(beta).replace('.','')}_k*.npz")))
        dws, taus = [], []
        for f in files:
            z = np.load(f)
            if z["dwells"].size:
                dws.append(z["dwells"])
                taus.append(float(z["tau"]))
        if dws:
            pooled = np.concatenate(dws)
            lo, hi = bootstrap_tau(dws)
            print(f"[stage-done] L={L} beta={beta}: n_traj={len(dws)} "
                  f"n_flips_total={pooled.size} tau_median="
                  f"{np.median(pooled):.4g} steps "
                  f"(bootstrap 68%: [{lo:.3g},{hi:.3g}])", flush=True)
        else:
            print(f"[stage-done] L={L} beta={beta}: NO flips anywhere "
                  f"-> tau > run lengths (bounds only)", flush=True)
    print(f"[tau2] campaign finished {time.strftime('%F %T')}", flush=True)


if __name__ == "__main__":
    main()