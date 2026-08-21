"""
Rerun L1000 and L500 fig3+ssb with the density-drift FIX, then reduce + plot.
Queued after the L1000 fig6 rerun frees the GPU.

Run:  python scripts/rerun_fig3_ssb_fixed.py [--Ls 1000 500]
Logs to results/rerun_fig3_ssb.log, auto-flushed.
"""
import os
import sys
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.remake_L import TeeLog
from scripts.unified_alpha09 import unified_alpha09, steps_per_site
from scripts.reduce_unified import group_chunks, reduce_fig3, reduce_ssb
from scripts.params_record import params_fig3_unified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", nargs="+", type=int, default=[1000, 500])
    ap.add_argument("--nrep", type=int, default=1024)
    ap.add_argument("--tag", default="unified")
    args = ap.parse_args()
    sys.stdout = TeeLog(os.path.join(ROOT, "results", "rerun_fig3_ssb.log"))
    print(f"RERUN fig3+ssb (density-fix) for Ls={args.Ls}", flush=True)
    for L in args.Ls:
        out = os.path.join(ROOT, "results", f"{args.tag}_L{L}")
        # high-res: use raw time-samples via unified_L1000 path if L>=1000
        if L >= 1000:
            from scripts.unified_L1000 import unified_L1000 as u1000
            # 128 reps, 100k steps/site (L-scaled), raw samples for fine bins
            u1000(L, out, nrep_per_beta=128, sample_every=100000,
                  chunk_raw=400)
            from scripts.reduce_L1000 import group_chunks as gc, reduce_fig3 as rf3, reduce_ssb as rsb
            by = gc(out)
            rf3(by, out, bins=96)
            rsb(by, out, 1)
        else:
            unified_alpha09(L, out, nrep_per_beta=args.nrep)
            by = group_chunks(out)
            reduce_fig3(by, out, 1)
            reduce_ssb(by, out, 1)
        params_fig3_unified(out, L, 128 if L >= 1000 else args.nrep,
                            steps_per_site(L), 5 if L >= 1000 else None,
                            20000, 100000 if L >= 1000 else 5000,
                            96 if L >= 1000 else 48)
        print(f"=== L={L} fig3+ssb rerun done ===", flush=True)
    print("ALL RERUN FIG3+SSB DONE", flush=True)


if __name__ == "__main__":
    main()
