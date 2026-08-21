"""
Redo the phase diagrams (fig2) for L in {200, 500, 1000} with the FIXED
HD/LD classifier. The old grids were classified with a bug that never
detected the broken-symmetry HD/LD phase.

Uses the fixed classify_phase (observables.py) via scan_phase_diagram_gpu.
Budget scaled by L: steps/site such that the SSB dense basin is reached
(needed for HD/LD). Logs to results/redo_fig2.log, writes params.json.

Run:  python scripts/redo_fig2.py [--Ls 200 500 1000]
"""
import os
import sys
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.remake_L import scan_phase_diagram, TeeLog
from scripts.params_record import params_fig2

# steps/site: L=200,500 cheap (HD/LD basin reached fast); L=1000 longer
SPS = {200: 10000, 500: 20000, 1000: 100000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", nargs="+", type=int, default=[200, 500, 1000])
    args = ap.parse_args()
    sys.stdout = TeeLog(os.path.join(ROOT, "results", "redo_fig2.log"))
    print(f"REDO fig2 (fixed HD/LD classifier) for {args.Ls}", flush=True)
    for L in args.Ls:
        sps = SPS.get(L, 10000)
        out = os.path.join(ROOT, "results", f"L{L}", "fig2")
        print(f"L={L}: {sps} steps/site", flush=True)
        scan_phase_diagram(L, out, L * sps, L * (sps // 5))
        params_fig2(out, L, sps, sps // 5, 16)
        # report phase counts
        g = np.load(os.path.join(out, "grid_full.npy"), allow_pickle=True)
        labels, counts = np.unique(g, return_counts=True)
        print(f"  L={L} grid: {dict(zip(labels, counts))}", flush=True)
    print("ALL REDO FIG2 DONE", flush=True)


if __name__ == "__main__":
    main()
