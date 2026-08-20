"""
Multi-L fig2 + fig6 comparison with the FIXED classifier (MC-by-current-
saturation + L-adaptive SSB threshold). Redoes L200 fig2 and adds L1000,
with a longer budget so large L equilibrates.

fig2: phase diagram grid (full + zoom) for L in {200, 500, 1000}.
fig6: currents/densities vs beta for alphas {0.1,0.8,0.9}, same Ls.
Budgets: light (fig2/fig6) at 3000 steps/site (equilibration for large L
improves with a longer run per the user's note).

Run:  python scripts/fig2_fig6_multil.py [--Ls 200 500 1000]
Logs to results/fig2_fig6_multil.log, saves params.json per L.
"""
import os
import sys
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.remake_L import scan_phase_diagram, scan_fig6, TeeLog
from scripts.params_record import params_fig2, params_fig6

LIGHT_SPS = 3000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", nargs="+", type=int, default=[200, 500, 1000])
    args = ap.parse_args()
    sys.stdout = TeeLog(os.path.join(ROOT, "results", "fig2_fig6_multil.log"))
    print(f"fig2+fig6 multi-L comparison: {args.Ls}", flush=True)
    for L in args.Ls:
        print(f"=== L={L} (fig2 3000 steps/site, fig6 3000 steps/site) ===",
              flush=True)
        out = os.path.join(ROOT, "results", f"L{L}")
        scan_phase_diagram(L, os.path.join(out, "fig2"), L * LIGHT_SPS, L * 300)
        scan_fig6(L, os.path.join(out, "fig6"), L * LIGHT_SPS, L * 300,
                  n_reps=32)
        params_fig2(os.path.join(out, "fig2"), L, LIGHT_SPS, 300, 16)
        params_fig6(os.path.join(out, "fig6"), L, LIGHT_SPS, 300, 32,
                    [0.1, 0.8, 0.9])
    print("ALL fig2+fig6 multi-L DONE", flush=True)


if __name__ == "__main__":
    main()
