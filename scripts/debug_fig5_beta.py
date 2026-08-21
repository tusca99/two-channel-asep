"""Debug the fig5 L=4000/8000 HD/LD boundary (nan)."""
import os, sys, warnings
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from scripts.fig5_boundaries import classify_point, _n_workers, phase_boundary_in_beta


def main():
    for L in [2000, 4000, 8000]:
        steps = int(L * 1e4)
        nw = _n_workers(L, steps)
        betas = np.linspace(0.05, 0.6, 24)
        grid = phase_boundary_in_beta([0.9], betas, L, steps, 1_000_000, 200,
                                      8, n_workers=nw)
        print(f"\nL={L} alpha=0.9 beta sweep:")
        for j in range(len(betas)):
            J1s, J2s, r1s, r2s = grid[0][j]
            lab, asym = classify_point(r1s, r2s, J1s, J2s, 0.9, betas[j], L)
            dens = np.maximum(np.array(r1s), np.array(r2s)).mean()
            print(f"  beta={betas[j]:.3f}: {lab:6s} dense={dens:.3f}")


if __name__ == "__main__":
    main()
