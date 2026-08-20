"""
Extend fig5 phase boundaries to large L (CPU). Run as a real file (not -c)
so multiprocessing's forkserver can re-import the driver. Writes merged
boundaries to results/fig5_big/fig5_boundaries_merged.npz.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    from scripts.run_bigger import fig5_large_L
    fig5_large_L(os.path.join(ROOT, "results", "fig5_big"), Ls=(800, 8000))
    print("fig5 DONE", flush=True)


if __name__ == "__main__":
    main()
