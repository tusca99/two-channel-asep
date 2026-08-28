#!/usr/bin/env python3
"""Plot tau(L) flipping times: basin-projection dwell detector.

tau = median dwell time in one SSB basin (|rho1-rho2| >= dmin), in MC steps.
"No flip within the run" -> lower-bound arrow at the run length.
Deep band (beta<=0.22): dmin=0.35. Edge (beta=0.26): dmin=0.15.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DIRS = [os.path.join(ROOT, "results", "tau"),
        os.path.join(ROOT, "results", "tau_smoothsign")]
OUT = os.path.join(ROOT, "results", "tau", "tau_L.png")

LS = [200, 500, 1000, 2000]
BETAS = [0.18, 0.22, 0.26]
COLORS = {0.18: "#1f77b4", 0.22: "#ff7f0e", 0.26: "#d62728"}


def collect():
    data = {}
    for d in DIRS:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not (f.startswith("tau_L") and f.endswith(".npz")):
                continue
            z = np.load(os.path.join(d, f))
            L, b, tau, steps = (int(z["L"]), float(z["beta"]),
                                float(z["tau"]), int(z["n_steps"]))
            if not np.isnan(tau):
                data[(L, b)] = (tau, steps, False)
            else:  # no flip -> lower bound, keep the LONGEST run
                if (L, b) not in data or data[(L, b)][0] < steps:
                    data[(L, b)] = (steps, steps, True)
    return data


def main():
    data = collect()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for b in BETAS:
        xs, ys, lb = [], [], []
        for L in LS:
            if (L, b) not in data:
                continue
            v, steps, is_lb = data[(L, b)]
            xs.append(L)
            ys.append(v)
            lb.append(is_lb)
        if not xs:
            continue
        xs, ys, lb = np.array(xs), np.array(ys), np.array(lb)
        ax.plot(xs[~lb], ys[~lb], "o-", color=COLORS[b], lw=1.6, ms=6,
                label=rf"$\beta={b}$")
        if lb.any():
            ax.plot(xs[lb], ys[lb], "v", color=COLORS[b], ms=8)
            for x, y in zip(xs[lb], ys[lb]):
                ax.annotate("no flip\n(lower bound)", (x, y),
                            textcoords="offset points", xytext=(6, -2),
                            fontsize=6.5, color=COLORS[b], alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\tau$ (MC steps, median dwell)")
    ax.set_title(rf"Flipping time $\tau(L)$, $\alpha=0.9$ — SSB basin dwell")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("saved", OUT)


if __name__ == "__main__":
    main()