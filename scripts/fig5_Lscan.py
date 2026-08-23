"""
fig5 — phase boundaries vs channel length L (paper Fig 5), CPU multi-core.

Panels
------
(a) asym -> LD boundary in beta vs L, at alpha=0.9   (HD/LD-LD/LD vs LD)
(b) LD -> MC boundary in alpha vs L, at beta=1.0     (LD vs MC)

Protocol (paper Sec 3, lines "The number of effective Monte Carlo steps per
lattice site ... between 2e7 and 5e8 ... first 5% omitted"):
  * steps per replica: enough to reach steady state (verified ~1e4/site in our
    BKL kernel) with margin, i.e. ~ up to 5e4/site (within the paper band);
  * an ensemble of independent replicas classifies broken vs symmetric;
  * each (alpha, beta, L) point is fully independent and memory-streaming
    (uniforms generated in blocks), so large L is safe;
  * ALL per-replica observables are saved so a GPU rerun or an L extension
    can reclassify/add points without re-simulating.

Output (results/fig5_Lscan/):
  a_beta_L{L}.npz, b_alpha_L{L}.npz : per-L raw data (per-rep rho/dense/dilute
                                      + labels + asym + detected transition)
  fig5_phase_boundary_vs_L.png, fig5_boundaries_summary.npz

Run (CPU, 31 workers default):
    uv run python scripts/fig5_Lscan.py --L 1000,4000,10000 --nreps 48
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "fig5_Lscan")
os.makedirs(OUT, exist_ok=True)

from asep.parallel import scan_points
from fig5_boundaries import classify_point, _smooth_labels  # noqa


def _budget(L):
    """(total_steps, warmup) per L.

    Steps/site: 2e5 — above our measured BKL convergence (~1e4-5e4/site; our
    probes reach the true HD/LD state and it saturates there) while staying
    inside the paper's band (2e7-5e8/site is for tight MEAN error bars, not
    needed to LOCATE an SSB boundary, which we resolve via the replica
    ensemble). Warmup discarded to guarantee steady state before sampling.
    """
    total = int(max(2.0e8, 2.0e5 * L))
    warmup = int(min(0.7 * total, max(1.5e7, 2.0e4 * L)))
    return total, warmup


def _scan_one_dimension(points, fixed, is_beta, L, n_reps, workers, seed0=0):
    """Scan `points` (the varying axis) holding `fixed`.
    is_beta=True  -> (a): points are beta, fixed is alpha.
    is_beta=False -> (b): points are alpha, fixed is beta."""
    total, warmup = _budget(L)
    sample_every = max(1000, (total - warmup) // 100)
    rng = np.random.default_rng(seed0)
    tasks = []
    for p in points:
        a = fixed if is_beta else p
        b = p if is_beta else fixed
        for r in range(n_reps):
            tasks.append((a, b, L, total, warmup, sample_every,
                          int(rng.integers(1e12))))
    res = scan_points(tasks, n_workers=workers, desc=f"{'(a)' if is_beta else '(b)'} L={L}")

    labels, asyms = [], []
    # per-point raw observables
    R1, R2, DEN, DIL, J1m, J2m = [], [], [], [], [], []
    for j in range(len(points)):
        reps = res[j * n_reps:(j + 1) * n_reps]
        r1 = np.array([r[2] for r in reps], dtype=float)
        r2 = np.array([r[3] for r in reps], dtype=float)
        J1 = np.array([r[0] for r in reps], dtype=float)
        J2 = np.array([r[1] for r in reps], dtype=float)
        a = fixed if is_beta else points[j]
        b = points[j] if is_beta else fixed
        lab, asym = classify_point(r1.tolist(), r2.tolist(),
                                   J1.tolist(), J2.tolist(), a, b, L)
        labels.append(lab)
        asyms.append(asym)
        R1.append(r1); R2.append(r2)
        DEN.append(np.maximum(r1, r2))
        DIL.append(np.minimum(r1, r2))
        J1m.append(J1); J2m.append(J2)
    labels = _smooth_labels(labels)

    if is_beta:
        trans = None
        for j in range(1, len(labels)):
            if labels[j - 1] != "LD" and labels[j] == "LD":
                trans = 0.5 * (points[j - 1] + points[j])
    else:
        trans = None
        for j in range(1, len(labels)):
            if labels[j - 1] == "LD" and labels[j] == "MC":
                trans = 0.5 * (points[j - 1] + points[j])

    np.savez(os.path.join(OUT, f"{'a_beta' if is_beta else 'b_alpha'}_L{L}.npz"),
             axis=points, labels=np.array(labels), asym=np.array(asyms, dtype=float),
             transition=np.float64(trans) if trans is not None else np.float64(np.nan),
             rho1=np.array(R1, dtype=object), rho2=np.array(R2, dtype=object),
             dense=np.array(DEN, dtype=object), dilute=np.array(DIL, dtype=object),
             J1=np.array(J1m, dtype=object), J2=np.array(J2m, dtype=object),
             L=L, n_reps=n_reps, warmup=warmup, total=total, fixed=float(fixed),
             is_beta=np.bool_(is_beta))
    print(f"  L={L} {'(a) beta-boundary' if is_beta else '(b) alpha-boundary'}"
          f" = {trans}  labels={labels}", flush=True)
    return trans


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", default="1000,4000,10000")
    ap.add_argument("--nreps", type=int, default=48)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    args = ap.parse_args()
    Ls = [int(x) for x in args.L.split(",")]
    nreps, workers = args.nreps, args.workers
    print(f"L={Ls} nreps={nreps} workers={workers}", flush=True)

    alpha_fixed = 0.9
    betas = np.linspace(0.20, 0.40, 16)          # (a) around paper ~0.258-0.28
    beta_boundary = []
    for L in Ls:
        beta_boundary.append(_scan_one_dimension(
            betas, alpha_fixed, True, L, nreps, workers, seed0=100_000 + L))

    beta_fixed = 1.0
    alphas = np.linspace(0.55, 0.95, 16)         # (b) around paper ~0.7-0.8
    alpha_boundary = []
    for L in Ls:
        alpha_boundary.append(_scan_one_dimension(
            alphas, beta_fixed, False, L, nreps, workers, seed0=200_000 + L))

    a_theory = alpha_fixed / (1 + alpha_fixed + alpha_fixed ** 2)
    b_theory = 2 * beta_fixed / (4 * beta_fixed - 1)

    def _clean(v):
        return np.where(np.isnan(v), np.nan, v)

    a_boundary = np.array(beta_boundary, dtype=float)
    b_boundary = np.array(alpha_boundary, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.errorbar(Ls, _clean(a_boundary), fmt="o-", ms=6, capsize=4, capthick=1,
                label="MC asym$\\to$LD boundary")
    ax.axhline(a_theory, color="k", ls=":", lw=1.2,
               label=f"MFT (eq 23) = {a_theory:.3f}")
    ax.set_xscale("log"); ax.grid(alpha=0.2, which="both")
    ax.set_xlabel(r"$L$"); ax.set_ylabel(r"$\beta$ boundary")
    ax.set_title(rf"(a) asym$-$LD boundary, $\alpha={alpha_fixed}$")
    ax.set_ylim(0.2, 0.4)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.errorbar(Ls, _clean(b_boundary), fmt="o-", ms=6, capsize=4, capthick=1,
                label="MC LD$/$MC boundary")
    ax.axhline(b_theory, color="k", ls=":", lw=1.2,
               label=f"MFT MC$/$LD (eq 10) = {b_theory:.3f}")
    ax.set_xscale("log"); ax.grid(alpha=0.2, which="both")
    ax.set_xlabel(r"$L$"); ax.set_ylabel(r"$\alpha$ boundary")
    ax.set_title(rf"(b) MC$/$LD boundary, $\beta={beta_fixed}$")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_phase_boundary_vs_L.png"), dpi=150)
    np.savez(os.path.join(OUT, "fig5_boundaries_summary.npz"),
             Ls=np.array(Ls), beta_boundary=a_boundary,
             alpha_boundary=b_boundary, theory_beta=a_theory,
             theory_alpha=b_theory, n_reps=nreps)
    print(f"\nBeta boundary (asym->LD): {beta_boundary}", flush=True)
    print(f"Alpha boundary (LD->MC) : {alpha_boundary}", flush=True)
    print(f"saved {OUT}/fig5_phase_boundary_vs_L.png", flush=True)


if __name__ == "__main__":
    main()
