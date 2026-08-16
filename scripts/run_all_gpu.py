"""
Unified GPU scan: produce ALL the MC data for every figure in one pass.

Runs the GPU ensemble kernel over the union of parameter points needed by the
figures, and saves raw data to results/gpu/*.npz. The plotting scripts
(plot_all.py) then only READ these files — no MC is re-run at plot time.

Outputs (results/gpu/):
  fig2/           phase diagram grid (full + zoom)
  fig6/alpha*.npz currents/densities vs beta
  fig3_points.npz joint (rho1,rho2) samples for the P(rho1,rho2) figure
  ssb_beta*.npz   SSB order parameter scan vs beta

Usage:
  python scripts/run_all_gpu.py          # do the full scan (slow, GPU)
  python scripts/run_all_gpu.py --load   # skip; just re-plot from saved data
"""
import os
import sys
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "gpu")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.9
L = 1000


def scan_phase_diagram():
    """fig2: classify phases on (alpha,beta) grid (full + zoom)."""
    from asep.parallel import scan_phase_diagram_gpu
    import numpy as np

    alphas = np.linspace(0.05, 0.95, 31)
    betas = np.linspace(0.05, 0.95, 31)
    grid = scan_phase_diagram_gpu(alphas, betas, L, 2_000_000, 200_000, 400,
                                  n_reps=16, seed=0)
    d = os.path.join(OUT, "fig2")
    os.makedirs(d, exist_ok=True)
    np.save(f"{d}/grid_full.npy", grid, allow_pickle=True)
    np.save(f"{d}/alphas_full.npy", alphas)
    np.save(f"{d}/betas_full.npy", betas)

    zoom_a = np.linspace(0.05, 0.95, 31)
    zoom_b = np.linspace(0.2, 0.4, 21)
    zoom = scan_phase_diagram_gpu(zoom_a, zoom_b, L, 2_000_000, 200_000, 400,
                                  n_reps=16, seed=1)
    np.save(f"{d}/grid_zoom.npy", zoom, allow_pickle=True)
    np.save(f"{d}/alphas_zoom.npy", zoom_a)
    np.save(f"{d}/betas_zoom.npy", zoom_b)
    print("fig2 saved", flush=True)


def scan_fig6():
    """fig6: currents & densities vs beta for alpha in {0.1,0.8,0.9}."""
    from asep.parallel import scan_beta_gpu

    betas = np.linspace(0.05, 0.95, 30)
    d = os.path.join(OUT, "fig6")
    os.makedirs(d, exist_ok=True)
    for alpha in [0.1, 0.8, 0.9]:
        res = scan_beta_gpu(alpha, betas, L, 3_000_000, 300_000, 400,
                            n_reps=32, seed=0)
        (J1, J2, rho1, rho2, eJ1, eJ2, erho1, erho2,
         dense, dilute, edense, edilute) = res
        np.savez(f"{d}/alpha{alpha}.npz",
                 J1=J1, J2=J2, rho1=rho1, rho2=rho2,
                 eJ1=eJ1, eJ2=eJ2, erho1=erho1, erho2=erho2,
                 dense=dense, dilute=dilute, edense=edense, edilute=edilute,
                 betas=betas)
        print(f"fig6 alpha={alpha} saved", flush=True)


def scan_fig3():
    """fig3: joint (rho1,rho2) samples over beta range for P(rho1,rho2)."""
    from asep.cuda_ensemble import run_ensemble_cuda

    snapshot_betas = [0.23, 0.245, 0.255, 0.258, 0.2595, 0.262, 0.2685,
                      0.28, 0.95]
    anim_betas = np.linspace(0.04, 0.35, 40).tolist()
    all_betas = sorted(set(snapshot_betas) |
                       set(round(b, 6) for b in anim_betas))
    n_reps = 2048
    steps = 2_000_000
    sample_every = 50000

    pts = {}
    chunk_betas = 8
    for c0 in range(0, len(all_betas), chunk_betas):
        chunk = all_betas[c0:c0 + chunk_betas]
        nrep = len(chunk) * n_reps
        bb = np.repeat(np.array(chunk), n_reps)
        res = run_ensemble_cuda(0.0, 0.0, L, steps, nrep, seed=7,
                                sample_every=sample_every, warmup=200000,
                                alphas=np.full(nrep, ALPHA), betas=bb)
        for i, b in enumerate(chunk):
            sl = slice(i * n_reps, (i + 1) * n_reps)
            pts[b] = np.stack([res["rho1"][sl], res["rho2"][sl]], axis=-1)
        print(f"  fig3 chunk {c0//chunk_betas+1}: beta {chunk[0]:.4f}.."
              f"{chunk[-1]:.4f}", flush=True)

    np.savez(f"{OUT}/fig3_points.npz",
             **{f"b{int(b*10000)}": pts[b] for b in all_betas},
             betas=np.array(all_betas))
    print("fig3 saved", flush=True)


def scan_ssb():
    """SSB: diff & std(rho1-rho2) vs beta, long runs at L."""
    from asep.cuda_ensemble import run_ensemble_cuda

    betas = [0.05, 0.08, 0.10, 0.12, 0.15, 0.2, 0.25, 0.3]
    nrep = 128
    chunk = 2_000_000
    nchunks = 50
    sample_every = 50000

    for b in betas:
        aa = np.full(nrep, ALPHA)
        bb = np.full(nrep, b)
        dens_buf = np.zeros((nrep, nchunks))
        dilu_buf = np.zeros((nrep, nchunks))
        std_buf = np.zeros((nrep, nchunks))
        cur1 = np.zeros(nrep); cur2 = np.zeros(nrep); ttime = np.zeros(nrep)
        for ci in range(nchunks):
            res = run_ensemble_cuda(0.0, 0.0, L, chunk, nrep, seed=1000 + ci,
                                    sample_every=sample_every, warmup=0,
                                    alphas=aa, betas=bb)
            for i in range(nrep):
                dens_buf[i, ci] = max(res["rho1"][i], res["rho2"][i])
                dilu_buf[i, ci] = min(res["rho1"][i], res["rho2"][i])
                std_buf[i, ci] = res["std_diff"][i]
            cur1 += res["cur1"]; cur2 += res["cur2"]; ttime += res["ttime"]
        np.savez(f"{OUT}/ssb_beta{b}.npz", dense=dens_buf, dilute=dilu_buf,
                 std=std_buf, cur1=cur1, cur2=cur2, ttime=ttime,
                 alpha=ALPHA, L=L, chunk=chunk, nchunks=nchunks, nrep=nrep)
        print(f"  ssb beta={b} saved", flush=True)


def main():
    if "--load" not in sys.argv:
        scan_phase_diagram()
        scan_fig6()
        scan_fig3()
        scan_ssb()
    print("ALL GPU DATA READY")


if __name__ == "__main__":
    main()
