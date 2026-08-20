"""
Write a params.json describing the MC/scan parameters for every figure run,
so the professor can see exactly what was done for each plot.

Each figure's run data lives in a results/<fig> dir; we drop a params.json
there. Re-running any figure with different params overwrites it.
"""
import os
import json
import numpy as np


def write_params(path, **params):
    """Dump params to <path>/params.json (JSON-serializable)."""
    os.makedirs(path, exist_ok=True)
    def _jsonify(o):
        if isinstance(o, (np.ndarray, np.generic)):
            return o.tolist()
        return o
    clean = {k: _jsonify(v) for k, v in params.items()}
    with open(os.path.join(path, "params.json"), "w") as f:
        json.dump(clean, f, indent=2, sort_keys=True)
    print(f"  params -> {path}/params.json", flush=True)
    return clean


def load_params(path):
    """Read <path>/params.json if present."""
    p = os.path.join(path, "params.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Known runs we already did this session (write accurate records).
# ---------------------------------------------------------------------------

def params_fig5_corrected(out_dir):
    return write_params(
        out_dir,
        figure="fig5_phase_boundaries_vs_L",
        paper="Pronina & Kolomeisky 2007, Fig 5",
        boundary_a="asym->LD at alpha=0.9 (beta scan)",
        boundary_b="LD->MC at beta=1.0 (alpha scan)",
        Ls=[200, 500, 1000, 2000, 4000, 8000],
        n_reps=8,
        n_boot=200,
        steps_rule="steps = int(L*1e4)  (10k steps/site)",
        steps_by_L={L: int(L * 1e4) for L in [200, 500, 1000, 2000, 4000, 8000]},
        warmup=1_000_000,
        betas_a=np.linspace(0.05, 0.6, 24),
        alphas_b=np.linspace(0.2, 0.95, 24),
        sample_every_a=200,
        sample_every_b=50,
        classifier="classify_point (MC by current saturation + L-adaptive SSB threshold)",
        backend="CPU ProcessPool (run_bkl_fenwick)",
        n_workers_rule="memory-aware _n_workers",
        seed=0,
        theory_hdld="alpha/(1+alpha+alpha^2)=0.3321",
        theory_mc="2*beta/(4*beta-1)=0.6667",
    )


def params_fig2(out_dir, L, steps_per_site, warmup_site, n_reps=16):
    return write_params(
        out_dir,
        figure="fig2_phase_diagram",
        paper="Pronina & Kolomeisky 2007, Fig 2",
        L=L,
        grid="full (31x31) + zoom (31x21)",
        alphas_full=np.linspace(0.05, 0.95, 31),
        betas_full=np.linspace(0.05, 0.95, 31),
        alphas_zoom=np.linspace(0.05, 0.95, 31),
        betas_zoom=np.linspace(0.2, 0.4, 21),
        steps_per_site=steps_per_site,
        steps=L * steps_per_site,
        warmup=L * warmup_site,
        sample_every=400,
        n_reps=n_reps,
        seeds=[0, 1],
        classifier="scan_phase_diagram_gpu (density-distribution over ensemble)",
    )


def params_fig6(out_dir, L, steps_per_site, warmup_site, n_reps, alphas):
    return write_params(
        out_dir,
        figure="fig6_currents_densities",
        L=L,
        alphas=list(alphas),
        betas=np.linspace(0.05, 0.95, 30),
        steps_per_site=steps_per_site,
        steps=L * steps_per_site,
        warmup=L * warmup_site,
        sample_every=400,
        n_reps=n_reps,
        seed=0,
        observables="J1,J2,rho1,rho2,dense,dilute (+ errors)",
    )


def params_fig3_unified(out, L, nrep_per_beta, steps_per_site, chunks,
                        chunk_site, sample_every, bins):
    return write_params(
        out,
        figure="fig3_P_rho1_rho2 (unified with ssb)",
        L=L,
        alpha=0.9,
        nrep_per_beta=nrep_per_beta,
        steps_per_site=steps_per_site,
        nchunks=chunks,
        chunk_steps=L * chunk_site,
        sample_every=sample_every,
        bins=bins,
        backend="GPU run_ensemble_cuda_continue (continuous trajectory)",
        ssb_betas=[0.05, 0.08, 0.10, 0.12, 0.15, 0.2, 0.25, 0.3],
    )


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("write params for existing runs", flush=True)
    params_fig5_corrected(os.path.join(ROOT, "results", "fig5_corrected"))
    params_fig2(os.path.join(ROOT, "results", "L200", "fig2"), 200, 3000, 300, 16)
    params_fig2(os.path.join(ROOT, "results", "L500", "fig2"), 500, 3000, 300, 16)
    params_fig6(os.path.join(ROOT, "results", "L200", "fig6"), 200, 3000, 300, 32, [0.1, 0.8, 0.9])
    params_fig6(os.path.join(ROOT, "results", "L500", "fig6"), 500, 3000, 300, 32, [0.1, 0.8, 0.9])
    params_fig3_unified(os.path.join(ROOT, "results", "unified_L500"),
                        500, 1024, 50000, 2, 20000, 5000, 48)
    print("done")
