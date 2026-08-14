"""
Helpers for embarrassingly-parallel Monte Carlo scans.

Each (alpha, beta) grid point is an independent simulation, so phase-diagram,
current, and density scans can be parallelized across CPU cores.
"""
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from asep import TwoChannelASEP


def _run_point(task):
    """Run one (alpha, beta, L, params) point; return (J1, J2, rho1, rho2)."""
    alpha, beta, L, n_steps, warmup, sample_every, seed = task
    sim = TwoChannelASEP(L=L, alpha=alpha, beta=beta, seed=seed)
    sim.run(n_steps=n_steps, sample_every=sample_every, warmup=warmup)
    J1, J2 = sim.get_currents()
    r1, r2 = sim.get_bulk_densities()
    return J1, J2, r1, r2


def _run_point_samples(task):
    """Run one point; return (J1, J2, rho1, rho2, joint_samples)."""
    alpha, beta, L, n_steps, warmup, sample_every, seed = task
    sim = TwoChannelASEP(L=L, alpha=alpha, beta=beta, seed=seed)
    sim.run(n_steps=n_steps, sample_every=sample_every, warmup=warmup)
    J1, J2 = sim.get_currents()
    r1, r2 = sim.get_bulk_densities()
    return J1, J2, r1, r2, sim.get_joint_density_samples()


def scan_points(tasks, n_workers=None, desc="scan"):
    """
    Run a list of independent (alpha, beta, L, n_steps, warmup, sample_every, seed)
    tasks in parallel and return their (J1, J2, rho1, rho2) results, preserving order.
    Shows a tqdm progress bar.
    """
    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_run_point, t): j for j, t in enumerate(tasks)}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc=desc):
            results[futures[fut]] = fut.result()
    return results


def scan_points_samples(tasks, n_workers=None, desc="scan"):
    """
    Like scan_points but also returns the joint density samples per point,
    for the density-distribution phase classification.
    """
    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_run_point_samples, t): j for j, t in enumerate(tasks)}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc=desc):
            results[futures[fut]] = fut.result()
    return results


def make_tasks(alphas, betas, L, n_steps, warmup, sample_every, seed=0):
    """Build a task list for a full (alphas x betas) grid scan."""
    rng = np.random.default_rng(seed)
    tasks = []
    for a in alphas:
        for b in betas:
            tasks.append((a, b, L, n_steps, warmup, sample_every,
                          int(rng.integers(1e9))))
    return tasks


def grid_to_labels(results, alphas, betas, classify_fn):
    """Reshape flat results into a 2D grid of phase labels via classify_fn."""
    grid = np.empty((len(alphas), len(betas)), dtype=object)
    k = 0
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            J1, J2, r1, r2 = results[k]
            grid[i, j] = classify_fn(J1, J2, r1, r2, a, b)
            k += 1
    return grid
