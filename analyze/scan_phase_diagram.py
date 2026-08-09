"""
Phase diagram scanner: classify (alpha, beta, L) parameter points by phase.

Strategy:
1. For each (alpha, beta) point, run a simulation
2. Measure currents J1, J2 and densities rho1, rho2
3. Classify the phase based on the measured observables
"""
import numpy as np
from asep import TwoChannelASEP
from asep.observables import classify_phase
from itertools import product
import pickle
from pathlib import Path
import time


def scan_phase_diagram(
    alpha_values,
    beta_values,
    L=100,
    n_steps=200000,
    sample_every=200,
    warmup=20000,
    seed=42,
    verbose=True,
    save_path=None,
    parallel=False,
    n_jobs=4,
):
    """
    Scan (alpha, beta) parameter space and classify phases.

    Parameters
    ----------
    alpha_values, beta_values : array-like
        Parameter scan grid
    L : int
        Lattice size
    n_steps : int
        MC steps per point
    sample_every : int
        Sample density every N steps
    warmup : int
        Warmup steps to discard
    parallel : bool
        If True, use joblib for parallel execution
    n_jobs : int
        Number of parallel jobs

    Returns
    -------
    results : dict
    """
    results = {
        'alpha': [], 'beta': [], 'phase': [],
        'rho1': [], 'rho2': [], 'J1': [], 'J2': [],
        'asymmetry': []
    }

    grid = list(product(alpha_values, beta_values))
    n_points = len(grid)
    t_start = time.time()

    def run_one(alpha, beta):
        sim = TwoChannelASEP(L=L, alpha=alpha, beta=beta, seed=seed)
        sim.run(n_steps=n_steps, sample_every=sample_every, warmup=warmup)
        J1, J2 = sim.get_currents()
        rho1, rho2 = sim.get_bulk_densities()
        label, asymm = classify_phase(J1, J2, rho1, rho2, alpha, beta, L)
        return label, asymm, rho1, rho2, J1, J2

    for i, (alpha, beta) in enumerate(grid):
        label, asymm, rho1, rho2, J1, J2 = run_one(alpha, beta)

        results['alpha'].append(alpha)
        results['beta'].append(beta)
        results['phase'].append(label)
        results['rho1'].append(rho1)
        results['rho2'].append(rho2)
        results['J1'].append(J1)
        results['J2'].append(J2)
        results['asymmetry'].append(asymm)

        if verbose:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (n_points - i - 1)
            print(f"[{i+1}/{n_points}] α={alpha:.3f}, β={beta:.3f} -> {label} "
                  f"(J={ (J1+J2)/2:.3f}, Δρ={asymm:.3f}) "
                  f"[{elapsed:.1f}s, ETA: {eta:.0f}s]")

    for k in ['alpha', 'beta', 'rho1', 'rho2', 'J1', 'J2', 'asymmetry']:
        results[k] = np.array(results[k])
    results['L'] = L
    results['n_steps'] = n_steps

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

    return results


def plot_phase_diagram(results, save_path=None):
    """Plot phase diagram with phase labels and asymmetry maps."""
    import matplotlib.pyplot as plt

    alphas = np.unique(results['alpha'])
    betas = np.unique(results['beta'])
    L = results['L']

    def to_grid(arr):
        grid = np.full((len(betas), len(alphas)), np.nan)
        for a, b, v in zip(results['alpha'], results['beta'], arr):
            ai = np.argwhere(alphas == a)[0, 0]
            bi = np.argwhere(betas == b)[0, 0]
            grid[bi, ai] = v
        return grid

    phase_map = {
        'LD': 0, 'MC': 1, 'HD/LD': 2, 'LD/HD': 2,
        'LD/LD': 3, 'HD': 4
    }
    phase_nums = np.array([phase_map.get(p, -1) for p in results['phase']])
    phase_grid = to_grid(phase_nums)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Phase diagram
    phase_labels = ['LD', 'MC', 'HD/LD', 'LD/LD', 'HD']
    im0 = axes[0].imshow(phase_grid, origin='lower', aspect='auto',
                         extent=[alphas[0], alphas[-1], betas[0], betas[-1]],
                         cmap='tab10', vmin=-0.5, vmax=4.5)
    axes[0].set_xlabel('α (entrance rate)')
    axes[0].set_ylabel('β (exit rate)')
    axes[0].set_title(f'Phase Diagram (L={L})')

    # Current asymmetry
    delta_J = to_grid(results['J1']) - to_grid(results['J2'])
    im1 = axes[1].imshow(delta_J, origin='lower', aspect='auto',
                         extent=[alphas[0], alphas[-1], betas[0], betas[-1]],
                         cmap='RdBu_r', vmin=-0.15, vmax=0.15)
    axes[1].set_xlabel('α')
    axes[1].set_title('J₁ − J₂ (current asymmetry)')
    plt.colorbar(im1, ax=axes[1], label='ΔJ')

    # Density asymmetry
    delta_rho = to_grid(results['rho2']) - to_grid(results['rho1'])
    im2 = axes[2].imshow(delta_rho, origin='lower', aspect='auto',
                         extent=[alphas[0], alphas[-1], betas[0], betas[-1]],
                         cmap='RdBu_r', vmin=-0.4, vmax=0.4)
    axes[2].set_xlabel('α')
    axes[2].set_title('ρ₂ − ρ₁ (density asymmetry)')
    plt.colorbar(im2, ax=axes[2], label='Δρ')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Quick test: scan a small region
    alphas = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    betas = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])

    print(f"Scanning {len(alphas)}x{len(betas)} parameter grid (L=200)...")
    results = scan_phase_diagram(
        alphas, betas,
        L=200,
        n_steps=200000,
        sample_every=200,
        warmup=20000,
        verbose=True,
    )

    print("\nResults summary:")
    for p in sorted(set(results['phase'])):
        mask = np.array(results['phase']) == p
        print(f"  {p}: {mask.sum()} points")
