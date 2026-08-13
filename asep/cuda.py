"""
Numba CUDA Monte Carlo for the two-channel ASEP.

Strategy: one CUDA thread per (alpha, beta) grid point. Each thread runs the
full sequential Gillespie MC loop for its own independent system, so the GPU's
parallelism comes from running many grid points simultaneously. This is ideal
for phase-diagram scans (thousands of independent systems).

Randomness
----------
Each thread owns an independent xoroshiro128p RNG state (seeded from a
caller-supplied seed + thread index), so results are reproducible for a given
seed. The kernel consumes uniforms via xoroshiro128p_uniform_float64.
"""
import numpy as np
from numba import cuda
from numba.cuda import random as curand


@cuda.jit(device=True)
def _collect_moves(lane1, lane2, alpha, beta, L,
                   move_types, move_sites, move_rates):
    """Collect all possible moves and their rates (device function)."""
    n_moves = 0
    for i in range(L - 1):
        if lane1[i] == 1 and lane1[i + 1] == 0:
            move_types[n_moves] = 0
            move_sites[n_moves] = i
            move_rates[n_moves] = 1.0
            n_moves += 1
        if lane2[i + 1] == 1 and lane2[i] == 0:
            move_types[n_moves] = 1
            move_sites[n_moves] = i + 1
            move_rates[n_moves] = 1.0
            n_moves += 1
    if lane1[0] == 0 and lane2[0] == 0:
        move_types[n_moves] = 2
        move_sites[n_moves] = 0
        move_rates[n_moves] = alpha
        n_moves += 1
    if lane2[L - 1] == 0 and lane1[L - 1] == 0:
        move_types[n_moves] = 3
        move_sites[n_moves] = L - 1
        move_rates[n_moves] = alpha
        n_moves += 1
    if lane1[L - 1] == 1:
        move_types[n_moves] = 4
        move_sites[n_moves] = L - 1
        move_rates[n_moves] = beta
        n_moves += 1
    if lane2[0] == 1:
        move_types[n_moves] = 5
        move_sites[n_moves] = 0
        move_rates[n_moves] = beta
        n_moves += 1
    return n_moves


@cuda.jit(device=True)
def _execute_move(lane1, lane2, mtype, site, L):
    """Execute a single move on the lattices (device function)."""
    if mtype == 0:
        lane1[site] = 0
        lane1[site + 1] = 1
    elif mtype == 1:
        lane2[site] = 0
        lane2[site - 1] = 1
    elif mtype == 2:
        lane1[0] = 1
    elif mtype == 3:
        lane2[L - 1] = 1
    elif mtype == 4:
        lane1[L - 1] = 0
    elif mtype == 5:
        lane2[0] = 0


# cuda.local.array needs a compile-time constant size, so we cap the lattice
# size at MAX_L and guard L <= MAX_L at launch time.
MAX_L = 4096


@cuda.jit
def mc_scan_kernel_fixed(lane1_all, lane2_all, alphas, betas,
                         n_steps, warmup, sample_every,
                         out_J1, out_J2, out_rho1, out_rho2,
                         rng_states):
    """
    One independent system per thread, with compile-time MAX_L scratch.

    Requires L <= MAX_L. Each thread runs n_steps of Gillespie MC and writes
    its steady-state currents and bulk densities.
    """
    tid = cuda.grid(1)
    if tid >= alphas.shape[0]:
        return

    alpha = alphas[tid]
    beta = betas[tid]
    L = lane1_all.shape[1]

    lane1 = lane1_all[tid]
    lane2 = lane2_all[tid]

    move_types = cuda.local.array(8194, dtype=np.int8)
    move_sites = cuda.local.array(8194, dtype=np.int64)
    move_rates = cuda.local.array(8194, dtype=np.float64)

    total_time = 0.0
    n_exit1 = 0
    n_exit2 = 0
    rho1_sum = 0.0
    rho2_sum = 0.0
    n_samples = 0

    for step in range(n_steps):
        n_moves = _collect_moves(lane1, lane2, alpha, beta, L,
                                 move_types, move_sites, move_rates)
        if n_moves == 0:
            total_time += 0.001
        else:
            total_rate = 0.0
            for j in range(n_moves):
                total_rate += move_rates[j]
            u1 = curand.xoroshiro128p_uniform_float64(rng_states, tid)
            dt = -np.log(u1) / total_rate
            total_time += dt
            u2 = curand.xoroshiro128p_uniform_float64(rng_states, tid)
            r = u2 * total_rate
            cum = 0.0
            selected = n_moves - 1
            for j in range(n_moves):
                cum += move_rates[j]
                if r <= cum:
                    selected = j
                    break
            mtype = move_types[selected]
            site = move_sites[selected]
            if mtype == 4:
                n_exit1 += 1
            elif mtype == 5:
                n_exit2 += 1
            _execute_move(lane1, lane2, mtype, site, L)

        if step > warmup and step % sample_every == 0:
            s1 = 0.0
            s2 = 0.0
            for i in range(L):
                s1 += lane1[i]
                s2 += lane2[i]
            rho1_sum += s1 / L
            rho2_sum += s2 / L
            n_samples += 1

    out_J1[tid] = n_exit1 / total_time if total_time > 0 else 0.0
    out_J2[tid] = n_exit2 / total_time if total_time > 0 else 0.0
    out_rho1[tid] = rho1_sum / n_samples if n_samples > 0 else 0.0
    out_rho2[tid] = rho2_sum / n_samples if n_samples > 0 else 0.0


def run_scan(alphas, betas, L, n_steps, warmup, sample_every, seed=0,
             block=256):
    """
    Run a batch of independent (alpha, beta) systems on the GPU.

    Parameters
    ----------
    alphas, betas : array-like of equal length (one system per element)
    L : int, lattice size (must be <= MAX_L)
    n_steps, warmup, sample_every : MC parameters
    seed : int, RNG seed for reproducibility

    Returns
    -------
    (J1, J2, rho1, rho2) : numpy arrays of shape (n_points,)
    """
    alphas = np.ascontiguousarray(alphas, dtype=np.float64)
    betas = np.ascontiguousarray(betas, dtype=np.float64)
    n_points = alphas.shape[0]
    assert L <= MAX_L, f"L={L} exceeds MAX_L={MAX_L}"

    d_lane1 = cuda.device_array((n_points, L), dtype=np.int8)
    d_lane2 = cuda.device_array((n_points, L), dtype=np.int8)
    d_alphas = cuda.to_device(alphas)
    d_betas = cuda.to_device(betas)
    d_J1 = cuda.device_array(n_points, dtype=np.float64)
    d_J2 = cuda.device_array(n_points, dtype=np.float64)
    d_rho1 = cuda.device_array(n_points, dtype=np.float64)
    d_rho2 = cuda.device_array(n_points, dtype=np.float64)

    # One RNG state per thread, seeded deterministically
    rng_states = curand.create_xoroshiro128p_states(n_points, seed=seed)

    grid = (n_points + block - 1) // block
    mc_scan_kernel_fixed[grid, block](
        d_lane1, d_lane2, d_alphas, d_betas,
        n_steps, warmup, sample_every,
        d_J1, d_J2, d_rho1, d_rho2, rng_states,
    )
    cuda.synchronize()

    return (d_J1.copy_to_host(), d_J2.copy_to_host(),
            d_rho1.copy_to_host(), d_rho2.copy_to_host())
