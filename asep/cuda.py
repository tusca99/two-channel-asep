"""
Numba CUDA Monte Carlo for the two-channel ASEP.

Two kernel designs:

1. `mc_scan_kernel` (one thread per grid point): simple, but each thread's
   lattice lives in global memory, which is slow. Good for correctness, poor
   for throughput.

2. `mc_scan_kernel_block` (one BLOCK per grid point, threads = sites):
   the lattice is loaded into SHARED memory (fast, replicated per block),
   and the total-rate sum is a parallel reduction across the block's threads.
   This is the high-throughput design: shared memory is ~100x faster than
   global, and many blocks run concurrently for high occupancy.

Randomness
----------
Each block owns an independent xoroshiro128p RNG state (seeded from a
caller-supplied seed + block index), so results are reproducible for a given
seed. The kernel consumes uniforms via xoroshiro128p_uniform_float64.
"""
import numpy as np
from numba import cuda
from numba.cuda import random as curand


# --- Device helpers (operate on shared-memory lattices) -------------------

@cuda.jit(device=True)
def _total_rate(lane1, lane2, alpha, beta, L):
    """Sum the rates of all possible moves (pass 1)."""
    total = 0.0
    for i in range(L - 1):
        if lane1[i] == 1 and lane1[i + 1] == 0:
            total += 1.0
        if lane2[i + 1] == 1 and lane2[i] == 0:
            total += 1.0
    if lane1[0] == 0 and lane2[0] == 0:
        total += alpha
    if lane2[L - 1] == 0 and lane1[L - 1] == 0:
        total += alpha
    if lane1[L - 1] == 1:
        total += beta
    if lane2[0] == 1:
        total += beta
    return total


@cuda.jit(device=True)
def _select_and_execute(lane1, lane2, alpha, beta, L, r):
    """
    Pass 2: walk the lattice accumulating rates until reaching r, then
    execute the selected move. Returns (exited_ch1, exited_ch2).
    """
    cum = 0.0
    for i in range(L - 1):
        if lane1[i] == 1 and lane1[i + 1] == 0:
            cum += 1.0
            if r <= cum:
                lane1[i] = 0
                lane1[i + 1] = 1
                return 0, 0
        if lane2[i + 1] == 1 and lane2[i] == 0:
            cum += 1.0
            if r <= cum:
                lane2[i + 1] = 0
                lane2[i] = 1
                return 0, 0
    if lane1[0] == 0 and lane2[0] == 0:
        cum += alpha
        if r <= cum:
            lane1[0] = 1
            return 0, 0
    if lane2[L - 1] == 0 and lane1[L - 1] == 0:
        cum += alpha
        if r <= cum:
            lane2[L - 1] = 1
            return 0, 0
    if lane1[L - 1] == 1:
        cum += beta
        if r <= cum:
            lane1[L - 1] = 0
            return 1, 0
    if lane2[0] == 1:
        cum += beta
        if r <= cum:
            lane2[0] = 0
            return 0, 1
    return 0, 0


# --- Kernel 1: one thread per grid point (simple, global-mem lattice) -----

@cuda.jit
def mc_scan_kernel(lane1_all, lane2_all, alphas, betas,
                   n_steps, warmup, sample_every,
                   out_J1, out_J2, out_rho1, out_rho2,
                   rng_states):
    """One independent system per thread (grid point)."""
    tid = cuda.grid(1)
    if tid >= alphas.shape[0]:
        return

    alpha = alphas[tid]
    beta = betas[tid]
    L = lane1_all.shape[1]

    lane1 = lane1_all[tid]
    lane2 = lane2_all[tid]

    total_time = 0.0
    n_exit1 = 0
    n_exit2 = 0
    rho1_sum = 0.0
    rho2_sum = 0.0
    n_samples = 0

    for step in range(n_steps):
        total_rate = _total_rate(lane1, lane2, alpha, beta, L)
        if total_rate == 0.0:
            total_time += 0.001
        else:
            u1 = curand.xoroshiro128p_uniform_float64(rng_states, tid)
            total_time += -np.log(u1) / total_rate
            u2 = curand.xoroshiro128p_uniform_float64(rng_states, tid)
            r = u2 * total_rate
            e1, e2 = _select_and_execute(lane1, lane2, alpha, beta, L, r)
            n_exit1 += e1
            n_exit2 += e2

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


# --- Kernel 2: one block per grid point, lattice in shared memory ---------

@cuda.jit
def mc_scan_kernel_block(lane1_all, lane2_all, alphas, betas,
                        n_steps, warmup, sample_every,
                        out_J1, out_J2, out_rho1, out_rho2,
                        rng_states):
    """
    One independent system per BLOCK; lattice in shared memory.

    grid = (n_points,), block = (block,). Threads cooperatively load the
    lattice into shared memory, then thread 0 runs the sequential MC loop
    against shared memory (fast). The total-rate sum is a parallel reduction
    across the block's threads.
    """
    bid = cuda.blockIdx.x
    if bid >= alphas.shape[0]:
        return
    tid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x
    L = lane1_all.shape[1]

    alpha = alphas[bid]
    beta = betas[bid]

    # Shared-memory lattices (compile-time max size MAX_L)
    s_lane1 = cuda.shared.array(MAX_L, dtype=np.int8)
    s_lane2 = cuda.shared.array(MAX_L, dtype=np.int8)
    s_partial = cuda.shared.array(MAX_BLOCK, dtype=np.float64)

    # Cooperative load from global to shared
    for i in range(tid, L, nthreads):
        s_lane1[i] = lane1_all[bid, i]
        s_lane2[i] = lane2_all[bid, i]
    cuda.syncthreads()

    total_time = 0.0
    n_exit1 = 0
    n_exit2 = 0
    rho1_sum = 0.0
    rho2_sum = 0.0
    n_samples = 0

    for step in range(n_steps):
        # Parallel reduction of total rate across threads
        # Each thread sums its slice of bulk hops
        local = 0.0
        for i in range(tid, L - 1, nthreads):
            if s_lane1[i] == 1 and s_lane1[i + 1] == 0:
                local += 1.0
            if s_lane2[i + 1] == 1 and s_lane2[i] == 0:
                local += 1.0
        # Thread 0 adds boundary terms
        if tid == 0:
            if s_lane1[0] == 0 and s_lane2[0] == 0:
                local += alpha
            if s_lane2[L - 1] == 0 and s_lane1[L - 1] == 0:
                local += alpha
            if s_lane1[L - 1] == 1:
                local += beta
            if s_lane2[0] == 1:
                local += beta
        # Block reduction (simple: thread 0 sums all partials via shared)
        s_partial[tid] = local
        cuda.syncthreads()
        if tid == 0:
            total_rate = 0.0
            for j in range(nthreads):
                total_rate += s_partial[j]
            if total_rate == 0.0:
                total_time += 0.001
            else:
                u1 = curand.xoroshiro128p_uniform_float64(rng_states, bid)
                total_time += -np.log(u1) / total_rate
                u2 = curand.xoroshiro128p_uniform_float64(rng_states, bid)
                r = u2 * total_rate
                e1, e2 = _select_and_execute(s_lane1, s_lane2, alpha, beta, L, r)
                n_exit1 += e1
                n_exit2 += e2
                if step > warmup and step % sample_every == 0:
                    s1 = 0.0
                    s2 = 0.0
                    for i in range(L):
                        s1 += s_lane1[i]
                        s2 += s_lane2[i]
                    rho1_sum += s1 / L
                    rho2_sum += s2 / L
                    n_samples += 1
        cuda.syncthreads()

    if tid == 0:
        out_J1[bid] = n_exit1 / total_time if total_time > 0 else 0.0
        out_J2[bid] = n_exit2 / total_time if total_time > 0 else 0.0
        out_rho1[bid] = rho1_sum / n_samples if n_samples > 0 else 0.0
        out_rho2[bid] = rho2_sum / n_samples if n_samples > 0 else 0.0


# Compile-time constants for shared arrays
MAX_L = 4096
MAX_BLOCK = 1024


def run_scan(alphas, betas, L, n_steps, warmup, sample_every, seed=0,
             block=256, use_block_kernel=True):
    """
    Run a batch of independent (alpha, beta) systems on the GPU.

    Parameters
    ----------
    alphas, betas : array-like of equal length (one system per element)
    L : int, lattice size (must be <= MAX_L for the block kernel)
    n_steps, warmup, sample_every : MC parameters
    seed : int, RNG seed for reproducibility
    use_block_kernel : bool, use the shared-memory block kernel (default True)

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

    # One RNG state per block (or per thread for the simple kernel)
    if use_block_kernel:
        rng_states = curand.create_xoroshiro128p_states(n_points, seed=seed)
        mc_scan_kernel_block[n_points, block](
            d_lane1, d_lane2, d_alphas, d_betas,
            n_steps, warmup, sample_every,
            d_J1, d_J2, d_rho1, d_rho2, rng_states,
        )
    else:
        rng_states = curand.create_xoroshiro128p_states(n_points, seed=seed)
        grid = (n_points + block - 1) // block
        mc_scan_kernel[grid, block](
            d_lane1, d_lane2, d_alphas, d_betas,
            n_steps, warmup, sample_every,
            d_J1, d_J2, d_rho1, d_rho2, rng_states,
        )
    cuda.synchronize()

    return (d_J1.copy_to_host(), d_J2.copy_to_host(),
            d_rho1.copy_to_host(), d_rho2.copy_to_host())
