"""
Ensemble GPU kernel: one CUDA thread per independent BKL(Fenwick) ASEP replica.

Layout is REPLICA-MAJOR: every per-replica array is laid out contiguously, and
thread `t` owns the slab `[t*L, (t+1)*L)`. Each thread keeps its own lane1/lane2,
rates[], and Fenwick tree, so no inter-thread communication is ever needed.

The workload is random-access (each thread touches its own chosen site), so the
access pattern is latency/branch bound; this layout benchmarks fastest on the
2060 SUPER. Uses numba.cuda xoroshiro128p (one independent stream per replica).
"""
import numpy as np
from math import log
from numba import cuda
from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_uniform_float64)


@cuda.jit(device=True)
def _site_rate(lane1, lane2, base, alpha, beta, L, i):
    """Rate of all moves involving site i of the replica starting at `base`."""
    r = 0.0
    if i < L - 1 and lane1[base + i] == 1 and lane1[base + i + 1] == 0:
        r += 1.0
    if i == 0 and lane1[base] == 0 and lane2[base] == 0:
        r += alpha
    if i == L - 1 and lane1[base + L - 1] == 1:
        r += beta
    if i > 0 and lane2[base + i] == 1 and lane2[base + i - 1] == 0:
        r += 1.0
    if i == L - 1 and lane2[base + L - 1] == 0 and lane1[base + L - 1] == 0:
        r += alpha
    if i == 0 and lane2[base] == 1:
        r += beta
    return r


@cuda.jit(device=True)
def _bit_build(rates, bit, base, L):
    for i in range(L):
        bit[base + i + 1] = rates[base + i]
    for i in range(1, L + 1):
        j = i + (i & -i)
        if j <= L:
            bit[base + j] += bit[base + i]


@cuda.jit(device=True)
def _bit_update(bit, base, i, delta, L):
    idx = i + 1
    while idx <= L:
        bit[base + idx] += delta
        idx += idx & -idx


@cuda.jit(device=True)
def _bit_find(bit, base, r, L):
    idx = 0
    bitmask = 1
    while bitmask << 1 <= L:
        bitmask <<= 1
    while bitmask:
        t = idx + bitmask
        if t <= L and bit[base + t] < r:
            idx = t
            r -= bit[base + t]
        bitmask >>= 1
    return idx


@cuda.jit(device=True)
def _execute(lane1, lane2, base, L, site, mtype):
    if mtype == 0:      # hop1_right
        lane1[base + site] = 0
        lane1[base + site + 1] = 1
    elif mtype == 1:    # hop2_left
        lane2[base + site] = 0
        lane2[base + site - 1] = 1
    elif mtype == 2:    # enter1
        lane1[base] = 1
    elif mtype == 3:    # enter2
        lane2[base + L - 1] = 1
    elif mtype == 4:    # exit1
        lane1[base + L - 1] = 0
        return 1, 0
    elif mtype == 5:    # exit2
        lane2[base] = 0
        return 0, 1
    return 0, 0


@cuda.jit(device=True)
def _pick_move(lane1, lane2, base, alpha, beta, L, site, r):
    cum = 0.0
    if site < L - 1 and lane1[base + site] == 1 and lane1[base + site + 1] == 0:
        cum += 1.0
        if r < cum:
            return _execute(lane1, lane2, base, L, site, 0)
    if site == 0 and lane1[base] == 0 and lane2[base] == 0:
        cum += alpha
        if r < cum:
            return _execute(lane1, lane2, base, L, site, 2)
    if site == L - 1 and lane1[base + L - 1] == 1:
        cum += beta
        if r < cum:
            return _execute(lane1, lane2, base, L, site, 4)
    if site > 0 and lane2[base + site] == 1 and lane2[base + site - 1] == 0:
        cum += 1.0
        if r < cum:
            return _execute(lane1, lane2, base, L, site, 1)
    if site == L - 1 and lane2[base + L - 1] == 0 and lane1[base + L - 1] == 0:
        cum += alpha
        if r < cum:
            return _execute(lane1, lane2, base, L, site, 3)
    if site == 0 and lane2[base] == 1:
        cum += beta
        if r < cum:
            return _execute(lane1, lane2, base, L, site, 5)
    return 0, 0


@cuda.jit
def _run_kernel(lane1, lane2, rates, bit, cur1, cur2, tot_time,
                alpha, beta, L, steps, states):
    t = cuda.grid(1)
    if t >= cur1.shape[0]:
        return
    base = t * L

    # init lanes at density ~0.4
    for i in range(L):
        u = xoroshiro128p_uniform_float64(states, t)
        lane1[base + i] = 1 if u < 0.4 else 0
        u = xoroshiro128p_uniform_float64(states, t)
        lane2[base + i] = 1 if u < 0.4 else 0

    for i in range(L):
        rates[base + i] = _site_rate(lane1, lane2, base, alpha, beta, L, i)
    _bit_build(rates, bit, base, L)

    # total rate kept as a per-thread scalar, updated incrementally on refresh
    total_rate = 0.0
    idx = L
    while idx > 0:
        total_rate += bit[base + idx]
        idx -= idx & -idx

    ttime = 0.0
    c1 = 0
    c2 = 0
    for step in range(steps):
        if total_rate == 0.0:
            ttime += 0.001
            continue

        u1 = xoroshiro128p_uniform_float64(states, t)
        ttime += -log(u1) / total_rate

        u2 = xoroshiro128p_uniform_float64(states, t)
        chosen = _bit_find(bit, base, u2 * total_rate, L)

        u3 = xoroshiro128p_uniform_float64(states, t)
        e1, e2 = _pick_move(lane1, lane2, base, alpha, beta, L, chosen,
                            u3 * rates[base + chosen])
        c1 += e1
        c2 += e2

        lo = chosen - 1
        if lo < 0:
            lo = 0
        hi = chosen + 1
        if hi > L - 1:
            hi = L - 1
        for i in range(lo, hi + 1):
            new_r = _site_rate(lane1, lane2, base, alpha, beta, L, i)
            old_r = rates[base + i]
            if new_r != old_r:
                delta = new_r - old_r
                _bit_update(bit, base, i, delta, L)
                total_rate += delta
                rates[base + i] = new_r

    tot_time[t] = ttime
    cur1[t] = c1
    cur2[t] = c2


def run_ensemble_cuda(alpha, beta, L, steps, n_replicas, seed=0, block=256):
    """Run n_replicas independent ASEP BKL-Fenwick sims on GPU (replica-major).
    Returns (cur1, cur2, ttime) arrays of shape (n_replicas,)."""
    lane1 = cuda.device_array(n_replicas * L, dtype=np.int8)
    lane2 = cuda.device_array(n_replicas * L, dtype=np.int8)
    rates = cuda.device_array(n_replicas * L, dtype=np.float32)
    bit = cuda.device_array(n_replicas * (L + 1), dtype=np.float32)
    cur1 = cuda.device_array(n_replicas, dtype=np.float64)
    cur2 = cuda.device_array(n_replicas, dtype=np.float64)
    ttime = cuda.device_array(n_replicas, dtype=np.float64)
    states = create_xoroshiro128p_states(n_replicas, seed=seed)

    grid = (n_replicas + block - 1) // block
    _run_kernel[grid, block](lane1, lane2, rates, bit, cur1, cur2, ttime,
                             alpha, beta, L, steps, states)
    cuda.synchronize()
    return cur1.copy_to_host(), cur2.copy_to_host(), ttime.copy_to_host()
