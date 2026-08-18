"""
Isolate the per-replica GPU bottleneck: global-memory Fenwick latency vs the
serial BKL dependency chain.

Key constraint discovered: a Fenwick tree for L=1000 needs 8KB/thread. That
does NOT fit in registers (~1KB/thread) and shared memory is per-BLOCK not
per-thread (~64 bytes/thread at full occupancy). So one replica's full tree
cannot live in fast per-thread memory on a 2060S — it must live in global
memory (or spill to local=global-backed).

This test checks the hypothesis "global-memory latency is the main cost" by
varying L: at small L the tree fits in registers (cuda.local.array), at large L
it spills. If per-replica throughput is roughly FLAT vs L, the cost is the
serial dependency chain, not memory latency. If it drops with L, latency wins.
"""
import numpy as np
from math import log
from numba import cuda
from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_uniform_float64)


@cuda.jit(device=True)
def _site_rate_sm(lane1, lane2, alpha, beta, L, i):
    r = 0.0
    if i < L - 1 and lane1[i] == 1 and lane1[i + 1] == 0:
        r += 1.0
    if i == 0 and lane1[0] == 0 and lane2[0] == 0:
        r += alpha
    if i == L - 1 and lane1[L - 1] == 1:
        r += beta
    if i > 0 and lane2[i] == 1 and lane2[i - 1] == 0:
        r += 1.0
    if i == L - 1 and lane2[L - 1] == 0 and lane1[L - 1] == 0:
        r += alpha
    if i == 0 and lane2[0] == 1:
        r += beta
    return r


@cuda.jit(device=True)
def _bit_find_sm(bit, r, L):
    idx = 0
    bitmask = 1
    while bitmask << 1 <= L:
        bitmask <<= 1
    while bitmask:
        t = idx + bitmask
        if t <= L and bit[t] < r:
            idx = t
            r -= bit[t]
        bitmask >>= 1
    return idx


@cuda.jit(device=True)
def _bit_update_sm(bit, i, delta, L):
    idx = i + 1
    while idx <= L:
        bit[idx] += delta
        idx += idx & -idx


@cuda.jit(device=True)
def _execute_sm(lane1, lane2, L, site, mtype):
    if mtype == 0:
        lane1[site] = 0; lane1[site + 1] = 1
    elif mtype == 1:
        lane2[site] = 0; lane2[site - 1] = 1
    elif mtype == 2:
        lane1[0] = 1
    elif mtype == 3:
        lane2[L - 1] = 1
    elif mtype == 4:
        lane1[L - 1] = 0
    elif mtype == 5:
        lane2[0] = 0


@cuda.jit(device=True)
def _pick_move_sm(lane1, lane2, alpha, beta, L, site, r):
    cum = 0.0
    if site < L - 1 and lane1[site] == 1 and lane1[site + 1] == 0:
        cum += 1.0
        if r < cum:
            _execute_sm(lane1, lane2, L, site, 0); return 0, 0
    if site == 0 and lane1[0] == 0 and lane2[0] == 0:
        cum += alpha
        if r < cum:
            _execute_sm(lane1, lane2, L, site, 2); return 0, 0
    if site == L - 1 and lane1[L - 1] == 1:
        cum += beta
        if r < cum:
            _execute_sm(lane1, lane2, L, site, 4); return 1, 0
    if site > 0 and lane2[site] == 1 and lane2[site - 1] == 0:
        cum += 1.0
        if r < cum:
            _execute_sm(lane1, lane2, L, site, 1); return 0, 0
    if site == L - 1 and lane2[L - 1] == 0 and lane1[L - 1] == 0:
        cum += alpha
        if r < cum:
            _execute_sm(lane1, lane2, L, site, 3); return 0, 0
    if site == 0 and lane2[0] == 1:
        cum += beta
        if r < cum:
            _execute_sm(lane1, lane2, L, site, 5); return 0, 1
    return 0, 0


@cuda.jit
def _run_kernel_local(lane1, lane2, rates, bit, cur1, cur2, tot_time,
                      alpha_arr, beta_arr, L, steps, warmup, states,
                      sample_every, stats_acc):
    """Each thread uses cuda.local.array (registers when small, else spills)."""
    t = cuda.grid(1)
    if t >= cur1.shape[0]:
        return
    alpha = alpha_arr[t]
    beta = beta_arr[t]

    lr1 = cuda.local.array(4096, dtype=np.int8)
    lr2 = cuda.local.array(4096, dtype=np.int8)
    lrates = cuda.local.array(4096, dtype=np.float64)
    lbit = cuda.local.array(4097, dtype=np.float64)

    n1 = 0; n2 = 0
    for i in range(L):
        u = xoroshiro128p_uniform_float64(states, t)
        lr1[i] = 1 if u < 0.4 else 0; n1 += lr1[i]
        u = xoroshiro128p_uniform_float64(states, t)
        lr2[i] = 1 if u < 0.4 else 0; n2 += lr2[i]

    for i in range(L):
        lrates[i] = _site_rate_sm(lr1, lr2, alpha, beta, L, i)
    for i in range(L):
        lbit[i + 1] = lrates[i]
    for i in range(1, L + 1):
        j = i + (i & -i)
        if j <= L:
            lbit[j] += lbit[i]

    total_rate = 0.0
    idx = L
    while idx > 0:
        total_rate += lbit[idx]; idx -= idx & -idx

    ttime = 0.0; c1 = 0; c2 = 0
    for step in range(steps + warmup):
        if total_rate == 0.0:
            ttime += 0.001; continue
        u1 = xoroshiro128p_uniform_float64(states, t)
        ttime += -log(u1) / total_rate
        u2 = xoroshiro128p_uniform_float64(states, t)
        chosen = _bit_find_sm(lbit, u2 * total_rate, L)
        u3 = xoroshiro128p_uniform_float64(states, t)
        e1, e2 = _pick_move_sm(lr1, lr2, alpha, beta, L, chosen,
                               u3 * lrates[chosen])
        c1 += e1; c2 += e2
        lo = chosen - 1
        if lo < 0: lo = 0
        hi = chosen + 1
        if hi > L - 1: hi = L - 1
        for i in range(lo, hi + 1):
            new_r = _site_rate_sm(lr1, lr2, alpha, beta, L, i)
            old_r = lrates[i]
            if new_r != old_r:
                delta = new_r - old_r
                _bit_update_sm(lbit, i, delta, L)
                total_rate += delta
                lrates[i] = new_r

    tot_time[t] = ttime
    cur1[t] = c1
    cur2[t] = c2


def run_local(L, steps, n_replicas, seed=0, block=256):
    nrep = int(n_replicas); L = int(L); steps = int(steps); seed = int(seed)
    block = int(block)
    cur1 = cuda.device_array(nrep, dtype=np.float64)
    cur2 = cuda.device_array(nrep, dtype=np.float64)
    ttime = cuda.device_array(nrep, dtype=np.float64)
    alpha_d = cuda.to_device(np.full(nrep, 0.9, dtype=np.float64))
    beta_d = cuda.to_device(np.full(nrep, 0.1, dtype=np.float64))
    states = create_xoroshiro128p_states(nrep, seed=seed)
    # dummy global arrays (unused by local kernel, but kept for signature)
    lane1 = cuda.device_array(nrep * L, dtype=np.int8)
    lane2 = cuda.device_array(nrep * L, dtype=np.int8)
    rates = cuda.device_array(nrep * L, dtype=np.float64)
    bit = cuda.device_array(nrep * (L + 1), dtype=np.float64)
    stats = cuda.device_array(nrep * 8, dtype=np.float64)
    grid = (nrep + block - 1) // block
    _run_kernel_local[grid, block](lane1, lane2, rates, bit, cur1, cur2, ttime,
                                   alpha_d, beta_d, L, steps, 0, states, 0, stats)
    cuda.synchronize()
    return cur1.copy_to_host(), cur2.copy_to_host(), ttime.copy_to_host()


def main():
    import time
    steps = 2_000_000
    nrep = 512
    print("per-thread LOCAL-memory Fenwick: flat vs L tells us if it's latency.")
    print("(at small L the tree may fit in registers; at large L it spills to")
    print(" local/global memory)")
    for L in [50, 100, 200, 500, 1000, 2000]:
        t0 = time.perf_counter()
        run_local(L, steps, nrep, seed=0)
        t1 = time.perf_counter()
        per = steps / (t1 - t0) / 1e3
        print(f"L={L:4d}: {per:7.1f} kstep/s/rep")
    print("DONE")


if __name__ == "__main__":
    main()
