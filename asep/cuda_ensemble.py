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
    """Execute move; returns (exited_ch1, exited_ch2, d_occ1, d_occ2)."""
    if mtype == 0:      # hop1_right (occupancy unchanged within lane1)
        lane1[base + site] = 0
        lane1[base + site + 1] = 1
        return 0, 0, 0, 0
    elif mtype == 1:    # hop2_left
        lane2[base + site] = 0
        lane2[base + site - 1] = 1
        return 0, 0, 0, 0
    elif mtype == 2:    # enter1
        lane1[base] = 1
        return 0, 0, 1, 0
    elif mtype == 3:    # enter2
        lane2[base + L - 1] = 1
        return 0, 0, 0, 1
    elif mtype == 4:    # exit1
        lane1[base + L - 1] = 0
        return 1, 0, -1, 0
    elif mtype == 5:    # exit2
        lane2[base] = 0
        return 0, 1, 0, -1
    return 0, 0, 0, 0


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
    return 0, 0, 0, 0


@cuda.jit(device=True)
def _advance_replica(lane1, lane2, rates, bit, cur1, cur2, tot_time,
                     t, alpha, beta, L, steps, warmup, states,
                     sample_every, stats_acc, init_lattice,
                     raw_samples, sample_count, capture_raw):
    """Run one replica's BKL-Fenwick trajectory; shared by the ensemble kernels.

    `raw_samples`/`sample_count` may be passed as zero-length (never None) so
    numba can type them. If `capture_raw`, each sample also writes (rho1,rho2)
    into raw_samples[t, s] for high-resolution P(rho1,rho2).
    """
    base = t * L

    # occupancy counters for O(1) bulk-density updates
    n1 = 0
    n2 = 0
    if init_lattice:
        for i in range(L):
            u = xoroshiro128p_uniform_float64(states, t)
            lane1[base + i] = 1 if u < 0.4 else 0
            n1 += lane1[base + i]
            u = xoroshiro128p_uniform_float64(states, t)
            lane2[base + i] = 1 if u < 0.4 else 0
            n2 += lane2[base + i]
        for i in range(L):
            rates[base + i] = _site_rate(lane1, lane2, base, alpha, beta, L, i)
        _bit_build(rates, bit, base, L)
        # total rate kept as a per-thread scalar, updated incrementally
        idx = L
        total_rate = 0.0
        while idx > 0:
            total_rate += bit[base + idx]
            idx -= idx & -idx
    else:
        # continuation: lattice/Fenwick/RNG persisted from the previous launch;
        # only rebuild the total-rate scalar (Fenwick tree is already built).
        for i in range(L):
            n1 += lane1[base + i]
            n2 += lane2[base + i]
        idx = L
        total_rate = 0.0
        while idx > 0:
            total_rate += bit[base + idx]
            idx -= idx & -idx

    ttime = 0.0
    c1 = 0
    c2 = 0
    n_sample = 0
    for step in range(steps + warmup):
        if total_rate == 0.0:
            ttime += 0.001
            continue

        u1 = xoroshiro128p_uniform_float64(states, t)
        ttime += -log(u1) / total_rate

        u2 = xoroshiro128p_uniform_float64(states, t)
        chosen = _bit_find(bit, base, u2 * total_rate, L)

        u3 = xoroshiro128p_uniform_float64(states, t)
        e1, e2, dn1, dn2 = _pick_move(lane1, lane2, base, alpha, beta, L, chosen,
                                      u3 * rates[base + chosen])
        c1 += e1
        c2 += e2
        n1 += dn1
        n2 += dn2

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

        # accumulate bulk-density statistics on-device (O(1) per sample)
        if (sample_every > 0 and step >= warmup
                and (step - warmup) % sample_every == 0):
            # Recompute occupancy DIRECTLY from the lattice at each sample.
            # The running integer counters n1/n2 can drift from the true
            # occupancy over very long runs (seen as rho2<0 or >1 at L=1000),
            # so we don't trust them for the density: sum the lattice instead.
            # O(L) per sample is negligible vs ~1e5 steps between samples.
            n1 = 0
            n2 = 0
            for i in range(L):
                n1 += lane1[base + i]
                n2 += lane2[base + i]
            r1 = n1 / L
            r2 = n2 / L
            d = r1 - r2
            mm = r1 if r1 > r2 else r2
            mn = r1 if r1 < r2 else r2
            # per-replica running sums: rho1_sum, rho1_sq, rho2_sum, rho2_sq,
            # diff_sq, dense_sum, dilute_sum, n
            stats_acc[t * 8 + 0] += r1
            stats_acc[t * 8 + 1] += r1 * r1
            stats_acc[t * 8 + 2] += r2
            stats_acc[t * 8 + 3] += r2 * r2
            stats_acc[t * 8 + 4] += d * d
            stats_acc[t * 8 + 5] += mm
            stats_acc[t * 8 + 6] += mn
            stats_acc[t * 8 + 7] += 1.0

            if capture_raw:
                s = sample_count[t]
                if s < raw_samples.shape[1]:
                    raw_samples[t, s, 0] = r1
                    raw_samples[t, s, 1] = r2
                    sample_count[t] = s + 1

    tot_time[t] = ttime
    cur1[t] = c1
    cur2[t] = c2


@cuda.jit
def _run_kernel(lane1, lane2, rates, bit, cur1, cur2, tot_time,
                alpha_arr, beta_arr, L, steps, warmup, states,
                sample_every, stats_acc, init_lattice,
                raw_samples, sample_count):
    t = cuda.grid(1)
    if t >= cur1.shape[0]:
        return
    alpha = alpha_arr[t]
    beta = beta_arr[t]
    _advance_replica(lane1, lane2, rates, bit, cur1, cur2, tot_time,
                     t, alpha, beta, L, steps, warmup, states,
                     sample_every, stats_acc, init_lattice,
                     raw_samples, sample_count,
                     raw_samples.shape[1] > 0)


def run_ensemble_cuda(alpha, beta, L, steps, n_replicas, seed=0, block=256,
                      sample_every=0, warmup=0, alphas=None, betas=None,
                      n_raw_samples=0):
    """
    Run n_replicas independent ASEP BKL-Fenwick sims on GPU (replica-major).

    If `alphas`/`betas` are given (length n_replicas), each replica uses its
    own (alpha, beta) — a single-launch grid scan. Otherwise all replicas use
    the scalar `alpha`, `beta`.

    Bulk-density statistics are accumulated on-device (O(1) per sample) rather
    than stored raw, so sampling is cheap even for dense sample_every.

    Returns a dict with:
      cur1, cur2, ttime : (n_replicas,) currents and total time per replica
      rho1, rho2        : time-averaged bulk densities per replica
      std_diff          : std(rho1 - rho2) over samples per replica (SSB order
                          parameter, robust to state flipping)
      dense, dilute     : mean of max/min(rho1,rho2) over samples per replica
      n_samples         : number of density samples accumulated per replica
      alphas, betas     : per-replica parameters

    `warmup` steps are discarded before any density sampling; currents and
    total time still accumulate over the full run (matching the CPU path).
    """
    nrep = int(n_replicas)
    L = int(L)
    steps = int(steps)
    warmup = int(warmup)
    sample_every = int(sample_every)
    seed = int(seed)
    block = int(block)
    if alphas is None:
        alphas = np.full(nrep, alpha, dtype=np.float64)
    else:
        alphas = np.asarray(alphas, dtype=np.float64)
    if betas is None:
        betas = np.full(nrep, beta, dtype=np.float64)
    else:
        betas = np.asarray(betas, dtype=np.float64)

    lane1 = cuda.device_array(nrep * L, dtype=np.int8)
    lane2 = cuda.device_array(nrep * L, dtype=np.int8)
    rates = cuda.device_array(nrep * L, dtype=np.float32)
    bit = cuda.device_array(nrep * (L + 1), dtype=np.float32)
    cur1 = cuda.device_array(nrep, dtype=np.float64)
    cur2 = cuda.device_array(nrep, dtype=np.float64)
    ttime = cuda.device_array(nrep, dtype=np.float64)
    stats = cuda.device_array(nrep * 8, dtype=np.float64)
    alpha_d = cuda.to_device(alphas)
    beta_d = cuda.to_device(betas)
    states = create_xoroshiro128p_states(nrep, seed=seed)

    grid = (nrep + block - 1) // block
    if n_raw_samples > 0:
        raw_samples = cuda.device_array((nrep, n_raw_samples, 2),
                                        dtype=np.float64)
        sample_count = cuda.device_array(nrep, dtype=np.int32)
        sample_count[:] = 0
    else:
        raw_samples = cuda.device_array((nrep, 0, 2), dtype=np.float64)
        sample_count = cuda.device_array(nrep, dtype=np.int32)
    _run_kernel[grid, block](lane1, lane2, rates, bit, cur1, cur2, ttime,
                             alpha_d, beta_d, L, steps, warmup, states,
                             sample_every, stats, True,
                             raw_samples, sample_count)
    cuda.synchronize()

    c1 = cur1.copy_to_host()
    c2 = cur2.copy_to_host()
    tt = ttime.copy_to_host()
    st = stats.copy_to_host().reshape(nrep, 8)

    out = {
        "cur1": c1, "cur2": c2, "ttime": tt,
        "alphas": alphas, "betas": betas,
    }
    if sample_every > 0:
        ns = st[:, 7]
        rho1 = np.zeros(nrep)
        rho2 = np.zeros(nrep)
        std_diff = np.zeros(nrep)
        dense = np.zeros(nrep)
        dilute = np.zeros(nrep)
        for i in range(nrep):
            n = ns[i]
            if n > 0:
                rho1[i] = st[i, 0] / n
                rho2[i] = st[i, 2] / n
                var = st[i, 4] / n - (rho1[i] - rho2[i]) ** 2
                std_diff[i] = np.sqrt(max(var, 0.0))
                dense[i] = st[i, 5] / n
                dilute[i] = st[i, 6] / n
        out["rho1"] = rho1
        out["rho2"] = rho2
        out["std_diff"] = std_diff
        out["dense"] = dense
        out["dilute"] = dilute
        out["n_samples"] = ns.astype(np.int64)
    if n_raw_samples > 0:
        out["raw_samples"] = raw_samples.copy_to_host()  # (nrep, nsamp, 2)
        out["sample_count"] = sample_count.copy_to_host()
    return out


def run_ensemble_cuda_continue(L, n_replicas, seed=0, block=256, n_raw_samples=0):
    """
    Create a persistent GPU ensemble state for continuous (multi-chunk) runs.

    Unlike run_ensemble_cuda, which allocates fresh device arrays and
    re-initializes a random lattice every call, this returns a state object
    whose device arrays (lattice, Fenwick, RNG states, running stats, currents,
    times) persist across successive `advance` calls. This gives a true
    continuous trajectory per replica (no per-chunk reseeding), which is what
    reaches the HD/LD basin — the per-chunk-reseed bug in run_all_gpu.py did not.

    Returns
    -------
    A dict with persistent device arrays and an `advance(...)` bound method.
    See `_AdvanceState.advance`.
    """
    nrep = int(n_replicas)
    L = int(L)
    seed = int(seed)
    block = int(block)

    lane1 = cuda.device_array(nrep * L, dtype=np.int8)
    lane2 = cuda.device_array(nrep * L, dtype=np.int8)
    rates = cuda.device_array(nrep * L, dtype=np.float32)
    bit = cuda.device_array(nrep * (L + 1), dtype=np.float32)
    cur1 = cuda.device_array(nrep, dtype=np.float64)
    cur2 = cuda.device_array(nrep, dtype=np.float64)
    ttime = cuda.device_array(nrep, dtype=np.float64)
    stats = cuda.device_array(nrep * 8, dtype=np.float64)
    stats[:] = 0.0
    states = create_xoroshiro128p_states(nrep, seed=seed)
    alpha_d = cuda.to_device(np.full(nrep, 0.9, dtype=np.float64))
    beta_d = cuda.to_device(np.full(nrep, 0.1, dtype=np.float64))
    if n_raw_samples > 0:
        raw_samples = cuda.device_array((nrep, n_raw_samples, 2),
                                        dtype=np.float64)
        sample_count = cuda.device_array(nrep, dtype=np.int32)
        sample_count[:] = 0
    else:
        raw_samples = cuda.device_array((nrep, 0, 2), dtype=np.float64)
        sample_count = cuda.device_array(nrep, dtype=np.int32)
    n_raw_samples_store = int(n_raw_samples)

    state = {
        "L": L, "nrep": nrep, "block": block,
        "lane1": lane1, "lane2": lane2, "rates": rates, "bit": bit,
        "cur1": cur1, "cur2": cur2, "ttime": ttime, "stats": stats,
        "states": states, "alpha_d": alpha_d, "beta_d": beta_d,
        "raw_samples": raw_samples, "sample_count": sample_count,
        "n_raw_samples": n_raw_samples_store,
    }
    state["advance"] = lambda steps, warmup=0, sample_every=0: _advance(
        state, steps, warmup, sample_every)
    return state


def _advance(state, steps, warmup=0, sample_every=0):
    """Run `steps` more MC steps on the persistent ensemble state.

    The first call initializes the random lattice (init_lattice=True); later
    calls continue the existing lattice/Fenwick/RNG (init_lattice=False), so
    the trajectory is continuous across calls.

    Returns the same dict format as run_ensemble_cuda (host copies of the
    accumulated observables).
    """
    L = state["L"]
    nrep = state["nrep"]
    block = state["block"]
    lane1 = state["lane1"]; lane2 = state["lane2"]
    rates = state["rates"]; bit = state["bit"]
    cur1 = state["cur1"]; cur2 = state["cur2"]; ttime = state["ttime"]
    stats = state["stats"]; states = state["states"]
    alpha_d = state["alpha_d"]; beta_d = state["beta_d"]
    raw_samples = state["raw_samples"]
    sample_count = state["sample_count"]
    n_raw_samples = state.get("n_raw_samples", 0)

    init = not state.get("_initialized", False)
    grid = (nrep + block - 1) // block
    _run_kernel[grid, block](lane1, lane2, rates, bit, cur1, cur2, ttime,
                             alpha_d, beta_d, L, steps, warmup, states,
                             sample_every, stats, init,
                             raw_samples, sample_count)
    cuda.synchronize()
    state["_initialized"] = True

    c1 = cur1.copy_to_host()
    c2 = cur2.copy_to_host()
    tt = ttime.copy_to_host()
    st = stats.copy_to_host().reshape(nrep, 8)

    out = {
        "cur1": c1, "cur2": c2, "ttime": tt,
        "alphas": alpha_d.copy_to_host(), "betas": beta_d.copy_to_host(),
    }
    if sample_every > 0:
        ns = st[:, 7]
        rho1 = np.zeros(nrep); rho2 = np.zeros(nrep)
        std_diff = np.zeros(nrep); dense = np.zeros(nrep); dilute = np.zeros(nrep)
        for i in range(nrep):
            n = ns[i]
            if n > 0:
                rho1[i] = st[i, 0] / n
                rho2[i] = st[i, 2] / n
                var = st[i, 4] / n - (rho1[i] - rho2[i]) ** 2
                std_diff[i] = np.sqrt(max(var, 0.0))
                dense[i] = st[i, 5] / n
                dilute[i] = st[i, 6] / n
        out["rho1"] = rho1; out["rho2"] = rho2
        out["std_diff"] = std_diff; out["dense"] = dense
        out["dilute"] = dilute
        out["n_samples"] = ns.astype(np.int64)
    if n_raw_samples > 0:
        out["raw_samples"] = raw_samples.copy_to_host()  # (nrep, nsamp, 2)
        out["sample_count"] = sample_count.copy_to_host()
    return out
