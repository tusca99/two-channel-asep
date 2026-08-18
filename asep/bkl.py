"""
Bortz-Kalos-Lebowitz (BKL) Monte Carlo for the two-channel ASEP.

Standard Gillespie scans the whole lattice every step to find all moves
(O(L) per step). BKL maintains a list of "active" sites (sites with at
least one possible move) and only recomputes the activity of sites near a
move when it executes. At low density the number of active sites is small,
so the per-step cost drops from O(L) to O(active) — a large constant-factor
win on both CPU and GPU.

Per-site rate
-------------
For site i, rate[i] is the sum of all moves involving site i:
  lane1: hop right (i<L-1, lane1[i]=1, lane1[i+1]=0) rate 1
         enter (i=0, lane1[0]=0, lane2[0]=0) rate alpha
         exit  (i=L-1, lane1[L-1]=1) rate beta
  lane2: hop left (i>0, lane2[i]=1, lane2[i-1]=0) rate 1
         enter (i=L-1, lane2[L-1]=0, lane1[L-1]=0) rate alpha
         exit  (i=0, lane2[0]=1) rate beta

A move at site i can only change the activity of sites i-1, i, i+1 (and the
boundary sites 0, L-1), so after each move we recompute rate[] on that small
window and update the active-site list.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def _site_rate(lane1, lane2, alpha, beta, L, i):
    """Total rate of all moves involving site i."""
    r = 0.0
    # lane1
    if i < L - 1 and lane1[i] == 1 and lane1[i + 1] == 0:
        r += 1.0
    if i == 0 and lane1[0] == 0 and lane2[0] == 0:
        r += alpha
    if i == L - 1 and lane1[L - 1] == 1:
        r += beta
    # lane2
    if i > 0 and lane2[i] == 1 and lane2[i - 1] == 0:
        r += 1.0
    if i == L - 1 and lane2[L - 1] == 0 and lane1[L - 1] == 0:
        r += alpha
    if i == 0 and lane2[0] == 1:
        r += beta
    return r


@njit(cache=True, fastmath=True)
def _rebuild_active(lane1, lane2, alpha, beta, L, rates, active, n_active):
    """Recompute rates and the active-site list from scratch."""
    n_active = 0
    for i in range(L):
        r = _site_rate(lane1, lane2, alpha, beta, L, i)
        rates[i] = r
        if r > 0:
            active[n_active] = i
            n_active += 1
    return n_active


@njit(cache=True, fastmath=True)
def _refresh_window(lane1, lane2, alpha, beta, L, rates, active, n_active, center):
    """
    Recompute rates for sites in [center-1, center+1] (clamped to [0, L-1])
    and update the active list incrementally. Returns the new n_active.

    A move at `center` can only change the activity of sites center-1,
    center, center+1 (and boundary sites 0, L-1). For each such site we
    recompute its rate and add/remove it from the active list in O(1).
    """
    lo = max(0, center - 1)
    hi = min(L - 1, center + 1)
    for i in range(lo, hi + 1):
        new_r = _site_rate(lane1, lane2, alpha, beta, L, i)
        old_r = rates[i]
        if old_r == 0 and new_r > 0:
            # became active: append
            active[n_active] = i
            n_active += 1
        elif old_r > 0 and new_r == 0:
            # became inactive: remove (swap-with-last)
            for k in range(n_active):
                if active[k] == i:
                    n_active -= 1
                    active[k] = active[n_active]
                    break
        rates[i] = new_r
    return n_active


@njit(cache=True, fastmath=True)
def _execute(lane1, lane2, L, site, mtype):
    """Execute a move; returns (exited_ch1, exited_ch2)."""
    if mtype == 0:  # hop1_right
        lane1[site] = 0
        lane1[site + 1] = 1
    elif mtype == 1:  # hop2_left
        lane2[site] = 0
        lane2[site - 1] = 1
    elif mtype == 2:  # enter1
        lane1[0] = 1
    elif mtype == 3:  # enter2
        lane2[L - 1] = 1
    elif mtype == 4:  # exit1
        lane1[L - 1] = 0
        return 1, 0
    elif mtype == 5:  # exit2
        lane2[0] = 0
        return 0, 1
    return 0, 0


@njit(cache=True, fastmath=True)
def _pick_move(lane1, lane2, alpha, beta, L, site, r):
    """
    Given a selected active site and a uniform r in [0, rate[site]),
    pick and execute one of the site's moves. Returns (exited_ch1, exited_ch2).
    """
    cum = 0.0
    # lane1
    if site < L - 1 and lane1[site] == 1 and lane1[site + 1] == 0:
        cum += 1.0
        if r < cum:
            return _execute(lane1, lane2, L, site, 0)
    if site == 0 and lane1[0] == 0 and lane2[0] == 0:
        cum += alpha
        if r < cum:
            return _execute(lane1, lane2, L, site, 2)
    if site == L - 1 and lane1[L - 1] == 1:
        cum += beta
        if r < cum:
            return _execute(lane1, lane2, L, site, 4)
    # lane2
    if site > 0 and lane2[site] == 1 and lane2[site - 1] == 0:
        cum += 1.0
        if r < cum:
            return _execute(lane1, lane2, L, site, 1)
    if site == L - 1 and lane2[L - 1] == 0 and lane1[L - 1] == 0:
        cum += alpha
        if r < cum:
            return _execute(lane1, lane2, L, site, 3)
    if site == 0 and lane2[0] == 1:
        cum += beta
        if r < cum:
            return _execute(lane1, lane2, L, site, 5)
    return 0, 0


@njit(cache=True, fastmath=True)
def _bit_build(rates, bit):
    """Build a Fenwick tree (1-indexed, size L) over per-site rates in O(L)."""
    L = rates.shape[0]
    for i in range(L):
        bit[i + 1] = rates[i]
    for i in range(1, L + 1):
        j = i + (i & -i)
        if j <= L:
            bit[j] += bit[i]


@njit(cache=True, fastmath=True)
def _bit_update(bit, i, delta):
    """Add delta to site i (0-indexed); O(log L)."""
    L = bit.shape[0] - 1
    idx = i + 1
    while idx <= L:
        bit[idx] += delta
        idx += idx & -idx


@njit(cache=True, fastmath=True)
def _bit_find(bit, r):
    """
    Return the smallest 0-indexed site i whose prefix-sum >= r.
    r is in [0, total_rate). O(log L).
    """
    L = bit.shape[0] - 1
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
    return idx  # number of elements with prefix < original r -> 0-indexed site


@njit(cache=True, fastmath=True)
def _refresh_window_fenwick(lane1, lane2, alpha, beta, L, rates, bit, center):
    """
    Recompute rates for sites in [center-1, center+1] and update both the
    rates[] cache and the Fenwick tree incrementally. Returns the number of
    sites whose rate changed (for bookkeeping). No active list needed.
    """
    lo = max(0, center - 1)
    hi = min(L - 1, center + 1)
    n_changed = 0
    for i in range(lo, hi + 1):
        new_r = _site_rate(lane1, lane2, alpha, beta, L, i)
        old_r = rates[i]
        if new_r != old_r:
            _bit_update(bit, i, new_r - old_r)
            rates[i] = new_r
            n_changed += 1
    return n_changed


@njit(cache=True, fastmath=True)
def run_bkl_fenwick(lane1, lane2, alpha, beta, n_steps, uniforms, u_idx):
    """
    BKL MC with a Fenwick (binary indexed) tree over per-site rates.

    Removes BOTH O(n_active) scans of the classic BKL kernel in run_bkl:
      - total rate comes from the tree root (O(1))
      - active-site selection becomes a tree "find" in O(log L)
      - activity bookkeeping is a few O(log L) point updates on the window

    Same model, same semantics, same return tuple
    (total_time, n_exit1, n_exit2, new_u_idx). Consumes the same 3 uniforms
    per step as run_bkl so results are comparable for equal seeds.
    """
    L = lane1.shape[0]
    rates = np.zeros(L, dtype=np.float64)
    bit = np.zeros(L + 1, dtype=np.float64)
    n_active = _rebuild_active(lane1, lane2, alpha, beta, L, rates, np.empty(L, dtype=np.int64), 0)
    _bit_build(rates, bit)

    total_time = 0.0
    n_exit1 = 0
    n_exit2 = 0

    for _ in range(n_steps):
        if n_active == 0:
            total_time += 0.001
            continue

        # Total rate = prefix-sum of all L sites via the tree (O(log L)).
        total_rate = 0.0
        idx = L
        while idx > 0:
            total_rate += bit[idx]
            idx -= idx & -idx

        u1 = uniforms[u_idx]
        u_idx += 1
        total_time += -np.log(u1) / total_rate

        u2 = uniforms[u_idx]
        u_idx += 1
        r = u2 * total_rate
        chosen = _bit_find(bit, r)

        u3 = uniforms[u_idx]
        u_idx += 1
        e1, e2 = _pick_move(lane1, lane2, alpha, beta, L, chosen, u3 * rates[chosen])
        n_exit1 += e1
        n_exit2 += e2

        _refresh_window_fenwick(lane1, lane2, alpha, beta, L, rates, bit, chosen)

    return total_time, n_exit1, n_exit2, u_idx


@njit(cache=True, fastmath=True, parallel=True)
def run_bkl_fenwick_batch(lane1, lane2, alphas, betas, n_steps, uniforms):
    """
    Run n_steps of BKL-Fenwick MC for an array of independent replicas.

    lane1, lane2 : int8 arrays of shape (n_replicas, L), modified in-place.
    alphas, betas: float64 arrays of length n_replicas (one parameter set per
        replica), enabling a single-launch heterogeneous scan.
    uniforms     : float64 array of pre-generated uniform(0,1) draws. Each
        replica consumes up to 3 uniforms per step from its own contiguous
        slice (offset i * n_steps * 3), so replicas stay independent and
        reproducible given a seed.

    Returns (total_times, n_exit1, n_exit2) as arrays of length n_replicas.

    Replicas are independent (no cross-replica coupling), so they are executed
    across threads with prange. Each replica still runs the serial Fenwick
    kernel (its data-dependent find/update chains are not SIMD-friendly), so
    the win is multi-core scaling, not AVX.
    """
    N, L = lane1.shape
    total_time = np.zeros(N, dtype=np.float64)
    n_exit1 = np.zeros(N, dtype=np.int64)
    n_exit2 = np.zeros(N, dtype=np.int64)
    for i in prange(N):
        dt, e1, e2, _ = run_bkl_fenwick(lane1[i], lane2[i], alphas[i], betas[i],
                                        n_steps, uniforms, i * n_steps * 3)
        total_time[i] = dt
        n_exit1[i] = e1
        n_exit2[i] = e2
    return total_time, n_exit1, n_exit2


@njit(cache=True, fastmath=True)
def run_bkl_profiled(lane1, lane2, alpha, beta, n_steps, uniforms, u_idx,
                     use_fenwick=False):
    """
    Run BKL MC and return per-phase operation counts for profiling.

    Numba cannot call time.perf_counter(), so instead of wall-clock timers this
    counts the number of elementary operations in each phase — which, per unit
    cost, is the best deterministic picture of where time goes. For wall-clock
    attribution of the compiled kernel, run the process with
    ``NUMBA_ENABLE_PROFILING=1`` (the `@njit` functions then appear in the
    cProfile output).

    Returns a 6-tuple
    (total_time, n_exit1, n_exit2, new_u_idx, counts, avg_active) where
      counts     = [n_scans_active, n_win_refreshes, n_logL_prefix, n_logL_find]
      avg_active = mean number of active sites per executed step.
    """
    use_fenwick = int(use_fenwick)
    L = lane1.shape[0]
    rates = np.zeros(L, dtype=np.float64)
    active = np.empty(L, dtype=np.int64)
    bit = np.zeros(L + 1, dtype=np.float64)
    n_active = _rebuild_active(lane1, lane2, alpha, beta, L, rates, active, 0)
    if use_fenwick:
        _bit_build(rates, bit)

    total_time = 0.0
    n_exit1 = 0
    n_exit2 = 0
    n_scans = 0       # classic: active-list scans (sum of n_active per step)
    n_win = 0         # refresh-window calls (approx constant 3 sites each)
    n_prefix = 0      # fenwick: O(log L) prefix sums
    n_find = 0        # fenwick: O(log L) selection finds
    n_blocked = 0
    ev = 0

    for _ in range(n_steps):
        if use_fenwick:
            total_rate = 0.0
            idx = L
            n_prefix += 1
            while idx > 0:
                total_rate += bit[idx]
                idx -= idx & -idx
            if total_rate == 0.0:
                total_time += 0.001
                n_blocked += 1
                continue
            u1 = uniforms[u_idx]; u_idx += 1
            total_time += -np.log(u1) / total_rate
            u2 = uniforms[u_idx]; u_idx += 1
            chosen = _bit_find(bit, u2 * total_rate)
            n_find += 1
            u3 = uniforms[u_idx]; u_idx += 1
            e1, e2 = _pick_move(lane1, lane2, alpha, beta, L, chosen, u3 * rates[chosen])
            n_exit1 += e1; n_exit2 += e2
            _refresh_window_fenwick(lane1, lane2, alpha, beta, L, rates, bit, chosen)
            n_win += 1
        else:
            if n_active == 0:
                total_time += 0.001
                n_blocked += 1
                continue
            n_scans += n_active
            total_rate = 0.0
            for k in range(n_active):
                total_rate += rates[active[k]]
            u1 = uniforms[u_idx]; u_idx += 1
            total_time += -np.log(u1) / total_rate
            u2 = uniforms[u_idx]; u_idx += 1
            r = u2 * total_rate
            cum = 0.0
            chosen = active[n_active - 1]
            for k in range(n_active):
                cum += rates[active[k]]
                if r <= cum:
                    chosen = active[k]
                    break
            n_scans += n_active
            u3 = uniforms[u_idx]; u_idx += 1
            e1, e2 = _pick_move(lane1, lane2, alpha, beta, L, chosen, u3 * rates[chosen])
            n_exit1 += e1; n_exit2 += e2
            n_active = _refresh_window(lane1, lane2, alpha, beta, L, rates, active, n_active, chosen)
            n_win += 1
        ev += 1

    avg_active = 0.0
    if ev > 0:
        avg_active = n_scans / (2.0 * ev)
    counts = np.array([n_scans, n_win, n_prefix, n_find], dtype=np.float64)
    return total_time, n_exit1, n_exit2, u_idx, counts, avg_active


@njit(cache=True, fastmath=True)
def run_bkl(lane1, lane2, alpha, beta, n_steps, uniforms, u_idx):
    """
    Run n_steps of BKL MC. Consumes uniforms from a seeded numpy stream.

    Returns (total_time, n_exit1, n_exit2, new_u_idx).
    """
    L = lane1.shape[0]
    rates = np.zeros(L, dtype=np.float64)
    active = np.empty(L, dtype=np.int64)
    n_active = _rebuild_active(lane1, lane2, alpha, beta, L, rates, active, 0)

    total_time = 0.0
    n_exit1 = 0
    n_exit2 = 0

    for _ in range(n_steps):
        if n_active == 0:
            total_time += 0.001
            continue

        # Total rate = sum over active sites
        total_rate = 0.0
        for k in range(n_active):
            total_rate += rates[active[k]]

        # Time advance
        u1 = uniforms[u_idx]
        u_idx += 1
        total_time += -np.log(u1) / total_rate

        # Select an active site proportional to its rate
        u2 = uniforms[u_idx]
        u_idx += 1
        r = u2 * total_rate
        cum = 0.0
        chosen = active[n_active - 1]
        for k in range(n_active):
            cum += rates[active[k]]
            if r <= cum:
                chosen = active[k]
                break

        # Pick and execute a move at the chosen site
        u3 = uniforms[u_idx]
        u_idx += 1
        e1, e2 = _pick_move(lane1, lane2, alpha, beta, L, chosen, u3 * rates[chosen])
        n_exit1 += e1
        n_exit2 += e2

        # Refresh activity around the moved site
        n_active = _refresh_window(lane1, lane2, alpha, beta, L, rates, active,
                                   n_active, chosen)

    return total_time, n_exit1, n_exit2, u_idx
