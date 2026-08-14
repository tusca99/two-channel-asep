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
from numba import njit

# xorshift64* multiplier
_XOR_MULT = np.uint64(2685821657736338717)


@njit(cache=True)
def _xor_uniform(state):
    """
    One xorshift64* step. state is a uint64 (passed by value; returned updated).
    Returns (uniform01, new_state).
    """
    x = state
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    u = (x * _XOR_MULT) >> np.uint64(11)
    return np.float64(u) / 9007199254740992.0, x


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


@njit(cache=True, fastmath=True)
def run_bkl_xor(lane1, lane2, alpha, beta, n_steps, seed):
    """
    BKL MC with an inline xorshift64* RNG (no numpy uniform stream).

    Avoids the overhead of generating and reading a numpy RNG array. Results
    are reproducible for a given `seed`. Uses 3 xorshift draws per step
    (time, site selection, move selection).

    Returns (total_time, n_exit1, n_exit2).
    """
    L = lane1.shape[0]
    rates = np.zeros(L, dtype=np.float64)
    active = np.empty(L, dtype=np.int64)
    n_active = _rebuild_active(lane1, lane2, alpha, beta, L, rates, active, 0)

    state = np.uint64(seed) & np.uint64(0xFFFFFFFFFFFFFFFF)
    if state == np.uint64(0):
        state = np.uint64(88172645463325252)

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
        u1, state = _xor_uniform(state)
        total_time += -np.log(u1) / total_rate

        # Select an active site proportional to its rate
        u2, state = _xor_uniform(state)
        r = u2 * total_rate
        cum = 0.0
        chosen = active[n_active - 1]
        for k in range(n_active):
            cum += rates[active[k]]
            if r <= cum:
                chosen = active[k]
                break

        # Pick and execute a move at the chosen site
        u3, state = _xor_uniform(state)
        e1, e2 = _pick_move(lane1, lane2, alpha, beta, L, chosen, u3 * rates[chosen])
        n_exit1 += e1
        n_exit2 += e2

        # Refresh activity around the moved site
        n_active = _refresh_window(lane1, lane2, alpha, beta, L, rates, active,
                                   n_active, chosen)

    return total_time, n_exit1, n_exit2
