"""
Numba-accelerated Monte Carlo step for the two-channel ASEP.

This module provides a compiled (Numba/JIT) version of the Gillespie
continuous-time MC step for the two-channel ASEP with narrow entrances.

Performance: ~50-100x faster than the pure Python step() in model.py.
"""
import numpy as np
from numba import njit, types
from numba.typed import List


@njit(cache=True, fastmath=True)
def _collect_moves(lane1, lane2, alpha, beta):
    """
    Collect all possible moves and their rates.

    Returns arrays of (type, site, rate) and count.
    """
    L = lane1.shape[0]
    max_moves = 2 * L + 2
    move_types = np.empty(max_moves, dtype=np.int8)
    move_sites = np.empty(max_moves, dtype=np.int64)
    move_rates = np.empty(max_moves, dtype=np.float64)
    n_moves = 0

    # Bulk hopping
    for i in range(L - 1):
        if lane1[i] == 1 and lane1[i + 1] == 0:
            move_types[n_moves] = 0  # hop1_right
            move_sites[n_moves] = i
            move_rates[n_moves] = 1.0
            n_moves += 1
        if lane2[i + 1] == 1 and lane2[i] == 0:
            move_types[n_moves] = 1  # hop2_left
            move_sites[n_moves] = i + 1
            move_rates[n_moves] = 1.0
            n_moves += 1

    # Entrance events
    if lane1[0] == 0 and lane2[0] == 0:
        move_types[n_moves] = 2  # enter1
        move_sites[n_moves] = 0
        move_rates[n_moves] = alpha
        n_moves += 1

    if lane2[L - 1] == 0 and lane1[L - 1] == 0:
        move_types[n_moves] = 3  # enter2
        move_sites[n_moves] = L - 1
        move_rates[n_moves] = alpha
        n_moves += 1

    # Exit events
    if lane1[L - 1] == 1:
        move_types[n_moves] = 4  # exit1
        move_sites[n_moves] = L - 1
        move_rates[n_moves] = beta
        n_moves += 1

    if lane2[0] == 1:
        move_types[n_moves] = 5  # exit2
        move_sites[n_moves] = 0
        move_rates[n_moves] = beta
        n_moves += 1

    return move_types[:n_moves], move_sites[:n_moves], move_rates[:n_moves]


@njit(cache=True, fastmath=True)
def _execute_move(lane1, lane2, mtype, site):
    """Execute a single move on the lattices (in-place)."""
    L = lane1.shape[0]

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
    elif mtype == 5:  # exit2
        lane2[0] = 0


@njit(cache=True, fastmath=True)
def mc_step_numba(lane1, lane2, alpha, beta):
    """
    One Gillespie (continuous-time) step, JIT-compiled with Numba.

    Uses numba's internal RNG (np.random.random() and np.random.exponential()).

    Parameters
    ----------
    lane1, lane2 : int8 arrays of shape (L,)
        Occupation variables (0 or 1), modified in-place
    alpha : float
        Entrance rate
    beta : float
        Exit rate

    Returns
    -------
    dt : float
        Time advance for this step
    exited_ch1 : int
        1 if a particle exited channel 1, else 0
    exited_ch2 : int
        1 if a particle exited channel 2, else 0
    """
    move_types, move_sites, move_rates = _collect_moves(lane1, lane2, alpha, beta)
    n_moves = move_types.shape[0]

    if n_moves == 0:
        return 0.001, 0, 0

    # Total rate
    total_rate = 0.0
    for j in range(n_moves):
        total_rate += move_rates[j]

    # Time advance
    dt = -np.log(np.random.random()) / total_rate

    # Select move
    r = np.random.random() * total_rate
    cum = 0.0
    selected = n_moves - 1
    for j in range(n_moves):
        cum += move_rates[j]
        if r <= cum:
            selected = j
            break

    mtype = move_types[selected]
    site = move_sites[selected]

    # Track if this was an exit
    exited_ch1 = 1 if mtype == 4 else 0
    exited_ch2 = 1 if mtype == 5 else 0

    _execute_move(lane1, lane2, mtype, site)

    return dt, exited_ch1, exited_ch2


@njit(cache=True, fastmath=True)
def run_mc_batched(lane1, lane2, alpha, beta, n_steps):
    """
    Run n_steps of continuous-time MC.

    Returns: total_time, n_exit1, n_exit2
    """
    total_time = 0.0
    n_exit1 = 0
    n_exit2 = 0

    for _ in range(n_steps):
        dt, e1, e2 = mc_step_numba(lane1, lane2, alpha, beta)
        total_time += dt
        n_exit1 += e1
        n_exit2 += e2

    return total_time, n_exit1, n_exit2
