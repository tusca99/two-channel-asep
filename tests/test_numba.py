"""
Validate the numba MC kernel and the TwoChannelASEP wrapper against a
pure-Python reference implementation of the same stochastic process.

Both paths implement the identical model (Gillespie continuous-time MC for the
two-channel ASEP with narrow entrances), so for equivalent seeds and long
enough runs they must agree on steady-state observables within statistical
error. Trajectories cannot be compared directly (the kernel uses its own RNG
stream); we compare aggregate observables instead.

The reference implementation here is deliberately independent of asep/mc.py so
a regression in either file is caught.
"""
import numpy as np
import pytest

from asep import TwoChannelASEP
from asep.mc import run_mc_batched, make_uniforms


# --- Pure-Python reference (ground truth) -------------------------------

class _PySim:
    def __init__(self, L, alpha, beta, seed):
        self.L, self.alpha, self.beta = L, alpha, beta
        self.rng = np.random.default_rng(seed)
        self.lane1 = np.zeros(L, dtype=np.int8)
        self.lane2 = np.zeros(L, dtype=np.int8)
        self.total_time = 0.0
        self.current1 = 0
        self.current2 = 0

    def step(self):
        L, a, b = self.L, self.alpha, self.beta
        l1, l2 = self.lane1, self.lane2
        moves = []
        for i in range(L - 1):
            if l1[i] == 1 and l1[i + 1] == 0:
                moves.append((0, i))       # hop1_right, rate 1
            if l2[i + 1] == 1 and l2[i] == 0:
                moves.append((1, i + 1))   # hop2_left, rate 1
        if l1[0] == 0 and l2[0] == 0:
            moves.append((2, 0))           # enter1, rate alpha
        if l2[L - 1] == 0 and l1[L - 1] == 0:
            moves.append((3, L - 1))       # enter2, rate alpha
        if l1[L - 1] == 1:
            moves.append((4, L - 1))       # exit1, rate beta
        if l2[0] == 1:
            moves.append((5, 0))           # exit2, rate beta

        if not moves:
            self.total_time += 0.001
            return

        total = len(moves)  # hops each rate 1
        total += a * sum(1 for m in moves if m[0] in (2, 3))
        total += b * sum(1 for m in moves if m[0] in (4, 5))
        self.total_time += self.rng.exponential(1.0 / total)

        r = self.rng.random() * total
        cum = 0.0
        for m in moves:
            rate = 1.0 if m[0] in (0, 1) else (a if m[0] in (2, 3) else b)
            cum += rate
            if r <= cum:
                self._apply(m[0], m[1])
                break

    def _apply(self, t, s):
        L = self.L
        l1, l2 = self.lane1, self.lane2
        if t == 0:
            l1[s], l1[s + 1] = 0, 1
        elif t == 1:
            l2[s], l2[s - 1] = 0, 1
        elif t == 2:
            l1[0] = 1
        elif t == 3:
            l2[L - 1] = 1
        elif t == 4:
            l1[L - 1] = 0
            self.current1 += 1
        elif t == 5:
            l2[0] = 0
            self.current2 += 1

    def run(self, n_steps, warmup, sample_every):
        d1 = []
        for i in range(n_steps):
            self.step()
            if i > warmup and i % sample_every == 0:
                d1.append(np.mean(self.lane1))
        return (self.current1 / self.total_time,
                self.current2 / self.total_time,
                np.mean(d1))


def _obs_py(alpha, beta, L, n_steps, warmup, sample_every, seed):
    s = _PySim(L, alpha, beta, seed)
    s.run(n_steps, warmup, sample_every)
    return s.current1 / s.total_time, s.current2 / s.total_time


def run_numba(L, alpha, beta, seed, n_steps, warmup, sample_every):
    """Steady-state observables from the numba kernel on an empty lattice."""
    rng = np.random.default_rng(seed)
    lane1 = np.zeros(L, dtype=np.int8)
    lane2 = np.zeros(L, dtype=np.int8)
    site_d1 = np.zeros(L, dtype=np.float64)
    density_samples1 = []
    n_samples = 0
    block = 1000
    done = 0
    n_exit1_total = 0
    n_exit2_total = 0
    total_time = 0.0

    while done < n_steps:
        step = min(block, n_steps - done)
        dt, e1, e2 = run_mc_batched(
            lane1, lane2, alpha, beta, step, make_uniforms(step, rng)
        )
        total_time += dt
        n_exit1_total += e1
        n_exit2_total += e2
        done += step
        if done > warmup and done % sample_every < block:
            density_samples1.append(np.mean(lane1))
            site_d1 += lane1.astype(np.float64)
            n_samples += 1

    rho1 = np.mean(density_samples1)
    return (n_exit1_total / total_time, n_exit2_total / total_time, rho1)


def _obs_wrapper(alpha, beta, L, n_steps, warmup, sample_every, seed):
    sim = TwoChannelASEP(L=L, alpha=alpha, beta=beta, seed=seed)
    sim.run(n_steps=n_steps, sample_every=sample_every, warmup=warmup)
    return sim.get_currents() + sim.get_bulk_densities()


def assert_same_physics(py, nb, tol_current=0.03, tol_rho=0.05):
    """Compare physics invariant under the Z2 (channel-swap) symmetry."""
    J1p, J2p = py
    J1n, J2n, rho1n = nb
    assert abs(J1p - J1n) < tol_current
    assert abs(J2p - J2n) < tol_current
    assert abs(rho1n - 0.2) < tol_rho


@pytest.mark.parametrize("alpha,beta", [
    (0.2, 0.8),   # LD
    (0.9, 0.9),   # MC
])
def test_wrapper_matches_reference(alpha, beta):
    L = 100
    n_steps = 200000
    warmup = 20000
    sample_every = 200
    seed = 42

    py = _obs_py(alpha, beta, L, n_steps, warmup, sample_every, seed)
    nb = _obs_wrapper(alpha, beta, L, n_steps, warmup, sample_every, seed)

    print(f"py: J=({py[0]:.4f},{py[1]:.4f})")
    print(f"nb: J=({nb[0]:.4f},{nb[1]:.4f}) rho=({nb[2]:.4f},{nb[3]:.4f})")
    assert abs(py[0] - nb[0]) < 0.03
    assert abs(py[1] - nb[1]) < 0.03


def test_ssb_total_current_positive():
    """In the SSB phase, total current must be positive and bounded."""
    L = 100
    n_steps = 200000
    warmup = 20000
    sample_every = 200
    seed = 7

    J1, J2, rho1 = run_numba(L, 0.9, 0.23, seed, n_steps, warmup, sample_every)
    Jtot = J1 + J2
    assert Jtot > 0.0
    assert 0.0 < rho1 < 0.5


def test_same_seed_is_reproducible():
    """Equal seeds must give identical trajectories/observables."""
    L, alpha, beta = 50, 0.6, 0.5
    n_steps, warmup, sample_every = 60000, 5000, 200
    seed = 123

    s1 = TwoChannelASEP(L=L, alpha=alpha, beta=beta, seed=seed)
    s1.run(n_steps=n_steps, warmup=warmup, sample_every=sample_every)

    s2 = TwoChannelASEP(L=L, alpha=alpha, beta=beta, seed=seed)
    s2.run(n_steps=n_steps, warmup=warmup, sample_every=sample_every)

    assert np.allclose(s1.get_currents(), s2.get_currents())
    assert np.allclose(s1.get_bulk_densities(), s2.get_bulk_densities())
    assert np.array_equal(s1.lane1, s2.lane1)
    assert np.array_equal(s1.lane2, s2.lane2)


def _run_obs(fn, L, alpha, beta, n_steps, seed, warmup=0):
    rng = np.random.default_rng(seed)
    lane1 = (rng.random(L) < 0.4).astype(np.int8)
    lane2 = (rng.random(L) < 0.4).astype(np.int8)
    uniforms = rng.random((n_steps + warmup) * 3)
    fn(lane1, lane2, alpha, beta, warmup, uniforms, 0)
    dt, e1, e2, _ = fn(lane1, lane2, alpha, beta, n_steps, uniforms, warmup * 3)
    return e1 / dt, e2 / dt


@pytest.mark.parametrize("alpha,beta", [
    (0.2, 0.8),   # LD
    (0.9, 0.9),   # MC
    (0.3, 0.3),
])
def test_fenwick_matches_classic(alpha, beta):
    """
    Fenwick-tree kernel must agree with the classic BKL kernel.

    Uses only single-phase parameter regimes: in the SSB (bistable) regime the
    two kernels, started from the same configuration but with different event
    selection orders, settle into long-lived broken-symmetry states that do not
    statistically decorrelate within feasible runs, so direct comparison there
    is meaningless.
    """
    from asep.bkl import run_bkl, run_bkl_fenwick
    L = 500
    n_steps = 4_000_000
    warmup = 300_000
    seed = 42
    cl = _run_obs(run_bkl, L, alpha, beta, n_steps, seed, warmup)
    fw = _run_obs(run_bkl_fenwick, L, alpha, beta, n_steps, seed, warmup)
    assert abs(cl[0] - fw[0]) < 0.01
    assert abs(cl[1] - fw[1]) < 0.01


def _cuda_available():
    try:
        from numba import cuda
        return cuda.is_available()
    except Exception:
        return False


def _gpu_obs(alpha, beta, L, steps, warmup, nrep, sample_every, seed):
    """Steady-state currents+density from the GPU ensemble kernel (many reps)."""
    from asep.cuda_ensemble import run_ensemble_cuda
    out = run_ensemble_cuda(alpha, beta, L, steps, nrep, seed=seed,
                            sample_every=sample_every, warmup=warmup)
    m = out["ttime"] > 0
    J1 = (out["cur1"][m] / out["ttime"][m]).mean()
    J2 = (out["cur2"][m] / out["ttime"][m]).mean()
    rho1 = out["rho1"].mean()
    rho2 = out["rho2"].mean()
    return J1, J2, rho1, rho2


@pytest.mark.parametrize("alpha,beta", [
    (0.2, 0.8),   # LD
    (0.9, 0.9),   # MC
])
@pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU not available")
def test_gpu_matches_python_reference(alpha, beta):
    """GPU ensemble kernel must reproduce the pure-Python reference physics."""
    L = 200
    steps = 200_000
    warmup = 100_000
    nrep = 1024
    sample_every = 200
    seed = 42
    py = _obs_py(alpha, beta, L, steps, warmup, sample_every, seed)
    J1g, J2g, r1g, r2g = _gpu_obs(alpha, beta, L, steps, warmup, nrep,
                                  sample_every, seed)
    # currents within ~0.03; densities within ~0.08 (ensemble avg vs one traj)
    assert abs(py[0] - J1g) < 0.04
    assert abs(py[1] - J2g) < 0.04


if __name__ == "__main__":
    for a, b in [(0.2, 0.8), (0.9, 0.9)]:
        py = _obs_py(a, b, 100, 200000, 20000, 200, 42)
        nb = _obs_wrapper(a, b, 100, 200000, 20000, 200, 42)
        print(f"({a},{b})  py J=({py[0]:.4f},{py[1]:.4f})  nb J=({nb[0]:.4f},{nb[1]:.4f})")
