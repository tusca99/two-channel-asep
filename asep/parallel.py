"""
Helpers for embarrassingly-parallel Monte Carlo scans.

Each (alpha, beta) grid point is an independent simulation. Scans can be run
either on CPU cores (ProcessPoolExecutor) or on the GPU as an ensemble of
independent replicas (asep.cuda_ensemble), which is ~3x faster than 12 CPU
cores for the same physics.
"""
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from asep import TwoChannelASEP


def _run_point(task):
    """Run one (alpha, beta, L, params) point; return (J1, J2, rho1, rho2)."""
    alpha, beta, L, n_steps, warmup, sample_every, seed = task
    sim = TwoChannelASEP(L=L, alpha=alpha, beta=beta, seed=seed)
    sim.run(n_steps=n_steps, sample_every=sample_every, warmup=warmup)
    J1, J2 = sim.get_currents()
    r1, r2 = sim.get_bulk_densities()
    return J1, J2, r1, r2


def _run_point_adaptive(task, tol=1e-3, min_steps=0):
    """Like _run_point but stops early once the current plateaus.

    Uses TwoChannelASEP.run_adaptive, which stops when the windowed current
    matches the cumulative current to within `tol`. Returns the same
    (J1, J2, rho1, rho2) tuple as _run_point.
    """
    alpha, beta, L, n_steps, warmup, sample_every, seed = task
    sim = TwoChannelASEP(L=L, alpha=alpha, beta=beta, seed=seed)
    sim.run_adaptive(max_steps=n_steps, sample_every=sample_every,
                      warmup=warmup, tol=tol, min_steps=min_steps)
    J1, J2 = sim.get_currents()
    r1, r2 = sim.get_bulk_densities()
    return J1, J2, r1, r2


def _run_point_samples(task):
    """Run one point; return (J1, J2, rho1, rho2, joint_samples)."""
    alpha, beta, L, n_steps, warmup, sample_every, seed = task
    sim = TwoChannelASEP(L=L, alpha=alpha, beta=beta, seed=seed)
    sim.run(n_steps=n_steps, sample_every=sample_every, warmup=warmup)
    J1, J2 = sim.get_currents()
    r1, r2 = sim.get_bulk_densities()
    return J1, J2, r1, r2, sim.get_joint_density_samples()


def scan_points(tasks, n_workers=None, desc="scan", adaptive=False,
                tol=1e-3, min_steps=0):
    """
    Run a list of independent (alpha, beta, L, n_steps, warmup, sample_every, seed)
    tasks in parallel and return their (J1, J2, rho1, rho2) results, preserving order.
    Shows a tqdm progress bar.

    If `adaptive` is True, each point stops early once its current plateaus
    (see TwoChannelASEP.run_adaptive), which can cut runtime several-fold in
    phases where the current converges quickly.
    """
    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        if adaptive:
            futures = {ex.submit(_run_point_adaptive, t, tol, min_steps): j
                       for j, t in enumerate(tasks)}
        else:
            futures = {ex.submit(_run_point, t): j for j, t in enumerate(tasks)}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc=desc):
            results[futures[fut]] = fut.result()
    return results


def scan_points_samples(tasks, n_workers=None, desc="scan"):
    """
    Like scan_points but also returns the joint density samples per point,
    for the density-distribution phase classification.
    """
    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_run_point_samples, t): j for j, t in enumerate(tasks)}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc=desc):
            results[futures[fut]] = fut.result()
    return results


def scan_points_batch(tasks, n_workers=None, desc="scan batch", chunk=None):
    """
    Run independent tasks with the parallel BKL-Fenwick kernel
    (asep.bkl.run_bkl_fenwick_batch) instead of ProcessPool.

    Tasks share the same (L, n_steps, warmup, sample_every); only
    (alpha, beta, seed) vary per task. Each task's RNG stream is seeded from
    its own seed, so replicas are independent and reproducible.

    Returns (J1, J2, rho1, rho2) arrays, one per task (order preserved).
    This is a multi-core (prange) path: replicas are independent but each runs
    the serial Fenwick kernel, so the gain over ProcessPool is modest (~10-15%).
    """
    from asep.bkl import run_bkl_fenwick_batch

    if len(tasks) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)

    L = tasks[0][2]
    n_steps = tasks[0][3]
    alphas = np.array([t[0] for t in tasks], dtype=np.float64)
    betas = np.array([t[1] for t in tasks], dtype=np.float64)
    N = len(tasks)

    lane1 = np.zeros((N, L), dtype=np.int8)
    lane2 = np.zeros((N, L), dtype=np.int8)

    J1 = np.zeros(N)
    J2 = np.zeros(N)
    r1 = np.zeros(N)
    r2 = np.zeros(N)

    if chunk is None:
        chunk = N
    for c0 in tqdm(range(0, N, chunk), desc=desc):
        sl = slice(c0, min(c0 + chunk, N))
        n = sl.stop - sl.start
        # one independent uniform stream per replica
        uf = np.empty(n * n_steps * 3, dtype=np.float64)
        off = 0
        for j in range(n):
            rng = np.random.default_rng(int(tasks[c0 + j][6]))
            uf[off:off + n_steps * 3] = rng.random(n_steps * 3)
            off += n_steps * 3
        tt, e1, e2 = run_bkl_fenwick_batch(
            lane1[sl], lane2[sl], alphas[sl], betas[sl], n_steps, uf)
        J1[sl] = e1 / tt
        J2[sl] = e2 / tt
        r1[sl] = lane1[sl].mean(axis=1)
        r2[sl] = lane2[sl].mean(axis=1)
    return J1, J2, r1, r2


def make_tasks(alphas, betas, L, n_steps, warmup, sample_every, seed=0):
    """Build a task list for a full (alphas x betas) grid scan."""
    rng = np.random.default_rng(seed)
    tasks = []
    for a in alphas:
        for b in betas:
            tasks.append((a, b, L, n_steps, warmup, sample_every,
                          int(rng.integers(1e9))))
    return tasks


# --- GPU ensemble backend -------------------------------------------------

def _cuda_available():
    try:
        from numba import cuda
        return cuda.is_available()
    except Exception:
        return False


def scan_grid_gpu(alphas, betas, L, n_steps, warmup, sample_every, n_reps=1,
                  seed=0, block=256, chunk=1024, desc="gpu scan"):
    """
    Run a full (alphas x betas) grid on the GPU as an ensemble of replicas.

    Each (alpha,beta) point is replicated n_reps times. Replicas are run in
    batches of `chunk` per kernel launch (one thread per replica), with a
    tqdm progress bar across batches — so a hung or pathologically slow batch
    is visible instead of a silent stall. Returns a list aligned with the CPU
    scan_points_samples output: one (J1, J2, rho1, rho2, samples) tuple per
    (alpha, beta, rep).
    """
    from asep.cuda_ensemble import run_ensemble_cuda

    na, nb = len(alphas), len(betas)
    n_points = na * nb
    nrep = int(n_points * n_reps)

    # per-replica (alpha, beta): point-major so results are easy to reshape
    aa = np.repeat(np.array([a for a in alphas for b in betas]), n_reps)
    bb = np.repeat(np.array([b for a in alphas for b in betas]), n_reps)

    m = np.zeros(nrep, dtype=bool)
    cur1 = np.zeros(nrep)
    cur2 = np.zeros(nrep)
    ttime = np.zeros(nrep)
    rho1 = np.zeros(nrep)
    rho2 = np.zeros(nrep)

    nb_chunks = (nrep + chunk - 1) // chunk
    for ic in tqdm(range(nb_chunks), desc=desc, total=nb_chunks):
        sl = slice(ic * chunk, min((ic + 1) * chunk, nrep))
        out = run_ensemble_cuda(
            0.0, 0.0, L, n_steps, sl.stop - sl.start, seed=seed + ic, block=block,
            sample_every=sample_every, warmup=warmup,
            alphas=aa[sl], betas=bb[sl],
        )
        idx = np.arange(sl.start, sl.stop)
        tm = out["ttime"] > 0
        m[idx] = tm
        cur1[idx] = out["cur1"]
        cur2[idx] = out["cur2"]
        ttime[idx] = out["ttime"]
        if sample_every > 0:
            rho1[idx] = out["rho1"]
            rho2[idx] = out["rho2"]

    results = []
    for i in range(nrep):
        J1 = cur1[i] / ttime[i] if m[i] else 0.0
        J2 = cur2[i] / ttime[i] if m[i] else 0.0
        if sample_every > 0:
            # per-replica density sample for ensemble SSB detection
            samples = np.array([[rho1[i], rho2[i]]])
            results.append((J1, J2, rho1[i], rho2[i], samples))
        else:
            results.append((J1, J2, rho1[i], rho2[i]))
    return results


def scan_beta_gpu(alpha, betas, L, n_steps, warmup, sample_every, n_reps=1,
                  seed=0):
    """Like scan_beta_stats but on the GPU: for fixed alpha, mean+/-sem over
    beta across n_reps seeds. Returns (J1,J2,r1,r2,eJ1,eJ2,er1,er2,
    dense,dilute,edense,edilute) matching scan_beta_stats."""
    res = scan_grid_gpu([alpha], betas, L, n_steps, warmup, sample_every,
                        n_reps=n_reps, seed=seed)
    nb = len(betas)
    J1 = np.zeros(nb); J2 = np.zeros(nb); r1 = np.zeros(nb); r2 = np.zeros(nb)
    eJ1 = np.zeros(nb); eJ2 = np.zeros(nb); er1 = np.zeros(nb); er2 = np.zeros(nb)
    dense = np.zeros(nb); dilute = np.zeros(nb)
    edense = np.zeros(nb); edilute = np.zeros(nb)
    for j in range(nb):
        vals = res[j * n_reps:(j + 1) * n_reps]
        Jv = np.array([v[0] for v in vals]); J2v = np.array([v[1] for v in vals])
        r1v = np.array([v[2] for v in vals]); r2v = np.array([v[3] for v in vals])
        J1[j] = Jv.mean(); eJ1[j] = Jv.std() / np.sqrt(n_reps)
        J2[j] = J2v.mean(); eJ2[j] = J2v.std() / np.sqrt(n_reps)
        r1[j] = r1v.mean(); er1[j] = r1v.std() / np.sqrt(n_reps)
        r2[j] = r2v.mean(); er2[j] = r2v.std() / np.sqrt(n_reps)
        d = np.maximum(r1v, r2v); dil = np.minimum(r1v, r2v)
        dense[j] = d.mean(); edense[j] = d.std() / np.sqrt(n_reps)
        dilute[j] = dil.mean(); edilute[j] = dil.std() / np.sqrt(n_reps)
    return (J1, J2, r1, r2, eJ1, eJ2, er1, er2, dense, dilute, edense, edilute)


def scan_phase_diagram_gpu(alphas, betas, L, n_steps, warmup, sample_every,
                           n_reps=1, seed=0):
    """GPU version of phase_diagram.scan_phase_diagram: returns 2D array of
    phase labels using the density-distribution method over an ensemble."""
    from asep.observables import classify_phase
    res = scan_grid_gpu(alphas, betas, L, n_steps, warmup, sample_every,
                        n_reps=n_reps, seed=seed)
    na, nb = len(alphas), len(betas)
    grid = np.empty((na, nb), dtype=object)
    k = 0
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            # aggregate across n_reps replicas for this point
            reps = res[k * n_reps:(k + 1) * n_reps]
            J1 = np.mean([r[0] for r in reps])
            J2 = np.mean([r[1] for r in reps])
            r1 = np.mean([r[2] for r in reps])
            r2 = np.mean([r[3] for r in reps])
            samples = np.vstack([r[4] for r in reps]) if n_reps > 0 and reps[0][4].size else None
            grid[i, j] = classify_phase(J1, J2, r1, r2, a, b, L,
                                        samples=samples, j_current=0.25)[0]
            k += 1
    return grid


def grid_to_labels(results, alphas, betas, classify_fn):
    """Reshape flat results into a 2D grid of phase labels via classify_fn."""
    grid = np.empty((len(alphas), len(betas)), dtype=object)
    k = 0
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            J1, J2, r1, r2 = results[k]
            grid[i, j] = classify_fn(J1, J2, r1, r2, a, b)
            k += 1
    return grid


def scan_points_gpu(tasks, n_reps=1, seed=0, chunk=2048, desc="gpu scan",
                    use_density=False):
    """
    GPU drop-in for scan_points: same task format and return layout.

    `tasks` is a list of (alpha, beta, L, n_steps, warmup, sample_every, seed).
    All points are grouped by (alpha,beta,L,n_steps,warmup,sample_every) and
    each group is run as a single GPU ensemble with n_reps replicas per point
    (extra independent seeds for statistics). Returns a flat list of
    (J1, J2, rho1, rho2) tuples, one per (point x replica).
    """
    from asep.cuda_ensemble import run_ensemble_cuda

    # group tasks by their (alpha,beta,L,n_steps,warmup,sample_every)
    groups = {}
    order = []
    for t in tasks:
        key = (t[0], t[1], t[2], t[3], t[4], t[5])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(t)

    results = []
    for key in order:
        t = groups[key][0]
        alpha, beta, L, n_steps, warmup, sample_every = key
        n_pts = len(groups[key])
        nrep = int(n_pts * n_reps)
        aa = np.repeat(np.array([g[0] for g in groups[key]]), n_reps)
        bb = np.repeat(np.array([g[1] for g in groups[key]]), n_reps)

        out = run_ensemble_cuda(
            0.0, 0.0, int(L), int(n_steps), nrep, seed=int(seed), block=256,
            sample_every=int(sample_every), warmup=int(warmup),
            alphas=aa, betas=bb,
        )
        m = out["ttime"] > 0
        for i in range(nrep):
            J1 = out["cur1"][i] / out["ttime"][i] if m[i] else 0.0
            J2 = out["cur2"][i] / out["ttime"][i] if m[i] else 0.0
            r1 = out["rho1"][i]
            r2 = out["rho2"][i]
            results.append((J1, J2, r1, r2))
    return results
