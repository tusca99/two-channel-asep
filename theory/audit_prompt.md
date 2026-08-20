# New-session audit prompt: validate the code & tests, compare to the paper

You are auditing the two-channel ASEP Monte Carlo reproduction of Pronina &
Kolomeisky, J. Phys. A 40, 2275 (2007). A subtle GPU kernel bug was found and
fixed this session; your job is to re-validate the whole pipeline against the
paper and confirm the fix is correct and complete.

## Repo orientation
- `asep/model.py`  — TwoChannelASEP (pure-Python Gillespie reference)
- `asep/bkl.py`    — serial Fenwick BKL (`run_bkl_fenwick`) + batch variant
- `asep/cuda_ensemble.py` — GPU ensemble: `run_ensemble_cuda`, `run_ensemble_cuda_continue`
- `asep/parallel.py` — GPU grid/beta scans; `asep/observables.py` — phase classifier
- `scripts/`       — figure generators + `params_record.py` (writes params.json)
- `tests/test_numba.py` — GPU tests incl. 3 new physical-coverage tests
- `theory/`        — paper PDF + derivations; `results/L*/params.json` = run params
- CLAUDE.md / todo.md — model reference, performance facts, workflow rules

## The bug just fixed (READ first, verify it)
Symptom: at L=1000, per-replica rho2 went NEGATIVE or >1 in long runs.
Root cause: the running integer occupancy counters n1/n2 in
`asep/cuda_ensemble.py` drifted from the true lattice occupancy over very
long runs (the counters are incremented/decremented by `_execute`'s returned
delta). The time-averaged `rho1/rho2` and the raw samples used n1/n2, so they
went unphysical. The fix: at each sample, recompute occupancy DIRECTLY from
the lattice (O(L) per sample, negligible vs ~1e5 steps between samples).

## What to do
1. **Audit the fix.** Read `_advance_replica` in `asep/cuda_ensemble.py`.
   Confirm the sample block now sums `lane1`/`lane2` from the lattice, and that
   `capture_raw` uses the corrected r1/r2. Check that the running counters n1/n2
   are STILL used correctly for the current d (currents) — those must not be
   affected by the fix (currents come from exit counters, not occupancy).
2. **Run the tests.** `python -m pytest tests/ -q` — expect 19 passed. Add more
   physical coverage if you find a gap (see ideas below).
3. **Validate against the paper.** For each claim in the paper table
   (TASEP phases LD/HD/MC, currents J=rho(1-rho), SSB bimodality, the
   MFT-vs-MC boundary deviations), check the reproduced numbers in
   `results/L*/`. Confirm:
   - bulk densities in [0,1], currents >= 0 and J <= 1/4 in MC
   - LD phase: rho ~ alpha, J ~ alpha(1-alpha)
   - HD phase: rho ~ 1-beta
   - MC phase: rho ~ 1/2, J ~ 1/4
   - narrow-entrance symmetry: at alpha=beta symmetric, J1~J2 and rho1~rho2
4. **Regression sweep.** Run the GPU ensemble at several (alpha,beta,L) and
   assert every per-replica rho1,rho2 in [0,1] (this is what the new
   `test_gpu_density_bounds_L1000` guards). Test L in {200, 1000, 4000} and
   long run lengths (>= 1e7 steps/site).
5. **Check raw-sample path.** The high-res fig3 uses `run_ensemble_cuda_continue`
   with `n_raw_samples`; verify `raw_samples`/`sample_count` are populated and
   in-bounds, and that resetting `sample_count[:]=0` between chunks works.
6. **Report** concisely: (a) is the fix correct & complete? (b) any remaining
   unphysical values? (c) test count + which new tests you added. Cite paper
   section numbers for any physics claim.

## Suggested additional physical tests (add if you can)
- `test_gpu_conservation`: for a closed-ish set of moves, total particles in a
  lane changes only by injection/exit — verify rho1+rho2 stays in a sane band.
- `test_gpu_symmetry`: symmetric alpha=beta -> J1≈J2, rho1≈rho2.
- `test_gpu_ld_hd_mc`: verify the three TASEP phase expectations above.
- `test_serial_fenwick_density_bounds`: same in-[0,1] check on the CPU
  `run_bkl_fenwick` path (should be trivially true, but guards parity).

Do not edit results/ figures unless asked; the goal is to VALIDATE code + tests.
