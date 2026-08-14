# TODO — Two-Channel ASEP

## Current State: Paper Reading Phase
- [x] Repo scaffold created (public, uv-based)
- [x] Basic TwoChannelASEP class (pure Python, Gillespie step) — WORKING
- [x] Phase scanner (classifies LD/MC/HD-LDD/LD-LDL) — WORKING
- [x] Theory notes draft — MFT derivation, phase table
- [x] OpenCode agent context (.opencode/, CLAUDE.md) — set up
- [x] Numba acceleration module — written, tested (reproducibility bug fixed: kernel now consumes seeded numpy RNG stream)

## Next: Code Structure Design (Tomorrow)
After reading the paper, decide on:
- [ ] Core abstraction: Model class vs. Simulator class
- [ ] MC backend strategy: pure Python → numba → CUDA (when needed)
- [ ] Observation strategy: streaming vs. batch sampling
- [ ] Phase detection: threshold-based vs. clustering vs. density distribution P(ρ₁,ρ₂)
- [ ] Finite-size scaling plan: which L values, which observables
- [ ] Reproducibility: checkpoints, random seeds, data logging format

## Long Term (Post Paper Read)
### Phase 2: Reproduce Key Results
- [x] Reproduce Figure 2: Phase diagram (MFT lines + MC points, scripts/phase_diagram.py)
- [x] Reproduce Figure 6: Currents & densities (scripts/currents_densities.py)
- [x] Finite-size scaling of LD/LD phase (scripts/ssb_finite_size.py)
- [x] Density distribution P(ρ₁,ρ₂) for SSB detection (scripts/ssb_analysis.py)

### Phase 2.5: Improve currents/densities figure (error bars + L)
- [ ] Error bars: add n_reps replicas per beta (done in code, needs rerun). Current errors are small (~0.009, < dot size); density errors are large (~0.18) in SSB phase because replicas land in different broken states (genuine bimodality, not noise)
- [ ] Increase L to ~1000 (paper used 1000-12000) to close finite-size gap vs MFT. NOTE: larger L does NOT reduce statistical error bars (those come from steps+replicas); it only fixes finite-size bias. 100k is beyond paper range and overkill
- [ ] Note in legend if error bars are smaller than the dots

### Phase 2.6: KEY IDEA — MF transition positions are NOT exact (professor's focus)
The paper's central point: unlike standard single-lane TASEP (where MFT gives exact phase boundaries), here the MFT transition positions deviate from MC. Investigate in detail:
- [x] Add embarrassingly-parallel MC scans (asep/parallel.py, ~7-10x speedup on 12 cores) — no more waiting on serial beta scans
- [x] Measure the LD/MC boundary shift vs MFT (scripts/mf_mc_boundary.py): MC boundary sits at HIGHER beta than MFT; deviation grows with alpha (+0.01 at a=0.8 -> +0.075 at a=1.0)
- [ ] For each boundary, measure the MC transition position vs the MFT prediction (e.g. α=2β/(4β−1) for LD/MC) and plot the deviation vs β
- [ ] Check whether the deviation shrinks with L (finite-size) or persists in the thermodynamic limit (genuine MF failure) — NEED LONGER RUNS at L=800 (equilibration-limited, not parallelization-limited)
- [ ] Compare with standard TASEP: confirm MFT is exact there (α=β=1/2 lines) vs. not exact here — this contrast is the paper's message
- [ ] Explain WHY: MFT neglects correlations; the narrow-entrance coupling creates boundary correlations that MFT misses (effective impurity at the boundary)
- [ ] Add a slide/figure in the presentation dedicated to this MF-vs-MC discrepancy

### Phase 3: Extension
- [ ] Asymmetric rates? Wider entrances? 3-channel?

### Phase 3.5: GPU port (numba CUDA) — CONCLUSION: NOT WORTH IT FOR PRODUCTION
Hardware: RTX 2060 (8GB), nvcc 12.9, numba CUDA works. Chose numba CUDA over torch (torch is bad fit for scalar per-system kernel; numba CUDA already installed, zero new deps).
- [x] asep/cuda.py: thread kernel (one thread per grid point) + block kernel (one block per point, lattice in shared memory, parallel rate reduction)
- [x] Per-thread/block xoroshiro128p RNG (numba curand), reproducible (verified same seed = same result)
- [x] Correctness verified vs physics: LD/MC/HD-LD/LD-LD all match
- [x] BENCHMARK (2000 pts, L=200, 200k steps): GPU block kernel = 29 pts/s, CPU 12-core = 24 pts/s → GPU only ~1.2x faster. NOT worth the complexity.
- [x] Shared-mem block kernel is 4x faster than thread kernel (52s->13s for 100 pts), but still ~= CPU parallel
- [x] ROOT CAUSE (structural, NOT numba): MC is a serial chain — each move depends on the previous. GPU can't parallelize within a trajectory; only runs many systems concurrently. Occupancy capped at 4 blocks/SM (shared mem), per-step syncthreads barrier serializes. 34-43% efficiency is inherent.
- [x] L2 cache / more concurrency won't fix it: already at occupancy ceiling.
- [ ] DECISION: use CPU parallel (ProcessPool) + BKL for production scans. GPU port kept as a learning exercise / optional backend, not default.
- [ ] (optional) BKL CUDA kernel: BKL removes the per-step reduction, so thread 0 could run the active list against shared mem without the syncthreads barrier — the one design that might beat CPU. Low priority.

### Phase 3.6: BKL (active-site list) — DONE, 2.2x speedup
- [x] asep/bkl.py: Bortz-Kalos-Lebowitz, incremental O(1) active-list update in 3-site window. Cuts per-step cost from O(L) to O(active). Verified correct vs Gillespie; ~2.2x speedup across densities.
- [x] Wire BKL into TwoChannelASEP.run() as an option (use_bkl=True) so scans use it by default
- [x] Add xorshift64* inline RNG (run_bkl_xor) as a numpy-free path. Result: only 1.07x faster. Corrected measurement: RNG gen is ~3.9% of the full run (35ms of 914ms), not 30% — an earlier mislabel from comparing against the raw kernel only. RNG was never the bottleneck; xorshift kept as optional backend, not default.
- [x] Parallelization strategy CONFIRMED: one thread per run (independent realizations in parallel via ProcessPool). MC is serial within a trajectory, so parallelizing steps is not possible; the only parallelism is across realizations.
- [x] C vs numba: wrote a clean C BKL (gcc -O2). numba BKL (326 ns/step) BEATS the C version (559 ns/step). LLVM backend > hand-written C here. C would NOT drastically help.
- [ ] OPTIONAL: xorshift in model.run() as default if we want to drop the numpy uniform stream (minor gain)

### Phase 4: Presentation
- [ ] Beamer slides
- [ ] Live demo notebook
