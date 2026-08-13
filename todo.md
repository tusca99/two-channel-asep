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

### Phase 3.5: GPU port (numba CUDA) — IN PROGRESS
Hardware: RTX 2060 (8GB), nvcc 12.9, numba CUDA works. Chose numba CUDA over torch (torch is bad fit for scalar per-system kernel; numba CUDA already installed, zero new deps). If numba CUDA hits a wall, fall back to raw CUDA in C (nvcc).
- [x] asep/cuda.py: kernel `mc_scan_kernel_fixed` — one thread per (alpha,beta) grid point, each runs full sequential Gillespie loop. Compile-time MAX_L=4096 scratch (cuda.local.array needs const size)
- [x] Per-thread xoroshiro128p RNG (numba curand), seeded from caller seed + thread index → reproducible (verified same seed = same result)
- [x] Correctness verified vs physics: LD (J~0.14,rho~0.17), MC (J~0.24,rho~0.47), HD/LD asymmetric (rho~0.80/0.08), LD/LD asymmetric
- [ ] BENCHMARK: CPU(12-core ProcessPool) vs GPU(RTX2060) speedup — NOT DONE, benchmark timed out. nvtop showed 33% efficiency / 100% usage during runs
- [ ] OPTIMIZE: 33% efficiency suggests occupancy/divergence issues. Ideas: (a) reduce local-array scratch (8194 floats/thread is huge → low occupancy), (b) use shared memory for lattice, (c) try raw CUDA in C via nvcc for max control, (d) check block size / grid config
- [ ] Add tests for CUDA correctness vs pure-Python reference (tests/test_cuda.py)
- [ ] Wire CUDA backend into phase-diagram / boundary scan scripts (swap scan_points → run_scan)
- [ ] Consider: one block per grid point (threads=sites) instead of one thread per point, for large-L single runs

### Phase 4: Presentation
- [ ] Beamer slides
- [ ] Live demo notebook
