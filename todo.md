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

### Phase 2.7: FIGURE 3 — MUST REVISE (current version is poor vs paper)
Paper's Fig 3: 3D plots of P(ρ₁,ρ₂) with CLEAR bimodal peaks (SSB), the highest region projected on the bottom plane (B&W in paper, we can do color). Our current 3D surfaces are "garbage" — peaks not distinguishable.
- [ ] Diagnose WHY our P(ρ₁,ρ₂) lacks clear peaks: system flips between broken states at L=1000, washing out the time-averaged distribution. Need to verify with vision model / longer runs / different beta range.
- [ ] Consider: sample the joint density over SHORT time windows (to catch the broken state before it flips) instead of the full run, OR increase L / steps as the paper (2e7-5e8 steps).
- [ ] Match paper's beta range (0.23-0.28) if possible, or justify our range.
- [ ] Add colored contour projection on the bottom plane so peak positions are clearly visible (paper is B&W, we can do better).

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

### Phase 3.5: Optimization exploration — SUMMARY (concluded, keep code lean)
Explored and measured, then removed unused code (cuda.py, xorshift) to keep repo clean. Final production path:
- [x] **BKL active-site list** (asep/bkl.py, wired into TwoChannelASEP.run default): ~2.2x over Gillespie
- [x] **ProcessPool parallel scans** (asep/parallel.py): ~4x (12 cores); one thread per run — MC is serial within a trajectory, so parallelism is only across realizations
- [x] **Numba > hand-written C**: clean C BKL (gcc -O2) = 559 ns/step vs numba 326 ns/step. C would NOT drastically help. GPU not worth it either (~1.2x vs CPU parallel, serial-chain structural limit). RNG was never a bottleneck (~4% of runtime).

### Phase 4: Presentation
- [ ] Beamer slides
- [ ] Live demo notebook
