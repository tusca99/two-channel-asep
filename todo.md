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
- [ ] Reproduce Figure 2: Phase diagram
- [ ] Reproduce Figure 6: Currents & densities
- [ ] Finite-size scaling of LD/LD phase
- [ ] Density distribution P(ρ₁,ρ₂) for SSB detection

### Phase 3: Extension
- [ ] Asymmetric rates? Wider entrances? 3-channel?
- [ ] GPU port

### Phase 4: Presentation
- [ ] Beamer slides
- [ ] Live demo notebook
