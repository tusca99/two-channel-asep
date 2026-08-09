# TODO — Two-Channel ASEP

## Phase 1: Foundation
- [x] Basic TwoChannelASEP class (Gillespie/MC step)
- [x] Test LD, MC, HD/LD phases
- [x] Verify entrance logic (blocked by exit site of other lane)
- [x] Density profile measurement

## Phase 2: Reproduce Key Results
- [x] Phase diagram scanner working (classifies LD, MC, HD/LD, LD/LD)
- [x] Verify MFT deviations at large α,β (confirmed: MC current ~0.21 vs 0.25)
- [ ] Refine classification thresholds (some borderline points misclassified)
- [ ] Reproduce Figure 6: Currents & densities for α=0.9
- [ ] Finite-size scaling (L=100,500,1000,2000) for HD/LD and LD/LD phases
- [ ] Finite-size scaling of LD/LD phase (does it disappear?)
- [ ] Density distribution P(ρ₁,ρ₂) for symmetry breaking detection

## Phase 3: Extension (if time allows)
- [ ] Option A: Asymmetric hopping rates (v₁≠v₂)
- [ ] Option B: Wider entrance coupling (bulk ↔ bulk)
- [ ] GPU port with CUDA (if A100/H100 needed)

## Phase 4: Presentation
- [ ] Beamer slides (60-min structure)
- [ ] Live code demo notebook
- [ ] Theory summary handout
