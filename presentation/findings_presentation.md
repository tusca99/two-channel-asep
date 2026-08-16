# Key findings for the presentation

Collected notes for the talk on the two-channel ASEP with narrow entrances
(Pronina & Kolomeisky 2007). See also `notes_ssb_discrepancy.md` for the
technical SSB analysis.

## 1. Model reproduction (working)

- Model reproduced exactly: channel 1 hops right, channel 2 hops left, hard-core
  exclusion, narrow-entrance coupling (enter α only if other channel's exit
  site empty), independent exit β.
- Phase diagram, currents, densities and P(ρ1,ρ2) all reproduced.
- MFT boundaries from eqs 10/13/23/33 match the paper's mean-field lines.

## 2. MFT vs MC: transition positions are NOT exact (professor's focus)

- Unlike single-lane TASEP (where MFT gives exact boundaries at α=β=1/2), here
  the MFT transition positions deviate from MC.
- The MC LD/MC boundary sits at HIGHER β than MFT; deviation grows with α
  (+0.01 at α=0.8 → +0.075 at α=1.0).
- Cause: MFT neglects correlations. The narrow-entrance coupling creates
  boundary correlations (an effective impurity at the boundary) that MFT misses.
- This is the paper's central message and our cleanest presentation slide.

## 3. Spontaneous symmetry breaking (SSB)

- Two asymmetric phases: HD/LD (first order) and LD/LD (continuous).
- Robust order parameter: std(ρ1−ρ2) over an ensemble (not the time-averaged
  |ρ1−ρ2|, which the state-flipping washes out).
- At L=1000 our SSB lives at LOWER β than the paper (β≈0.06–0.12 vs paper's
  0.23). Long-run β-scan: diff/std decrease monotonically with β
  (0.069/0.091 at β=0.05 → 0.029/0.021 at β=0.30).
- Finite-size: at L=200 strong immediate HD/LD; at L=1000 broken states flip.

## 4. Computational: BKL + GPU ensemble

- Fenwick-tree BKL (O(log L) selection) is ~4x faster than classic double-loop
  BKL; identical physics.
- GPU ensemble (one thread per replica) reaches ~85 Msteps/s on RTX 2060 SUPER,
  ~19x single-thread CPU, ~2.9x 12-core CPU.
- "One giant run → derive all figures offline" workflow (`run_all_gpu.py` +
  `plot_all.py`) avoids re-running MC per figure.
- Cost model: classic BKL ∝ n_active (fast at low β), Fenwick/GPU ∝ log L
  (wins at high β) — see notes_ssb_discrepancy.md.

## 5. Open questions / next steps

- Whether the LD/LD band survives at L=1000 and moves with L (finite-size).
- Whether the MFT-vs-MC boundary deviation shrinks with L or persists (genuine
  MF failure). Needs longer runs at L=800 (equilibration-limited).
- Going beyond 2nd order / adding correlation parameters might improve MFT;
  needs reading the follow-up literature.
