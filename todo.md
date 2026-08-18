# TODO — Two-Channel ASEP

## Current State (18 Aug 2026) — SESSION HANDOFF
**Tutto committato e pushato (origin/main). Working tree pulito.**

### Fatto
- **Kernel**: Fenwick BKL (`run_bkl_fenwick`, O(log L), ~4x classic) + GPU ensemble
  (`cuda_ensemble.py`, thread=replica, RNG on-device, float32, stats on-device).
  `model.py` usa Fenwick di default. 13/13 test passano.
- **Figure L=200** in `results/L200/` (fig2, fig6, fig3 snapshots+anim, ssb, mft_vs_mc).
  `run_all_gpu.py`/`plot_all.py` parametrizzati con `--L`/`--out`.
- **fig3**: ensemble GPU, snapshots densi + animazioni 2D/3D (β 0.04→0.95).
- **fig5**: boundary vs L su CPU (L=200,500,1000) — solo 2 punti validi, abbandonata.
- **MFT-vs-MC**: boundary teorici verificati (eq 10/13/23/33); deviazione J_MC-J_MFT
  misurata (max ~0.036). Presentazione Beamer (33 frame) in `presentation/`.

### Scoperte chiave (vedi presentation/notes_ssb_discrepancy.md)
- SSB a L=1000 è RISOLTO: il "dense~0.27, non vero HD" era un bug dei dati
  (per-chunk reseed in run_all_gpu.py, non un run continuo). Un run continuo
  raggiunge vero HD/LD (dense~0.89). Vedi theory/fig5_hdld_equilibration.md.
- **CPU per-thread 4.4M step/s vs GPU 35k/s** → per run lunghe (equilibrazione) la CPU
  è 100x meglio; GPU vince solo col parallelismo di massa (phase diagram, ensemble).
- Per L=200 la CPU è 13x più veloce della GPU per ~100 realizzazioni.

### Prossimo (non fatto)
- [x] **Convergenza adattiva** (punto 2): `TwoChannelASEP.run_adaptive` ferma la
  run quando la corrente a finestra (ultimi `win_steps`) coincide con la corrente
  cumulativa entro `tol`. In MC (α=0.9, β=0.9, L=200) si ferma a ~29% dei max_steps
  con |ΔJ|~0.01. Test `test_adaptive_convergence` aggiunto (14/14 passano).
  Wired into `scan_points(adaptive=True)` + scripts fig6/currents_densities/mf_mc_boundary.
  **Speedup misurato (L=200, 2M max_steps): MC/HD ~3-4x, LD ~20x** (la corrente
  converge presto).
- [ ] **AVX across trajectories** (punto 3): MISURATO e RIDIMENSIONATO. Il gain è
  multi-core, non SIMD: `run_bkl_fenwick_batch` (prange su repliche indipendenti,
  ognuna col suo slice di uniforms) = 14.0 Mstep/s su 12 core vs 2.9 Mstep/s
  seriale (~4.8x), pari a ProcessPool (~5x). Le catene Fenwick find/update sono
  data-dependent e non vettorizzabili. `scan_points_batch` NON è più veloce di
  ProcessPool (overhead Python per-replica lo cancella). Test batch==serial
  bit-identico aggiunto (15/15 passano).
- [x] **Classifier fig5/fig2**: MC via current-saturation (J plateau ~1/4, metodo
  paper Fig 4) + rho>0.45 invece di rho>0.35 → boundary MC/LD α ~0.70 (prima
  0.477; il paper ~0.7-0.8). Soglia SSB L-adattiva 0.04·√(1000/L) → fix NaN a
  L=200 (boundary asym→LD ~0.325). n_reps>=16 → fix outlier L=2000 (0.373→0.325).
  VERIFICATO su /tmp/opencode/fig5_mcld.npy e fig5_L2000.npy.
- [ ] **HD/LD RISOLTO**: NON era equilibration-limited. Calibration probe
  (/tmp/opencode/calib_probe.py, α=0.9 β=0.1 L=1000, 12 rep, fino 500M step/rep,
  ~10 min): dense≈0.89 dilute≈0.06, saturo da ~100M step, 12/12 rep. Il claim
  "dense mai >0.5" derivava da un BUG dati: `run_all_gpu.py` resetta ogni chunk
  con seed nuovo (traiettorie corte indipendenti, non un run continuo). Action:
  rifare ssb scan GPU con traiettorie continue. Vedi
  theory/fig5_hdld_equilibration.md. Backend (GPU/Rust/AVX) tutti ~entro 2x
  (serial chain); il lever è un run continuo corretto (~10 min L=1000), non
  budget enormi.
- [x] **GPU BKL profiling (Nsight Compute)**: il 35 kstep/s/replica è REALE e
  spiegato: 95.8% stall = long_scoreboard (latenza memoria globale sul Fenwick
  scattered per-thread), NON Numba/RNG/matematica. Persiste a 57.8% occupancy.
  Fenwick è già meglio di classic BKL (che peggiorerebbe). Rewrite CUDA C NON
  aiuta (stessa global mem). Aggregate GPU ~72 Mstep/s vs CPU 12-core ~58
  (~1.2x). Vedi theory/gpu_bkl_profiling.md. Lever = saturare occupancy (nrep
  migliaia), non backend.
- [ ] **New paper (Jimenez & Ortiz 2015, OkMC GPU)**: theory/1-s2.0-...pdf.
  Parallel event-selection per-particle. MA: particelle indipendenti (no
  exclusion); la nostra exclusion coupling rende "parallel across particles"
  harder. Da valutare per l'extension, non per il kernel attuale.
- [ ] Error bars in fig6 (n_reps già nel codice, serve rerun).
- [ ] Presentazione: aggiungere figure L200, sezione MFT-vs-MC.

## Paper Reading Phase
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

### Phase 2.7: FIGURE 3 — REVISED (ensemble reconstruction + animation)
Paper's Fig 3: 3D plots of P(ρ₁,ρ₂) with CLEAR bimodal peaks (SSB). Root cause of our poor version: at L=1000 a single MC run gets STUCK in one broken state (one peak); the paper's P shows two peaks because their sampling visits both broken states.
- [x] Fix: average P(ρ₁,ρ₂) over an ENSEMBLE of seeds (each lands in a random broken state) → reconstructs the two-peak structure
- [x] Colored contour projected on bottom plane (paper is B&W; we use turbo colormap) so peak locations are clear
- [x] Snapshots at the paper's exact β values (0.23-0.95)
- [x] 3D animation sweeping β (0.04-0.35, parallelized precompute) showing peaks emerge/merge
- [ ] NOTE: our SSB lives at lower β than the paper (0.06-0.12 vs paper's 0.23) — the MFT-vs-MC discrepancy. Snapshots at paper's β show diagonal peaks (symmetric); the off-diagonal peaks appear in our β range.
- [ ] Run full 80-frame animation with fine β steps (0.0001) for the slow, sensitive region

### Phase 2.5: Improve currents/densities figure (error bars + L)
- [ ] Error bars: add n_reps replicas per beta (done in code, needs rerun). Current errors are small (~0.009, < dot size); density errors are large (~0.18) in SSB phase because replicas land in different broken states (genuine bimodality, not noise)
- [ ] Increase L to ~1000 (paper used 1000-12000) to close finite-size gap vs MFT. NOTE: larger L does NOT reduce statistical error bars (those come from steps+replicas); it only fixes finite-size bias. 100k is beyond paper range and overkill
- [ ] Note in legend if error bars are smaller than the dots

### Phase 2.6: KEY IDEA — MF transition positions are NOT exact (professor's focus)
The paper's central point: unlike standard single-lane TASEP (where MFT gives exact phase boundaries), here the MFT transition positions deviate from MC. Investigate in detail:
- [x] Add embarrassingly-parallel MC scans (asep/parallel.py, ~7-10x speedup on 12 cores) — no more waiting on serial beta scans
- [x] Measure the LD/MC boundary shift vs MFT (scripts/mf_mc_boundary.py): MC boundary sits at HIGHER beta than MFT; deviation grows with alpha (+0.01 at a=0.8 -> +0.075 at a=1.0)
- [x] Verify MFT boundary formulas against the paper (eq 10/13, 23, 33) — confirmed in phase_diagram.py. For alpha=0.9: HD/LD↔LD/LD at beta=0.3321, LD/LD↔LD at beta=0.3324 (nearly coincident); for beta=1.0: MC/LD at alpha=0.6667. NOTE: at L=1000 our SSB is a weak LD/LD-type asymmetry (dense~0.27, not a true HD), so the paper's HD/LD boundary is not directly reproducible; see presentation/notes_ssb_discrepancy.md.
- [x] fig5 phase boundaries vs L (scripts/fig5_boundaries.py, CPU): asym->LD boundary in beta drops with L (0.349 at L=500 -> 0.301 at L=1000, MFT 0.332); LD->MC boundary in alpha ~0.477 constant (MFT 0.667, large MFT-vs-MC deviation). L=200 points are NaN (SSB too strong at small L for the fixed threshold).
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
