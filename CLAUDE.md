# CLAUDE.md - Project Context for OpenCode

## Project: Two-Channel ASEP MC Reproductions

Reproduce: Pronina & Kolomeisky, J. Phys. A 40, 2275 (2007)

### Quick Model Reference
- `Lane 1` (→): enters site 0, exits site L-1, hops right (rate 1)
- `Lane 2` (←): enters site L-1, exits site 0, hops left (rate 1)
- **Narrow entrance**: enter lane1 ⟺ lane2[0] (exit of ch2) is empty
- **Narrow entrance**: enter lane2 ⟺ lane1[L-1] (exit of ch1) is empty
- Exit rate β is independent of the other channel

### TASEP Phase Reference (from course L18-19)
| Phase | Condition | Bulk Density | Current |
|-------|-----------|-------------|---------|
| LD    | α<β, α<1/2 | α | α(1−α) |
| HD    | α>β, β<1/2 | 1−β | β(1−β) |
| MC    | α>1/2, β>1/2 | 1/2 | 1/4 |

### Code Structure
```
asep/          ← simulation code (importable)
  model.py     ← TwoChannelASEP class (uses run_bkl_fenwick by default)
  bkl.py       ← run_bkl (classic), run_bkl_fenwick (Fenwick, O(log L)), run_bkl_profiled
  cuda_ensemble.py ← GPU: thread=replica, RNG on-device, float32, stats on-device
  parallel.py  ← scan_points/scan_grid_gpu/scan_beta_gpu/scan_phase_diagram_gpu
scripts/       ← run_all_gpu.py (scan), plot_all.py (figures), fig3_plot.py, fig3_extra.py,
                 fig5_boundaries.py, gen_L200_cpu.py, plot_mft_vs_mc.py, limit_gpu.sh
tests/         ← must pass (13 tests)
presentation/  ← slides.tex/pdf (33 frames), notes_ssb_discrepancy.md, findings_presentation.md
theory/        ← notes, derivations, paper PDF
results/       ← raw data + figures (gitignored); results/L200/ = L=200 figure set
```

### Key Performance Facts (measured)
- CPU Fenwick: ~4.4M step/s per core; classic BKL ~1.8M/s. Fenwick ~2x classic on CPU too.
- GPU per-thread: ~35k step/s (BKL serial, latency-bound). GPU wins ONLY via mass
  parallelism (2048 threads → ~80M/s aggregate).
- For long equilibration runs (many steps per single trajectory), CPU is ~100x faster
  than GPU per-thread. For L=200 with ~100 realizations, CPU is 13x faster than GPU.
- GPU ensemble is best for: phase diagram (many points), P(ρ1,ρ2) ensemble, big statistics.

### Workflow Rules
1. Small changes — test before you commit
2. Verify MFT vs MC at low α,β first (LD phase), then scale up
3. Keep responses lean — no unnecessary text
4. Always cite the paper section when referencing results
5. For L≤1000 single-trajectory runs, prefer CPU (12-core) over GPU
6. results/ is gitignored — figures are regenerable, don't commit them

### Next Steps (check /todo.md — has full session handoff)
