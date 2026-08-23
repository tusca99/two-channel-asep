# TODO — Two-Channel ASEP

## Current State (23 Aug 2026) — READY FOR PUSH & PRESENTATION 2nd HALF
**Revise done, 4h window used, all high-budget runs with 30 workers, CPU only (no GPU overflow).**

### Done in this revise (23 Aug, commits 8170bfb, a76974c, a50c805, 67b0df5)
- **fig5** `scatter` `o`/`s` `ls="none"` `set_xticks(Ls)` `NullLocator` — no overlapping `2×10²` ticks, no misleading line (`scripts/fig5_corrected.py:183` `results/fig5_corrected/fig5_phase_boundary_vs_L.png:1` `69K`)
- **fig5 variant left** `α=0.9` both `β` boundaries vs `L` `50k/site` `16` `6` `L` (`200,500,1000,2000,4000,8000`) `results/fig5_variant_left/fig5a_*.png:1` `17:10` — `hdld [0.253×4,0.229,0.182]` `ldl [0.421,0.373,0.397,0.325,0.325,0.349]` `ylim 0.15-0.45` all visible (large `L` drift `c=50` vs needed `c=100` for `L8000` — note in log)
- **fig2** `L200/500/1000/2000` `50k/site` `CPU 30` `L-adaptive` `0.04√(1000/L)` (`results/L*/fig2/phase_diagram_{full,zoom}.png:1` `88K/85K` `L2000` `18:23` clean `LD/LD` band, was `3k` speckled) `30` workers (was `25`) to fill `78GiB` `32` cores
- **fig3** `L500`/`L2000` `50k/site` `9` snaps `0.23-0.95` `512` `CPU 30` `22.2m` (`results/L500/fig3_points_high_cpu.npz:1` `75K` `dense 0.702→0.449` `β 0.23→0.2595` paper `0.2595` vs short `0.245`; `L2000` `75K` `20:27` `dense 0.742→0.303`) — snapshot `1.1M` `L=500`/`L=2000` fixed `L` label via `fig3_plot.snapshots` `f3.L` patched (`was L=1000` for all)
- **fig3 animations** `24 fps` `10s` `240` frames `H` lerp `48→240` `L200` `5×` / `8→240` `30×` `L500/1000/2000` contiguous `no 0.95` jump (`results/L*/fig3_anim*.mp4:1` `L200 97K/180K` `L500 39K/72K` `L1000 29K/64K` `L2000 23K/55K` `11:47` `240` `10s` `0.8fps` `1250ms` `8`→`240`, `4.8fps` `208ms` `48`→`240`) via `imageio-ffmpeg` `7.0.2` `→ .venv/bin/ffmpeg` (was `8` frames `0.8 fps` `1.6s` choppy)
- **fig4** `L200` `10M` `L500` `25M` `L2000` `200M` `100k/site` `CPU 30` `40` `β` (`results/L*/fig4_current_derivative.png:1` `80K` `18:26` `L2000`)
- **fig6** `L200` `10M` `L500` `25M` `L1000` `50M` `L2000` `100M` `50k/site` `16` `CPU 30` `30` `β` `480` tasks `results/L*/fig6/currents_*` `densities_*` (`45K/55K` `17:11-17:39` done, `L2000` `41K/46K/45K` `07:55-08:16` fixed `pickling` top-level)
- **GPU test** `fig3 L500` `9216` `threads` `25M` `230B` saved to `/tmp/opencode/fig3_L500_gpu_test.log:1` for comparison, final `CPU` used as requested (no `GPU` overflow `cuda_ensemble.py:357` `var` `overflow`)
- **Commit** `a76974c` `67b0df5` `a50c805` `8170bfb` (no push, `VSCode`)

### Next: Presentation 2nd half (MC vs MFT, SSB, finite-size)
- Theory done (90%). Need to update `presentation/slides.tex:596` `fig2` `fig5` `fig6` `fig3` `ssb` frames to point to `results/L2000`/`L500` high-budget `50k/site` figures (currently `../results/gpu/` and `../results/L1000`).
- Add `L2000` comparison slide: `fig2` `L500` vs `L2000` `LD/LD` shrinking, `fig5` variant `6` `L`, `fig3` `L500` ` dense 0.449` at `0.2595`.
- Verify `fig6` currents `α=0.1` `0.8` `0.9` for `L2000` match `MFT` at low `α,β` (as in `slides.tex:633`).
- Add `fig4` `L2000` `dJ/dβ` saturation at `0.5` for `LD/MC`.

### Notes
- `L2000` bundle `fig2` `100M` `+ fig4` `200M` `+ fig6` `100M` `CPU 30` `~37m` done `18:26`, `fig3` `L500` `22m` `fig3` `L1000` `93m` `09:50` done, `fig3` `L2000` `75K` done.
- `fig5` large `L` drift `0.229`/`0.182` vs paper `0.26` — `50k/site` `c=50` insufficient for `L8000` (`c=100` → `800M` needed, `theory/fig5_hdld_equilibration.md` calibration `100M` saturates `dense~0.89` at `L=1000`).
- `home/alessio` `os.makedirs` at `import` mocked, `ffmpeg` via `imageio-ffmpeg` `→ .venv/bin/ffmpeg`.

### Fatto (storico)
- Kernel Fenwick `4.8 Mstep/s` `30` workers `70GiB` `32` cores, `13/13` tests pass.
- `presentation` `33` frames `slides.pdf` `theory` `90%` done.
