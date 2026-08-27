# TODO — Two-Channel ASEP

## Current State (27 Aug 2026) — PRESENTATION 44pp, audited, ready for push
**Revise done 23 Aug (4h), recheck 27 Aug: data-integrity fix, programmatic + vision audit clean.**

### Presentation (done 27 Aug)
- `presentation/slides.tex` 44 pp (was 42), `xelatex` ×2: 0 err, 0 overfull h/vbox, 0 `\pause`, semantic colors, 10pt, graphicspath `L2000>L500>L1000>L200`.
- **Fix 27 Aug (frame 27, `mft_mc_deviation`)**: caption L=200→L=2000, recomputed from `L2000/fig6` npz: dev ≈0 deep, ~0.04 dips, HD/LD 0.23 vs MFT 0.33 (papers: `scripts/plot_mft_vs_mc.py`, cross-checked npz). Recompiled clean.
- After theory (~frame 22): all figures `results/L2000|L500|L1000|L200` high-budget 50k/site (old `results/gpu/` dead, replaced `figures/tasep_*.png` with TikZ/pgfplots, `\movie`→poster+`\href`).
- Frames: MC runs table (c·L), MC-vs-MFT L2000 fig2, fig6 trio, HD/LD jump, deviation, fig4 L2000, L500-vs-L2000 band, fig5a 6L, SSB L1000 (explicit), fig3 512 replicas + mp4, References, Thank-you, 2 backups ✅.
- Skill `.opencode/skills/beamer/` installed, verified identical to `Noi1r/beamer-skill` (SKILL.md diff clean).
- Vision path: local `gemma4:e4b` hung/empty (4-worker contention bug + 500tok think); pulled `gemma4:31b-cloud` (ollama cloud) → 3 tok/page, 2 min for 44pp audit.

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

### Next: Post-theory analysis — full results/figure review + new ToC (needs new session)
- **Handoff prompt at bottom of file** — next session must: load beamer skill, vision via `gemma4:31b-cloud` (ollama cloud, ~2 min/44pp, local `e4b` unreliable), inspect every `results/L*/` figure, propose a revised analysis ToC (structure, cuts, new comparisons).
- Theory done except remaining `\scriptsize` density flags on frames 13/14/18–20 (6 frames with >2 display eqs, beamer-mandated split candidate).

### Notes
- `L2000` bundle `fig2` `100M` `+ fig4` `200M` `+ fig6` `100M` `CPU 30` `~37m` done `18:26`, `fig3` `L500` `22m` `fig3` `L1000` `93m` `09:50` done, `fig3` `L2000` `75K` done.
- `fig5` large `L` drift `0.229`/`0.182` vs paper `0.26` — `50k/site` `c=50` insufficient for `L8000` (`c=100` → `800M` needed, `theory/fig5_hdld_equilibration.md` calibration `100M` saturates `dense~0.89` at `L=1000`).
- `home/alessio` `os.makedirs` at `import` mocked, `ffmpeg` via `imageio-ffmpeg` `→ .venv/bin/ffmpeg`.

### TODO (added 28 Aug) — for slide 46 / outlook
- [ ] New-papers digest done: `theory/new_papers_summary.md:1` (Xiao 2010 cluster $C$ + Tian 2017 $N$-cluster spurious LD/LD). Use its one-liners in outlook (slide 36) and speaker notes — no need to read papers.
- [ ] Slide 46 backup: consider adding a 1-slide “New papers at a glance” figure (2 columns: 2010 $C$ shift vs 2017 exponential LD/LD) if asked.

### Handoff — copy/paste into a NEW session (analysis part after theory)

Paste this as the first user message of the new chat:

```
You are working on two-channel-asep (Pronina & Kolomeisky J. Phys. A 40, 2275 2007) at /home/alessio/Documenti/two-channel-asep.

Context: presentation/slides.tex is 44 pp, beamer/Madrid, xelatex-clean (0 err/0 overfull). Beamer skill is at .opencode/skills/beamer/ (Noi1r/beamer-skill, SKILL.md identical to GitHub). 27 Aug recheck: frame 27 mft_mc_deviation caption fixed L=200→L=2000 (recomputed from L2000/fig6 npz, ~0.04 dips, 0.23 vs MFT 0.33).

Vision you MUST use (local gemma4:e4b is broken at this load): ollama cloud gemma4:31b-cloud
— already pulled (ollama list: gemma4:31b-cloud, ef09f235533c). It needs num_predict=700 (think-then-answer) and responds CLEAN vs ISSUE. A working harness is /tmp/opencode/vision_audit.py (API http://localhost:11434/api/generate, base64 images). 44 pp via that model is ~2 min, 3 tok/page. Reference image render: pdftoppm -jpeg -r 110 ... or pdftoppm -gray -r 100 ... . For raw results figures: results/L500 and L2000 fig2 zoom, fig6, fig4, fig5_variant_left, fig3_joint, etc. (check present with: ls results/L*/ and results/L*/fig*/)

Your task (analysis part AFTER theory):
1) Load the beamer skill (skill tool).
2) Systematically inspect EVERY results/ figure that the deck currently uses + candidates not yet used. For each, use gemma4:31b-cloud vision on the actual PNG (base64 via python like in vision_audit.py) to assess: what's shown, is it sharp, is the shrink/growth/band actually visible, are axes/legends projector-legible, is there clutter or duplicated content.
3) Cross-check against data provenance: L2000/fig6 npz for MFT-vs-MC, L500 vs L2000 fig2 grids/pngs (mean abs diff ~1.6, 15.9k px >30, max 61 — same band structure, marker jitter), fig5 6L drift, fig3 512-replica joint densities. Mark stale/underpowered figures (L large c=50 vs needed c=100).
4) Propose a revised post-theory ToC: section order, keep/cut/replace each slide, new comparisons (e.g. make L500→L2000 shrink POP rather than two nearly-identical zooms), what to drop if talk is 40 vs 60 min, which backup slides to add. Structure your output as: A) Figure-by-figure audit table (figure | verdict | notes) and B) Proposed ToC with per-slide title + figure path + speaker note cue (gate: ask before editing slides.tex).
5) Constraints to preserve: 10pt, 169, Madrid, \pos/\con/\HL, 0 \pause, graphicspath L2000>L500>L1000>L200, 50k/site budget caveat for L≥4000.
```

### Fatto (storico)
- Kernel Fenwick `4.8 Mstep/s` `30` workers `70GiB` `32` cores, `13/13` tests pass.
- `presentation` `33` frames `slides.pdf` `theory` `90%` done.
