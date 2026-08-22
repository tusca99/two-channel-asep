# New-session handoff: two-channel ASEP (Pronina & Kolomeisky 2007)

Continue from commit `ac3d08c`. Read `CLAUDE.md` (model + workflow rules) and
`todo.md` first. Key: background runs MUST log to `results/<tag>.log`
(TeeLog now captures stderr, so tqdm appears). Never block on a probe; poll
the log. Run long jobs with `setsid ... > /dev/null 2>&1 & disown`.

## Current state (what's committed)
- Density-drift fix (`0dc7705`): GPU kernel now recomputes occupancy from the
  lattice at each sample. Physical tests added (20 pass).
- HD/LD classifier fix (`7794fb9`): uses `dense=max(rho1,rho2)>0.5` instead of
  std(rho1-rho2) (which was ~0 for a stuck replica). Phase diagrams now show
  the broken-symmetry region.
- `fig5_corrected` now uses an L-scaled budget for the beta boundary.

## IMPORTANT — a background job is likely STILL RUNNING
`fig5_corrected.py` was launched and is grinding the heavy L=4000/8000 scans.
Check FIRST:
```
ps aux | grep fig5_corrected | grep -v grep
```
- If running: let it finish, then `git add results/fig5_corrected/ && commit`.
  The npz currently has L200-2000 → beta_asym=0.253, alpha_mcld=0.673;
  L4000/8000 should append beta_asym≈0.253 too (or NaN if under-equilibrated —
  then recheck the budget in `run_one_L`).
- If it died: relaunch `taskset -c 0-15 .venv/bin/python scripts/fig5_corrected.py`.

## Open items (pick up here)
1. **Verify the HD/LD classifier** (the user is skeptical: "blocks of points
   higher than the HD/LD threshold, and some below"). Compare each grid cell's
   classifier label vs the physically-motivated rule `dense>0.5`:
   ```
   alpha | last HD/LD beta | MFT alpha/(1+alpha+alpha^2) | match
   ```
   Confirm L200/L500/L1000 phase diagrams look right (in results/L*/fig2/).
   Expect MFT-vs-MC deviation at high alpha (the paper's point), but the
   boundary should track MFT at mid-alpha.
2. **Plot the corrected phase diagrams** (L200/L500/L1000 already regenerated
   with the fixed classifier; make sure fig2 PNGs are current) and commit.
3. **Commit the final fig5** npz + the stray `theory/fig5_phase_boundary_vs_L.png`.
4. Cleanup: `scripts/debug_fig5_beta.py` was a scratch debug — remove or keep
   (it's harmless).

## Known items (do NOT redo unless asked)
- fig3 is FINAL for all L (it's just P(rho1,rho2), doesn't use the classifier;
  density samples already fixed). The β~0.22 "artifact" is finite-replica
  basin-bias (all reps land in one basin), not a bug.
- L1000 raw chunks are in `results/L1000/chunks/` (gitignored, derivable).

## Useful commands
- Run fig5: `taskset -c 0-15 .venv/bin/python scripts/fig5_corrected.py`
- Check fig5 progress: mtime of `results/fig5_corrected/fig5_boundaries_corrected.npz`
  (it saves incrementally per-L now).
- Kill: `ps aux | grep fig5_corrected | awk '{print $2}' | xargs -r kill -9`
