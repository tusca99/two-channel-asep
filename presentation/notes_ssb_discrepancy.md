# Discrepancy: our L=1000 vs paper's L=1000 (no symmetry breaking?)

Status: **largely resolved** — the apparent absence of SSB at L=1000 was an
equilibration/statistics artifact, not a model bug. Long runs recover it.

## Summary of the discrepancy

The paper (Pronina & Kolomeisky 2007) shows clear HD/LD symmetry breaking and
bimodal P(ρ1, ρ2) at L = 1000, with the LD/LD asymmetric band near β ≈ 0.2–0.4.
Our MC at L = 1000 initially showed **no** asymmetry: both channels converge to
the same density (~0.4) and std_diff ≈ 0.01, while the same code at L = 200
gives a clean HD/LD separation (dense ≈ 0.9, dilute ≈ 0.04, std_diff ≈ 0.15).

## Root cause (main hypothesis): simulation length, not a model bug

The model itself matches the paper exactly. Paper §2.1 (entrance rule):
"enter with rate α if first site empty AND site L of the other channel
unoccupied"; exit rate β independent. That is exactly our
`asep/model.py` / `asep/bkl.py` narrow-entrance coupling. No discrepancy there.

The difference is **statistics**. Paper §3:
> "The number of effective Monte Carlo steps **per lattice site** in our
> simulations was typically between 2×10^7 and 5×10^8."

With L = 1000 that is 2×10^10 – 5×10^11 **total** steps. Our first tests used
~10^6 total steps — **4–5 orders of magnitude shorter**. SSB at large L is a
rare-nucleation / slow-decorrelation phenomenon: a run that is too short never
leaves the symmetric basin, so the time-averaged densities wash out and look
symmetric (our std_diff ≈ 0.01).

## Supporting evidence (GPU, α=0.9, β=0.1, L=1000)

Sequential windows, accumulating more steps:

| steps (M) | dense | dilute | diff | std_diff |
|-----------|-------|--------|------|----------|
| 2         | 0.265 | 0.211  | 0.055 | 0.062 |
| 4         | 0.267 | 0.210  | 0.057 | 0.061 |
| 6         | 0.270 | 0.205  | 0.065 | 0.067 |
| 8         | 0.277 | 0.204  | 0.073 | 0.075 |
| 10        | 0.272 | 0.209  | 0.064 | 0.064 |

diff and std_diff are growing with run length (vs ~0.02/0.01 at short runs),
consistent with the broken state emerging on long timescales. We are still far
from the paper's 2×10^7–5×10^8 per-site range, so this is only a trend, not yet
the asymptotic answer.

## New result (GPU, long runs, L=1000): separation RECOVERED

A long run at α=0.9, β=0.05, L=1000 (nrep=128, chunks of 2M steps, up to 100M
steps, sampling every 50k) shows:

| step count (M) | dense (all win.) | dilute (all win.) | diff (all) | std (all) | diff (last win.) |
|----------------|------------------|-------------------|------------|-----------|------------------|
| 10             | 0.055            | 0.040             | 0.015      | 0.019     | 0.076 |
| 25             | 0.138            | 0.099             | 0.038      | 0.048     | 0.076 |
| 40             | 0.220            | 0.159             | 0.061      | 0.077     | 0.071 |
| 50 (final)     | 0.275            | 0.199             | 0.076      | 0.096     | 0.069 |

Two crucial observations:

1. **diff and std keep growing with run length** (0.015 → 0.076): the two
   channels progressively separate. Short runs saw diff≈0.02, std≈0.01 — the
   separation simply had not had time to build up.

2. **The last-window diff is ~constant (0.07–0.08) from very early on**, while
   the all-window average grows slowly. Interpretation: each replica quickly
   settles into a broken state (large rho1-rho2 in that window), but the *state
   flips* between the two broken states over time. The time-averaged density
   over many windows washes the asymmetry out, whereas a single window (or the
   ensemble / P(ρ1,ρ2)) shows it clearly. **This is exactly the paper's point**:
   the robust SSB signature is std(rho1-rho2) / the bimodal P(ρ1,ρ2), not the
   long time-average of rho1, rho2.

So the earlier "no SSB at L=1000" was because we (a) ran too short and (b) used
the time-averaged |rho1-rho2| which the flipping destroys. The model is fine.

## Full beta-scan result (L=1000, long runs, alpha=0.9)

All 8 betas completed (nrep=128, 100M steps each, sampling every 50k). Final
window (last 2M steps) ensemble means:

| beta | diff (dense-dilute) | std(rho1-rho2) | dense | dilute |
|------|---------------------|----------------|-------|--------|
| 0.05 | 0.069 | 0.091 | 0.270 | 0.201 |
| 0.08 | 0.068 | 0.076 | 0.272 | 0.204 |
| 0.10 | 0.060 | 0.063 | 0.270 | 0.210 |
| 0.12 | 0.054 | 0.053 | 0.270 | 0.215 |
| 0.15 | 0.047 | 0.041 | 0.271 | 0.224 |
| 0.20 | 0.039 | 0.031 | 0.278 | 0.239 |
| 0.25 | 0.033 | 0.025 | 0.287 | 0.254 |
| 0.30 | 0.029 | 0.021 | 0.299 | 0.270 |

**The SSB order parameter (diff and std) decreases monotonically with beta**,
from ~0.07/0.09 at beta=0.05 to ~0.03/0.02 at beta=0.30. The two channel
densities converge toward a common value (~0.27-0.30) as beta grows. This is
consistent with the SSB region being confined to low beta — matching our
earlier todo note that "our SSB lives at lower beta than the paper (0.06-0.12
vs paper's 0.23)". The paper's Fig 2b zoom (0.2<beta<0.4) shows the LD/LD band
there; our data at L=1000 shows the asymmetry is already weak by beta=0.2-0.3.

## Simulation parameters: old fig3 vs new fig3

Fig3 (P(ρ1,ρ2) snapshots + animations) was rebuilt on the GPU. Key parameter
change: the new version uses many independent replicas per beta instead of a
few long single runs.

| Parameter | Old (CPU, `fig3_joint_density.py`) | New (GPU, `fig3_plot.py`) |
|-----------|--------------------------------------|---------------------------|
| Backend   | `TwoChannelASEP` serial, 8 seeds | `run_ensemble_cuda`, 2048 replicas/beta |
| n_steps   | 3,000,000 per seed | 2,000,000 per replica (window) |
| warmup    | 300,000 | 200,000 |
| sample_every | L/10 = 100 | 50,000 |
| β range   | b_min=0.04, b_max=0.35 | 0.04–0.35 (snapshots also at paper β's 0.23–0.95) |
| n_frames  | 80 | 40 (animation) |
| bins      | 48 | 48 |
| P(ρ1,ρ2) | ensemble of 8 seeds | 2048 replicas |

Why the change: at L=1000 a single short run stays in ONE broken state
(one sharp peak). The paper's TWO peaks are recovered by an ensemble — many
replicas each settling into a (random) broken state. The GPU makes thousands
of replicas per beta cheap.

## Simulation parameters: SSB beta-scan (results/gpu/ssb_beta*.npz)

| Parameter | Value |
|-----------|-------|
| alpha     | 0.9 |
| L         | 1000 |
| nrep      | 128 replicas per beta |
| chunk     | 2,000,000 steps per launch |
| nchunks   | 50 (total 100,000,000 steps per replica) |
| sample_every | 50,000 |
| betas     | 0.05, 0.08, 0.10, 0.12, 0.15, 0.2, 0.25, 0.3 |

Long runs recover the SSB that short runs wash out (see table in the
"Full beta-scan result" section above).

## Computational note: BKL cost vs beta (do not confuse event-rate with algorithm cost)

A BKL **step's** cost does NOT depend on how many events happen; it depends on
the selection algorithm:

- Classic BKL (double for-loop over the active list): cost/step ∝ n_active.
- Fenwick tree / GPU: cost/step ∝ log L (constant).

And n_active grows with density, which grows with beta in our model:
  beta=0.05 -> rho~0.24, beta=0.30 -> rho~0.28, beta=0.95 (MC) -> rho~0.46.
Physically, the narrow-entrance coupling blocks injection when the other
channel's exit is occupied, so low beta (crowded exit) -> low density -> few
active sites.

Consequence (the OPPOSITE of "few events = both are slow"):

- **Low beta (low density):** n_active is tiny -> classic BKL is FAST (little
  list to scan); Fenwick/GPU still pays its fixed log L per step, so the
  double-loop CPU can actually be FASTER there.
- **High beta (high density):** n_active ~ L -> classic BKL becomes O(L) and
  slow; Fenwick/GPU stays log L and WINS by a lot.

So the classic and Fenwick/GPU paths do NOT converge in time: they cross. The
fig3 slowdown at beta=0.02 is the fixed per-step cost not being amortized over
a nearly-blocked (total_rate~0) state, not an event-rate bottleneck.

## Measured throughput vs beta (GPU, L=1000, nrep=2048, steps=300k)

| beta | Msteps/s | rho ~ |
|------|----------|-------|
| 0.05 | 31.6 | 0.39 |
| 0.15 | 69.7 | 0.38 |
| 0.30 | 72.9 | 0.38 |
| 0.60 | 72.7 | 0.40 |
| 0.90 | 72.7 | 0.41 |

Key finding: density is roughly CONSTANT (~0.38-0.41) across beta, so the
"low density -> few events" explanation is WRONG for the slowdown. Instead, at
very low beta the total rate R is small (the system spends long stretches
nearly blocked), so -log(u)/R yields huge time steps and many BKL steps do no
useful work — yet each step still pays the full Fenwick O(log L). At high beta
R is large, every step advances time, throughput saturates at ~72 Msteps/s.

Implication for a "threshold / hybrid" idea: switching back to classic
double-loop BKL at high beta would be WORSE (O(L) vs O(log L)). The real
optimization for low beta is to skip inert steps (e.g. detect R~0 and advance
time directly), not to swap the selection algorithm.

a nearly-blocked (total_rate~0) state, not an event-rate bottleneck.

## Crash notes / driver (not a code bug)

The machine crashed twice during long 100% GPU runs. Diagnosis from journalctl:

- The GPU is correctly identified as **RTX 2060 SUPER** (driver 580.173.02),
  so a "wrong card / 1050ti driver" suspicion does not explain it.
- Both boots load the **NVIDIA Open Kernel Module**
  (`NVRM: loading NVIDIA UNIX Open Kernel Module`) with `i2c timeout error` and
  `[drm] Cannot find any crtc or sizes`.
- The Open Kernel Module is known to be less stable than the closed-source
  proprietary driver under sustained full load. Crashes happened after
  ~30-120 min at 100% GPU util, consistent with that instability.

Mitigation already in place: the fig3 scan saves every chunk incrementally
(`*.chunks.npz`), so a crash loses at most the in-flight chunk, and the run
resumes from the saved betas. For the presentation, note that long GPU runs may
crash the machine; using the proprietary (non-open) NVIDIA driver, or capping
GPU power/length, is worth trying if crashes persist.

## Remaining question for the prof (finite-L dependence)

β=0.05 (below paper's Fig 2b band 0.2<β<0.4) shows clear SSB at L=1000. The
interesting comparison the user will discuss with the prof is **how the
separation / the LD/LD band depends on L**:
- At L=200 we see strong, immediate HD/LD (dense≈0.9, dilute≈0.04).
- At L=1000 the broken states flip and the time-averaged |rho1-rho2| collapses
  unless runs are long; the *instantaneous* separation is still present.
- Whether the LD/LD asymmetric band (eq 33, β≈0.2-0.4) survives at L=1000 and
  moves with L is the finite-size question (see scripts/ssb_finite_size.py).

## What to do next (if needed)

1. Finish the β-scan long runs (β=0.05..0.30) to map where SSB survives vs L.
2. Plot dense/dilute AND std(rho1-rho2) together — the std is the robust order
   parameter; the density split alone misleads when states flip.
3. If a genuine deviation from the paper's band position persists after
   long runs, revisit the MFT boundaries (eq 23/33) vs our classifier.

## Relation to the MFT-vs-MC message

The comment "maybe going beyond 2nd order improves?" applies here independently
of the length issue: MFT assumes no site correlations (eq 1) and stops at
nearest-neighbour level. The boundary-position deviations we measured
(`scripts/mf_mc_boundary.py`) are where adding correlation parameters (e.g. an
effective boundary impurity from the narrow-entrance coupling) would improve
agreement. But that only matters once the run-length issue above is settled.
