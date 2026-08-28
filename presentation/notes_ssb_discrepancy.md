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

## Measured throughput vs beta (GPU, L=1000, nrep=2048, warmup=200k)

Earlier measurements (with too-short warmup) appeared to show a large low-beta
slowdown, but with a properly equilibrated warmup the throughput is essentially
FLAT in beta:

| beta | Msteps/s (kernel) | rho ~ |
|------|-------------------|-------|
| 0.05 | 67.1 | 0.35 |
| 0.30 | 72.0 | 0.35 |
| 0.90 | 71.9 | 0.41 |

So the Fenwick/GPU kernel is uniformly fast (~70 Msteps/s) across beta; the
earlier "31 vs 72" gap was an artefact of an un-equilibrated low-beta run.
Also, density DOES increase with beta (0.35 at low beta -> 0.41 at beta=0.9),
so the "constant rho" claim in a draft of these notes was wrong too.

Consequence: there is NO serious performance problem at low beta and no need
for an R=0 / hybrid optimization. (A "skip inert steps" idea would in any case
be invalid: every BKL step is a real event, and dropping events corrupts
currents and density dynamics.)

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

---

# 28 Aug session — papers digest, τ(L) campaign, outlook slide

## What was done

1. **Follow-up papers retrieved & read in full** (in `theory/`):
   - `zhu2012_pre.pdf` — Zhu et al., PRE 85, 041132 (2012)
   - `tian2017_chinphysb.pdf` — Tian et al., CPB 26, 020503 (2017) (+ arXiv 1605.01817, 13 pp)
   - `xiao2010_chinphysb.pdf` — Xiao et al., CPB 19, 090202 (2010)
   - `pronina2007_arxiv.pdf` — the original paper's arXiv preprint (cond-mat/0611472)
   Full verified digest: `theory/new_papers_summary.md`.

2. **Who studies what (corrected twice — read the PDFs, don't trust secondary sources):**
   - **Tian 2017 = exactly our model** (same rules). N-cluster MF (N=1..6) + current
     minimization ⇒ LD/LD "should not exist"; but the exponential decay is in **N (cluster
     size), NOT L**: β_c(N→∞) = 0.28871 at α=0.9 (vs MF 0.332, vs our SIM ≈0.25). Their MC is a
     single L=10⁴, which still sees the band. **Nobody has done the MC L-scaling** — our
     200/500/1000/2000 scan is the only one.
   - **Zhu 2012 = our model + leaky entrance**: injection at rate pα even when the other lane's
     exit is occupied (p=0 → Pronina, p=1 → decoupled). Both broken phases die at p_c≈0.6
     (SIM) / 0.5 (MF) — coupling is the SSB engine, our p=0 is maximal coupling. Plus τ ∝ e^L
     (their Fig. 3).
   - **Xiao 2010 = different process** (Y-junction TASEP, α₁≠α₂, p₁,p₂,p₃). No correlator, no
     SSB. Template only for the α₁≠α₂ extension.

3. **τ(L) campaign (new, 28 Aug)** — `scripts/tau_flipping.py`, `scripts/tau_plot.py`,
   data `results/tau/`, figure `results/tau/tau_L.png`. New slide inserted after P(ρ₁,ρ₂)
   ("How long-lived is a broken state?"). Deck now 45 pp, 0 err, 0 overfull. Tests 17/17.
   Everything committed & pushed (through 7ff618b).

## What is a "basin" (the talk-ready explanation)

In the SSB region the system has two stable broken states: HD/LD (d = ρ₁−ρ₂ > 0) and LD/HD
(d < 0). Landscape picture: **two valleys**. A **basin** = one valley + the region around its
minimum; the trajectory wanders inside a valley for a very long time (**dwell**), then makes a
rare fast crossing over the saddle (**flip**) into the other valley.
τ = median dwell time in one basin. Detector (final version): in-basin iff |d| ≥ dmin
(0.35 deep band, 0.15 near edge); flip = leave basin (+), enter basin (−).

**Key message (on the new slide): SSB = metastability.** τ ~ e^{L/ξ} with measured
ξ ≈ 390 at the band edge (β=0.26): τ ≈ 1.4·10⁴·e^{L/390}, ×100 per 1800 sites.
Deep band (β=0.18, 0.22): no flip at all in 6·10⁹ steps at L=2000 ⇒ τ ≳ 10¹⁰.
Consequences, all the same physics: (i) a finite run at large L samples ONE basin → that's
*why* single runs look symmetry-broken; (ii) equilibration needs c ∝ L steps/site (the fig5
rule); (iii) ensemble std(ρ₁−ρ₂) is the right order parameter — different replicas land in
different basins, so time-averaged |ρ₁−ρ₂| washes out but std stays finite.

## τ campaign results (α=0.9, single long trajectories, 8700K)

| L | β=0.18 | β=0.22 | β=0.26 |
|---|---|---|---|
| 200 | ≥10⁸ | 2.2·10⁵ | 1.3·10⁴ |
| 500 | ≥4·10⁸ | ≥4·10⁸ | 6.5·10⁴ |
| 1000 | ≥2·10⁹ | ≥2·10⁹ | 3.8·10⁵ |
| 2000 | ≥6·10⁹ | ≥6·10⁹ | 1.7·10⁶ |

- Detector lesson: first attempt (smoothed sign crossings) failed at the band edge — noise
  crossings near d≈0 made τ *shrink* with L. Basin projection (threshold on |d|) fixed it.
- Figure: log-log τ vs L; ▽ = "no flip" lower bound (marker at the run length, true τ above).
  β=0.18 is entirely lower bounds; β=0.22 measured only at L=200.
- Scope honesty (user's correction): method reproduced from **Zhu Fig. 3** (done for their
  leaky model A at p>0, α=0.5); we extend to **p=0** (original Pronina model — not covered by
  Zhu), at α=0.9, plus the β-ladder. Reproduction + extension, NOT "nobody did it".

## Why α=0.9 in all the SSB analyses

(i) Pronina & Kolomeisky's Fig. 5(a) boundary is drawn at α=0.9 → our fig5 reproduces *their*
figure at *their* operating point. (ii) Deep in the SSB region: wide broken bands, clean
signal. (iii) Tian 2017 quotes β_c(∞)=0.28871 exactly at α=0.9 → apples-to-apples with their
N→∞ extrapolation. The full α-plane scans (fig2/fig4/fig6) are unchanged — only the
boundary/SSB/τ runs pin α=0.9.

## RNG caveat (Tian's ref [53]) — noted on the outlook slide

At 10¹¹+ draws, PRNG artifacts become a real worry. Plan: per-replica counter-based streams
(Philox on GPU) or the AES-CTR-seeded Trivium bank from the FPGA-project percolation core
(64 provably independent streams, period ≥2¹⁴⁴). Validation precedent there: 2-D
**directed-percolation** p_c reproduced to 2·10⁻⁴ — different universality class from ASEP,
it validates the *streams*, not any ASEP physics (user's explicit caveat).

## Boundary reruns: cancelled (user decision)

The fig5 boundary L-scan stands as is; no c=100 rerun, no L=16000–20000. The τ campaign
answers the "does SSB survive" question from the dynamics side instead.
