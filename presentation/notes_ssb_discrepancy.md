# SSB order parameter — corrected definition & data-quality note (30 Aug 2026)

Context: slide "SSB: a robust order parameter" (deck section *Finite size and
SSB*). Triggered by the question "isn't the order-parameter plot a repetition
of the fig6 densities plot?"

## 1. The original slide's orange curve was the wrong object

`ssb_order_vs_beta.png` (plot_all.py / reduce_L1000.py) plotted, as "std":

    std_diff = per-replica TIME std of (rho1 - rho2), averaged over replicas
               (cuda_ensemble.py stats accumulator st[4], divided by n
               samples; ~0.115-0.17 flat across beta).

This is a fluctuation size ("temperature"), NOT the SSB diagnostic:

- In the broken phase each replica sits in ONE basin, its own time std is just
  the intra-basin fluctuation width (~0.12).
- In the SYMMETRIC phase it is also ~0.12 (same fluctuation width) — zero
  discriminating power. The number plateaus at all beta, which is why the old
  orange curve looked uninformative.
- Crucially, std_ens over replicas of the mean gap = sqrt(m^2/2 + s^2) for a
  50/50 sign-bimodal ensemble: the old value 0.125 at beta=0.05 was an
  arithmetic accident of the true broken moment m=0.885 (m/sqrt(2)+s). It was
  NOT measuring "different replicas break different ways".

The old slide text ("std stays finite precisely because single trajectories
flip sign") was correct about the time-averaged |rho1-rho2| collapse but the
plotted quantity was not the ensemble std the text described.

## 2. Correct order parameters (both ENSEMBLE level)

For each beta: run N independent replicas, per replica record the time-averaged
gap g_i = <rho1 - rho2> (steady state):

- broken moment        m(beta) = <|g|> over replicas
                       ("magnetization" of the Ising analogy; equals the
                       vertical HD/LD branch gap of fig6 — connected to it,
                       but on the ensemble definition needed for SSB)
- broken fraction      f_br(beta) = fraction of replicas with |g_i| > 0.1
                       (1 = every replica broken; 0 = symmetric phase.
                       Replaces the degenerate std curve; visualizes the
                       coexistence wedge at the band edge.)
- (if wanted) std_ens(g) is then genuinely the 2-basin signature: m/sqrt(2)
  when 50/50, ->0 when unbroken. Redundant with (m, f_br) so we drop it from
  the figure; mention in notes.

Analogy summary for the talk:
  m  = |magnetization| magnitude;  f_br = how many copies of the system broke;
  single-replica time-avg |g| = would-be magnetization of ONE sample — washes
  out only if flips are frequent (small L); tau(L) slide shows why they aren't.

## 3. DATA QUALITY: the old fig3_points.npz is part-corrupted (do not reuse)

Diagnosis (30 Aug): in results/L1000/fig3_points.npz, 11/49 betas have a
NONZERO ensemble mean of the signed gap (|mean| up to 0.67; frac(gap>0)=1.000),
and 3 more tail betas (0.318-0.334, outside the SSB band) have f_br~0.45.
Independent SSB ensembles must split ~50/50 between (+,-) and (-,+): a
one-basin ensemble at beta deep in the band is impossible for independent
replicas.

Root cause: fig3_plot.py's ANIM pass calls run_ensemble_cuda(..., seed=7) with
ONE shared generator seeding a whole 512-replica ensemble:
  - same-seed reruns are bit-identical (verified),
  - lattices are init-random-0.4 per replica but every replica draws from
    statistically-synchronized streams -> basin choice is correlated, and at
    certain betas ALL replicas correlate into the SAME basin.
Control: betas run through the continue-kernel path (snapshot set, e.g.
b2300/b2450/b2546/b2620 in the npz) show the correct 50/50 structure; L=200
npz (same protocol, different runs) shows 0 corrupted betas. CPU Fenwick
cross-check (48 indep reps, L=64, b=0.10): frac>0 = 0.500 exactly.

Also note the OLD ssb_beta*.npz (128 reps, continue-kernel) has frac(gap>0)=1
at every beta BUT that is expected: dense-dilute = <|rho1-rho2|> >= 0 by
construction (sign-blind). Fine for the moment curve; unusable for f_br.

Consequences:
- Never present f_br / signed diagnostics from fig3 anim-pass betas.
- plots from fig3_points.npz must exclude betas with |mean gap| > 0.15
  (0.064, 0.096, 0.215, 0.223, 0.231, 0.239, 0.247, 0.255, 0.259, 0.318, 0.334)
  and treat 0.318-0.334 as suspect anyway.
- If a slide uses P(rho1,rho2) heatmaps from those betas (0.23/0.255/0.2595
  among them are ANIM-corrupted; 0.245/0.258/0.262/0.2685 are snapshot-clean),
  the "correlated-ensemble" caveat applies (basin split may be hidden).

## 4. The rerun (clean data for the slide)

scripts/ssb_rerun.py (30 Aug, ~1.5 h GPU): fresh dir results/ssb_rerun,
SEED=20260830, continue-kernel (no per-launch reseed), 49 betas x 128 reps,
6 chunks x 2e7 steps = 120k steps/site (c=120 > c=100 rule), per-chunk
incremental save, independence monitor per chunk (min broken-frac across
betas, signed-gap range). Reduction + figure: scripts/plot_ssb_order.py
(v2: m(beta) + f_br(beta), twin axis; text label for band edge ~0.25-0.26).

Slide text rewritten around two sentences:
  1. order parameter must be an ENSEMBLE statistic of the mean gap
     (broken moment m and broken fraction f_br), not a single-trajectory
     average (washes out via flips) and not per-replica time std (just
     fluctuation width, same in both phases);
  2. m decays 0.93 -> 0 by beta~0.30 while f_br stays 1 then falls through a
     two-phase coexistence wedge (band edge beta ~ 0.25-0.26) to 0 — the
     SSB region from the phase diagram, seen in the order parameter itself.