# Papers beyond Pronina & Kolomeisky (2007) — verified from the PDFs (28 Aug 2026)

> Sources read in full: `theory/zhu2012_pre.pdf` (8 pp, PRE 85, 041132),
> `theory/tian2017_chinphysb.pdf` (6 pp) + arXiv v1 `1605.01817` (13 pp, fuller),
> `theory/xiao2010_chinphysb.pdf` (11 pp). Pronina 2007 arXiv preprint:
> `theory/pronina2007_arxiv.pdf` (cond-mat/0611472).

## 0. Model check — who studies what
| Paper | Process | Relation to ours |
|---|---|---|
| Pronina & Kolomeisky 2007 | bidirectional 2-lane, **narrow entrances** | **our model** |
| **Zhu et al. 2012** (PRE 85, 041132) | **our model + leaky entrance**: injection at rate $p\alpha$ also when the other lane's exit is occupied | generalization; $p{=}0$ → Pronina, $p{=}1$ → independent ASEPs |
| **Tian et al. 2017** (CPB 26, 020503) | **same model**, identical rules | the follow-up on our model |
| Xiao et al. 2010 (CPB 19, 090202) | **different**: 2-input 1-output **Y-junction** TASEP, α₁≠α₂, p₁,p₂,p₃ | NOT our model; method template only |

## 1. Zhu et al., PRE 85, 041132 (2012) — our model with a coupling knob
- Model A (ours + knob): entrance injects at rate $\alpha$ if the other lane's exit site is empty,
  at rate $p\alpha$ if it is occupied. $p$ tunes the entrance coupling between the two lanes.
- **Result:** five phases at small $p$ (asym LD-LD + HD/LD, sym HD/MC/LD); both asymmetric phases
  shrink as $p$ grows; LD/LD unresolvable by $p=0.4$; **SSB completely gone at $p_c\approx0.6$**
  (SIM) / $p_c=0.5$ (MF — exact result at $p=0.5$). SSB dies *before* coupling is eliminated.
- MF: effective rates $\alpha_1=\alpha[1-(1-p)m_1]$, $\alpha_2=\alpha[1-(1-p)p_L]$ — boundary
  densities of the *other* lane fed back self-consistently (simple-MF level, factorized joints).
- **Flipping time $\tau\propto e^{L}$** (their Fig. 3) → SSB genuine at finite $p$; $\tau$ and its
  growth rate decrease as coupling weakens.
- Lesson for us: **the entrance coupling is the SSB engine** — our $p=0$ is the maximal-coupling
  case, the best possible arena for SSB; and $\tau(L)$ is measurable on our CPU farm.
- Historical note: Tian 2017 corrects Zhu's LD/LD existence condition ($\rho<1/2 \to \rho<\beta$,
  Tian §3.2.1). Zhu's conclusion also: narrow *exits* alone give no SSB; both entrance+exit
  under investigation (their outlook).

## 2. Tian et al., CPB 26, 020503 (2017) — our model; LD/LD attacked in MF, not in MC
- **Method:** N-cluster MF (N = 1…6): N consecutive sites from the exit of one lane + N from the
  entrance of the other; states grow as 2^{2N}; only one approximation left (p₂ = 1−ρ₂ at the
  cluster edge). N=1 is the 2-site vertical cluster (keeps the entrance joint states P₀₀,P₀₁,P₁₀,P₁₁).
  Solution count: N ≤ 4 → sym-LD + one asym-LD/LD pair; **N ≥ 5 adds a second asym-LD/LD pair (II)**
  — five intersection points A–E (their Fig. 5).
- **Current minimization** (eqs. (2)–(4), Figs. 2/6): inside the LD/LD window β_L < β < β_R the
  asymmetric current J_A exceeds both the symmetric-LD current J_B and the HD/LD current J_C ⇒
  **the asym-LD/LD phase should not exist** (holds at every N; "not rigorously proved").
- **Exponential decay is in N, not L:** boundaries vs cluster size follow β_c − β_{c,∞} ∝ e^{aN+b}
  (Fig. 8). Extrapolated N→∞ at **α=0.9: β_{c,∞}=0.28871** (vs simple-MF 0.332 = α/(1+α+α²), vs
  our SIM ≈ 0.253) → closes ~half the MFT-vs-SIM gap. α_{c,∞}=0.2362 (β=0.2), 0.233355 (β=0.4).
- **Simulations:** MC at **L = 10⁴ still shows the LD/LD band** (Fig. 7) — they do *not* claim MC
  disappearance and do **no L-scaling study of the band**; they explicitly call for larger careful
  runs (RNG-quality caveat, their ref [53]). Contrast Erickson et al. 2005 (bridge model), where
  high-precision MC killed one broken phase.
- Boundary convergence: N-cluster boundaries approach the L=10⁴ MC ones as N grows (Fig. 7) —
  entrance correlations, not bulk MF, set the boundary position.

## 3. Xiao et al., CPB 19, 090202 (2010) — different process; usable as a template
- Y-junction: two input lanes merge into one output lane; TASEP (one direction), α₁ ≠ α₂,
  bulk rates p₁, p₂, p₃; simple 1-site MF with effective junction rates (β_eff, α_eff) from
  current conservation + the reduced-rate TASEP solution.
- Five phases (LD/LD/LD, HD/LD/MC, HD/LD/HD, HD/HD/HD, HD/HD/MC); unequal rates shift the
  vertical boundary left and horizontal boundary down.
- **No correlator C, no SSB analysis** — value for us: method template for the α₁ ≠ α₂ extension
  of our model (our Open-questions frame).

## How to say it in 60 s (outlook cue) — corrected
> *Our β boundary disagreement (MFT 0.33 vs SIM 0.25 at α=0.9) is an entrance-correlation effect:
> Tian's N-site vertical clusters move the boundary to 0.289 at N→∞ — half the gap closes. And the
> entrance coupling is the SSB engine itself: Zhu's leaky entrance kills both broken phases at
> p_c≈0.6, so our p=0 is where SSB is strongest. Tian's current-minimization argument says the
> LD/LD band should not exist at all — yet his own L=10⁴ simulations still see it. The verdict
> needs exactly what we have: c≈100 equilibration scaling to L≈10⁴, a few hours on our CPU farm.*

## Action items for us
1. ~~Cheap first: rerun the existing fig5(a) L=8000 point at c=100~~ — user decided: enough boundary runs, keep the existing L-scan as is.
2. ~~fig5(a) full rerun at c=100~~ — cancelled (see 1).
3. ~~std(ρ₁−ρ₂) vs L in the band~~ — superseded by the τ campaign (done, see below).
4. **DONE 28 Aug — τ(L) campaign** (`scripts/tau_flipping.py`, `scripts/tau_plot.py`, `results/tau/`):
   - **Scope (honest):** method reproduced from Zhu 2012 (their Fig. 3 did model A at p>0,
     α=0.5, β=0.2); we apply it at **p=0** (the original Pronina model — not covered by Zhu's
     τ analysis), at our **α=0.9** operating point, plus a **β-ladder** (0.18/0.22/0.26).
     Reproduction + extension, not "nobody did it".
   - **Detector:** basin projection — in-basin iff |ρ₁−ρ₂| ≥ dmin (0.35 deep band, 0.15 edge);
     flip = leave one basin, enter the other; τ = median dwell. (First attempt with smoothed
     sign-crossings failed at the band edge: noise crossings made τ *shrink* with L. Fixed.)
   - **Results** (single long trajectories, 8700K, α=0.9):
     | L | β=0.18 | β=0.22 | β=0.26 |
     |---|---|---|---|
     | 200 | ≥10⁸ | 2.2·10⁵ | 1.3·10⁴ |
     | 500 | ≥4·10⁸ | ≥4·10⁸ | 6.5·10⁴ |
     | 1000 | ≥2·10⁹ | ≥2·10⁹ | 3.8·10⁵ |
     | 2000 | ≥6·10⁹ | ≥6·10⁹ | 1.7·10⁶ |
   - **β=0.26 (band edge, measured):** τ ≈ 1.4·10⁴ · exp(L/390) — clean exponential (ξ≈390),
     ~×100 per 1800 sites. Deep band: zero flips in 6·10⁹ steps at L=2000 → τ ≳ 10¹⁰.
   - **Key message:** SSB is metastability — basins are exponentially long-lived, so at large L
     a finite run samples ONE basin: this is *why* the bands shrink in the phase diagram and
     *why* equilibration needs c∝L. The τ campaign and the fig5 L-scan are the same physics
     seen two ways.
5. 4-site (N=4) cluster MF as the next theory step — N≥5 only adds the spurious solution II.
6. RNG for 10¹¹-draw runs: per-replica counter-based seeding (Philox/Threefry on GPU, or an
   AES-CTR-seeded Trivium bank as in the FPGA-project percolation core — 64 provably independent
   streams, period ≥2¹⁴⁴). Validation precedent: 2-D directed-percolation p_c reproduced to 2e-4 —
   different universality class from ASEP, but a clean end-to-end test of the streams themselves.
7. α₁≠α₂ extension following Xiao's effective-rate junction method.

**Why α=0.9 everywhere in the SSB analyses:** (i) Pronina & Kolomeisky's Fig. 5(a) boundary is
drawn at α=0.9 — our fig5 is a direct reproduction at their operating point; (ii) deep in the
SSB region, both broken phases exist with wide bands (clean signal); (iii) Tian 2017 quotes
β_c(∞)=0.28871 exactly at α=0.9 — apples-to-apples with their N→∞ extrapolation. The rest of
the deck (fig2/fig4/fig6) scans the full α-plane; only fig5-type boundary runs + SSB + τ fix α.

**Refs on slide:** `zhu2012` + `tian2017` (primary); `xiao2010` (template for unequal rates).