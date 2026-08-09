# OpenCode Agent Context — Two-Channel ASEP Project

## Project Identity

```yaml
project:
  name: "Two-Channel ASEP with Narrow Entrances"
  paper: "Pronina & Kolomeisky, J. Phys. A 40, 2275 (2007)"
  type: "numerical-reproduction-exam"
  course: "Physics of Complex Systems — University of [redacted]"

context:
  mode: "build"
  style: "lean-code-minimal-output"
  change_hpolicy: "small-meaningful-commits"
```

## Core Principles

1. **Minimal output** — no unnecessary paragraphs, keep responses under 4 lines
2. **Small, meaningful changes** — never 1000-line edits
3. **Verify before move on** — tests must pass before adding features
4. **Document as you build** — inline + /theory/ notes
5. **Portable** — everything in repo, no external deps for core logic

## Coding Rules

- Python 3.10+
- `asep/` — simulation code only (no I/O)
- `tests/` — must pass on every commit
- `notebooks/` — analysis & visualization (reproduction of paper figures)
- `theory/` — notes, derivations, paper annotations
- `presentation/` — beamer slides

## Key Concepts (for context)

```
TASEP phases (from L18-19):
  LD: rho=α,     J=α(1−α)      [α<β, α<1/2]
  HD: rho=1−β,  J=β(1−β)      [α>β, β<1/2]
  MC: rho=1/2,  J=1/4         [α>1/2, β>1/2]

Two-channel model:
  - Two lanes, particles move in opposite directions
  - Narrow entrance: enter lane1 blocked by lane2 exit site (lane2[0])
  - Narrow entrance: enter lane2 blocked by lane1 exit site (lane1[L-1])
  - Both exit at rate β (independent of other channel)
  
MFT predictions (§2.2):
  α1 = α(1−m1)   [effective entrance rate for ch1]
  α2 = α(1−pL)   [effective entrance rate for ch2]
  m1 = exit density of channel 2
  pL = exit density of channel 1

Expected phases (MFT):
  1. LD    — symmetric, both channels LD
  2. MC    — symmetric, both channels MC
  3. HD/LD — asymmetric, ch1 HD & ch2 LD (or vice versa)
  4. LD/LD — asymmetric, both LD but different densities
```

## File Conventions

```
# Each file starts with:
# """One-line description."""

# Tests: test_*.py
# Sim: asep/model.py, asep/mc.py, asep/observables.py
# Data: results/ (gitignored)
# Notes: theory/*.md
```

## Commit Messages

```
feat: implement BKL step for two-channel ASEP
test: verify HD/LD asymmetry with α=0.9, β=0.23
fix: correct entrance condition (check exit site, not entrance site)
doc: add theory notes on MFT derivation
perf: add numba-compiled inner loop
```
