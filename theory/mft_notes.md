# Mean-Field Theory (MFT) Notes — §2.2

## Core Assumption
No correlations: ⟨p_i p_j⟩ = ⟨p_i⟩⟨p_j⟩, ⟨m_i m_j⟩ = ⟨m_i⟩⟨m_j⟩

## Effective Entrance Rates

Each channel is a single-channel ASEP with effective entrance rates modified by the narrow entrance coupling:

```
α₁ = α(1 − m₁)    [channel 1 entrance blocked by exit site of channel 2]
α₂ = α(1 − p_L)   [channel 2 entrance blocked by exit site of channel 1]
```

where:
- m₁ = ⟨density at site 0 of channel 2⟩ (the exit site of channel 2)
- p_L = ⟨density at site L-1 of channel 1⟩ (the exit site of channel 1)

## Single-Channel ASEP Phases

| Phase | Condition             | J        | ρ_bulk   |
|-------|----------------------|----------|----------|
| LD    | α_eff < β, α_eff < 1/2 | α(1-α_eff)| α_eff   |
| HD    | α_eff > β, β < 1/2    | β(1-β)   | 1-β      |
| MC    | α_eff > 1/2, β > 1/2  | 1/4      | 1/2      |

## Symmetric Case (p_i = m_{L-i+1})

In symmetric phases, ρ₁ = ρ₂ and J₁ = J₂.

### Symmetric LD Phase
J = α₁(1−α₁), and from J₂ = βm₁ = α₁(1−α₁):
→ m₁ = α₁(1−α₁)/β

Substituting into α₁ = α(1 − m₁):
→ α₁ = [α + β − √((α+β)² − 4α²β)] / (2α)
```

## Asymmetric Phases

### HD/LD Phase
- Channel 1: HD → J₁ = β(1−β), ρ₁ = 1−β
- Channel 2: LD → J₂ = α₂(1−α₂), ρ₂ = α₂
- From J₂ = βm₁ → m₁ = α₂(1−α₂)/β
- From α₁ = α(1−m₁) and α₂ = α(1−p_L) = α(1−(1−β)) = αβ
- Existence: α > β(1 + α + α²)

### MC Phase
- Both channels: J = 1/4
- J₂ = βm₁ = 1/4 → m₁ = 1/(4β)
- α₁ = α(1−1/(4β))
- Existence: α > 2β/(4β−1)
```

## Phase Diagram (MFT)

From Figure 2 of the paper:

```
β
↑
1 |     MC (symmetric)
  |
  |  HD/LD (asymm)
  |
  |  LD/LD (asym) ← narrow wedge
0.5|-------- LD (symmetric)
  |
  |-------- 
0 +----------------→ α
  0    0.5      1
```

Key boundaries:
- LD: α < 2β/(4β−1)
- HD/LD: α > β(1+α+α²), β < 0.5
- LD/LD: narrow band, likely finite-size effect per paper §3
```

## Important Notes from the Paper

1. **MC phase does NOT exist** in single-lane two-species ASEP — only appears here with two channels
2. **LD/LD asymmetric phase is controversial** — paper's MC simulations suggest it shrinks with L and disappears in thermodynamic limit
3. **MFT deviations**: agreement only at low α,β. At large α,β, correlations matter — the paper explicitly states this
4. **SSB is boundary-induced**: the narrow entrance acts as an effective boundary defect
```
