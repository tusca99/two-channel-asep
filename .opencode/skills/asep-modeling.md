# ASEP Modeling Skill

## Quick Start

```python
from asep import TwoChannelASEP

sim = TwoChannelASEP(L=1000, alpha=0.9, beta=0.3)
sim.run(n_steps=100000, sample_every=100, warmup=5000)
J1, J2 = sim.get_currents()
rho1, rho2 = sim.get_bulk_densities()
prof1, prof2 = sim.get_site_density_profiles()
```

## Key Parameters

| Parameter | Meaning          | Typical Range |
|-----------|------------------|---------------|
| `L`       | lattice size     | 100–12000     |
| `alpha`   | entrance rate    | 0–1           |
| `beta`    | exit rate        | 0–1           |

## Phases to Target (from paper Fig 2)

| Phase | α range | β range | What to look for |
|-------|---------|---------|-----------------|
| LD    | < 0.5   | > α, > 0.5 | J ≈ α(1−α), ρ ≈ α |
| MC    | > 0.5   | > 0.5    | J ≈ 0.25, ρ ≈ 0.5 |
| HD/LD | > 0.5   | < 0.5, α/β > ... | One channel HD (ρ≈1−β), one LD (ρ≈small) |
| LD/LD | borderline | narrow band | Both LD, ρ₁≠ρ₂, very small window |

## Performance Tips

- Use `numba` for the inner MC loop
- For finite-size scaling, run multiple L values in parallel
- GPU port: each site is independent → perfect for SIMT
