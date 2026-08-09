# Two-Channel ASEP with Narrow Entrances

Reproduction and extension of:

> Pronina & Kolomeisky, *J. Phys. A: Math. Theor.* **40**, 2275 (2007)
> "Spontaneous symmetry breaking in two-channel asymmetric exclusion processes with narrow entrances"

## Model

Two parallel 1D lattices (channels), each of length `L`, with particles:
- Channel 1: hop **right** (rate 1)
- Channel 2: hop **left** (rate 1)
- Hard-core exclusion: max 1 particle per site
- **No inter-channel hopping**

### Narrow entrance coupling

- Particle enters channel 1 at site 0 **only if** site `L-1` of channel 2 is empty.
- Particle enters channel 2 at site `L-1` **only if** site 0 of channel 1 is empty.
- Exit rate `β` is independent of the other channel.

### Parameters

| Symbol | Meaning          | Range        |
|--------|------------------|--------------|
| `L`    | Lattice size     | integer      |
| `α`    | Entrance rate    | (0, 1]       |
| `β`    | Exit rate        | (0, 1]       |

### Phases (mean-field)

| Phase   | Description                         |
|---------|-------------------------------------|
| LD      | Low-density, symmetric (both lanes) |
| MC      | Maximal current, symmetric          |
| HD/LD   | Asymmetric: one HD, one LD          |
| LD/LD   | Asymmetric: both LD, different ρ    |

## Setup

```bash
uv venv
uv pip install -e .
```

## Usage

```python
from asep.model import TwoChannelASEP

# Initialize a system
sim = TwoChannelASEP(L=1000, alpha=0.9, beta=0.3)

# Run for some steps
sim.run(n_steps=100000, sample_every=100)

# Get observables
J1, J2 = sim.get_currents()
rho1, rho2 = sim.get_bulk_densities()
```

## Project Structure

```
.
├── theory/        # Paper notes, MFT derivations
├── asep/          # Core simulation code
├── analyze/       # Phase diagram scans, finite-size scaling
├── notebooks/     # Reproduction notebooks (Fig 2, Fig 6)
├── presentation/  # Beamer slides
└── results/       # Raw simulation data (gitignored)
```

## License

MIT
