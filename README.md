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

- Particle enters channel 1 at site 0 **only if** site 0 of channel 2 (its exit) is empty.
- Particle enters channel 2 at site `L-1` **only if** site `L-1` of channel 1 (its exit) is empty.
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

## Reproduction workflow (GPU ensemble)

The figures are produced by **one GPU ensemble scan** followed by **offline
plotting** from the saved data — MC is never re-run at plot time.

```bash
# 1. Run the full GPU scan (slow; needs a CUDA GPU, e.g. RTX 2060 SUPER).
#    Saves raw data to results/gpu/*.npz
python scripts/run_all_gpu.py

# 2. Plot every figure from the saved data (fast, no MC).
python scripts/plot_all.py
```

What `run_all_gpu.py` produces (each is one ensemble launch on the GPU —
one thread per (α, β, replica), ~85 Msteps/s on a 2060 SUPER):

| Figure | Data        | What it shows |
|--------|-------------|---------------|
| fig2   | `fig2/`     | Phase diagram (full + zoom) |
| fig3   | `fig3_points.npz` | P(ρ1,ρ2) snapshots + 2D/3D animations |
| fig6   | `fig6/alpha*.npz` | Currents & densities vs β |
| ssb    | `ssb_beta*.npz` | SSB order parameter vs β (long runs) |

`scripts/fig3_plot.py` is the plotting module for fig3 (used by `plot_all`).

> **SSB / long runs:** at L=1000 a short run stays in one broken-symmetry
> state, so time-averaged |ρ1−ρ2| looks symmetric. The robust signature is
> std(ρ1−ρ2) or the bimodal P(ρ1,ρ2) over an ensemble. Our long-run β-scan
> shows diff/std decreasing monotonically with β (0.069/0.091 at β=0.05 →
> 0.029/0.021 at β=0.30) — see `presentation/notes_ssb_discrepancy.md`.

## Project Structure

```
.
├── theory/        # Paper notes, MFT derivations
├── asep/          # Core simulation code (model, bkl, cuda_ensemble, parallel)
├── scripts/       # run_all_gpu.py (scan) + plot_all.py (figures)
├── presentation/  # Slides + SSB-discrepancy notes
└── results/       # Raw data + figures (gitignored)
```

## License

MIT
