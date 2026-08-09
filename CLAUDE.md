# CLAUDE.md - Project Context for OpenCode

## Project: Two-Channel ASEP MC Reproductions

Reproduce: Pronina & Kolomeisky, J. Phys. A 40, 2275 (2007)

### Quick Model Reference
- `Lane 1` (→): enters site 0, exits site L-1, hops right (rate 1)
- `Lane 2` (←): enters site L-1, exits site 0, hops left (rate 1)
- **Narrow entrance**: enter lane1 ⟺ lane2[0] (exit of ch2) is empty
- **Narrow entrance**: enter lane2 ⟺ lane1[L-1] (exit of ch1) is empty
- Exit rate β is independent of the other channel

### TASEP Phase Reference (from course L18-19)
| Phase | Condition | Bulk Density | Current |
|-------|-----------|-------------|---------|
| LD    | α<β, α<1/2 | α | α(1−α) |
| HD    | α>β, β<1/2 | 1−β | β(1−β) |
| MC    | α>1/2, β>1/2 | 1/2 | 1/4 |

### Code Structure
```
asep/          ← simulation code (importable)
  model.py     ← TwoChannelASEP class (Gillespie step)
tests/         ← must pass
notebooks/     ← reproduction notebooks
theory/        ← notes, derivations
results/       ← raw data (gitignored)
```

### Workflow Rules
1. Small changes — test before you commit
2. Verify MFT vs MC at low α,β first (LD phase), then scale up
3. Keep responses lean — no unnecessary text
4. Always cite the paper section when referencing results

### Next Steps (check /todo.md)
