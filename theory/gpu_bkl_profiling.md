# GPU BKL Profiling — Nsight Compute (18 Aug 2026)

## Question
Why is a single GPU trajectory ~35 kstep/s vs ~4.8 Mstep/s on one CPU core
(~36x)? Is it Numba slowness, memory, RNG, or the serial BKL structure?

## Method
Profiled the numba CUDA BKL-Fenwick kernel (`asep/cuda_ensemble.py`) with
Nsight Compute (`ncu`, Nsight Compute 2025.2). Grid filled to 79 blocks on the
RTX 2060 SUPER (34 SMs, CC 7.5), 512->20000 threads, short step count for fast
replay.

## Result (full-occupancy grid, 57.8% achieved occupancy)

Warp stall breakdown:

| stall reason | % |
|--------------|---|
| long_scoreboard (global/local mem latency) | **95.77** |
| wait (fixed-latency dependency)            | 1.03 |
| math_pipe_throttle (RNG, -log, div)        | 0.02 |
| short_scoreboard (shared mem latency)      | 0.01 |

Other:
- Issue slots busy ~10.5% (scheduler mostly idle waiting on memory)
- FP64 pipeline 74.9% (but 0.02% stalled on math -> not the limit)
- L1/TEX hit 41.8%, L2 hit 100%, DRAM 0.02% (tiny data, latency-bound not BW-bound)
- Memory accesses uncoalesced: 3.5/32 bytes per sector (each thread's scattered
  Fenwick walk)
- Registers/thread = 61

## Interpretation
The per-replica 35 kstep/s is REAL and fully explained: each GPU thread runs a
serial BKL trajectory whose Fenwick tree (`_bit_find`/`_bit_update`) does ~10
scattered global loads per event at ~400-600 cycle latency. 95.8% of stalls are
this global-memory latency. It is NOT:
- Numba overhead (a CUDA C rewrite hits the same global memory)
- RNG / FP64 math (0.02% math-pipe stall)
- Insufficient occupancy alone (persists at 57.8% occupancy)

It IS the serial BKL structure + Fenwick-in-global-memory on a single thread.
Fundamental: BKL is a serial chain (each event depends on the previous), so a
single GPU thread can't hide latency; the GPU only wins via mass parallelism.

## Aggregate (the number that matters)
- GPU aggregate: ~72 Mstep/s (mass-parallel, latency hidden by occupancy)
- CPU 12-core aggregate: ~58 Mstep/s
- GPU is ~1.2x CPU in aggregate. Per-replica 35k is misleading.

## Algorithm implications
- Fenwick is already the BETTER GPU choice vs classic BKL: classic BKL's
  O(n_active) active-site scan is an even longer serial global-memory walk.
  Switching to classic BKL makes the 95.8% latency stall WORSE.
- Alias method: O(1) selection but no cheap incremental updates -> loses to
  Fenwick when the tree changes every event.
- Rejection sampling: O(1) per trial, high rejection at low density (SSB phase)
  -> needs measurement, may or may not beat Fenwick.
- The real lever: saturate occupancy (nrep -> thousands) so latency is hidden
  behind other warps -> ~72 Mstep/s aggregate, >= CPU.

## Takeaway for the repo
Don't rewrite in CUDA C for per-replica speed (same global-memory latency).
If a speedup is wanted, the promising directions are:
  1. occupancy saturation (already the production path: scan_grid_gpu ensembles)
  2. an algorithm that avoids per-thread scattered global Fenwick walks
     (e.g. rejection sampling / per-warp shared Fenwick, if L small enough)
  3. the Jimenez & Ortiz (2015) GPU OkMC idea, assessed for hard-core exclusion
     (their particles are independent walkers; our exclusion coupling makes
     "parallel across particles" harder)

## Verdict on the remaining ideas (18 Aug 2026, after math + analysis)

Per-run CPU speed on GPU is FUNDAMENTALLY unreachable, and the ideas fail on
aggregate grounds:

1. **Shared-memory Fenwick (CUDA C or Numba)**: shared mem is PER-BLOCK, each
   replica needs its OWN 4KB Fenwick (float32, L=1000). A 64KB block fits only
   ~16 trees -> 16 replicas/block. Aggregate/SM:
     - current (2048 thr/SM, global, 35k): 2048*35k = 72M step/s/SM
     - shared (16 thr/SM, ~10x latency cut -> 350k): 16*350k = 5.6M step/s/SM
   Shared memory LOSES by ~13x because occupancy collapses. CUDA C doesn't
   change the per-thread-vs-shared-capacity tradeoff.

2. **Rejection sampling**: active fraction ~ rho(1-rho).
     - MC (rho~0.5): active ~25% -> ~4 trials x ~3 loads = ~12 loads/event (same
       as Fenwick's ~10).
     - SSB (rho~0.27): active ~20% -> ~5 trials = ~15 loads/event (WORSE).
   Not better anywhere. Not worth testing.

3. **Classic BKL (no Fenwick)**: O(n_active) serial scan = longer scattered
   global walk. Worse.

Conclusion: per-thread 35k is a hard floor on GPU. The GPU's value is AGGREGATE
(~72 Mstep/s, ~1.2x CPU 12-core), as a massively-parallel ensemble engine for
phase diagrams / fig5. The real lever is ALGORITHMIC step reduction (continuous
runs, adaptive stopping), not per-step backend speed.

## CPU profile (for reference)
Already have it: run_bkl_profiled counts per-phase ops; ~4.8 Mstep/s/core.
CPU wins because the Fenwick fits in L1 (~30 cyc) vs GPU global (~400-600 cyc).

