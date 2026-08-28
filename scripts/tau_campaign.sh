#!/usr/bin/env bash
# Launch the tau(L) campaign: 12 parallel single-trajectory runs (8700K, 6c/12t -> 12 processes on 6 physical cores, Fenwick is latency-bound so HT helps slightly).
set -u
cd "$(dirname "$0")/.."
mkdir -p results/tau
LOG=results/tau/tau.log
echo "=== tau campaign started $(date) ===" >> $LOG

# Ladder: (L, beta, n_steps, seed, sample_every, smooth_w)
# - L=200:  tau ~ 2.4e6 steps measured -> 1e8 steps = ~40 dwells, 2min
# - L=500:  expect tau ~ x10-x30 -> 4e8 steps, 8 min
# - L=1000: unknown, could be >>1e9; try 2e9 (40 min). If no flip, we get a LOWER bound.
# - L=2000: 6e9 steps as a lower-bound probe (2 h)
# betas: inside band (0.22), near upper edge (0.28) -> tau should grow as beta -> beta_c from below
run () {
  L=$1; B=$2; S=$3; SEED=$4; SE=$5; SM=$6
  nohup python scripts/tau_flipping.py $L $B $S $SEED $SE $SM \
    >> results/tau/tau.log 2>&1 &
  echo "launched L=$L beta=$B steps=$S pid=$!"
}

# group A: L=200 (fast, 2 min each) x 3 betas
run 200 0.18 1e8 11 2000  300
run 200 0.22 1e8 12 2000  300
run 200 0.26 1e8 13 2000  300
# group B: L=500 x 3 betas (8 min each)
run 500 0.18 4e8 21 5000  150
run 500 0.22 4e8 22 5000  150
run 500 0.26 4e8 23 5000  150
# group C: L=1000 x 3 betas (35 min each)
run 1000 0.18 2e9 31 10000 80
run 1000 0.22 2e9 32 10000 80
run 1000 0.26 2e9 33 10000 80
# group D: L=2000 x 3 betas (~2.5 h each)
run 2000 0.18 6e9 41 20000 40
run 2000 0.22 6e9 42 20000 40
run 2000 0.26 6e9 43 20000 40

wait
echo "=== tau campaign finished $(date) ===" >> $LOG