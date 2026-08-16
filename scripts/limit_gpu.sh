#!/bin/bash
# Limit GPU power/temperature to stabilize long runs (avoid driver crashes).
#
# IMPORTANT: requires root (sudo). Run:  sudo ./scripts/limit_gpu.sh
#
# Notes on this GPU (RTX 2060 SUPER):
#   - power limit range is [125, 215] W; the floor is 125 W, so values like
#     105/120 W are NOT settable on this card.
#   - There is no -ltc/-tcl thermal-limit option on this nvidia-smi/driver
#     build, and the NVIDIA hwmon does not expose a writable thermal file.
#     So we cap power at the floor (125 W) and lock clocks to a moderate SM
#     clock as a proxy for temperature control.
#
# Safe default: power 125 W (minimum), SM clock capped at 1500 MHz.

POWER=${1:-125}      # W, within [125, 215]
SM_CLOCK=${2:-1500}  # MHz, moderate cap

echo "Setting GPU power limit to ${POWER} W ..."
nvidia-smi -pl "$POWER"

echo "Locking SM clocks to <= ${SM_CLOCK} MHz ..."
nvidia-smi -lgc 300,"$SM_CLOCK"

echo "Done. Current state:"
nvidia-smi --query-gpu=power.limit,power.draw,clocks.sm,temperature.gpu --format=csv

echo
echo "To verify stability on a long run, e.g.:"
echo "  nvidia-smi -pl 125 && nvidia-smi -lgc 300,1500"
echo "To reset:"
echo "  nvidia-smi -pm 0 && nvidia-smi -rgc"
