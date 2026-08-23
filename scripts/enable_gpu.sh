#!/usr/bin/env bash
# Build a minimal CUDA-12 toolkit layout in /tmp/opencode/CUDAHOME for the
# H100 (sm_90), from the pip-installed nvidia wheels in this venv.
#
# Why: numba 0.66 (this venv) only supports CUDA <= 12 via its CTK table, but
# the venv ships the CUDA 13 nvidia wheels. We point numba at:
#   - the cua 12 runtime (libcudart.so.12)  -> version detection sees 12.x
#   - the cu13 libnvvm                      -> ABI-compatible, supports sm_90
#   - libdevice.10.bc                        -> for the float device functions
#
# Usage (in the repo):  source scripts/enable_gpu.sh
# Then:  uv run python scripts/fig5_Lscan_gpu.py --L 1000 --nrep 48
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SP="$ROOT/.venv/lib/python3.13/site-packages/nvidia"
CH="/tmp/opencode/CUDAHOME"

mkdir -p "$CH/nvvm/lib64" "$CH/nvvm/libdevice" "$CH/lib64"
# nvvm  (cu13 ABI is the only one with a working sm_90 build we have)
ln -sf "$SP/cu13/lib/libnvvm.so.4" "$CH/nvvm/lib64/libnvvm.so"
# libdevice
ln -sf "$SP/cu13/nvvm/libdevice/libdevice.10.bc" "$CH/nvvm/libdevice/libdevice.10.bc"
# cudart (cu12 for numba table) + nvrtc
ln -sf "$SP/cuda_runtime/lib/libcudart.so.12" "$CH/lib64/libcudart.so.12"
[ -e "$SP/cuda_nvrtc/lib/libnvrtc.so.12" ] && ln -sf "$SP/cuda_nvrtc/lib/libnvrtc.so.12" "$CH/lib64/libnvrtc.so.12"
chmod 755 "$SP/cu13/lib/libnvvm.so.4" 2>/dev/null || true

export CUDA_HOME="$CH"
export LD_LIBRARY_PATH="$CUDA_HOME/nvvm/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "GPU env ready: CUDA_HOME=$CUDA_HOME"
