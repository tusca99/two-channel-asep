#!/usr/bin/env bash
cd /home/user_vr/Documents/two-channel-asep
exec uv run python scripts/fig5_Lscan.py "$@" >/tmp/opencode/fig5_Lscan.log 2>&1
