#!/usr/bin/env bash
# Render all diagrams/*.drawio to PNG using the draw.io flatpak CLI.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

APP="com.jgraph.drawio.desktop"
EXE="/app/main/drawio"

for f in *.drawio; do
  out="${f%.drawio}.png"
  echo "Rendering $f -> $out"
  flatpak run --command="$EXE" "$APP" \
    --export --format png --output "$out" --no-sandbox "$f" 2>/dev/null \
    || { echo "FAILED: $f"; exit 1; }
done

echo "Done."
