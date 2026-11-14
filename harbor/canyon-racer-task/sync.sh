#!/bin/bash
# Vendor the current project snapshot (src, ROM, pinned constraints) into the sidecar and verifier build contexts.
set -e
T="$(cd "$(dirname "$0")" && pwd)"; P="$T/../.."
for d in "$T/environment/mcp-server" "$T/tests"; do
  rm -rf "$d/game"; mkdir -p "$d/game"
  cp -r "$P/src" "$d/game/src"; find "$d/game" -name __pycache__ -prune -exec rm -rf {} +
  mkdir -p "$d/game/roms"; cp "$P"/roms/canyon_racer_0.7.7_mbc5_rumble.{gb,noi,cdb} "$d/game/roms/"
  (cd "$P" && uv export --frozen --no-dev --no-hashes --no-emit-project 2>/dev/null | grep -v '^ *#' > "$d/game/constraints.txt")
done
