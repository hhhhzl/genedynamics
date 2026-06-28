#!/bin/bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

# Humanoid Corridor 2D: zones A-D x 5 algorithms (mbd, ebmbd, mdoc, mdcoas, twogo).
# Each config carries seeds: [0,1,2,3,4] and device: cuda.
echo "=== Humanoid Corridor 2D (zones A-D): main experiments ==="
for zone in a b c d; do
  for algo in mbd ebmbd mdoc mdcoas twogo; do
    cfg="configs/humanoid/corridor_2d/main/${algo}_zone_${zone}.yaml"
    echo ">>> $cfg"
    python -m genedynamics.experiments.runner "$cfg"
    echo ""
  done
done

echo "Done."
