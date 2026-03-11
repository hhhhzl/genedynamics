#!/usr/bin/env bash
# Setup SoftZoo assets in data/softzoo/assets
# Download from: https://drive.google.com/drive/folders/1AYeZsr2ZMb1DkeOndQM0nBlNfx7dorUL

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSETS_DIR="${ASSETS_DIR:-$ROOT/data/softzoo/assets}"

echo "=== SoftZoo Assets Setup ==="
echo "Target: $ASSETS_DIR"
echo ""

mkdir -p "$ASSETS_DIR"/{meshes/pcd,meshes/stl,textures,skybox}

if [[ -d "$ROOT/third_party/environments/softzoo/softzoo/assets" ]]; then
    echo "Copying vendor assets (textures, stl) to data/softzoo/assets..."
    rsync -a --ignore-existing \
        "$ROOT/third_party/environments/softzoo/softzoo/assets/" \
        "$ASSETS_DIR/" 2>/dev/null || cp -r "$ROOT/third_party/environments/softzoo/softzoo/assets"/* "$ASSETS_DIR/" 2>/dev/null || true
fi

echo ""
echo "PCD meshes (Caterpillar.pcd, etc.) must be downloaded manually:"
echo "  1. Open https://drive.google.com/drive/folders/1AYeZsr2ZMb1DkeOndQM0nBlNfx7dorUL"
echo "  2. Download meshes/pcd/*.pcd"
echo "  3. Place in $ASSETS_DIR/meshes/pcd/"
echo ""
echo "Required for crawling: Caterpillar.pcd"
echo ""
echo "Done. Assets dir: $ASSETS_DIR"
