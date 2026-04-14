#!/usr/bin/env bash
#
# Download TUM RGB-D sequences (freiburg1 subset by default).
#
# Usage:
#   bash scripts/tasks/3dgs/download_tum.sh [output_dir]
#
# Default output: data/tum

set -e

OUT_DIR="${1:-data/tum}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

BASE_URL="${TUM_BASE_URL:-https://vision.in.tum.de/rgbd/dataset/freiburg1}"

SEQUENCES=(
  rgbd_dataset_freiburg1_desk
  rgbd_dataset_freiburg1_room
  rgbd_dataset_freiburg1_xyz
)

for seq in "${SEQUENCES[@]}"; do
  if [ -d "$seq" ]; then
    echo "[skip] $seq already exists."
    continue
  fi
  echo "=== Downloading $seq ==="
  wget -q --show-progress "$BASE_URL/$seq.tgz" -O "$seq.tgz"
  tar -xzf "$seq.tgz"
  rm "$seq.tgz"
done

echo "Done. Sequences installed under: $(pwd)"
