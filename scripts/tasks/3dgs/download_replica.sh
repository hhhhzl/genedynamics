#!/usr/bin/env bash
#
# Download the Replica dataset as released by the iMAP / NICE-SLAM authors.
#
# Usage:
#   bash scripts/tasks/3dgs/download_replica.sh [output_dir]
#
# Default output: data/replica
# Size: ~70 GB unzipped for the 8 standard sequences (room_0..office_4).
# If you only need a smoke test, comment out most of SEQUENCES below.

set -e

OUT_DIR="${1:-data/replica}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

# Official release (iMAP / Nice-SLAM) hosts a tar.gz per sequence.
BASE_URL="${REPLICA_BASE_URL:-https://cvg-data.inf.ethz.ch/nice-slam/data/Replica}"

SEQUENCES=(
  room_0 room_1 room_2
  office_0 office_1 office_2 office_3 office_4
)

for seq in "${SEQUENCES[@]}"; do
  if [ -d "$seq" ]; then
    echo "[skip] $seq already exists."
    continue
  fi
  echo "=== Downloading $seq ==="
  wget -q --show-progress "$BASE_URL/$seq.tar.gz" -O "$seq.tar.gz"
  tar -xzf "$seq.tar.gz"
  rm "$seq.tar.gz"
done

echo "Done. Sequences installed under: $(pwd)"
