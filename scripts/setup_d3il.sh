#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"


cd $ROOT_DIR/third_party/environments/d3il
./install_py.sh

export PYTHONPATH=$ROOT_DIR/third_party/environments/d3il
echo "export PYTHONPATH=$ROOT_DIR/third_party/environments/d3il" >> ~/.bashrc

# if rendering is required, install mesa dependencies
apt-get update && apt-get install -y libosmesa6-dev libgl1-mesa-glx