#!/usr/bin/env bash

./scripts/setup/setup_d3il.sh

pip install -e .

# Install PyTorch with CUDA after -e ., so PyPI does not overwrite with CPU build (torch 2.10 has no cu124 wheel, use cu126)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

python -c "import jax; print(jax.devices())"
python -c "import torch; print(torch.cuda.is_available())"
