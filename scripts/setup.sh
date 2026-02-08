#!/usr/bin/env bash

. scripts/setup_d3il.sh

pip install -e .

python -c "import jax; print(jax.devices())"

python -c "import torch; print(torch.cuda.is_available())"
