#!/bin/bash
# Run from repo root so mdoc/... paths resolve.
cd "$(dirname "$0")/../../.." || exit 1

# Run all MDCOAS experiments/baselines
python -m genedynamics.experiments.runner configs/single_2d/mbd.yaml
python -m genedynamics.experiments.runner configs/single_2d/ebmbd.yaml
python -m genedynamics.experiments.runner configs/single_2d/mdoc.yaml
python -m genedynamics.experiments.runner configs/single_2d/mdcoas.yaml
python -m genedynamics.experiments.runner configs/single_2d/mdcoas-a.yaml
python -m genedynamics.experiments.runner configs/single_2d/mdcoas-f.yaml

# D3IL avoiding
python -m genedynamics.experiments.runner configs/d3il_avoiding/ebmbd.yaml
python -m genedynamics.experiments.runner configs/d3il_avoiding/mdoc.yaml
python -m genedynamics.experiments.runner configs/d3il_avoiding/mdcoas.yaml
python -m genedynamics.experiments.runner configs/d3il_avoiding/mdcoas-a.yaml
python -m genedynamics.experiments.runner configs/d3il_avoiding/mdcoas-f.yaml
python -m genedynamics.experiments.runner configs/d3il_avoiding/safediffuser.yaml
# python -m genedynamics.experiments.runner configs/d3il_avoiding/safediffuser_9d.yaml
python -m genedynamics.experiments.runner configs/d3il_avoiding/dpcc.yaml