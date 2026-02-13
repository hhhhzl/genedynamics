#!/bin/bash
# Run from repo root so mdoc/... paths resolve.
cd "$(dirname "$0")/.." || exit 1

# Run all MDCOAS experiments/baselines
python -m enerdynamics.experiments.runner configs/single_2d/mbd.yaml
python -m enerdynamics.experiments.runner configs/single_2d/ebmbd.yaml
python -m enerdynamics.experiments.runner configs/single_2d/mdoc.yaml
python -m enerdynamics.experiments.runner configs/single_2d/mdcoas.yaml
python -m enerdynamics.experiments.runner configs/single_2d/mdcoas-a.yaml
python -m enerdynamics.experiments.runner configs/single_2d/mdcoas-f.yaml