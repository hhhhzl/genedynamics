#!/bin/bash

OUT_ROOT_DIR=./local/cppn
SEED=100

args=(
    # Misc
    --out-dir "${OUT_ROOT_DIR}"
    --env aquatic_environment
    --env-config-file no_grad/movement_speed/ocean.yaml
    --num-workers 0
    --seed $SEED
    --torch-seed $SEED
    --config-path "./config/neat.cfg"
    # Env
    --n-frames 100
    # Designer
    --optimize-design-types geometry actuator
    # Controller
    --update-controller
    --controller-type sin_wave_open_loop
    --n-sin-waves 4
    --actuation-omega 20. 100
)

python -m algorithms.cppn_neat.run "${args[@]}"
