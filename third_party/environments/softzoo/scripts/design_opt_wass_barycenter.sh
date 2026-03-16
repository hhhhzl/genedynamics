#!/bin/bash


OUT_ROOT_DIR="./local/design_opt/wass_barycenter"
CONTROLLER_CKPT="./local/swimmer_controller/iter_0030.ckpt"
SEED=100

args=(
    ### General
    --out-dir "${OUT_ROOT_DIR}/${SEED}"
    --save-every-iter 10
    --render-every-iter 10
    --seed $SEED
    --torch-seed $SEED
    --save-controller # NOTE: for baselines
    ### Env
    --env aquatic_environment
    --env-config-file ocean_vbr.yaml
    --set-design-types geometry softness actuator
    --n-frames 100
    ### Loss
    --n-iters 101
    --loss-types PerStepCoVLoss
    ### Designer 
    --optimize-designer
    --optimize-design-types actuator softness geometry
    --designer-type wass_barycenter
    --designer-geometry-offset 0.0
    --designer-lr 0.01
    # ### Controller
    --load-controller ${CONTROLLER_CKPT}
    --actuation-omega 20. 100.
    --log-reward
)
python -m algorithms.diffsim.run "${args[@]}"
