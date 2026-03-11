#!/bin/bash


ENV="ground"
ANIMAL="Caterpillar"
SEED=100
OUT_DIR="./local/render/${ENV}-${ANIMAL}-${SEED}"
CKPT_DIR="./local/diffsim/${ENV}-${ANIMAL}-${SEED}/ckpt"

args=(
    --out-dir "${OUT_DIR}"
    --seed $SEED
    --torch-seed $SEED
    --env land_environment
    --env-config-file render/${ENV}.yaml
    --save-every-iter 999
    --render-every-iter 1
    --n-iters 1
    --n-frames 40
    --set-design-types geometry softness actuator actuator_direction
    --designer-type annotated_pcd
    --annotated-pcd-path ./softzoo/assets/meshes/pcd/${ANIMAL}.pcd
    --annotated-pcd-passive-geometry-mul 0.2
    # --load-args ${CKPT_DIR}/args.json # NOTE: uncomment to load pretrained model
    # --load-controller ${CKPT_DIR}/controller/iter_0010.ckpt
)
python -m algorithms.diffsim.run "${args[@]}"
