#!/bin/bash


ENV="ground" # "ground" "desert" "wetland" "clay" "ice" "snow" "shallow_water" "ocean"
ANIMAL="Caterpillar" # "BabySeal" "Caterpillar" "Panda" "Fish" "Fish_2" "GreatWhiteShark" "Orca"
SEED=100
OUT_DIR="./local/diffsim/${ENV}-${ANIMAL}-${SEED}"

args=(
    --out-dir $OUT_DIR
    # --env "aquatic_environment" # NOTE: uncomment this if using ocean environment
    --env-config-file ${ENV}.yaml
    --save-every-iter 10
    --render-every-iter 1
    --n-iters 31
    --n-frames 200
    --optimize-controller
    --loss-types FinalStepCoMLoss
    --designer-type annotated_pcd
    --annotated-pcd-path ./softzoo/assets/meshes/pcd/${ANIMAL}.pcd
    --annotated-pcd-passive-geometry-mul 0.2
    --actuation-omega 20. 100.
    --controller-lr 0.1
    --torch-seed $SEED
)
python -m algorithms.diffsim.run "${args[@]}"
