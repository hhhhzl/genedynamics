#!/bin/bash


args=(
    --out-dir "./local/demo_manipulation"
    --env manipulation_environment
    --env-config-file demo_manipulation.yaml
    --save-every-iter 10
    --render-every-iter 10
    --n-iters 101
    --n-frames 200
    # Loss
    --loss-types ObjectVelocityLoss
    --obj-particle-id 2
    --obj-x-mul 1. 0. 0.
    --obj-v-mul 1. 0. 0.
    # Design
    --set-design-types geometry softness actuator
    --designer-type annotated_pcd
    --annotated-pcd-path ./softzoo/assets/meshes/pcd/Hand.pcd
    --annotated-pcd-passive-geometry-mul 0.2
    # Control
    --optimize-controller
    --controller-type trajopt
    --controller-lr 0.1
)

python -m algorithms.diffsim.run "${args[@]}"
