#!/bin/bash


OUT_ROOT_DIR="./local/design_opt/sdf_basis"
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
    --env-config-file ocean.yaml
    --set-design-types geometry softness actuator actuator_direction
    --n-frames 100
    ### Loss
    --n-iters 101
    --loss-types PerStepCoVLoss
    ### Designer 
    --optimize-designer
    --optimize-design-types geometry
    --designer-type sdf_basis
    --sdf-basis-pcd-paths ./softzoo/assets/meshes/pcd/Orca.pcd
                          ./softzoo/assets/meshes/pcd/GreatWhiteShark.pcd
                          ./softzoo/assets/meshes/pcd/Fish_2.pcd
    --sdf-basis-mesh-paths ./softzoo/assets/meshes/stl/Orca.stl
                           ./softzoo/assets/meshes/stl/GreatWhiteShark.stl
                           ./softzoo/assets/meshes/stl/Fish_2.stl
    --sdf-basis-init-coefs-geometry 0.1 0.8 0.1
    --sdf-basis-init-coefs-softness 0.1 0.8 0.1
    --sdf-basis-init-coefs-actuator 0.1 0.8 0.1
    --sdf-basis-init-coefs-actuator-direction 0.1 0.8 0.1
    --designer-lr 0.001
    --sdf-basis-passive-geometry-mul 0.2
    # ### Controller
    --load-controller ${CONTROLLER_CKPT}
    --actuation-omega 20. 100.
    --log-reward
)
python -m algorithms.diffsim.run "${args[@]}"
