#!/bin/bash


WORKSPACE=${HOME}/workspace/softzoo
LOG_ROOT=${WORKSPACE}/local/rl
YAML_FILE=${WORKSPACE}/algorithms/rl/hyperparams/ver0/ppo_sin_2.yml

ANIMAL="Caterpillar" # "BabySeal" "Caterpillar" "Panda" "Fish"
MATERIAL="Ground" # "Ground" "Desert" "Wetland" "Clay" "Ice" "Snow" "Shallow_Water" "Ocean"
TASK="MovementSpeed"

ENV="${MATERIAL}-${ANIMAL}-${TASK}-v0"
echo $ENV
args=(
    --algo ppo
    --env $ENV
    --save-freq 10000
    --log-folder $LOG_ROOT
    --tensorboard-log ${LOG_ROOT}/tb/
    --eval-freq -1 # no evaluation
    --yaml-file $YAML_FILE 
    --vec-env dummy
)
python -m algorithms.rl.train "${args[@]}"
