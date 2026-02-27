#!/bin/bash
# Test script for visualizing joint-space primitive actions.
#
# Executes a fixed sequence of primitives (no policy warmup):
#   Forward → Down → Right → Left → Up → Retreat
# Each primitive runs for primitive_horizon steps (default 50).

cd ./third_party/Robotwin

policy_name=pi05
task_config=demo_clean
seed=1
task_name=handover_block
tag=primitive_test

# Primitive test parameters
nudge_distance=0.05     # Meters for each primitive displacement
primitive_horizon=50   # Number of interpolation steps per primitive
camera_name=head_camera

PYTHONWARNINGS=ignore::UserWarning \
python script/test_primitives.py \
    --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --tag ${tag} \
    --nudge_distance ${nudge_distance} \
    --primitive_horizon ${primitive_horizon} \
    --camera_name ${camera_name} \
    $@
