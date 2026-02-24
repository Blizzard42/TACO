# Please modify "policy_path", "tag", "cfn_ckpt_path" !!!!

# your cfn_ckpt_path !
# Here, we recommend using the absolute path.
# adjust_bottle
# stack_blocks_two
# place_container_plate
# beat_block_hammer
# handover_block
# move_can_pot
# place_object_stand

cd ./third_party/Robotwin

policy_name=pi05
policy_path="/nethome/gpatlin3/flash/huggingface/hub/models--rhodes-team-teleai--pi05_TACO_robotwin2_finetuned/snapshots/0f000e2748bd1fcb43027d8790f81fcccaa04670/"
task_config=demo_clean
seed=1
task_name=handover_block
tag=debug

# Steering Parameters:
compute_mmd=False
num_mmd_samples=100
mmd_gamma=10.0
use_pivot_steering=False
use_primitive_steering=False
guidance_scale=1.0
ensemble_weights="[1.0,0.0]" # [pivot_weight, primitive_weight]
vlm_server_url="http://localhost:8000"
vlm_model_name="Qwen/Qwen2.5-VL-72B-Instruct"
vlm_prompt_path="placeholder"
# output dir is "./third_party/Robotwin/eval_result/{tag}"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_lerobot_torch_pi05.py \
    --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --tag ${tag} \
    --policy_path $policy_path \
    --compute_mmd ${compute_mmd} \
    --num_mmd_samples ${num_mmd_samples} \
    --mmd_gamma ${mmd_gamma} \
    --use_pivot_steering ${use_pivot_steering} \
    --use_primitive_steering ${use_primitive_steering} \
    --guidance_scale ${guidance_scale} \
    --ensemble_weights ${ensemble_weights} \
    --vlm_server_url ${vlm_server_url} \
    --vlm_model_name ${vlm_model_name} \
    --vlm_prompt_path ${vlm_prompt_path} \
    $@

