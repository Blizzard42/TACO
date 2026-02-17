# Please modify "policy_path", "tag", "cfn_ckpt_path" !!!!

# your cfn_ckpt_path !
# Here, we recommend using the absolute path.
# adjust_bottle_cfn.pt
# beat_block_hammer_cfn.pt
# handover_block_cfn.pt
# move_can_pot_cfn.pt
# place_object_stand_cfn.pt

export cfn_ckpt_path="/nethome/gpatlin3//flash/huggingface/hub/models--rhodes-team-teleai--pi05_TACO_robotwin2_finetuned/snapshots/0f000e2748bd1fcb43027d8790f81fcccaa04670/cfns/place_object_stand_cfn.pt" 

cd ./third_party/Robotwin

policy_name=pi05
policy_path="/nethome/gpatlin3/flash/huggingface/hub/models--rhodes-team-teleai--pi05_TACO_robotwin2_finetuned/snapshots/0f000e2748bd1fcb43027d8790f81fcccaa04670/" # model report id, or trained lerobot model checkpoint, e.g. ./outputs/pi05_training/checkpoints/100000/pretrained_model
task_config=demo_clean
seed=1
task_name=place_object_stand
tag=place_object_stand_pi05_taco
# output dir is "./third_party/Robotwin/eval_result/{tag}"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_lerobot_torch_pi05_taco.py \
    --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --tag ${tag} \
    --policy_path $policy_path



