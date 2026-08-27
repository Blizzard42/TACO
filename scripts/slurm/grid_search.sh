#!/bin/bash

# ==============================================================================
# GLOBAL HARDCODED CONFIGURATIONS
# ==============================================================================
policy_name="pi05"
policy_path="/nethome/gpatlin3/flash/huggingface/hub/models--rhodes-team-teleai--pi05_TACO_robotwin2_finetuned/snapshots/0f000e2748bd1fcb43027d8790f81fcccaa04670/"
task_config="demo_clean"
compute_mmd="True"
num_mmd_samples=20
mmd_gamma="median"
use_pivot_steering="False"
use_primitive_steering="False"
act_steps=15
vlm_model_name="Qwen/Qwen2.5-VL-72B-Instruct"
pivot_prompt_path="/nethome/gpatlin3/flash/TACO/third_party/Robotwin/steering/prompts/pivot_template.txt"
primitive_prompt_path="/nethome/gpatlin3/flash/TACO/third_party/Robotwin/steering/prompts/primitive_template.txt"
test_num=48

# ==============================================================================
# GRID SEARCH SPACE
# ==============================================================================
GUIDANCE_SCALES=(2.5)
EMA_ALPHAS=(0.8)
ENSEMBLE_WEIGHTS=("[0.9,0.1]")
MMD_THRESHOLDS=(1.05)

SEEDS=(0 1 2 3 4 5)
TASKS=(
# "adjust_bottle"
# "beat_block_hammer"
# "move_can_pot"
# "place_container_plate"
# "place_object_stand"

# "place_can_basket"
# "place_bread_skillet"
# "shake_bottle"
# "hanging_mug"
# "stack_bowls_three"
# "place_phone_stand"
# "place_object_scale"
# "place_object_basket"
# "move_stapler_pad"
# "place_burger_fries"

# "click_bell"
# "stamp_seal"
# "place_dual_shoes"
# "place_cans_plasticbox"

#"place_bread_basket"

# "place_mouse_pad"
#"blocks_ranking_rgb"
#"pick_diverse_bottles"
#"stack_bowls_two"
#"handover_block"
#"rotate_qrcode"
#"move_pillbottle_pad"
#"turn_switch"
#"dump_bin_bigbin"
#"place_empty_cup"
#"place_fan"
#"open_microwave"
#"move_playingcard_away"
#"place_shoe"
#"stack_blocks_two"
#"stack_blocks_three"
#"place_a2b_right"
#"pick_dual_bottles"
#"lift_pot"
#"open_laptop"
#"place_a2b_left"
#"grab_roller"
#"click_alarmclock"
#"shake_bottle_horizontally"
#"handover_mic"
#"press_stapler"
#"put_bottles_dustbin"
#"blocks_ranking_size"
#"put_object_cabinet"
)


BASE_EXP_NAME="mar/12/final_base_supplemental"

# Add or remove your VLM servers here
VLLM_SERVERS=(
    "http://optimistprime:38477"
    "http://clippy:56749"
    "http://shakey:53727"
    "http://cheetah:33793"
    "http://ig-88:56151"
)

# ==============================================================================
# EXECUTION LOOP
# ==============================================================================

NUM_SERVERS=${#VLLM_SERVERS[@]}
TOTAL_COMBOS=$(( ${#GUIDANCE_SCALES[@]} * ${#EMA_ALPHAS[@]} * ${#ENSEMBLE_WEIGHTS[@]} * ${#MMD_THRESHOLDS[@]} ))
CURRENT_COMBO=0

# echo "Sleeping for 5 hours and 20 minutes to allow previous job to complete"
# sleep 19200

echo "Starting Grid Search. Total Hyperparameter Combinations: $TOTAL_COMBOS"
echo "================================================================================"

for mmd in "${MMD_THRESHOLDS[@]}"; do
    for gs in "${GUIDANCE_SCALES[@]}"; do
        for ema in "${EMA_ALPHAS[@]}"; do
            for ens in "${ENSEMBLE_WEIGHTS[@]}"; do
                
                ((CURRENT_COMBO++))
                
                # Format ensemble string for directory naming (e.g. "[0.3,0.7]" -> "0.3_0.7")
                ens_clean=$(echo "$ens" | tr -d '[]' | tr ',' '_')
                EXP_TAG="gs_${gs}_mmd_${mmd}_ema_${ema}_ens_${ens_clean}"
                
                echo ""
                echo ">>> LAUNCHING BATCH $CURRENT_COMBO OF $TOTAL_COMBOS"
                echo ">>> Hyperparameters: Guidance=$gs | EMA=$ema | Ensemble=$ens | MMD=$mmd"
                echo ">>> Directory Tag: $EXP_TAG"
                
                JOB_COUNTER=0
                
                for task in "${TASKS[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        
                        # Distribute VLLM servers equally across the 21 jobs
                        SERVER_INDEX=$((JOB_COUNTER % NUM_SERVERS))
                        VLM_SERVER_URL="${VLLM_SERVERS[$SERVER_INDEX]}"
                        ((JOB_COUNTER++))

                        # Directory logic
                        SLURM_LOG_DIR="./runs/$BASE_EXP_NAME/$EXP_TAG/$task/seed_${seed}"
                        EVAL_LOG_DIR="./third_party/Robotwin/eval_result/$BASE_EXP_NAME/$EXP_TAG/$task/seed_${seed}"
                        
                        mkdir -p "$SLURM_LOG_DIR"
                        mkdir -p "$EVAL_LOG_DIR"
                        
                        echo "    - Submitting $task | Seed $seed [Server: $VLM_SERVER_URL]"
                        
                        # SBATCH HEREDOC
                        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=eval_${task}_${seed}
#SBATCH --output=${SLURM_LOG_DIR}/run_%j.out
#SBATCH --error=${SLURM_LOG_DIR}/run_%j.err
#SBATCH --partition=kira-lab
#SBATCH --account=kira-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --mem=64G
#SBATCH --exclude="droid,flexo,irona,calculon,dendrite,xaea-12,johnny5,synapse,major,qt-1"

nvidia-smi
export PYTHONIOENCODING=UTF-8
source ~/.bashrc
conda deactivate
conda activate taco
cd ~/flash/TACO/third_party/Robotwin

echo "Running evaluation for task: $task with seed: $seed"
echo "Grid Params: Guidance=$gs, EMA=$ema, Ensemble=$ens"

PYTHONWARNINGS=ignore::UserWarning \
PYTHONUNBUFFERED=1 TORCH_COMPILE_DISABLE=1 python script/eval_lerobot_torch_pi05.py \
    --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task} \
    --task_config ${task_config} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --tag ${BASE_EXP_NAME}/${EXP_TAG}/${seed} \
    --policy_path ${policy_path} \
    --compute_mmd ${compute_mmd} \
    --num_mmd_samples ${num_mmd_samples} \
    --mmd_gamma ${mmd_gamma} \
    --mmd_threshold ${mmd} \
    --use_pivot_steering ${use_pivot_steering} \
    --use_primitive_steering ${use_primitive_steering} \
    --guidance_scale ${gs} \
    --ema_alpha ${ema} \
    --act_steps ${act_steps} \
    --ensemble_weights "${ens}" \
    --vlm_server_url ${VLM_SERVER_URL} \
    --vlm_model_name ${vlm_model_name} \
    --pivot_prompt_path ${pivot_prompt_path} \
    --primitive_prompt_path ${primitive_prompt_path} \
    --test_num ${test_num}

echo "Completed seed: $seed"
EOT
                    done
                    # sleep 2400
                done
                
                # 30-minute delay logic
                if [ "$CURRENT_COMBO" -lt "$TOTAL_COMBOS" ]; then
                    echo "Batch $CURRENT_COMBO submitted successfully."
                    echo "Sleeping for 90 minutes (5400 seconds) to allow VLM server processing..."
                    sleep 0
                else
                    echo "Final batch submitted successfully!"
                fi
                
            done
        done
    done
done
echo "================================================================================"
echo "Grid search submission complete."