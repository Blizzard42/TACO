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
mmd_threshold=1.05
use_pivot_steering="False"
use_primitive_steering="False"
guidance_scale=2.5
ema_alpha=0.8
act_steps=15
ensemble_weights="[0.9,0.1]"
vlm_model_name="Qwen/Qwen2.5-VL-72B-Instruct"
pivot_prompt_path="/nethome/gpatlin3/flash/TACO/third_party/Robotwin/steering/prompts/pivot_template.txt"
primitive_prompt_path="/nethome/gpatlin3/flash/TACO/third_party/Robotwin/steering/prompts/primitive_template.txt"
test_num=48

# ==============================================================================
# EXPERIMENT QUEUE
# ==============================================================================

TASK_1="adjust_bottle"
TAG_1="eve_experiments/bigger_base_2"
EXTRA_ARGS_1=""
DELAY_1=0

TASK_2="stack_blocks_two"
TAG_2="eve_experiments/bigger_base_2"
EXTRA_ARGS_2=""
DELAY_2=0

TASK_3="place_container_plate"
TAG_3="eve_experiments/bigger_base_2"
EXTRA_ARGS_3=""
DELAY_3=0

TASK_4="beat_block_hammer"
TAG_4="eve_experiments/bigger_base_2"
EXTRA_ARGS_4=""
DELAY_4=0

TASK_5="handover_block"
TAG_5="eve_experiments/bigger_base_2"
EXTRA_ARGS_5=""
DELAY_5=0

TASK_6="move_can_pot"
TAG_6="eve_experiments/bigger_base_2"
EXTRA_ARGS_6=""
DELAY_6=0

TASK_7="place_object_stand"
TAG_7="eve_experiments/bigger_base_2"
EXTRA_ARGS_7=""
DELAY_7=0

# ==============================================================================
# MASTER CONFIGURATION
# ==============================================================================
# LIST THE CONFIGS TO RUN (In Order)
RUN_ORDER=(1 3 4 6 7)
SEEDS=(0 1 2 3 4 5)
BASE_EXP_NAME="mar/11/"

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

for idx in "${RUN_ORDER[@]}"; do
    # Load Configuration Variables dynamically
    curr_task_var="TASK_$idx"
    curr_tag_var="TAG_$idx"
    curr_extra_args_var="EXTRA_ARGS_$idx"
    curr_delay_var="DELAY_$idx"

    TASK="${!curr_task_var}"
    TAG="${!curr_tag_var}"
    EXTRA_ARGS="${!curr_extra_args_var}"
    DELAY="${!curr_delay_var}"

    echo "================================================================================"
    echo "STARTING BATCH SEQUENCE $idx"
    echo "Task: $TASK" 
    echo "Tag: $TAG"
    echo "Extra Args: $EXTRA_ARGS"
    echo "================================================================================"

    # Loop through seeds and submit jobs
    for seed in "${SEEDS[@]}"; do
        
        # Distribute VLLM servers equally
        SERVER_INDEX=$((seed % NUM_SERVERS))
        VLM_SERVER_URL="${VLLM_SERVERS[$SERVER_INDEX]}"

        # Output directory logic based on your script
        LOG_DIR="./third_party/Robotwin/eval_result/${TAG}/${TASK}/seed_${seed}"
        mkdir -p "$LOG_DIR"

        echo "  > Submitting seed $seed [Server: $VLM_SERVER_URL]"
        
        # SBATCH HEREDOC
        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=evalSteer_%j
#SBATCH --output=./runs/$BASE_EXP_NAME/$TASK/$seed/run_%j.out
#SBATCH --error=./runs/$BASE_EXP_NAME/$TASK/$seed/run_%j.err
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
# export HF_HOME="$HOME/flash/huggingface"
# source /nethome/gpatlin3/flash/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate taco
# cd $SLURM_SUBMIT_DIR
cd ~/flash/TACO/third_party/Robotwin

echo "Running evaluation for task: $TASK with seed: $seed"

PYTHONWARNINGS=ignore::UserWarning \
PYTHONUNBUFFERED=1 TORCH_COMPILE_DISABLE=1 python script/eval_lerobot_torch_pi05.py \
    --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${TASK} \
    --task_config ${task_config} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --tag ${BASE_EXP_NAME}/${TAG}/${seed} \
    --policy_path ${policy_path} \
    --compute_mmd ${compute_mmd} \
    --num_mmd_samples ${num_mmd_samples} \
    --mmd_gamma ${mmd_gamma} \
    --mmd_threshold ${mmd_threshold} \
    --use_pivot_steering ${use_pivot_steering} \
    --use_primitive_steering ${use_primitive_steering} \
    --guidance_scale ${guidance_scale} \
    --ema_alpha ${ema_alpha} \
    --act_steps ${act_steps} \
    --ensemble_weights "${ensemble_weights}" \
    --vlm_server_url ${VLM_SERVER_URL} \
    --vlm_model_name ${vlm_model_name} \
    --pivot_prompt_path ${pivot_prompt_path} \
    --primitive_prompt_path ${primitive_prompt_path} \
    --test_num ${test_num} \
    --save_data True \
    ${EXTRA_ARGS}

echo "Completed seed: $seed"
EOT
        
    done

    echo "Batch $idx submitted successfully."

    # Handle Delay
    if [ "$DELAY" -gt 0 ]; then
        echo "Waiting $DELAY minutes before launching next configuration..."
        sleep "$((DELAY * 60))"
    else
        echo "No delay configured or end of list."
    fi
    echo ""

done

echo "All configuration batches submitted!"