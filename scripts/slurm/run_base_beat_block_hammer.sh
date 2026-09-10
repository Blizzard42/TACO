#!/bin/bash
# ==============================================================================
# Base (no-steering) policy evaluation: beat_block_hammer, 3 seeds x 48 rollouts.
# Single configuration -- no hyperparameter grid.
#
# Run from the TACO repo root:  bash scripts/slurm/run_base_beat_block_hammer.sh
# ==============================================================================
set -euo pipefail

REPO_ROOT="/coc/testnvme/yali30/code/symbotic/TACO"
ROBOTWIN_DIR="${REPO_ROOT}/third_party/Robotwin"
CONDA_SH="/coc/testnvme/yali30/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="taco"

policy_name="pi05"
policy_path="/coc/testnvme/yali30/code/hf_cache/hub/models--rhodes-team-teleai--pi05_TACO_robotwin2_finetuned/snapshots/0f000e2748bd1fcb43027d8790f81fcccaa04670"
task_config="demo_clean"

# No steering, no MMD -- unmodified base policy.
compute_mmd="False"
use_pivot_steering="False"
use_primitive_steering="False"

act_steps=15
ema_alpha=0.8
test_num=48

TASK="beat_block_hammer"
SEEDS=(0 1 2)
BASE_EXP_NAME="base_no_steering/beat_block_hammer"

# Sanity checks before burning queue slots.
[ -f "${policy_path}/model.safetensors" ]        || { echo "ERROR: checkpoint missing: ${policy_path}"; exit 1; }
[ -f "${ROBOTWIN_DIR}/task_config/${task_config}.yml" ] || { echo "ERROR: task_config/${task_config}.yml missing"; exit 1; }
[ -f "${CONDA_SH}" ]                             || { echo "ERROR: conda.sh missing: ${CONDA_SH}"; exit 1; }

echo "Submitting ${#SEEDS[@]} jobs: ${TASK}, ${test_num} rollouts each (no steering, no MMD)"

for seed in "${SEEDS[@]}"; do
    RUN_DIR="${ROBOTWIN_DIR}/eval_result/${BASE_EXP_NAME}/${seed}"
    mkdir -p "${RUN_DIR}"

    echo "  - ${TASK} | seed ${seed} -> ${RUN_DIR}"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=base_${TASK}_${seed}
#SBATCH --output=${RUN_DIR}/run_%j.out
#SBATCH --error=${RUN_DIR}/run_%j.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="a40:1"
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

nvidia-smi
export PYTHONIOENCODING=UTF-8
export HF_HUB_CACHE=/coc/testnvme/yali30/code/hf_cache/hub

source ${CONDA_SH}
conda activate ${CONDA_ENV}
cd ${ROBOTWIN_DIR}

echo "Running BASE (no steering, no MMD) eval | task=${TASK} seed=${seed} test_num=${test_num}"

PYTHONWARNINGS=ignore::UserWarning \
PYTHONUNBUFFERED=1 TORCH_COMPILE_DISABLE=1 python script/eval_lerobot_torch_pi05.py \
    --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${TASK} \
    --task_config ${task_config} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --tag ${BASE_EXP_NAME}/${seed} \
    --policy_path ${policy_path} \
    --compute_mmd ${compute_mmd} \
    --use_pivot_steering ${use_pivot_steering} \
    --use_primitive_steering ${use_primitive_steering} \
    --act_steps ${act_steps} \
    --ema_alpha ${ema_alpha} \
    --test_num ${test_num}

echo "Completed seed: ${seed}"
EOT
done

echo "Done. Per-seed directory (slurm logs + eval output together):"
echo "  ${ROBOTWIN_DIR}/eval_result/${BASE_EXP_NAME}/<seed>/"
echo "    run_<jobid>.out / run_<jobid>.err"
echo "    ${TASK}/${policy_name}/${task_config}/None/<timestamp>/_result.txt"
