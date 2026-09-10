#!/bin/bash
# ==============================================================================
# Base (no-steering) policy evaluation with MMD logging ON.
# One job per (task, seed). Only the task list changes between experiments.
#
# Run from the TACO repo root:  bash scripts/slurm/run_base_mmd.sh
# ==============================================================================
set -euo pipefail

REPO_ROOT="/coc/testnvme/yali30/code/symbotic/TACO"
ROBOTWIN_DIR="${REPO_ROOT}/third_party/Robotwin"
CONDA_SH="/coc/testnvme/yali30/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="taco"

policy_name="pi05"
policy_path="/coc/testnvme/yali30/code/hf_cache/hub/models--rhodes-team-teleai--pi05_TACO_robotwin2_finetuned/snapshots/0f000e2748bd1fcb43027d8790f81fcccaa04670"
task_config="demo_clean"

# No steering, but MMD ON.
compute_mmd="True"
num_mmd_samples=10
mmd_gamma="median"
mmd_threshold=1.05
use_pivot_steering="False"
use_primitive_steering="False"

act_steps=15
ema_alpha=0.8
test_num=48

TASKS=(
    beat_block_hammer
    place_can_basket
    place_container_plate
    place_object_stand
    move_can_pot
)
SEEDS=(3 4 5)
BASE_EXP_NAME="icra/sep_9/base_mmd_on"

# Sanity checks before burning queue slots.
[ -f "${policy_path}/model.safetensors" ]              || { echo "ERROR: checkpoint missing: ${policy_path}"; exit 1; }
[ -f "${ROBOTWIN_DIR}/task_config/${task_config}.yml" ] || { echo "ERROR: task_config/${task_config}.yml missing"; exit 1; }
[ -f "${CONDA_SH}" ]                                    || { echo "ERROR: conda.sh missing: ${CONDA_SH}"; exit 1; }
for t in "${TASKS[@]}"; do
    [ -f "${ROBOTWIN_DIR}/envs/${t}.py" ] || { echo "ERROR: unknown task '${t}' (no envs/${t}.py)"; exit 1; }
    grep -q "^${t}:" "${ROBOTWIN_DIR}/task_config/_eval_step_limit.yml" || { echo "ERROR: no step limit for '${t}'"; exit 1; }
done

echo "Submitting $(( ${#TASKS[@]} * ${#SEEDS[@]} )) jobs: ${#TASKS[@]} tasks x ${#SEEDS[@]} seeds"
echo "  ${test_num} rollouts each | no steering | MMD ON (num_mmd_samples=${num_mmd_samples})"
echo "  under eval_result/${BASE_EXP_NAME}/"

for TASK in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # eval_lerobot_torch_pi05.py:205 builds
        #   eval_result/{tag}/{task}/{policy}/{task_config}/{ckpt}/{timestamp}
        # With tag = BASE/TASK/seed_N the tree is <task>/seed_N/, and the slurm logs
        # sit at the seed level. The task name repeats one level down -- unavoidable
        # without editing that shared line.
        RUN_DIR="${ROBOTWIN_DIR}/eval_result/${BASE_EXP_NAME}/${TASK}/seed_${seed}"
        mkdir -p "${RUN_DIR}"

        echo "  - ${TASK} | seed ${seed}"

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
#SBATCH --time=05:00:00

set -euo pipefail

nvidia-smi
export PYTHONIOENCODING=UTF-8
export HF_HUB_CACHE=/coc/testnvme/yali30/code/hf_cache/hub

source ${CONDA_SH}
conda activate ${CONDA_ENV}
cd ${ROBOTWIN_DIR}

echo "BASE eval (no steering, MMD ON) | task=${TASK} seed=${seed} test_num=${test_num} num_mmd_samples=${num_mmd_samples}"

PYTHONWARNINGS=ignore::UserWarning \
PYTHONUNBUFFERED=1 TORCH_COMPILE_DISABLE=1 python script/eval_lerobot_torch_pi05.py \
    --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${TASK} \
    --task_config ${task_config} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --tag ${BASE_EXP_NAME}/${TASK}/seed_${seed} \
    --policy_path ${policy_path} \
    --compute_mmd ${compute_mmd} \
    --num_mmd_samples ${num_mmd_samples} \
    --mmd_gamma ${mmd_gamma} \
    --mmd_threshold ${mmd_threshold} \
    --use_pivot_steering ${use_pivot_steering} \
    --use_primitive_steering ${use_primitive_steering} \
    --act_steps ${act_steps} \
    --ema_alpha ${ema_alpha} \
    --test_num ${test_num}

echo "Completed ${TASK} seed ${seed}"
EOT
    done
done

echo "Done. Per (seed, task) directory holds slurm logs + eval output together:"
echo "  ${ROBOTWIN_DIR}/eval_result/${BASE_EXP_NAME}/<task>/seed_<n>/"
echo "    run_<jobid>.out / run_<jobid>.err"
echo "    <task>/${policy_name}/${task_config}/None/<timestamp>/_result.txt"
