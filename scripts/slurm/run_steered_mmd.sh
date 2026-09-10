#!/bin/bash
# ==============================================================================
# ENSEMBLE STEERING (pivot + primitive) with per-task p99 MMD trigger.
#
# Thresholds are the 99th percentile of per-step MMD over SUCCESSFUL rollouts
# of the matching base run (icra/sep_9/base_mmd_on, 6 seeds x 48).
#
# Run from the TACO repo root:  bash scripts/slurm/run_steered_mmd.sh
# ==============================================================================
set -euo pipefail

REPO_ROOT="/coc/testnvme/yali30/code/symbotic/TACO"
ROBOTWIN_DIR="${REPO_ROOT}/third_party/Robotwin"
CONDA_SH="/coc/testnvme/yali30/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="taco"

policy_name="pi05"
policy_path="/coc/testnvme/yali30/code/hf_cache/hub/models--rhodes-team-teleai--pi05_TACO_robotwin2_finetuned/snapshots/0f000e2748bd1fcb43027d8790f81fcccaa04670"
task_config="demo_clean"

# Ensemble steering: BOTH steerers on, so ensemble_weights is actually applied.
use_pivot_steering="True"
use_primitive_steering="True"
ENSEMBLE_WEIGHTS="[0.5,0.5]"
EW_TAG="ew_0.5_0.5"
guidance_scale=2.5

# Gaussian noise (metres, world-space) on the EE waypoints drawn for the VLM, to
# separate visually near-identical candidates. Rendering only -- guidance uses the
# unperturbed action_samples. For scale: nudge_distance (a full corrective primitive)
# is 0.05 m.
traj_std_perturb=0.005

# cv2 line width for candidate trajectories in the pivot image. Default in the
# steerer is 2; open-pi-zero standardised on 1 for legibility when candidates overlap.
pivot_line_thickness=1

# MMD settings held identical to the base run so the comparison is clean.
compute_mmd="True"
num_mmd_samples=10
mmd_gamma="median"

act_steps=15
ema_alpha=0.8
test_num=48

# Per-task p95 MMD trigger, from base_mmd_on successful rollouts (~5% of steps fire,
# vs ~1.2% measured at p99).
declare -A MMD_THRESHOLD=(
    [place_object_stand]=1.017
    [beat_block_hammer]=0.970
    [place_container_plate]=1.037
    [place_can_basket]=1.077
    [move_can_pot]=0.988
)

# Local prompts (the gpatlin3 paths in grid_search.sh are not readable from this account).
PROMPT_DIR="${ROBOTWIN_DIR}/steering/prompts"
pivot_prompt_path="${PROMPT_DIR}/pivot_template.txt"
primitive_prompt_path="${PROMPT_DIR}/primitive_template.txt"

# These endpoints serve Qwen3.8-27B (verified via /v1/models), not the
# Qwen2.5-VL-72B hardcoded in grid_search.sh.
vlm_model_name="Qwen/Qwen3.8-27B"
VLLM_SERVERS=(
    "http://optimistprime:51995"
    "http://tachikoma:34873"
    "http://ig-88:47417"
    "http://synapse:59087"
    "http://kitt:34905"
)

TASKS=(place_object_stand)
SEEDS=(0 1 2 3 4 5)
BASE_EXP_NAME="icra/sep_9/steered_p95_mmd_perturb/${EW_TAG}"

# ---- pre-flight ----
[ -f "${policy_path}/model.safetensors" ]               || { echo "ERROR: checkpoint missing"; exit 1; }
[ -f "${ROBOTWIN_DIR}/task_config/${task_config}.yml" ] || { echo "ERROR: task_config missing"; exit 1; }
[ -f "${CONDA_SH}" ]                                    || { echo "ERROR: conda.sh missing"; exit 1; }
[ -f "${pivot_prompt_path}" ]                           || { echo "ERROR: pivot prompt missing"; exit 1; }
[ -f "${primitive_prompt_path}" ]                       || { echo "ERROR: primitive prompt missing"; exit 1; }
for t in "${TASKS[@]}"; do
    [ -f "${ROBOTWIN_DIR}/envs/${t}.py" ] || { echo "ERROR: unknown task '${t}'"; exit 1; }
    [ -n "${MMD_THRESHOLD[$t]:-}" ]       || { echo "ERROR: no MMD threshold for '${t}'"; exit 1; }
done

NUM_SERVERS=${#VLLM_SERVERS[@]}
JOB_COUNTER=0

echo "Submitting $(( ${#TASKS[@]} * ${#SEEDS[@]} )) STEERED jobs (pivot+primitive, ${ENSEMBLE_WEIGHTS})"
echo "  ${test_num} rollouts x ${#SEEDS[@]} seeds | guidance_scale=${guidance_scale} | traj_std_perturb=${traj_std_perturb} | lt=${pivot_line_thickness} | VLM=${vlm_model_name}"
echo "  under eval_result/${BASE_EXP_NAME}/"

for TASK in "${TASKS[@]}"; do
    mmd_threshold="${MMD_THRESHOLD[$TASK]}"
    for seed in "${SEEDS[@]}"; do
        VLM_SERVER_URL="${VLLM_SERVERS[$(( JOB_COUNTER % NUM_SERVERS ))]}"
        JOB_COUNTER=$(( JOB_COUNTER + 1 ))

        RUN_DIR="${ROBOTWIN_DIR}/eval_result/${BASE_EXP_NAME}/${TASK}/seed_${seed}"
        mkdir -p "${RUN_DIR}"

        echo "  - ${TASK} | seed ${seed} | mmd_thr=${mmd_threshold} | ${VLM_SERVER_URL}"

        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=steer_${TASK}_${seed}
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

echo "STEERED eval | task=${TASK} seed=${seed} mmd_threshold=${mmd_threshold} ew=${ENSEMBLE_WEIGHTS} server=${VLM_SERVER_URL}"

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
    --guidance_scale ${guidance_scale} \
    --traj_std_perturb ${traj_std_perturb} \
    --pivot_line_thickness ${pivot_line_thickness} \
    --ensemble_weights "${ENSEMBLE_WEIGHTS}" \
    --act_steps ${act_steps} \
    --ema_alpha ${ema_alpha} \
    --vlm_server_url ${VLM_SERVER_URL} \
    --vlm_model_name ${vlm_model_name} \
    --pivot_prompt_path ${pivot_prompt_path} \
    --primitive_prompt_path ${primitive_prompt_path} \
    --test_num ${test_num}

echo "Completed ${TASK} seed ${seed}"
EOT
    done
done

echo "Done. Results under:"
echo "  ${ROBOTWIN_DIR}/eval_result/${BASE_EXP_NAME}/<task>/seed_<n>/"
