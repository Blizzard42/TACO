#!/bin/bash
#SBATCH --job-name=evalSteer_%j
#SBATCH --output=./runs/run_%j.out
#SBATCH --error=./runs/run_%j.err
#SBATCH --partition=kira-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --mem=0
#SBATCH --exclude="xaea-12,deebot,samantha,shakey,baymax,irona,flexo,siri"

export PYTHONIOENCODING=UTF-8
source ~/.bashrc
# export HF_HOME="$HOME/flash/huggingface"
# source /nethome/gpatlin3/flash/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate taco
# cd $SLURM_SUBMIT_DIR
cd ~/flash/TACO

bash scripts/eval/eval_robotwin2_torch_pi05.sh
