#!/bin/bash
#SBATCH --job-name=evalSteer_%j
#SBATCH --output=./runs/run_%j.out
#SBATCH --error=./runs/run_%j.err
#SBATCH --partition=kira-lab
#SBATCH --account=kira-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --mem=64G
#SBATCH --exclude="droid,flexo,irona,calculon,dendrite,xaea-12,johnny5,synapse,major"
# #SBATCH --exclude="xaea-12,deebot,samantha,shakey,baymax,irona,flexo,siri"

nvidia-smi
export PYTHONIOENCODING=UTF-8
source ~/.bashrc
# export HF_HOME="$HOME/flash/huggingface"
# source /nethome/gpatlin3/flash/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate taco
# cd $SLURM_SUBMIT_DIR
cd ~/flash/TACO

bash scripts/eval/eval_robotwin2_torch_pi05.sh
