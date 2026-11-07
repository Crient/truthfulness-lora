#!/bin/bash
#SBATCH --job-name=qwen-f2-full
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --constraint=sm_80&vram16
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=07:55:00
#SBATCH --output=slurm_f2_full_%j.out

set -euo pipefail
cd "${TRUTHFULNESS_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source .venv/bin/activate
export HF_HOME="$PWD/hf_cache"
export PYTHONUNBUFFERED=1

python -u src/train_f2_lite.py --mode full
