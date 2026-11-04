#!/bin/bash
#SBATCH --job-name=qwen-f2-pilot
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --constraint=sm_80&vram16
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:55:00
#SBATCH --output=slurm_f2_pilot_%j.out

set -euo pipefail
cd "${TRUTHFULNESS_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source .venv/bin/activate
export HF_HOME="$PWD/hf_cache"
export PYTHONUNBUFFERED=1

# Only smoke or pilot is permitted in this short job.
case "${F2_MODE:-pilot}" in
    smoke|pilot) ;;
    *) echo "F2_MODE must be smoke or pilot" >&2; exit 2 ;;
esac

python -u src/train_f2_lite.py --mode "${F2_MODE:-pilot}"
