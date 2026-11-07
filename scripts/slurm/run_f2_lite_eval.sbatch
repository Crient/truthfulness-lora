#!/bin/bash
#SBATCH --job-name=qwen-f2-eval
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --constraint=sm_80&vram16
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:55:00
#SBATCH --output=slurm_f2_eval_%j.out

set -euo pipefail
cd "${TRUTHFULNESS_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source .venv/bin/activate
export HF_HOME="$PWD/hf_cache"
export PYTHONUNBUFFERED=1

ADAPTER="$PWD/adapters/f2_lite_seed42_full"
OUTROOT="$PWD/results/truthfulqa_f2_lite_seed42_full"

if [ ! -f "$ADAPTER/adapter_config.json" ]; then
    echo "Full F2 adapter is not ready: $ADAPTER" >&2
    exit 1
fi
if [ -e "$OUTROOT" ]; then
    echo "Evaluation already exists; refusing to overwrite: $OUTROOT" >&2
    exit 1
fi

mkdir -p "$OUTROOT"
python -m lm_eval run \
    --model hf \
    --model_args "pretrained=Qwen/Qwen3-1.7B,peft=$ADAPTER,dtype=float16" \
    --tasks truthfulqa_mc1 truthfulqa_mc2 \
    --device cuda:0 \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path "$OUTROOT/tuned" \
    --log_samples

echo "F2 FULL TRUTHFULQA EVALUATION COMPLETED"
