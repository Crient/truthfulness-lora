#!/bin/bash
#SBATCH --job-name=qwen-truthfulqa
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --constraint=sm_80&vram16
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:55:00
#SBATCH --output=slurm_truthfulqa_%j.out

set -euo pipefail

cd "${TRUTHFULNESS_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source .venv/bin/activate

export HF_HOME="$PWD/hf_cache"
export PYTHONUNBUFFERED=1

OUTROOT="$PWD/results/truthfulqa_leng_20261027"

# Preserve any earlier evaluation outputs.
if [ -e "$OUTROOT" ]; then
    echo "Output folder already exists: $OUTROOT"
    exit 1
fi

mkdir -p "$OUTROOT"

# First, verify that the trained adapter can be evaluated.
echo "STARTING TUNED MODEL SMOKE TEST"

python -m lm_eval run \
    --model hf \
    --model_args "pretrained=Qwen/Qwen3-1.7B,peft=$PWD/adapters/qwen3_halueval_seed42,dtype=float16" \
    --tasks truthfulqa_mc1 truthfulqa_mc2 \
    --device cuda:0 \
    --batch_size 1 \
    --num_fewshot 0 \
    --limit 20 \
    --output_path "$OUTROOT/tuned_smoke"

echo "TUNED MODEL SMOKE TEST PASSED"

# Evaluate the unchanged Qwen model.
echo "STARTING ORIGINAL QWEN EVALUATION"

python -m lm_eval run \
    --model hf \
    --model_args "pretrained=Qwen/Qwen3-1.7B,dtype=float16" \
    --tasks truthfulqa_mc1 truthfulqa_mc2 \
    --device cuda:0 \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path "$OUTROOT/baseline" \
    --log_samples

echo "BASELINE EVALUATION COMPLETED"

# Evaluate Qwen with your trained LoRA adapter.
echo "STARTING FINE-TUNED QWEN EVALUATION"

python -m lm_eval run \
    --model hf \
    --model_args "pretrained=Qwen/Qwen3-1.7B,peft=$PWD/adapters/qwen3_halueval_seed42,dtype=float16" \
    --tasks truthfulqa_mc1 truthfulqa_mc2 \
    --device cuda:0 \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path "$OUTROOT/tuned" \
    --log_samples

echo "FULL TRUTHFULQA EVALUATION COMPLETED"
