#!/bin/bash
#SBATCH --job-name=qwen-truthfulness
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --constraint=sm_80&vram16
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:55:00
#SBATCH --output=slurm_lora_%j.out

set -euo pipefail

cd "${TRUTHFULNESS_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source .venv/bin/activate
export HF_HOME="$PWD/hf_cache"
export PYTHONUNBUFFERED=1

python - <<'PY'
import json
from pathlib import Path

notebook_path = Path("notebooks/03_train_lora.ipynb")
cells = json.loads(notebook_path.read_text())["cells"]

context = {"__name__": "__main__"}

# Load the notebook's existing configuration, data, and training functions.
for index in (2, 4, 6):
    source = "".join(cells[index]["source"])
    exec(compile(source, f"notebook_cell_{index}", "exec"), context)

# Verify that the actual notebook tokenizer works before full training.
print("Checking training examples...")
sample = context["tokenize_split"]("train", 2)
assert len(sample) == 2
print("DATA PREFLIGHT PASSED")

# Start a fresh training run without modifying the notebook.
context["TRAINING_MODE"] = "fresh"

print("STARTING FULL LORA TRAINING")
source = "".join(cells[12]["source"])
exec(compile(source, "notebook_cell_12", "exec"), context)

print("FULL TRAINING COMPLETED")
PY
