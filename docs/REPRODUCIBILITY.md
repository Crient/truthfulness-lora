# Reproducibility Guide

**Status:** The saved scores can be independently recalculated from the exported samples. Full training is **not yet a one-command, portable reproduction**: model/data caches and adapter weights were excluded, exact historical dependencies and tokenizer revision are incompletely pinned, and Unity paths are hard-coded in the original scripts.

This document is an **instructional record**. No SLURM jobs, model inference, benchmark runs, or network downloads were executed in preparing the documentation.

## 1. Reproduction levels

| Level | Objective | Current status | Required materials |
|---|---|---|---|
| A — Arithmetic verification | Confirm JSONL question means equal aggregate scores | **Verified independently** | Saved sample JSONL and aggregate JSON |
| B — Paired statistical comparison | Recompute full-set deltas and paired-bootstrap CIs | **Verified independently** | Paired sample JSONL, NumPy |
| C — Evaluation rerun | Re-evaluate original and saved adapters | **Not performed in this audit** | Exact model/tokenizer files, adapters, Python environment, GPU, benchmark cache |
| D — Training reproduction | Train fresh adapters that replicate historical results | **Not guaranteed** | Source/split, full resolved args, exact revisions, initial seeds, GPU and package lock |
| E — Scientific replication | Test whether results generalize | **Not performed** | Multiple seeds, independent data/tasks, prespecified analysis |

## 2. Environment evidence

The audit's current installed-package inventory (recorded November 4) reports:

| Package | Version |
|---|---|
| Python | `3.12.3` (evaluation environment report) |
| PyTorch | `2.9.1+cu126` |
| Transformers | `4.57.6` |
| PEFT | `0.18.1` |
| Accelerate | `1.12.0` |
| Datasets | `4.4.1` |
| `lm_eval` | `0.4.11` |
| NumPy | `2.2.6` |
| tokenizers | `0.22.2` |
| huggingface_hub | `0.36.2` |
| safetensors | `0.8.0` |
| Evaluated GPU | NVIDIA A16, from evaluation JSON |

The trainer metadata contains checkpoint state but **not a complete historical package lock or all resolved TrainingArguments**. Treat the inventory as an environment reconstruction starting point, not proof of the versions loaded at every earlier execution.

## 3. Source and revision requirements

| Resource | Recorded identifier | Missing control |
|---|---|---|
| Base model | `Qwen/Qwen3-1.7B` | Training checkpoint snapshot SHA not captured independently |
| Evaluation model | JSON records SHA `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Confirm adapter/model/tokenizer compatibility |
| Tokenizer | Fast Qwen3 tokenizer/chat template | Exact tokenizer commit missing |
| HaluEval QA | `pminervini/HaluEval`, `qa`, `data` | Notebook omits explicit `revision` |
| HaluEval cache | Upstream parquet source commit `12a856119f03975a94509091e8cada3e6be6ead7` | Verify same rows and file checksum |
| Training split | `train_test_split(test_size=.05, seed=42)` | Pin dataset version and saved row/fingerprint identity |
| TruthfulQA | `truthfulqa/truthful_qa`, `multiple_choice`, validation | Dataset revision and 38-category `generation` mapping not saved |
| Harness | `lm_eval 0.4.11`, MC1 task v2, MC2 task v3 | Pin exact harness commit/task code for reruns |

The original run used offline model loading and project-local Hugging Face caches. A freshly cloned documentation-only project does not have these assets.

## 4. Verify the exported scores: CPU only

Use the large results JSONL from your **private** Unity project or audit ZIP in a local/private workspace. The public documentation bundle excludes those potentially license-sensitive benchmark per-item records.

From the project root, after making NumPy available:

```bash
python analysis/verify_truthfulqa_samples.py --root . --bootstrap 10000 --seed 42
```

The independent script locates the following relative subtrees by glob:

```text
results/truthfulqa_leng_20261027/baseline/
results/truthfulqa_leng_20261027/tuned/
results/truthfulqa_f2_lite_seed42_full/tuned/
```

Each must contain exactly one `samples_truthfulqa_mc1_*.jsonl`, one `samples_truthfulqa_mc2_*.jsonl` and one aggregate `results_*.json` somewhere under the directory. The script:

- Parses sample `doc_id`, full `doc` and `acc` fields.
- Checks uniqueness, sample counts and exact matching between paired documents.
- Verifies sample means against the aggregate JSON to within `1e-12`.
- Computes per-model MC1/MC2 percentages and three paired comparisons.
- Computes reproducible nominal percentile bootstrap intervals.
- Starts no training or evaluation and writes no files.

Expected full-run percentages, rounded to two decimals:

```text
Original     MC1 29.38%   MC2 45.87%
Approach 1   MC1 26.68%   MC2 43.97%
F2-Lite      MC1 26.81%   MC2 45.27%
```

If the script reports a mismatch in IDs, documents, means or task data, **stop before interpreting comparisons**. A successful arithmetic check validates the *saved files*, not the inference run that generated them.

## 5. Recreate the saved HaluEval data split

In a scratch reproduction copy, first stage the correct dataset version and use an environment with compatible `datasets`:

```python
from datasets import DatasetDict, load_dataset

ds = load_dataset(
    "pminervini/HaluEval",
    "qa",
    split="data",
    revision="12a856119f03975a94509091e8cada3e6be6ead7",
)
split = ds.train_test_split(test_size=0.05, seed=42)
prepared = DatasetDict({
    "train": split["train"],
    "validation": split["test"],
})
```

This illustrates the historical procedure while adding a known dataset revision. **Do not assume** the upstream commit is a complete guarantee of the exact original local Arrow row sequence; compare row counts, fingerprints, representative row hashes and any saved metadata. Do not overwrite `data/halueval_qa_seed42` in the original project.

## 6. Historical training commands (reference only)

The original files are tied to this research group's Unity workspace and intended for the SLURM environment. For a reproduction, first copy them into a separate directory, update all absolute paths, reconstruct caches, and choose new noncolliding output names.

### Approach 1

```bash
sbatch run_full_lora.sbatch
```

This runner loads notebook cells `(2, 4, 6)` and executes `(12)` after a small data/tokenization preflight. It requires a GPU allocation, and the notebook's `require_allocated_cuda()` checks `SLURM_JOB_ID` and CUDA. The run refuses populated output directories.

### Approach 2: preflight, pilot and full

```bash
# CPU-only tokenizer/data preflight, if the prepared split and tokenizer exist
python src/train_f2_lite.py --mode preflight

# GPU jobs: launch sequentially only if preceding steps succeed
sbatch run_f2_lite_pilot.sbatch
sbatch run_f2_lite_pilot_eval.sbatch
sbatch run_f2_lite_full.sbatch
sbatch run_f2_lite_eval.sbatch
```

The pilot evaluates **100** questions and is not a full benchmark. The full evaluation scripts refuse preexisting destination directories. No Slurm dependency chain is implemented; **do not submit all the commands at once**. Check contemporary account, QOS and partition guidance before allocating cluster resources.

These commands are *historical references*, not requests or recommendations to rerun the user’s current GPU workloads now.

## 7. Benchmark evaluation protocol

Original-run task settings recorded in `run_truthfulqa.sbatch` and `run_f2_lite_eval.sbatch`:

```bash
python -m lm_eval run \
  --model hf \
  --model_args "pretrained=Qwen/Qwen3-1.7B,peft=/path/to/adapter,dtype=float16" \
  --tasks truthfulqa_mc1 truthfulqa_mc2 \
  --device cuda:0 \
  --batch_size 1 \
  --num_fewshot 0 \
  --output_path /path/to/new/results \
  --log_samples
```

For the original baseline, omit the `peft` argument. In an actual reproducibility run, pin/verify the recorded model revision and tokenizer snapshot rather than relying on `main`; this may require adapting the argument string to your installed harness. Ensure MC1 v2 and MC2 v3, literal Q/A prompt, fixed six template demonstrations, `multiple_choice` validation split, dtype and all 817 questions remain consistent. A new harness version or altered prompt can change scores.

Check the saved warning that the evaluation loader found tokenizer vocabulary size **151669** versus model-config vocabulary size **151936** and resized embeddings. Preserve and investigate this behavior before attributing any difference to the trained adapter.

## 8. Why a clean rerun may differ

1. **Random initialization:** the code creates the LoRA adapter before the `Trainer` sets its default RNG seed. A fresh run may start from a different adapter initialization.
2. **Tokenizer/model pinning:** the initial training snapshot and tokenizer revision were not independently captured.
3. **Installed defaults:** omitted `TrainingArguments` fields depend on library version.
4. **Floating-point/GPU variance:** device, kernels, precision and nondeterministic operations may alter training trajectories.
5. **Dataset version and ordering:** cache metadata points to an upstream revision, but complete saved Arrow data was excluded.
6. **Checkpoint choice:** approach 1 exports the best evaluated checkpoint at 2,000; F2 exports at 6,000, not necessarily the final-epoch weights.
7. **Prompt formatting:** training uses Qwen chat templating while evaluation uses plain fixed Q/A likelihood examples.
8. **Benchmark availability:** TruthfulQA evaluation rows and source task code need fixed revisions and license review.

## 9. What to capture for Approach 3

Before launching any training, create one machine-readable run manifest with:

```yaml
approach: "approach_3_pending"
status: "planned"
experiment_date: null
git_commit: null
base_model_id: "Qwen/Qwen3-1.7B"
base_model_revision: null
tokenizer_revision: null
training_dataset_revision: null
train_split_fingerprint: null
validation_split_fingerprint: null
seed_before_model_init: null
resolved_training_arguments: null
lora_parameters: null
slurm_job_id: null
gpu_model: null
training_start_and_end: null
best_checkpoint: null
final_adapter_sha256: null
truthfulqa_task_versions: {mc1: 2, mc2: 3}
truthfulqa_prompt_hash: null
evaluation_result_files_sha256: null
```

Also preserve the precise question-ID mapping and explicit paired-bootstrap script, and avoid making the already-inspected TruthfulQA subgroup the sole final evaluation target.

## 10. Safety and publication

- Do not run GPU training on a login node.
- Do not run historical launchers against populated output directories.
- Preserve adapters, raw samples and the original 61-file audit privately unless sharing is authorized.
- Do not commit Hugging Face caches, private `.env` files, credentials, or SLURM account information.
- Verify third-party licensing and research group permissions before releasing source/data.
- See [`PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md) before GitHub publication.
