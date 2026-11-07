> **Historical document — November 4, 2026 release snapshot.** The original text below is preserved for research continuity. For the completed four-model study, use [current methods](APPROACHES.md), [matched results](RESULTS.md), [limitations](LIMITATIONS.md) and [release blockers](../release/BLOCKERS.md). Earlier scores and pending-work statements describe that snapshot.

# Chronological Experiment Log

**Snapshot through November 4, 2026.** This is a research audit trail intended to remain append-only as future work proceeds. Dates embedded in SLURM logs/result filenames are reproduced as recorded; their timezone was not independently established. Filenames, logs, code and trainer-state records are different types of evidence and are labeled accordingly.

## 1. Milestones and decisions

| Period | Event | Decision / observed outcome | Evidence |
|---|---|---|---|
| October 17, 2026 | Unity HPC GPU access established | Remote research environment and GPU training workflow prepared | Research conversation and environment notes |
| October 20–21 | Earlier notebook/setup activity | Existing notebook outputs contain references to another user's environment; not evidence for the final October 27 run | `verification/HISTORICAL_NOTEBOOK_OUTPUTS.md` in audit |
| October 27 | Fresh Approach 1 LoRA training | 9,500 QA records; rank-16 LoRA; 2,375 optimizer steps; validation checkpoint selected at step 2,000 | `run_full_lora.sbatch`, notebook 03, log job 66178521, trainer states |
| October 27 | TruthfulQA evaluation | 20-question tuned smoke, then 817-question original Qwen and tuned LoRA evaluations | `run_truthfulqa.sbatch`, log job 66187817, aggregate JSON + JSONL |
| October 27 | First documented research conclusion | Regular QA SFT reduced MC1 and MC2 point estimates versus original | Research report and notebook 04; original sample evidence |
| Early November | F2-Lite recipe developed | Three-record expansion: direct QA, knowledge retrieval, grounded QA; lower LR and longer context | `src/train_f2_lite.py`, Slurm scripts |
| November 4 | F2-Lite pilot (100 TruthfulQA questions) | MC1 pilot decreased, MC2 pilot increased slightly; noisy and underpowered | Pilot baseline/tuned result JSON + JSONL |
| November 4 | F2-Lite full training | 28,500 train + 1,500 validation derived records; 7,125 steps; best validation step 6,000 | `run_f2_lite_full.sbatch`, log job 66654973, trainer states |
| November 4 | F2-Lite full evaluation | 817-question MC1 26.81%; MC2 45.27% | `run_f2_lite_eval.sbatch`, log job 66684762, result JSON + JSONL |
| November 4 | Overall paired analysis | F2-Lite vs Approach 1 MC2 +1.30 pp; vs original −0.60 pp; minimal MC1 difference | Independent audit recalculation, `RESULTS.md` |
| November 4 | Answer-length exploratory analysis | No monotonic length-dependent pattern established | Numbers provided in research chat; analysis script not exported |
| November 4 | TruthfulQA categories joined and grouped | 817/817 categories reportedly joined; 66-question context-dependent group showed strong post-hoc MC2 difference | Numbers provided in research chat; join script not exported |
| November 4 | Read-only audit bundle created | 61 files, including source, notebooks, result samples, metadata and logs; SHA256 verification reported | `AUDIT_MANIFEST.md`, `SHA256SUMS`, inventory |
| Future | Approach 3 | No method, run, or result yet | To be specified before GPU training |

## 2. Historical training runs

### 2.1 Approach 1 — LoRA QA training

| Field | Recorded value |
|---|---|
| Script | `run_full_lora.sbatch` |
| Job ID appearing in log filename | `66178521` |
| Entrypoint | Notebook 03 cells 2,4,6; preflight; cell 12 |
| Output path | `results/qwen3_halueval_seed42/` |
| Exported adapter | `adapters/qwen3_halueval_seed42/` |
| Epochs | 1 |
| Optimizer steps completed | 2,375 |
| Train/validation items | 9,500 / 500 |
| Effective microbatch + accumulation | 1 × 4 per GPU |
| Learning rate | `2e-4` |
| Best checkpoint (trainer_state) | `checkpoint-2000` |
| Best validation loss | `0.8938544392585754` |
| Final Trainer-reported train loss | `0.9140742677387438` |
| Trainer-reported runtime | `3060.9147` seconds |
| Peak allocated GPU memory (code divided bytes by 1024³) | `3.85` GiB reported under a `GB` log label |
| Terminal log marker | `FULL TRAINING COMPLETED` |
| SLURM accounting | Unavailable in audit; exact exit code and allocated node unverified |

Validation checkpoints:

| Step | Evaluation loss |
|---:|---:|
| 500 | `0.9506149291992188` |
| 1,000 | `0.9319583773612976` |
| 1,500 | `0.9149497151374817` |
| 2,000 | `0.8938544392585754` |

The source selects the lowest periodically evaluated validation loss and reloads the best model before export. Although 2,375 steps completed, `checkpoint-2000` was the best recorded validation checkpoint. A lower validation loss does not by itself imply lower hallucination rates on TruthfulQA.

### 2.2 Approach 2 — F2-Lite full training

| Field | Recorded value |
|---|---|
| Script | `run_f2_lite_full.sbatch` |
| Job ID appearing in log filename | `66654973` |
| Entrypoint | `python -u src/train_f2_lite.py --mode full` |
| Output path | `results/f2_lite_seed42_full/` |
| Exported adapter | `adapters/f2_lite_seed42_full/` |
| Source item counts | 9,500 train / 500 validation |
| Expanded record counts | 28,500 train / 1,500 validation |
| Objective ratio by record count | 1:1:1 |
| Source items requiring knowledge truncation | 0 train, 0 validation |
| Epochs | 1 |
| Optimizer steps completed | 7,125 |
| Learning rate | `2e-5` |
| Maximum input length | 1,024 tokens |
| Best checkpoint (trainer_state) | `checkpoint-6000` |
| Best validation loss | `1.12755286693573` |
| Final Trainer-reported train loss | `1.2028051393743147` |
| Trainer-reported runtime | `10404.0068` seconds |
| Peak allocated GPU memory | `4.42` GiB reported under a `GB` log label |
| Terminal log marker | `F2-LITE FULL COMPLETED` |
| SLURM accounting | Unavailable in audit; exact exit code and allocated node unverified |

Validation checkpoints:

| Step | Evaluation loss |
|---:|---:|
| 1,500 | `1.168407917022705` |
| 3,000 | `1.1442012786865234` |
| 4,500 | `1.1322370767593384` |
| 6,000 | `1.12755286693573` |

F2 uses longer sequences, three targets and lower learning rate, so its loss should **not** be plotted against Approach 1's loss as a controlled comparison of training quality.

## 3. Historical benchmark runs

### 3.1 October 27 baseline/Approach 1 — job 66187817

Launcher `run_truthfulqa.sbatch` executed the following sequence under the same job:

1. Tuned adapter smoke evaluation with `--limit 20`; output saved without per-question JSONL.
2. Original Qwen3 full 817-question MC1 and MC2 evaluation, logging samples.
3. Trained LoRA full 817-question MC1 and MC2 evaluation, logging samples.

The log contains `TUNED MODEL SMOKE TEST PASSED`, `BASELINE EVALUATION COMPLETED` and `FULL TRUTHFULQA EVALUATION COMPLETED` terminal markers. The exported aggregate files independently identify the three numerical outputs.

| Run directory | Questions | MC1 | MC2 |
|---|---:|---:|---:|
| `results/truthfulqa_leng_20261027/baseline` | 817 | 29.38% | 45.87% |
| `results/truthfulqa_leng_20261027/tuned` | 817 | 26.68% | 43.97% |
| `results/truthfulqa_leng_20261027/tuned_smoke` | 20 | 25.00% | 38.73% |

### 3.2 November 4 F2 pilot

The package includes pilot result files, but the baseline pilot invocation and pilot training job IDs are not independently established. `run_f2_lite_pilot_eval.sbatch` records the tuned `--limit 100` invocation; pilot baseline output exists independently.

| Run directory | Questions | MC1 | MC2 |
|---|---:|---:|---:|
| `results/truthfulqa_f2_lite_seed42_pilot/baseline` | 100 | 30.00% | 47.72% |
| `results/truthfulqa_f2_lite_seed42_pilot/tuned` | 100 | 26.00% | 48.68% |

### 3.3 November 4 F2 full — job 66684762

`run_f2_lite_eval.sbatch` checked that the F2 full adapter existed and the destination was absent, then executed an 817-item `truthfulqa_mc1`/`truthfulqa_mc2` harness run with saved samples.

| Run directory | Questions | MC1 | MC2 |
|---|---:|---:|---:|
| `results/truthfulqa_f2_lite_seed42_full/tuned` | 817 | 26.81% | 45.27% |

The log ends with `F2 FULL TRUTHFULQA EVALUATION COMPLETED`.

## 4. Artifact map and identity of evidence

| Artifact type | Example | Reliability / limitation |
|---|---|---|
| Aggregate evaluation JSON | `results_*.json` | Direct stored metric totals and config; independent arithmetic checked against JSONL samples |
| Per-question JSONL | `samples_truthfulqa_mc1_*.jsonl`, `mc2` | Allows paired ID/document checks, deltas and confidence intervals; does not prove correctness of model execution |
| Notebook source | `03_train_lora.ipynb` | Exact inspected current source; historical file identity requires version-control provenance |
| F2 source | `src/train_f2_lite.py` | Records method and safe execution guards; depends on notebook cell indices |
| Trainer state | `checkpoint-2000/trainer_state.json`, etc. | Step count and best metric; excludes checkpoint weights and resolved TrainingArguments |
| Sanitized SLURM excerpts | `slurm_lora_66178521`, etc. | Contains training logs and completion markers; full scheduler state unavailable |
| Dataset metadata | `dataset_info.json`, `state.json` | Source schema, upstream URI, fingerprints; does not contain saved Arrow rows |
| Conversation analysis | Quartiles, category grouping, 66-question CI | Useful research lead but not independently reproduced from package |
| Audit manifest | `AUDIT_MANIFEST.md` | Read-only collection narrative and provenance hash inventory; derived secondary evidence |

## 5. Append-only entry template for Approach 3 and later runs

Copy this entry when a new run begins. Do not rewrite the November 4 entries after the fact.

```markdown
### YYYY-MM-DD — [Approach name / run ID]
- Hypothesis frozen before training:
- Code commit/hash:
- Dataset source/revision + split fingerprint:
- Model and tokenizer revisions:
- Seed set before LoRA/model initialization:
- Full resolved TrainingArguments:
- SLURM job ID, allocation, hardware and accounting evidence:
- Training record count, supervised-token budget and steps:
- Best checkpoint and selection criterion:
- Runtime, validation curve, warnings:
- Evaluation task/configuration and output SHA256:
- MC1/MC2 aggregate and paired CIs:
- Pre-specified subgroup analyses:
- Exploratory post-hoc analyses:
- Deviations from preregistration:
- Research decision / next step:
```

## 6. Why the audit ZIP remains separate from public GitHub

The comprehensive 61-file ZIP contains private research paths, log excerpts, per-question benchmark material and source copies. Its SHA256 inventory is valuable for **internal provenance**. A public project should ordinarily contain reviewed source, a safe summary of the metrics, methodology, and reproduction commands. Retain the original audit archive privately and track authorization for publishing code/data with the research supervisor.
