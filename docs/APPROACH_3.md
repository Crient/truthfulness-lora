# Approach 3 — Prospective Experiment Protocol

**Status:** Not implemented or run.  
**Opened:** November 4, 2026.  
**Owner / review:** Gechleng (Leng) Lim and research advisor/team, to confirm before training.

This is a **blank prospective research record**, not a claim that Approach 3 uses any particular method. Keep it distinct from the fixed, historical results for the original model, Approach 1 and F2-Lite. Do not fill results before they exist.

## 1. What Approaches 1 and 2 establish

The original Qwen3-1.7B checkpoint outperformed both fine-tuned versions on the complete TruthfulQA benchmark. Approach 1's correct-answer SFT decreased MC1 by about 2.69 percentage points. F2-Lite partially recovered MC2 relative to Approach 1 (+1.30 pp) but remained about 0.60 pp below original MC2.

The strongest exploratory lead is a post-hoc 66-question indexical/subjective group with a reported +14.09 pp F2-vs-original MC2 difference. The matching script/metadata was not included in the research audit. The next study should avoid using this selected subset as the **only final test**.

## 2. Scientific planning form: complete **before** the first GPU run

| Field | Required entry | State |
|---|---|---|
| Approach name and concise method definition | — | **TBD** |
| Scientific hypothesis, including direction and primary metric | — | **TBD** |
| Scientific rationale / citation | — | **TBD** |
| Exactly what differs from Approach 1 | — | **TBD** |
| Exactly what differs from F2-Lite | — | **TBD** |
| Training supervision and target format | — | **TBD** |
| Use of `hallucinated_answer` (if any) | — | **TBD** |
| Whether learning rate is held fixed | — | **TBD** |
| Whether max sequence length is held fixed | — | **TBD** |
| Whether source-item, supervised-token or compute budget is matched | — | **TBD** |
| Number of independent training seeds | — | **TBD** |
| Checkpoint-selection criterion | — | **TBD** |
| Primary comparison and primary metric | — | **TBD** |
| Secondary metrics and hypotheses | — | **TBD** |
| Pre-specified analysis of context-dependent questions | — | **TBD** |
| Independent final evaluation task | — | **TBD** |
| Failure/stop criterion and compute budget | — | **TBD** |

## 3. Candidate method directions (discussion only)

These are **options to consider**, not components already selected for Approach 3:

| Candidate | What it would test | Important design control |
|---|---|---|
| Explicit correct-vs-hallucinated contrastive objective | Whether negative HaluEval targets help distinguish plausible misinformation | Hold model, source examples, number of updates, context length and LR fixed; inspect labels/answer lengths |
| Pairwise preference training | Whether preference supervision on `right_answer` versus `hallucinated_answer` improves rankings | Decide objective/regularization; preserve a suitable reference model; separate preference tuning from extra SFT |
| F2-Lite component ablation | Which of direct QA, retrieval or grounded QA contributes to behavior | Match supervised-token or optimizer-step budget, not just epochs |
| Hyperparameter-controlled LoRA comparison | Whether the F2 gain is caused by changed LR, sequence length or update count | Change **one** factor at a time wherever feasible |
| Inference-only calibration or prompt study | Whether scoring changes come from Qwen thinking/chat-format and prompt templates | Freeze checkpoint/answer choices and modify one prompt condition |

The research team should select **one defensible primary hypothesis** and a feasible ablation design before spending cluster GPU time. If the method is inspired by a paper, document implementation differences precisely.

## 4. Comparison design

### 4.1 Fixed evaluation conditions

- All **817** original TruthfulQA `multiple_choice` validation documents, with recorded exact question IDs/content.
- MC1 task version 2 and MC2 task version 3 under an explicitly pinned harness commit.
- Same candidate answer ordering, labels, normalization, prompt Q/A exemplars, dtype and evaluation batch behavior.
- Consistent base model/tokenizer revisions, or repeat all baselines if they change.
- Log per-question data with `--log_samples` for paired analysis, while keeping raw benchmark files private as appropriate.
- Separately evaluate free-form generated answers if the scientific claim concerns practical hallucinations.

### 4.2 Fairness rules

Two legitimate comparison strategies answer different questions:

1. **Recipe comparison:** compare each approach as designed, while reporting the differing training budgets and other confounders. This is what the current Approach 1/F2 comparison does.
2. **Controlled mechanism test:** hold learning rate, data/token budget, sequence length, model revision, optimizer and training seeds as constant as feasible, varying only the target objective or one intended mechanism.

If compute permits, report both. A matching *number of records* may still expose different supervised-token counts when one target is a long knowledge passage.

### 4.3 Training seeds

Set seeds **before** model and LoRA adapter initialization; store the exact seed, optimizer state and resolved arguments. Prefer several independent seeds to estimate variability across full training runs. The existing paired question bootstrap captures benchmark-question variability for fixed checkpoints, not variability caused by training randomness.

## 5. Pre-specified statistical plan (fill before training)

| Item | Proposed baseline rule | Decision before run |
|---|---|---|
| Primary outcome | One of MC1 or MC2 on all 817 questions | **TBD** |
| Primary comparator | Original model and/or Approach 1/F2 (clearly ranked) | **TBD** |
| Paired-delta unit | TruthfulQA question | Accept/change |
| Bootstrap samples and RNG | 10,000 paired resamples, NumPy `default_rng(42)` with sorted IDs | Accept/change |
| Primary confidence level | 95% | Accept/change |
| Multiple-comparison policy | Declare main contrast; adjust/label other contrasts exploratory | **TBD** |
| Context-dependent subgroup | Freeze 66-item question IDs before seeing Approach 3 outcomes | **TBD / mapping not yet recovered** |
| Independent out-of-domain dataset | Name, version, and primary metric | **TBD** |
| Free-generation hallucination rubric | Prompt policy and evaluator agreement | **TBD** |
| Minimum practically meaningful gain | Predeclare in percentage points | **TBD** |

**Important:** A pre-specified 66-question subgroup on Approach 3 provides a stronger test of a *subgroup prediction* than a second post-hoc search on the same 66 outcomes. It still shares the training/benchmark development history, so a genuinely independent benchmark is preferable for a confirmatory claim.

## 6. Implementation / run ledger template

```yaml
approach_id: approach_3
approach_name: null
status: not_started
primary_hypothesis: null
git_commit: null
date_of_pre_registration: null
references: []
dataset:
  name: pminervini/HaluEval
  config: qa
  source_revision: null
  train_rows: null
  validation_rows: null
  saved_split_fingerprints: null
  sample_construction: null
model:
  base_model: Qwen/Qwen3-1.7B
  model_revision: null
  tokenizer_revision: null
  template_and_thinking_mode: null
training:
  method: null
  lora: {r: null, alpha: null, dropout: null, targets: null}
  learning_rate: null
  epochs: null
  microbatch_size: null
  gradient_accumulation: null
  max_sequence_length: null
  supervised_token_budget: null
  optimizer: null
  scheduler: null
  warmup: null
  seeds: []
  checkpoint_selection: null
compute:
  slurm_job_ids: []
  gpu_model: null
  peak_memory: null
  runtime_seconds: null
artifacts:
  adapter_sha256: null
  trainer_state: null
  result_jsons: []
  sample_jsonls: []
  warnings: []
evaluation:
  harness_revision: null
  truthfulqa_mc1: null
  truthfulqa_mc2: null
  sample_count: 817
  prompt_hash: null
  independent_benchmark: null
analysis:
  primary_pair_bootstrap: null
  multiple_testing_policy: null
  secondary_analyses: []
  deviations: []
```

## 7. Future result tables — do not fill until verified

### Aggregate metrics

| Evaluated system | MC1 | MC2 | Number of TruthfulQA questions |
|---|---:|---:|---:|
| Original | 29.38% | 45.87% | 817 |
| Approach 1 | 26.68% | 43.97% | 817 |
| Approach 2 (F2-Lite) | 26.81% | 45.27% | 817 |
| **Approach 3** | **PENDING** | **PENDING** | **PENDING** |

### Primary paired differences

| Comparison | MC1 change and 95% CI | MC2 change and 95% CI | Independent seeds |
|---|---|---|---|
| Approach 3 − original | Pending | Pending | Pending |
| Approach 3 − Approach 1 | Pending | Pending | Pending |
| Approach 3 − F2-Lite | Pending | Pending | Pending |

### Prespecified context-dependent group, if recovered and frozen

| Question group | Question IDs/revision frozen before training? | MC1 delta | MC2 delta / CI |
|---|---|---|---|
| 66 previously identified indexical/subjective questions | Pending | Pending | Pending |
| Remaining 751 questions | Pending | Pending | Pending |

## 8. After Approach 3 finishes

1. Run read-only artifact integrity and duplicate-ID checks.
2. Independently reproduce JSONL means and paired differences with the same analysis script and declared RNG behavior.
3. Append a new entry to [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) with actual job IDs, dataset/model revisions, resolved hyperparameters and checkpoint hashes.
4. Append a separate Approach 3 methods section to [`METHODOLOGY.md`](METHODOLOGY.md); preserve historical settings for Approaches 1 and 2 unchanged.
5. Add Approach 3 rows to [`RESULTS.md`](RESULTS.md), separating pre-specified from exploratory analyses and including every intended comparator.
6. Update the README summary and the machine-readable aggregate results table.
7. Note every deviation from this prospective record. Do not retroactively describe any decision made after inspecting Approach 3 results as preregistered.

**No Approach 3 GPU job, dataset transformation, adapter or benchmark score existed in the audit as of November 4, 2026.**
