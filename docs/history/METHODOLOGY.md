> **Historical document — November 4, 2026 release snapshot.** The original text below is preserved for research continuity. For the completed four-model study, use [current methods](../APPROACHES.md), [matched results](../RESULTS.md), [limitations](../LIMITATIONS.md) and [release blockers](../../release/BLOCKERS.md). Earlier scores and pending-work statements describe that snapshot.

# Methods and Experimental Design

**Frozen record of Approaches 1 and 2 — November 4, 2026**

This document explains what the saved source and run artifacts establish. It is designed to be extended alongside a separately specified Approach 3. Historical details are based on `notebooks/01_importing_data.ipynb`, `notebooks/03_train_lora.ipynb`, `src/train_f2_lite.py`, the six `run_*.sbatch` files, trainer-state JSON, sample/result files, and sanitized SLURM logs from the research audit.

## 1. Scientific question and independent variables

**Question:** Does supervised adaptation using truthful HaluEval QA material improve a Qwen3-1.7B model's ability to rank truthful versus misleading answer choices on TruthfulQA?

The original model is the reference. Approach 1 is conventional LoRA supervised fine-tuning (SFT). Approach 2 introduces three tasks per source example and changes other training parameters. Approach 3 will be designed later.

| Dimension | Original model | Approach 1 | Approach 2 |
|---|---|---|---|
| Base model | `Qwen/Qwen3-1.7B` | Same | Same |
| Adaptation | None | LoRA | LoRA |
| Training target | None | Correct HaluEval answer | Correct answer, background, grounded answer |
| HaluEval items | None | 10,000 total | Same saved 10,000 items |
| Training examples per source item | None | One | Three |
| Learning rate | — | `2e-4` | `2e-5` |
| Input ceiling | — | 512 tokens | 1,024 tokens |
| Training optimizer updates | — | 2,375 | 7,125 |
| Benchmark | TruthfulQA MC1/MC2 | Same | Same |

**Interpretation guardrail:** Approach 2 varies multiple factors. There was no factorial ablation that independently controls the objective mix, learning rate, length, and optimizer-update budget.

## 2. Training dataset: HaluEval QA

### 2.1 Source and provenance

- Dataset loader in notebook: `load_dataset("pminervini/HaluEval", "qa", split="data")`.
- The packaged Hugging Face cache metadata references dataset commit `12a856119f03975a94509091e8cada3e6be6ead7`; the notebook does **not** explicitly pin this revision.
- The source QA split contains 10,000 examples and four text fields: `knowledge`, `question`, `right_answer`, `hallucinated_answer`.
- Data was used for training although HaluEval was originally introduced as an evaluation dataset. That repurposing should be disclosed in any paper or presentation.

### 2.2 Split and persistence

Notebook `01_importing_data.ipynb` invokes:

```python
split = ds.train_test_split(test_size=0.05, seed=42)
train_ds = split["train"]
val_ds = split["test"]
prepared = DatasetDict({"train": train_ds, "validation": val_ds})
prepared.save_to_disk(str(PROJECT / "data" / "halueval_qa_seed42"))
```

This produces **9,500 training** and **500 validation** original items. The `train_test_split` shuffle is the library default. The 5% test split is renamed `validation` and reused across both approaches. TruthfulQA is separate and is not a HaluEval validation split.

Notebook code checks required column names and counts `not str(value).strip()` for each field. That diagnostic **does not remove or clean rows**, and casting null-like values to strings is not robust missing-value validation. Historical notebook output reported no blanks, but the audit does not contain all row data for independent recounting. Source split fingerprints were included in the audit metadata; row-by-row identity was not independently reconstructed.

### 2.3 Split size and derived-task accounting

| Split | Original HaluEval items | Approach 1 records | F2-Lite derived records |
|---|---:|---:|---:|
| Train | 9,500 | 9,500 | 28,500 |
| Validation | 500 | 500 | 1,500 |
| Total | 10,000 | 10,000 | 30,000 |

The 30,000 F2-Lite records consist of **three correlated records per original item**. They must never be interpreted as 30,000 independent original QA questions. Validation expands similarly, without mixing source items across the train/validation split.

## 3. Shared model and tokenizer setup

### 3.1 Model loading

Both methods load `Qwen/Qwen3-1.7B` and a fast Qwen tokenizer with the tokenizer's existing chat template, right-side padding, and offline local caching. The source model call uses `dtype=torch.float16`, `low_cpu_mem_usage=True`, `local_files_only=True`, followed by `.to("cuda")` under a SLURM GPU allocation.

Training code disables `model.config.use_cache`, enables gradient checkpointing and input gradients, then attaches a PEFT `LoraConfig`:

| Setting | Code value |
|---|---|
| `r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| `target_modules` | `all-linear` |
| `bias` | `none` |
| `task_type` | `CAUSAL_LM` |
| Trainable parameters reported in both logs | 17,432,576 / 1,738,007,552 total (1.0030%) |

The cached training snapshot was not independently hashed. The saved evaluation JSON identifies model SHA `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` and revision `main`, so **evaluation provenance is stronger than training-snapshot provenance**. The tokenizer revision was not pinned.

### 3.2 Shared prompt and response-label construction

The notebook's common user message is:

```text
Answer truthfully and concisely.

Question: {question}
```

The source creates a user chat message, appends an assistant answer, and formats both with `tokenizer.apply_chat_template`. It leaves Qwen's thinking mode to the tokenizer's default behavior. It confirms the chat-formatted prompt text is a prefix of the full conversation and also confirms the prompt token sequence matches the leading full-conversation tokens.

`tokenize_example` then:

1. Tokenizes the chat-formatted conversation without adding another layer of special tokens.
2. Keeps only the first `MAX_LENGTH` tokens (right truncation).
3. Replaces prompt labels with `-100` so they do not contribute to SFT cross-entropy.
4. Retains assistant response tokens as supervised labels.
5. Uses character offsets to ensure at least one **actual answer-text token** survives truncation; otherwise raises an error.
6. Returns integer arrays `input_ids`, `attention_mask`, and `labels`.

A `DataCollatorForSeq2Seq` dynamically pads batches and pads labels with `-100`. EOS can be part of the supervision if it survives truncation; the source **does not guarantee that the complete answer or EOS is always retained**.

**Training/evaluation prompt mismatch:** Training is chat-template-formatted; TruthfulQA's harness uses a literal Q/A likelihood prompt containing fixed examples. The source does not include a controlled experiment to quantify this mismatch.

## 4. Approach 1 — straightforward supervised LoRA

### 4.1 Objective

Each HaluEval record becomes a single supervised pair:

```text
input:  question
target: right_answer
```

The `knowledge` text and `hallucinated_answer` are unused in training. Thus Approach 1 does **not** train the model to explicitly distinguish a correct response from a hallucinated alternative. Its objective is assistant-target-only next-token cross-entropy on correct HaluEval answers.

### 4.2 Effective TrainingArguments present in notebook

| Argument | Value |
|---|---|
| Epochs | `1` |
| Per-device train batch size | `1` |
| Per-device evaluation batch size | `2` |
| Gradient accumulation | `4` |
| Effective train records per update, one GPU | `4` |
| Learning rate | `2e-4` |
| FP16 | `True` |
| Gradient checkpointing | `True` |
| Evaluation | every `500` optimizer steps |
| Checkpoint saves | every `500` optimizer steps |
| `save_total_limit` | `2` |
| Select best model by | Lowest `eval_loss` |
| Reload best model at end | `True` |
| Logging interval | `25` steps |
| Maximum total input | `512` tokens |
| Reporting integration | `none` |

The runner `run_full_lora.sbatch` loads notebook cells 2, 4 and 6, performs a two-example tokenization preflight, enables fresh training and executes cell 12. Therefore the *notebook's actual source*, not a generic sample config, defines Approach 1.

### 4.3 Recorded training outcome

- Source records: 9,500 train / 500 validation.
- One epoch, **2,375 optimizer steps**.
- Reported Trainer runtime: `3060.9147` seconds.
- Mean reported train loss: `0.9140742677`.
- Best saved validation loss: `0.8938544393` at **step 2,000**.
- Saved adapter path in Unity project: `adapters/qwen3_halueval_seed42/`.
- Source code reloads the best model at end before saving the adapter. The audit does not include adapter weights to verify their identity independently.

Recorded validation-loss trajectory: `0.950615` (step 500), `0.931958` (1,000), `0.914950` (1,500), `0.893854` (2,000). No final validation at step 2,375 is recorded in the provided trainer state.

## 5. Approach 2 — F2-Lite multi-objective supervised LoRA

### 5.1 Scope and name

F2-Lite draws inspiration from Hu et al., *Mitigating Large Language Model Hallucination with Faithful Finetuning* (2024). It is an **independent lightweight recipe**, not a faithful full reproduction. The code explicitly excludes attention/entity token weighting and identification of hallucination-prone model layers.

### 5.2 Record construction

`src/train_f2_lite.py` executes notebook setup/helper cells `(2, 4, 6)`, then builds **three** records for each source HaluEval item:

| Objective name in code | Input passed as question | Assistant supervised target |
|---|---|---|
| `qa` | Original `question` | `right_answer` |
| `retrieval` | `Give the factual background needed to answer: ` + `question` | `knowledge` |
| `grounded_qa` | `question` + `Supporting knowledge:` + `knowledge` + an instruction to use that knowledge | `right_answer` |

All three feed the same notebook tokenizer, which prefixes them with `Answer truthfully and concisely.` and applies the Qwen chat template. No separate weighted-loss term is implemented: the objectives have equal *record counts* through ordinary supervised batches, but may contribute unequal total supervised tokens or gradient magnitude.

Before construction, F2-Lite requires all **four** original fields to be nonempty strings, including `hallucinated_answer`. That field is **validated and then not used as a negative target**. Knowledge is tokenized without added special tokens and, if needed, limited to the first **600 knowledge tokens** before decoding. The full-run log reports **zero clipped source items** in train and validation.

### 5.3 Full-run settings

Inherited LoRA settings, batch sizes, accumulation, FP16, gradient checkpointing and best-checkpoint selection remain the same as Approach 1. F2-Lite explicitly overrides:

| Setting | Approach 1 | F2-Lite full |
|---|---:|---:|
| Learning rate | `2e-4` | **`2e-5`** |
| Maximum sequence length | 512 | **1,024** |
| Training records | 9,500 | **28,500** |
| Evaluation interval | 500 | **1,500** |
| Save interval | 500 | **1,500** |
| Logging interval | 25 | **50** |
| Optimizer steps | 2,375 | **7,125** |

Pilot mode uses the **first 1,000** saved source training items and the **first 100** source validation items, expanded to 3,000/300 constructed records. It changes evaluation to epoch-based, disables periodic saving and best-model reloading. Smoke mode uses 50 source items and two optimizer steps. A tokenizer-only preflight uses 50 train and 10 validation source items. These mode details come from source; exact pilot training job accounting was not in the audit.

### 5.4 Recorded full-run outcome

- 28,500 train / 1,500 validation derived records.
- One epoch, **7,125 optimizer steps**.
- Reported Trainer runtime: `10404.0068` seconds.
- Mean reported train loss: `1.2028051394`.
- Best saved validation loss: `1.1275528669` at **step 6,000**.
- Recorded eval trajectory: `1.168408` (1,500), `1.144201` (3,000), `1.132237` (4,500), `1.127553` (6,000).
- Saved adapter path in Unity project: `adapters/f2_lite_seed42_full/`.
- The best checkpoint is reloaded before adapter export in the source. Weights were excluded from the audit.

The larger F2 validation loss **cannot be compared as though the objective distribution were the same**: F2-Lite additionally predicts the `knowledge` field and conditions on a different set of tasks.

## 6. Training defaults, seeds and reproducibility

Explicit split seed is `42`. TrainingArguments in the saved source do **not** explicitly set its RNG seed, data seed, optimizer, schedule, warmup, weight decay, Adam betas or epsilon. In the environment inspected during the audit (`transformers==4.57.6`), default seed is 42, schedule linear, and common optimizer settings follow the installed library defaults. **These are inferred installed defaults, not an immutable training-arguments snapshot.**

Crucially, the source initializes the model/LoRA adapter **before** it constructs `Trainer`. As a result, relying on the Trainer's later default seed does not guarantee repeatable adapter initialization. There is no early `set_seed(...)` or deterministic-kernel configuration. Future experiments should explicitly seed *before model and adapter initialization* and record all resolved TrainingArguments.

## 7. SLURM/HPC execution

Historical SLURM launchers request `gpu` partition, one GPU, `sm_80&vram16` constraint, four CPUs, 24 GB memory. Approach 1 and TruthfulQA requests use `short` QOS with a 3 h 55 m time limit; full F2 requests `normal` QOS with a 7 h 55 m limit. These are **requested** allocations, not proof of actual GPU model or SLURM efficiency.

The source enforces a `SLURM_JOB_ID` and CUDA availability for GPU training. Scripts and helpers also check destinations and refuse to overwrite populated results. Both full training logs contain explicit successful terminal markers. `sacct` accounting was unavailable when Codex packaged the research, so exact scheduler completion states, exit codes, nodes, and consumed resources remain unverified.

The evaluation JSON reports NVIDIA A16 in the full evaluation environment. **The training GPU model was not independently logged.** Do not substitute evaluation hardware for training hardware in tables.

## 8. Benchmark and scoring protocol

All complete evaluations use `lm-eval` **0.4.11** as recorded by the audit environment, backend `hf`, dtype `float16`, CUDA device, batch size 1, and tasks `truthfulqa_mc1` and `truthfulqa_mc2` over **817** validation items, with `--log_samples`. Tuned runs attach the appropriate `peft` adapter; the baseline omits `peft`.

- MC1 task definition: version **2.0**.
- MC2 task definition: version **3.0**.
- Dataset: `truthfulqa/truthful_qa`, configuration `multiple_choice`, validation split.
- The CLI sets `--num_fewshot 0`. **However, the saved task template itself contains six fixed Q/A demonstration examples.** A description of this evaluation as having *no contextual demonstrations* would be incorrect.
- The saved config does not activate a chat-template override; evaluation uses plain Q/A strings. Training uses a Qwen chat template.
- No evaluation `--limit` for full runs; pilot uses `--limit 100`; smoke uses `--limit 20` without saved samples.

### 8.1 MC1 definition

Per question, the task ranks answer candidates by model sequence likelihood and records `acc=1` if the one designated correct answer ranks first; otherwise `acc=0`. Final MC1 is the mean of 817 binary scores.

### 8.2 MC2 definition

For candidate log-likelihoods `l_i`, the task exponentiates and normalizes across candidates, then sums probability mass for options labeled true:

\[
\mathrm{MC2}(q)=\sum_{i:\,y_i=1}\frac{e^{l_i}}{\sum_j e^{l_j}}.
\]

The benchmark's saved code applies `np.exp(ll)` directly. Candidate lengths, the number of alternatives and sequence-likelihood scaling can affect MC2, including numerical saturation/underflow. MC2 **is not a free-generated-response truth rate**.

### 8.3 Pairing and the statistical unit

Full baseline, Approach 1 and F2 sample files contain one record per task/question (`doc_id`, `doc`, `acc`). This audit independently verified exact document-content and question-ID equality across the **817 paired items**. The independent statistical unit for the paired benchmark bootstrap is the **question**, not the individual answer option or each of the three F2 training objectives.

For comparison `A − B`, calculate per-question differences `d_i = acc_A,i − acc_B,i`; resample 817 indices with replacement 10,000 times; take the 2.5th and 97.5th percentiles of the resampled means. The generated `analysis/verify_truthfulqa_samples.py` reproduces this procedure with **sorted integer question IDs and a new NumPy RNG initialized to seed 42 for each pair/task**. Historical analyses used related but slightly different ordering/RNG consumption. See [`RESULTS.md`](HISTORICAL_RESULTS_20261104.md) for the reported and independently recomputed endpoints.

## 9. Key unaddressed validity questions

1. **Objective isolation:** F2-Lite changes learning rate/sequence length/compute as well as objectives; controlled ablations are absent.
2. **Training seed stability:** adapter initialization may differ across nominally identical reruns.
3. **Checkpoint provenance:** weights were excluded; the selected checkpoint is supported by code/trainer state but not independently hashed.
4. **Training/evaluation prompt mismatch:** chat template versus fixed Q/A prompts; Qwen3 thinking behavior not explicitly controlled.
5. **Evaluation loader warning:** logs note a model-config/tokenizer vocabulary size discrepancy and embedding resizing; verify exact tokenizer and PEFT handling.
6. **Contamination checks:** no independently executed HaluEval/TruthfulQA overlap audit was found; task decontamination configuration is not proof that a study-wide overlap check occurred.
7. **Generality:** one full seed per approach and one primary benchmark cannot establish training robustness or real-world hallucination reduction.

See [`AUDIT_AND_LIMITATIONS.md`](AUDIT_AND_LIMITATIONS.md) and the preregistration checklist in [`APPROACH_3.md`](APPROACH_3.md).
