> **Historical snapshot — November 4, 2026.** Original README narrative retained below; local links adjusted for this archival location. The completed four-model study is documented in [the current README](../README.md) and [matched results](RESULTS.md). Historical six-demonstration scores must not be combined with native-chat scores.

# Truthfulness-Oriented LoRA Fine-Tuning of Qwen3-1.7B

**An experimental study of whether HaluEval-supervised adaptation improves out-of-domain TruthfulQA performance**

> **Research snapshot: November 4, 2026 · Approaches 1–2 completed · Approach 3 planned · Public release pending faculty review**

**Researcher:** Gechleng (Leng) Lim, undergraduate AI Research Assistant  
**Faculty advisor:** Professor Salem Othman  
**Institution:** Wentworth Institute of Technology  
**Compute:** UMass Unity HPC, NVIDIA GPU allocations through SLURM  
**Model:** [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B)  
**Training data:** HaluEval question answering (10,000 source items)  
**Evaluation:** TruthfulQA MC1 and MC2 (817 questions per complete run)

## Executive summary

We compared three checkpoints built around the same Qwen3-1.7B base model: the original, unadapted model; a conventional question-to-correct-answer LoRA adapter; and **F2-Lite**, a lightweight, *Faithful Finetuning (F2)-inspired* three-objective LoRA adapter. We trained on HaluEval QA and evaluated transfer to TruthfulQA multiple-choice answer rankings.

| Evaluated system | TruthfulQA MC1 ↑ | TruthfulQA MC2 ↑ | Difference in MC2 versus original |
|---|---:|---:|---:|
| Original Qwen3-1.7B | **29.38%** | **45.87%** | — |
| Approach 1 — standard LoRA SFT | 26.68% | 43.97% | −1.90 percentage points |
| Approach 2 — F2-Lite multi-objective SFT | 26.81% | 45.27% | −0.60 percentage points |
| Approach 3 — to be defined | — | — | Not run |

**Primary result:** Neither trained adapter improved overall TruthfulQA MC1 or MC2 compared with the original model. F2-Lite had a **+1.30 percentage-point MC2 advantage over Approach 1** on these fixed checkpoints and 817 questions. A new independent 10,000-resample paired bootstrap gives a *nominal* 95% interval of **[+0.05, +2.51] percentage points** for that comparison. It does not establish cross-seed robustness or an improvement versus the untouched base model.

**Exploratory research lead:** Additional analysis conducted on Unity, reported separately from the export, grouped 66 indexical/subjective TruthfulQA questions. On that post-hoc subset, F2-Lite reportedly exceeded the original model in MC2 by **+14.09 points**, with a nominal paired-bootstrap 95% interval of **[+3.83, +24.23] points**. The exported audit lacks the category-matching script and metadata required to independently reproduce this subgroup result. Treat it as **provisional, hypothesis-generating evidence**, not a validated conclusion.

**Interpretation:** MC1 measures whether the preferred correct answer wins a likelihood ranking. MC2 measures normalized likelihood mass across truthful answer *choices*. Neither number directly measures the correctness of an answer generated freely by the model.

## Research questions

1. Does ordinary supervised LoRA fine-tuning on HaluEval `right_answer` improve cross-benchmark truthfulness on TruthfulQA?
2. Does adding factual-background retrieval and knowledge-grounded QA supervision change that transfer?
3. How do changes in answer-choice probability mass (MC2) relate to changes in top-choice decisions (MC1)?
4. Are effects concentrated among particular question types, or associated with answer length?
5. Which design controls are needed before making stronger causal claims in Approach 3?

## Study at a glance

| Element | Approach 1 | Approach 2 (F2-Lite) |
|---|---|---|
| Starting checkpoint | `Qwen/Qwen3-1.7B` | Same base model identifier |
| Source data | HaluEval QA | Same saved HaluEval split |
| Source train / validation items | 9,500 / 500 | 9,500 / 500 |
| Supervised records | 9,500 / 500 | 28,500 / 1,500 (3 records per source item) |
| Targets | `question → right_answer` | Direct QA; background retrieval; knowledge-grounded QA |
| LoRA settings | rank 16, alpha 32, dropout 0.05, all-linear | Same |
| Learning rate | 2×10⁻⁴ | 2×10⁻⁵ |
| Maximum sequence length | 512 | 1,024 |
| Completed optimizer steps | 2,375 | 7,125 |
| Selected checkpoint | Step 2,000 | Step 6,000 |
| Training runtime reported by Trainer | 3,060.9 s (~51 min) | 10,404.0 s (~2 h 53 min) |

**Design caveat:** F2-Lite changes the **objective mix, learning rate, context length, and training-update budget at the same time**. Its results are evidence about this complete recipe. They cannot isolate the causal effect of three-objective training or reproduce the full published F2 method.

## Experimental development

1. **Environment and data preparation (October 2026).** Connected VS Code to Unity HPC; prepared a seed-42 split of HaluEval QA with 9,500 training and 500 validation source items; used locally cached Qwen model/tokenizer assets.
2. **Approach 1 (October 27).** Trained a rank-16 LoRA adapter on the correct-answer target and ran a 20-question evaluation smoke test, followed by full original-model and LoRA TruthfulQA evaluations. The trained model underperformed the baseline on both aggregate metrics.
3. **F2-Lite design (early November).** Reused the split and the original training notebook's tokenization/model helpers. Constructed three supervised tasks per source item, then applied preflight/smoke/pilot/full modes.
4. **Approach 2 (November 4).** Completed 7,125 steps and full 817-question evaluation. MC2 partially recovered relative to Approach 1, while overall baseline MC2 remained higher.
5. **Error analysis (November 4).** Inspected answer-length quartiles, MC1 question transitions, and TruthfulQA categories. No simple monotonic answer-length explanation emerged. The 66-item context-dependent subgroup became a candidate follow-up hypothesis.
6. **Approach 3 (future).** Method and results remain undecided. See [`docs/APPROACH_3.md`](APPROACH_3.md) for a prospective experiment protocol.

## What is and is not supported

**Directly supported by this audit:** saved aggregate JSON and sample JSONL scores; question-ID and content pairing; original notebook and F2 training code; dataset split procedure; checkpoint trainer states; SLURM script requests and terminal log markers; training losses/runtimes; evaluation task configurations; recomputed full-set paired confidence intervals.

**Not independently established by the export:** the post-hoc category metadata join and subgroup bootstrap implementation; historical exact training-environment lock; original training model/tokenizer cache SHA; adapter weights and their identity; SLURM accounting exit codes or actual training GPU model; dataset/benchmark overlap testing; repeated training seeds; free-generation hallucination outcomes.

The complete score evidence demonstrates a **negative overall result**, plus a narrower exploratory observation worth testing in an independently planned experiment.

## Read the research record

| Document | What it contains |
|---|---|
| [`docs/METHODOLOGY.md`](METHODOLOGY.md) | Data construction, LoRA settings, exact training objectives, checkpoint selection and evaluation scoring |
| [`docs/RESULTS.md`](HISTORICAL_RESULTS_20261104.md) | Full scores, reproduced paired bootstraps, pilot/smoke context, answer-length and subgroup interpretation |
| [`docs/EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) | Dated chronology, SLURM job IDs, training artifacts, status and source-of-evidence ledger |
| [`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Environment, CLI protocol, validation steps, rerun restrictions and known reproduction gaps |
| [`docs/APPROACH_3.md`](APPROACH_3.md) | Pre-run protocol, hypotheses/ablation controls, and blank future-result tables |
| [`docs/AUDIT_AND_LIMITATIONS.md`](AUDIT_AND_LIMITATIONS.md) | Claims ledger, inconsistencies corrected during review and research threats to validity |
| [`docs/PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md) | Responsible open-source release checklist and reproducibility priorities |
| [`results/aggregate_metrics.csv`](../results/aggregate_metrics.csv) | Machine-readable complete-run and pilot score summary |
| [`analysis/verify_truthfulqa_samples.py`](../analysis/verify_truthfulqa_samples.py) | Independent CPU-only verification of saved per-question results and paired bootstrap intervals |

## Research code and evidence

This documentation bundle is intended to accompany the **existing** Unity `truthfulness-lora` research repository. The project already contains:

```text
truthfulness-lora/
├── notebooks/                  # Setup, HaluEval split, LoRA training, evaluation comparisons
├── src/train_f2_lite.py       # Three-objective SFT construction; pilot/full runner
├── src/compare_f2_lite.py     # Original F2-versus-baseline paired bootstrap
├── run_full_lora.sbatch       # Approach 1 Slurm launcher
├── run_truthfulqa.sbatch      # Baseline and Approach 1 benchmark launcher
├── run_f2_lite_*.sbatch       # F2 pilot/full training and benchmark launchers
├── data/                       # HaluEval saved split; not in this documentation bundle
├── adapters/                   # Trained adapters; do not commit as regular source files
├── results/                    # Evaluations and checkpoints; review each before public release
└── docs/                       # This documentation set, once added
```

**Current reproducibility status: partial.** The historical code used Unity-specific absolute paths; release copies use the project working directory or `TRUTHFULNESS_PROJECT`. The code and F2-Lite executes source from particular cells of an existing notebook. Cached models, adapters, data rows, and exact historical dependency locks were deliberately excluded from the exported audit. The [`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md) file distinguishes reproducing scores from rerunning the training.

### Recheck the saved scores without training

After placing this documentation bundle in the research repo, run on a machine with Python and NumPy:

```bash
python analysis/verify_truthfulqa_samples.py --root . --bootstrap 10000 --seed 42
```

This checks saved results already present in `results/`. It **does not** download data, load model weights, start GPU jobs, modify results, or rerun the models. The large sample JSONL files are not included in the GitHub documentation bundle.

## Future research and Approach 3

Recommended scientific priority: freeze the next hypothesis **before** using the observed TruthfulQA subgroup to choose design details; control learning rate, sequence length, supervised-token exposure, and update budget; use explicit model initialization seeds; run multiple training seeds; and add an independent benchmark or generated-answer assessment. See [`docs/APPROACH_3.md`](APPROACH_3.md).

## References

1. [Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B), Qwen team.
2. Li et al., [*HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models*](https://aclanthology.org/2023.emnlp-main.397/), EMNLP 2023.
3. Lin, Hilton & Evans, [*TruthfulQA: Measuring How Models Mimic Human Falsehoods*](https://aclanthology.org/2022.acl-long.229/), ACL 2022.
4. Hu et al., [*Mitigating Large Language Model Hallucination with Faithful Finetuning*](https://arxiv.org/abs/2406.11267), 2024. **F2-Lite is an adaptation inspired by this work, not an implementation of the full method.**
5. [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), including its TruthfulQA task definitions.

---

**Publication status:** Draft for research-team review. Do not push the audit ZIP, raw per-question benchmark records, checkpoints, model weights, cached data, credentials, or shared Unity directories to a public repository without reviewing permissions and licenses. See [`docs/PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md).

*Last substantive audit: November 4, 2026. Historical experiment outputs are preserved; Approach 3 has not started.*

## Release copy

This isolated release preserves experimental source and cell ordering, clears notebook outputs and execution counts, and replaces private project paths. Run scripts from the project root; for notebook kernels started elsewhere, set `TRUTHFULNESS_PROJECT` to your clone root. Slurm launchers use that variable or `SLURM_SUBMIT_DIR`. Original research artifacts remain outside this copy. See [release preparation notes](RELEASE_PREPARATION.md).
