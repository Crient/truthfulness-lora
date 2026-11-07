# Truthfulness LoRA: four matched Qwen3-1.7B recipes

**Completed exploratory four-model comparison · Private review branch · Public release blocked.** Integrated from candidate `20261107` into the existing private [Crient/truthfulness-lora](https://github.com/Crient/truthfulness-lora) repository. No blanket project license or faculty approval is asserted.

**Researcher:** Gechleng (Leng) Lim, undergraduate AI Research Assistant  
**Faculty advisor:** Professor Salem Othman  
**Institution:** Wentworth Institute of Technology  
**Compute:** UMass Unity HPC, NVIDIA GPU allocations through Slurm

These attributions are retained from the existing research record. They do not imply publication authorization or institutional endorsement.

This research compares the unmodified released `Qwen/Qwen3-1.7B` checkpoint (Base), HaluEval supervised LoRA (SFT), HaluEval F2-Lite, and SFT-initialized exploratory UltraFeedback DPO. Base here is not the separately named `Qwen3-1.7B-Base` model. The experiment is complete; this candidate adapts its code for review and future fresh runs. The portable wrappers have not been validated by rerunning full training or GPU inference.

## Final matched results

All arms use the same 817 frozen TruthfulQA questions and question-only native-chat, summed EOS-inclusive answer likelihood protocol.

| Recipe | MC1 % | MC2 % |
|---|---:|---:|
| Base | 23.623 | 37.561 |
| SFT | 22.399 | 39.233 |
| F2-Lite | 21.053 | 38.176 |
| DPO | 23.501 | 42.050 |


DPO−SFT MC2 is +2.817 percentage points (nominal paired item-bootstrap 95% interval [1.783,3.891]); MC1 is +1.102 pp [−0.490,2.693]. DPO MC1 is one question below Base. MC2 is candidate-normalized true-choice probability mass, **not** the percentage of generated answers that are factual. [Full results](docs/RESULTS.md), [research report](docs/RESEARCH_REPORT.md), [limitations](docs/LIMITATIONS.md).

The recipes use different data and settings. UltraFeedback provides general GPT-4 preferences, not independent factual labels. Original quality and shortcut gates failed: 28/48 preferences supported (requirement ≥80%), 8/48 chosen unusable (maximum 10%), answer-only estimates 66.01%/74.69% (maximum 65%). One seed was used; bootstrap intervals omit training-seed variability and multiplicity adjustment. Higher MC2 does not establish free-generation factual accuracy or DPO-specific factual learning.

## What is included

- Portable Python package for input staging/preprocessing, extracted SFT/F2-Lite logic, exploratory DPO and matched four-arm evaluation.
- Final aggregate scores, paired intervals, model/data hashes and post-hoc diagnostic tables.
- CPU tests for aggregation, output safety, masks, objective and tiny synthetic checkpoint recovery.
- [Methods and approaches](docs/APPROACHES.md), [data audit](docs/DATA_AUDIT.md), [setup/reproduction](docs/REPRODUCTION.md), [rights/notices](docs/RIGHTS_AND_NOTICES.md), [release blockers](release/BLOCKERS.md).

Raw datasets, third-party responses, benchmark answer records, per-question outputs, reviewer packets, pretrained weights, checkpoints, environments and logs are excluded. [Candidate manifest](release/CANDIDATE_MANIFEST.md) and [exclusions](release/EXCLUDED_FILES.md) define the transfer scope. Numeric results are historical evidence; new code is a documented adaptation.

## Quick inspection on your Mac

From the repository root with Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m truthfulness_lora --help
python -m unittest discover -s tests -p test_public.py -v
shasum -a 256 -c SHA256SUMS
```

These commands do not train or download model assets. The package itself is installed locally; NumPy may be downloaded during dependency installation. Full GPU/CPU research dependencies are optional and described in [REPRODUCTION.md](docs/REPRODUCTION.md). No command runs training by default; each workload needs an explicit subcommand, inputs and a new output path. Cluster workloads require a Slurm allocation; `--allow-local-compute` is only for a dedicated non-cluster workstation.

## Repository layout and historical record

```text
configs/                         # Recorded settings and input hashes
src/truthfulness_lora/           # Portable, adapted research package
src/train_f2_lite.py             # Preserved historical runner
src/compare_f2_lite.py           # Preserved historical analysis
notebooks/                      # Five preserved historical notebooks
run_*.sbatch                    # Six preserved original Slurm launchers
docs/                           # Current methods/report plus dated history
results/                        # Explicitly allowlisted aggregate tables only
analysis/                       # Preserved historical sample verifier
tests/                          # Synthetic CPU tests
release/                        # Candidate provenance and integration audit
```

The November 4 six-demonstration harness evaluation and the final question-only native-chat evaluation are separate protocols. Their scores must not be pooled or used as a before/after training effect. Approach 3 is now exploratory SFT-initialized UltraFeedback DPO; the earlier prospective protocol is historical and does not establish preregistration of the completed DPO experiment.

- [Historical November 4 README](docs/HISTORICAL_README_20261104.md), [results](docs/HISTORICAL_RESULTS_20261104.md) and [October 27 report](docs/HISTORICAL_REPORT_20261027.md).
- Historical [methodology](docs/METHODOLOGY.md), [experiment log](docs/EXPERIMENT_LOG.md), [audit](docs/AUDIT_AND_LIMITATIONS.md), [reproduction notes](docs/REPRODUCIBILITY.md) and [Approach 3 proposal](docs/APPROACH_3.md), each labeled with its original scope.
- The original [aggregate metrics](results/aggregate_metrics.csv), notebooks, analysis scripts and Slurm launchers are preserved. Unity remains the original research archive; this checkout preserves the existing Git history and adds the reviewed portable candidate.

## Integration and publication status

Work is confined to `research/final-release-review`; `main` remains at its original commit. The private remote and its base branch were checked before integration. See the [integration review](release/INTEGRATION_REVIEW.md) for changed-file decisions, preservation checks, local test results and remaining gaps.

The original [candidate manifest](release/CANDIDATE_MANIFEST.md), [candidate QC](release/QC_REPORT.md) and [candidate checksum list](release/CANDIDATE_SHA256SUMS) describe the received ZIP, including preparation-time limitations; they are retained as provenance. The root `SHA256SUMS` covers the integrated review tree and is checked from the repository root. The original candidate checksum list is checked only against an untouched extraction of that ZIP.

Public publication remains blocked by rights/licensing, applicable faculty/institutional authorization, clean-install/full GPU replay gaps and scientific claim limits. See [release blockers](release/BLOCKERS.md) and [rights/notices](docs/RIGHTS_AND_NOTICES.md). A private branch push is authorized for review; no merge, public release or visibility change is authorized. Raw data, answer records, weights, checkpoints, operational logs, private review packets, credentials and transfer ZIPs remain excluded.
