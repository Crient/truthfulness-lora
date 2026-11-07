# Environment, setup and reproduction boundary

> **Local integration verification — November 7, 2026.** A fresh Python 3.12 environment was used for the minimal editable installation and synthetic CPU checks. A wheel-packaging gap was found: model verification expected a checkout-only configuration file. The integrated package now includes the unchanged input-identity JSON as package data. See [the integration review](../release/INTEGRATION_REVIEW.md) for wheel checks and exact test scope. Full clean CUDA-environment installation, original-input replay and GPU parity remain unverified. Preparation-time statements below describe the original candidate.

## Three distinct levels

1. Inspect/reconcile exported aggregate results and run lightweight synthetic tests: possible from this archive on Python3.12 with NumPy.
2. Reconstruct data staging/splits and inspect tensor/checkpoint behavior: needs application dependencies and separately obtained inputs; CPU compute on a scheduler or owned workstation.
3. Reproduce full training and four-arm scores: requires source data, base snapshot, exact historical adapters or newly trained replacements, compatible CUDA hardware and independent authorization/resources. This candidate has **not** completed level3 after adapting paths. New runs need not match historical bytes/scores.

## Minimal inspection

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m truthfulness_lora --help
python -m unittest discover -s tests -p test_public.py -v
```

CPU tensor tests additionally need torch/transformers/peft and the related pinned packages; then use `python -m unittest discover -s tests -v`. The optional real-template test needs a local pinned tokenizer via TL_TEST_TOKENIZER and otherwise reports a skip. Tests use invented examples and a tiny randomly initialized model, not pretrained weights or benchmark answer records.

## Recorded training environment

Python3.12.3, torch2.9.1+cu126, transformers4.57.6, peft0.18.1, tokenizers0.22.2, NumPy2.2.6 and supporting observed versions are pinned in [requirements-training.txt](../requirements-training.txt). It is not a full transitive/driver lock. The original DPO used a Quadro RTX8000, FP32/eager, disabled TF32 and19.852GiB peak allocated memory. Original SFT/F2 used A16 FP16; full replay needs compatible hardware. A Mac is suitable for inspection/lightweight tests, not this recorded CUDA training recipe.

For a separately provisioned Linux CUDA12.6 environment, a possible installation sequence is:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-training.txt
python -m pip install -e .
```

The CLI sets CUBLAS_WORKSPACE_CONFIG=:4096:8 before GPU initialization, as in the recorded deterministic execution. Set private cache/temp locations under your allocated research workspace when running workloads; do not use home storage for high-volume I/O.

This installation was not performed or clean-environment-tested during packaging. Confirm driver/platform compatibility and availability. No adapter or third-party data is distributed. Resolve source licenses and permissions before fetching or using inputs.

## Input acquisition and preparation

The commands below are future workload instructions, not work run by this packaging task. On an HPC cluster, execute them inside appropriately sized Slurm jobs. Inspect your current partitions/accounts/QOS/resources rather than copying another site's settings. stage-inputs downloads several GB; prepare-dpo scans the60,917-pair source with lexical/TF-IDF grouping and may require several GB of RAM. Use active research storage, never a login node for substantial I/O. On a dedicated non-cluster workstation, explicitly add --allow-local-compute. Do not use that flag to bypass cluster rules.

From the project root, choose new paths that do not yet exist:

```bash
python -m truthfulness_lora stage-inputs --output data/staged
python -m truthfulness_lora prepare-halueval --source data/staged/halueval --output data/halueval
python -m truthfulness_lora freeze-benchmark --source data/staged/benchmark --output data/benchmark
python -m truthfulness_lora prepare-dpo --model data/staged/model --source-parquet data/staged/ultrafeedback/data/train-00000-of-00001.parquet --benchmark data/benchmark/truthfulqa_817_frozen.jsonl --output data/dpo --acknowledge-exploratory-limitations
```

The first command is explicitly network-enabled input staging. Later model calls are local-only. Revisions are pinned in code and [input_identities.json](../configs/input_identities.json). The benchmark freezer refuses a hash differing from the recorded817 JSONL. HaluEval preserves the historical shuffled seed42 split; the DPO preprocessor preserves source strings/selection algorithms while omitting private reviewer packets. Rebuilt metadata manifests differ; do not claim their hashes equal the historical manifest. Inputs/data/ are ignored by Git.

## Fresh SFT and F2-Lite runs

Only within a suitable GPU allocation, after future resource/work authorization:

```bash
python -m truthfulness_lora train-supervised --recipe sft --model data/staged/model --split data/halueval/split --output outputs/sft-new
python -m truthfulness_lora train-supervised --recipe f2-lite --model data/staged/model --split data/halueval/split --output outputs/f2-new
```

SFT adapter: outputs/sft-new/adapters/sft. F2 adapter: outputs/f2-new/adapters/f2_lite_seed42_full. Best validation checkpoints are reloaded before export. Refuse existing output roots. Source model bytes are hash-checked before training; new adapter hashes may differ from the recorded ones due to platform/execution variation.

## Fresh exploratory DPO and matched evaluation

```bash
python -m truthfulness_lora train-dpo --model data/staged/model --sft outputs/sft-new/adapters/sft --prepared data/dpo --output outputs/dpo-new --acknowledge-exploratory-limitations
python -m truthfulness_lora evaluate --model data/staged/model --sft outputs/sft-new/adapters/sft --f2 outputs/f2-new/adapters/f2_lite_seed42_full --dpo-run outputs/dpo-new --prepared data/dpo --benchmark data/benchmark/truthfulqa_817_frozen.jsonl --output outputs/eval-new
```

Evaluation requires explicit SUCCESS and the valid selected immutable DPO checkpoint, not the stopping/latest checkpoint. Each path is explicit; CLI output claims reject existing/reused/input-overlapping paths. DPO inputs/code are bound at run creation and rechecked. Dataset manifests/selected checkpoint manifests are also checked. No automatic scheduler submit/requeue, continuation or implicit resume is performed. Atomic recovery primitives remain available for research review and CPU tests; CLI continuation must be designed/reviewed separately rather than pointed at a historical run. Do not load untrusted pickle checkpoints.

The evaluator writes raw item/choice outputs under ignored outputs/eval-new/evaluation; keep those private. If those private records are available, a separately scheduled CPU call can aggregate all six pairs:

```bash
python -m truthfulness_lora summarize --records outputs/eval-new/evaluation --output outputs/summary-new
```

The matched evaluator's saved table covers the original three DPO comparisons; summarize returns all six using the same shared item resamples. This archive supplies only aggregate historical data, so a reader cannot reconstruct the817 item bootstrap draws or independently inspect logits without additional private inputs.

Use ordinary Python, without -O: retained research integrity checks use assertions. These source-layout instructions use editable installation (-e). Minimal editable and non-editable wheel inspection were tested locally during branch integration; a full clean research/CUDA environment and model replay remain unverified. Wheels and source distributions are code-installation artifacts; use the Git checkout for the complete research documentation and aggregate evidence.

## Known gaps

Original SFT/F2 unpinned cache resolution; incomplete upstream preference binarization randomness; no independently factual-verified labels; no full dependency/driver/OS lock; one seed; no new GPU parity or full clean research-environment replay; unmeasured semantic/pretraining exposure; portable wrappers rather than original execution-time scripts. Input hashing establishes bytes, not factual validity or rights. Exported source hashes and adaptation notes are in [SOURCE_PROVENANCE.json](../release/SOURCE_PROVENANCE.json), with preparation-time verification scope in [QC_REPORT.md](../release/QC_REPORT.md) and local checks in [INTEGRATION_REVIEW.md](../release/INTEGRATION_REVIEW.md).
