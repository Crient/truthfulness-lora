# Private release-candidate manifest

Candidate: truthfulness-lora-20261107-rc1. **50 files**, excluding directory entries. Public release remains blocked; the user is inspecting a sanitized private candidate. Source is the complete original research project, not only the25-file private review ZIP. No Git metadata or third-party data/model records are included.

[Machine manifest](CANDIDATE_MANIFEST.json), [source hashes/adaptations](SOURCE_PROVENANCE.json), [exclusions](EXCLUDED_FILES.md), [QC](QC_REPORT.md), [blockers](BLOCKERS.md).

| Relative path | Purpose |
|---|---|
| `.gitignore` | Project setup, inspection entry point or checksum metadata |
| `README.md` | Project setup, inspection entry point or checksum metadata |
| `SHA256SUMS` | Project setup, inspection entry point or checksum metadata |
| `configs/dpo.json` | Observed configuration or safe pinned input hashes |
| `configs/f2_lite.json` | Observed configuration or safe pinned input hashes |
| `configs/input_identities.json` | Observed configuration or safe pinned input hashes |
| `configs/sft.json` | Observed configuration or safe pinned input hashes |
| `docs/APPROACHES.md` | Sanitized scientific/reproduction/rights narrative |
| `docs/DATA_AUDIT.md` | Sanitized scientific/reproduction/rights narrative |
| `docs/LIMITATIONS.md` | Sanitized scientific/reproduction/rights narrative |
| `docs/REPRODUCTION.md` | Sanitized scientific/reproduction/rights narrative |
| `docs/RESEARCH_REPORT.md` | Sanitized scientific/reproduction/rights narrative |
| `docs/RESULTS.md` | Sanitized scientific/reproduction/rights narrative |
| `docs/RIGHTS_AND_NOTICES.md` | Sanitized scientific/reproduction/rights narrative |
| `pyproject.toml` | Project setup, inspection entry point or checksum metadata |
| `release/BLOCKERS.md` | Candidate scope, blockers, provenance or QC metadata |
| `release/CANDIDATE_MANIFEST.json` | Candidate scope, blockers, provenance or QC metadata |
| `release/CANDIDATE_MANIFEST.md` | Candidate scope, blockers, provenance or QC metadata |
| `release/EXCLUDED_FILES.json` | Candidate scope, blockers, provenance or QC metadata |
| `release/EXCLUDED_FILES.md` | Candidate scope, blockers, provenance or QC metadata |
| `release/LICENSE_EVIDENCE.json` | Candidate scope, blockers, provenance or QC metadata |
| `release/QC_REPORT.json` | Candidate scope, blockers, provenance or QC metadata |
| `release/QC_REPORT.md` | Candidate scope, blockers, provenance or QC metadata |
| `release/SOURCE_PROVENANCE.json` | Candidate scope, blockers, provenance or QC metadata |
| `requirements-training.txt` | Project setup, inspection entry point or checksum metadata |
| `results/exploratory_diagnostics.json` | Sanitized aggregate historical evidence; no item text |
| `results/feature_bins.csv` | Sanitized aggregate historical evidence; no item text |
| `results/feature_correlations.csv` | Sanitized aggregate historical evidence; no item text |
| `results/matched_results.json` | Sanitized aggregate historical evidence; no item text |
| `results/mc1_mc2_decomposition.csv` | Sanitized aggregate historical evidence; no item text |
| `results/mc1_transitions.csv` | Sanitized aggregate historical evidence; no item text |
| `results/model_identities.json` | Sanitized aggregate historical evidence; no item text |
| `results/scores.csv` | Sanitized aggregate historical evidence; no item text |
| `results/training_summary.json` | Sanitized aggregate historical evidence; no item text |
| `src/truthfulness_lora/__init__.py` | Portable code, extracted/adapted original algorithms or explicit runtime wrapper |
| `src/truthfulness_lora/__main__.py` | Portable code, extracted/adapted original algorithms or explicit runtime wrapper |
| `src/truthfulness_lora/cli.py` | Portable code, extracted/adapted original algorithms or explicit runtime wrapper |
| `src/truthfulness_lora/dpo/__init__.py` | Portable code, extracted/adapted original algorithms or explicit runtime wrapper |
| `src/truthfulness_lora/dpo/common.py` | Portable code, extracted/adapted original algorithms or explicit runtime wrapper |
| `src/truthfulness_lora/dpo/core.py` | Portable code, extracted/adapted original algorithms or explicit runtime wrapper |
| `src/truthfulness_lora/dpo/dataset.py` | Portable code, extracted/adapted original algorithms or explicit runtime wrapper |
| `src/truthfulness_lora/dpo/evaluate.py` | Portable code, extracted/adapted original algorithms or explicit runtime wrapper |
| `src/truthfulness_lora/dpo/train.py` | Portable code, extracted/adapted original algorithms or explicit runtime wrapper |
| `src/truthfulness_lora/f2_lite.py` | Portable code, extracted/adapted original algorithms or explicit runtime wrapper |
| `src/truthfulness_lora/metrics.py` | Portable code, extracted/adapted original algorithms or explicit runtime wrapper |
| `src/truthfulness_lora/prepare.py` | Portable code, extracted/adapted original algorithms or explicit runtime wrapper |
| `src/truthfulness_lora/runtime.py` | Portable code, extracted/adapted original algorithms or explicit runtime wrapper |
| `src/truthfulness_lora/supervised.py` | Portable code, extracted/adapted original algorithms or explicit runtime wrapper |
| `tests/test_core.py` | Synthetic CPU verification; no third-party answer records |
| `tests/test_public.py` | Synthetic CPU verification; no third-party answer records |

## Integrity and integration

SHA256SUMS covers every candidate file except itself. The transfer ZIP has a separate external SHA256 sidecar. Manifest self-size/hash fields are intentionally null to avoid circularity. No raw source/private absolute path or operational identifier is needed to verify candidate hashes.

The archive extracts under one truthfulness-lora/ directory. Inspect beside the existing private checkout, reconcile project ownership/license and review file diffs before integrating on a review branch. This package does not authorize a push or public visibility change and does not overwrite the original research archive. Existing original code/notebooks/reports/checkpoints remain preserved; this candidate explicitly marks adapted code and sanitized narrative as derivatives.
