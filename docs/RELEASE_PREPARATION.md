# Release preparation — November 4, 2026

This is an isolated proposed source release. Original research files and historical outputs were not edited. The original `.git` directory is empty and read-only; Git reports no repository, so no local history, branch, or remote was available to audit. A new repository is initialized only in this release copy, on `release/research-docs-20261104`. No existing history is overwritten.

The destination is `https://github.com/Crient/truthfulness-lora.git`; the supplied screenshot identifies it as private. Live visibility/access is not independently verified. No commit or push is authorized yet.

## Changes from original source

- Two Python scripts and six Slurm launchers retain the experimental logic; private project roots are replaced with working-directory/environment configuration.
- Five notebooks retain cell ordering (training runners execute specific cell indices). Outputs, execution counts, cell metadata, and nonessential notebook metadata are cleared. A duplicated notebook link is fixed. Notebook kernels must start in the project root or use TRUTHFULNESS_PROJECT.
- The October report is retained as historical documentation; current documentation qualifies its seed and statistical claims. PDF/DOCX reports, raw audit exports, private logs, datasets, adapters, caches, checkpoints and per-item evaluation records are excluded.
- Slurm directives are historical examples, not verified current resource recommendations. Check Unity configuration before any future job. Older setup/data/evaluation notebook cells can download data or initiate computation when manually executed; no notebooks were executed during preparation.
- No license is invented: release licensing and faculty/publication review remain open in the publication checklist.

## Documentation archive comparison

- README.md: no existing target
- analysis/verify_truthfulqa_samples.py: no existing target
- docs/APPROACH_3.md: no existing target
- docs/AUDIT_AND_LIMITATIONS.md: no existing target
- docs/EXPERIMENT_LOG.md: no existing target
- docs/METHODOLOGY.md: no existing target
- docs/PUBLICATION_CHECKLIST.md: no existing target
- docs/REPRODUCIBILITY.md: no existing target
- docs/RESULTS.md: no existing target
- results/aggregate_metrics.csv: no existing target

## Validation performed

The CPU-only verifier passed against the original private saved results with 10,000 resamples and seed 42: 817 questions per task/run, equal aggregate/sample means, and matching paired documents. Original/A1/F2 MC1 percentages are 29.38/26.68/26.81 and MC2 percentages are 45.87/43.97/45.27. No inference, training, notebook execution, or Slurm submission occurred.

All 25 local Markdown/notebook links resolve. Python scripts parse and six shell launchers pass bash syntax checks. The proposed text files were scanned for common HF/GitHub/AWS credential formats, private-key headers, literal credential assignments, and private user paths; no matches were found. This pattern scan is not proof that arbitrary secrets are absent. Notebook outputs are empty. External URLs were inventoried, but live HTTP checks failed with network URL errors; remote visibility/access was not tested.

The complete initial-commit proposal, exact allowlist, file hashes and verification output are local ignored artifacts in release_review/. Git's index remains empty. No commit or push has occurred. Approve the exact allowlist before staging, committing, or pushing.
