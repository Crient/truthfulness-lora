# Repository presentation and publication scope

The README is the portfolio entry point. Current methods, full results and limitations stay in `docs/`; detailed provenance is linked rather than placed ahead of the research question.

## Public-facing structure

```text
truthfulness-lora/
├── README.md
├── CITATION.cff
├── pyproject.toml
├── MANIFEST.in                 # Explicit aggregate figure inputs in sdists
├── SHA256SUMS
├── requirements-training.txt
├── requirements-verification.txt
├── src/
│   ├── truthfulness_lora/      # Portable training/evaluation package
│   ├── train_f2_lite.py        # Preserved historical runner
│   └── compare_f2_lite.py      # Preserved historical comparison
├── configs/                   # Recorded settings and input hashes
├── results/
│   ├── figures/               # Aggregate comparison PNG/SVG + provenance
│   └── ...                    # Explicitly allowlisted aggregate tables
├── notebooks/                 # Five historical notebooks; outputs cleared
├── analysis/                  # Figure generator and historical verifier
├── scripts/slurm/             # Six unchanged historical launchers
├── tests/                     # Synthetic CPU checks
├── docs/
│   ├── RESEARCH_REPORT.md
│   ├── APPROACHES.md
│   ├── RESULTS.md
│   ├── LIMITATIONS.md
│   ├── REPRODUCTION.md
│   ├── DATA_AUDIT.md
│   ├── RIGHTS_AND_NOTICES.md
│   ├── REPOSITORY_GUIDE.md
│   └── history/               # Earlier methods, reports, results and plans
└── release/                   # Sanitized provenance and review snapshots
```

The Slurm scripts retain their bytes and resolve inputs from the submission/project root. [Submission examples](../scripts/slurm/README.md) reflect their new location. Historical notebooks and Python runners retain their original locations because the legacy code relies on them. Historical document links and current checksums are updated; original manifest paths describe the earlier snapshots rather than the current layout.

The owner has chosen public visibility **without a project-wide `LICENSE` for now**. A license can be considered later if the owner selects terms covering the included project-authored code; none is invented here. The citation file lists the established researcher; the faculty advisor remains acknowledged separately, without assuming coauthorship, a DOI or peer-reviewed publication.

## Public repository versus research archive

| Material | Recommendation |
|---|---|
| Research summary, full methods, honest limitations and references | Public |
| Project source, configs, synthetic tests and figure generator | Public, subject to the selected project-code terms |
| Aggregate scores, diagnostics and figures | Public; retain protocol and uncertainty descriptions |
| Historical notebooks, reports and Slurm launchers | Public with cleared outputs and historical labels |
| Source hashes, compact QC, provenance and checksum lists | Useful public evidence; keep snapshot dates explicit |
| Large anonymized exclusion inventory and branch-integration ledgers | Safe but optional public provenance after review; full records can stay in the research archive |
| Raw datasets, downloaded responses and per-question answer/logit evidence | Keep in the private research archive; use upstream acquisition instructions |
| Base weights, adapters, checkpoints and optimizer/RNG state | Keep outside this code repository; any future distribution is a separate source-terms decision |
| Reviewer packets/keys, raw operational logs, credentials, ZIPs, caches and environments | Exclude from version control |

The existing `release/EXCLUDED_FILES.json` contains safe aliases, categories, sizes and hashes, not original private filenames or answer records. It is retained to preserve evidence. It is not necessary to read it to understand the study. Every file in a public repository is visible: folders cannot be made private. Moving or removing a file from the latest tree does not hide older commits.

The accessible history was reviewed for prohibited artifacts, credential patterns, private machine paths and notebook outputs. No such finding requires a new repository or history rewrite. A fresh public repository is a presentation choice if a smaller provenance history is preferred; it is not required by a detected secret. Use the existing repository if the owner is comfortable making the reviewed historical provenance public. This is a bounded local audit, not a guarantee that pattern scanning detects every secret.

## GitHub publication metadata

**About description:**

> Exploratory comparison of SFT, F2-Lite, and DPO on Qwen3-1.7B using 817 TruthfulQA questions.

**Topics:** `lora`, `qwen3`, `dpo`, `peft`, `truthfulqa`, `fine-tuning`, `llm-evaluation`, `machine-learning`.

These are the owner-selected publication settings; their final values are verified on GitHub after the visibility change. [CITATION.cff](../CITATION.cff) supplies citation metadata without asserting a GitHub Release, ownership grant or publication venue.

## Publication decisions and optional future work

The researcher reports Professor Salem Othman has already given permission to share the educational work. That account is recorded; the same faculty permission is not requested again. [Source notices](RIGHTS_AND_NOTICES.md) distinguish that permission from third-party terms and project-code reuse licensing.

On November 7, 2026, the owner explicitly authorized merging the reviewed branch into `main` and changing the existing repository to public after final checks pass. Project-code licensing is deliberately deferred, and no GitHub Release, dataset/weight publication or GPU workload is authorized. The [final publication-readiness record](../release/PUBLICATION_READINESS.md) documents the checked tree and retained boundaries; GitHub's current repository and pull-request state record whether the authorized operations have completed.

Multiple training seeds, independent factual annotations, stronger decontamination and a full clean CUDA/GPU replay are **future research/reproduction improvements**, not prerequisites for publishing this accurately scoped educational experiment. Failed quality gates remain limitations rather than being recast as passes. The full [limitations](LIMITATIONS.md) and [reproduction boundary](REPRODUCTION.md) remain prominent.
