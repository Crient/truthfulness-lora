# Portfolio presentation review — November 7, 2026

This update builds on integration commit `5ecf3de76f3cdafec6a4277d182ed4d07d57bb53`, solely on `research/final-release-review`. The working tree was clean before this task. Main, visibility and releases were not changed.

## Presentation

The README now leads with the research title, concise summary, a grouped four-model chart and a qualified takeaway. Exact scores, the research objective, methodology, lightweight reproduction steps, limitations, references and researcher/faculty attribution remain visible. Deep audits and historical material are linked.

The figure is generated from the unchanged saved `scores.csv` and `matched_results.json`. Blue/green bars have a zero baseline and a labeled 0–50% range; saved nominal 95% item-bootstrap intervals are shown. Every recipe receives the same styling. PNG, SVG and a JSON source/output-hash record are included. No model inference, item resampling or generative image tool was used for the chart.

The standalone HTML preview uses GitHub-like typography, normal Markdown tables and local image data. Desktop and narrow-width renders were visually checked and refined. There was no connected browser, so these are local HTML layout renders, not screenshots of live GitHub. The actual review-branch README remains the authoritative GitHub presentation. Preview HTML/images and intermediate rendering files stay in the ignored local review directory, not in the public-facing source tree.

## Layout and preservation

- Six root Slurm scripts moved to `scripts/slurm/`, with their bytes unchanged. They still select the project root through `TRUTHFULNESS_PROJECT` / `SLURM_SUBMIT_DIR`. Six syntax checks and twelve stubbed launches covered submission-root and explicit-project-root cases without scheduling or running model work.
- Ten historical documents moved to `docs/history/`, with an index. Local Markdown links were rebased and Slurm submission examples updated. Scientific narrative, results and historical labels remain intact. Historical notebooks and legacy Python runners stay at their existing paths.
- Model/training/evaluation package source, configs, notebooks, tests and all prior numeric result files are unchanged. Only presentation tooling, packaging metadata and documentation were extended.
- Original candidate and integration JSON ledgers remain historical records. Their old paths describe the archived snapshots. Current relocation/hash mappings and check outcomes are in [PRESENTATION_QC.json](PRESENTATION_QC.json); root [SHA256SUMS](../SHA256SUMS) covers the current tree.

## Documentation corrections and publication scope

`RIGHTS_AND_NOTICES.md` now states that the repository was inspected. The research report now correctly says individual extreme-case records remain in the private archive; `EXCLUDED_FILES.md` describes the inventory and does not list those results.

The researcher's account of Professor Salem Othman's permission to share the educational project is recorded. Exact recorded Qwen, HaluEval mirror, Argilla and TruthfulQA notices/cards and the upstream HaluEval license were retrieved successfully; declared terms are Apache-2.0 or MIT. The two original cached-card hashes matched. See [SOURCE_NOTICE_CHECKS.json](SOURCE_NOTICE_CHECKS.json). No project-wide license grant or written institutional instrument is invented.

No raw datasets/responses, item-level answer/logit evidence, weights, checkpoints, reviewer packets, credentials, logs, ZIPs, caches or environments were added. The accessible existing history and proposed tree were checked for those file categories, credential patterns, private machine paths and notebook outputs. Sanitized provenance is retained and identified as optional public material in the [repository guide](../docs/REPOSITORY_GUIDE.md). Making the existing repository public would expose its history too; no finding requires a clean repository or history rewrite.

## Validation

- Existing synthetic CPU suite: **6 passed, 1 skipped**. The skip requires an optional local pinned tokenizer; no pretrained tokenizer was downloaded.
- Citation metadata validates against CFF 1.2.0. The file cites the established researcher and does not assume advisor coauthorship, a DOI, a public release or a project license.
- Chart inputs, labels, stored intervals and output hashes are checked; regenerating with the same plotting environment reproduces the PNG, SVG and provenance bytes.
- Local Markdown file/image targets and README anchors are checked, including relocated history. JSON/CSV/Python parsing, shell syntax, ignore protections and checksum coverage are checked.
- Editable installation, wheel/sdist builds, a wheel rebuilt from the source distribution, isolated wheel smoke tests, console/CLI help and dependency consistency are checked. The source distribution includes the plot generator and only the named aggregate inputs/outputs.
- Full clean CUDA-environment installation, original-input replay, real-template masking and GPU training/inference remain unverified. These are disclosed reproduction limits rather than requirements to turn an educational result into a publishable portfolio record.

Machine-readable counts, relocation mappings, preservation hashes and packaging checks are in [PRESENTATION_QC.json](PRESENTATION_QC.json). No training or scheduler job was launched.

## Recommendation

**Presentation ready for portfolio review.** The [public-facing structure and metadata recommendations](../docs/REPOSITORY_GUIDE.md) are concrete. Failed quality gates, single-seed/recipe confounds, nominal uncertainty and the distinction between MC2 and generated factual accuracy remain explicit. A positive result or a new GPU replay is not required for sharing these accurately described findings.

Publication remains the owner's explicit decision. Select any desired project-code reuse terms and review the retained provenance/history before changing visibility. [BLOCKERS.md](BLOCKERS.md) records those decisions without treating optional future research or already reported faculty permission as unresolved publication requirements. This branch update does not merge, publish a release, change visibility or assign a license.
