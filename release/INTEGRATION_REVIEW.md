# Private branch integration review — November 7, 2026

> **Historical integration snapshot for commit `5ecf3de`.** Original review findings are retained below. Current layout, reported faculty sharing permission and publication decisions are documented in [PRESENTATION_REVIEW.md](PRESENTATION_REVIEW.md) and [BLOCKERS.md](BLOCKERS.md).

Integrated `truthfulness-lora_release-candidate_20261107.zip` into the existing private `Crient/truthfulness-lora` repository on `research/final-release-review`. The base is `ead40cedad88e214d8033b3ecf0774e86e04e96a`. This review supports a private branch commit and push only; it does not authorize a merge or public publication.

## Initial state and preservation

- The checkout was on `main`, matching local `origin/main` and GitHub's live `main`. GitHub reported private visibility and push access. The remote URL was `https://github.com/Crient/truthfulness-lora.git`.
- No staged or unstaged tracked changes existed. The only untracked file was `.DS_Store`; three ZIPs and the existing private `release_review/` artifacts were ignored. Their original bytes were preserved. `.DS_Store` is now ignored too.
- The original commit remains the parent of the review work. No historical commit, notebook, script, result, private packet or Unity artifact was removed. The local `main` reference is unchanged.
- Five notebooks, six Slurm launchers, two historical Python runners, the historical sample verifier, the original aggregate CSV and verification requirements retain their original bytes. Historical document bodies remain intact, with scope banners and links redirected to historical results where appropriate.
- The original README narrative and results were preserved as [HISTORICAL_README_20261104.md](../docs/history/HISTORICAL_README_20261104.md) and [HISTORICAL_RESULTS_20261104.md](../docs/history/HISTORICAL_RESULTS_20261104.md). Only local links were adjusted in the archived README; the original README's exact bytes also remain available in the base commit.
- Researcher Gechleng (Leng) Lim, advisor Professor Salem Othman, Wentworth Institute of Technology and UMass Unity HPC attribution were retained. Attribution does not assert clearance or endorsement.

## Archive and integration decisions

The ZIP SHA-256 matches the user-supplied value:

```text
0ba169c78e53ac49f5c52a205ff73818163a7f7ae50dcc233967a094b53d598c
```

The archive has 50 files under one `truthfulness-lora/` prefix. CRC checks passed; extraction rejected unsafe paths, symlinks and duplicate entries. All 49 entries in its checksum list matched. Extraction and test environments were outside the repository. No nested repository directory was created.

Three files overlapped the existing tree:

| File | Integration decision |
|---|---|
| `.gitignore` | Retain existing protections; add build/output/OS exclusions and nine exact aggregate-result exceptions. No broad JSON/CSV allowance. |
| `README.md` | Use the completed four-model summary; retain authorship, historical context, negative findings and failed gates. Archive the earlier narrative. |
| `docs/RESULTS.md` | Use final native-chat results; preserve the earlier six-demonstration results in a dated document. Keep protocols separate. |

Other candidate files were added after review. Original manifests, exclusion aliases, source-provenance hashes, license evidence and QC remain preparation-time evidence. Their claims about original-input tests and item-level recalculation were not rerun locally. The [integration manifest](INTEGRATION_MANIFEST.json) maps every candidate path to its integrated path and records original tracked-file hashes.

The original checksum list is retained unchanged as [CANDIDATE_SHA256SUMS](CANDIDATE_SHA256SUMS), to verify an untouched extraction. The root [SHA256SUMS](../SHA256SUMS) verifies the integrated review tree, excluding itself. Neither list covers ignored private artifacts or implies availability of excluded datasets or weights.

### Packaging correction

The original wheel omitted `configs/input_identities.json`, while model verification expected that checkout-only location. A direct wheel test reproduced the missing-config error. The integrated package includes an identical copy as `truthfulness_lora/_data/input_identities.json`, declares it as package data and reads it through `importlib.resources`. The root configuration remains unchanged. The installed resource's bytes and SHA-256 match the root configuration.

This changes configuration lookup and packaging only. Losses, data selection, masks, optimizer logic, checkpoints and scoring algorithms were not changed. The original candidate remains verifiable against its ZIP. Future edits must keep the root and packaged identity JSON synchronized.

## Local validation

Tests used Python 3.12.14 on macOS ARM64, NumPy 2.2.6, torch 2.9.1, transformers 4.57.6, peft 0.18.1, tokenizers 0.22.2 and safetensors 0.8.0. Optional test dependencies were installed in a temporary environment; this is not the historical CUDA environment. No pretrained weights, datasets, private answer records or tokenizer were downloaded. Test execution disabled Hugging Face network access.

| Check | Local outcome and boundary |
|---|---|
| Minimal editable installation | Passed in a fresh temporary Python 3.12 environment. |
| Lightweight synthetic suite | 4 passed. Metrics/ties, stable softmax, pairing/bootstrap reproducibility and output refusal checks. |
| Full available synthetic CPU suite | 6 passed, 1 skipped. Includes DPO gradient orientation/reference detachment and exact tiny random PEFT model optimizer/scheduler/RNG recovery. Repeated against the installed wheel after the packaging correction. |
| Optional real-template/mask test | Skipped: `TL_TEST_TOKENIZER` absent. No pretrained tokenizer was fetched. |
| CLI inspection | All 9 root/subcommand help invocations and the console entry point passed. Missing subcommands/required arguments were rejected without creating outputs. |
| Wheel and source distribution | Built locally; source distribution rebuilt into a wheel. Package contents exclude research data/weights/private files; packaged input identities are present. |
| Clean minimal wheel installation | Passed in a second fresh environment, outside the checkout: 4 lightweight tests, all 9 CLI help invocations, package resource lookup and dependency consistency. |
| Dependency consistency | `pip check` passed in both local test environments. This does not validate the full historical dependency/driver lock. |
| Static/layout checks | Python AST, JSON/notebook and CSV parsing; empty historical notebook outputs; all six Slurm scripts pass `bash -n`. |
| Scientific aggregate consistency | Four score rows and their stored intervals agree across CSV/JSON/headline prose; all six paired mean differences and MC1 transition counts reconcile. MC1 exact-binomial/Holm arithmetic, MC1/MC2 decompositions, feature-bin weighted means and diagnostic gain/loss mass reconcile. These are aggregate checks, not item-bootstrap or inference replay. |
| References | Local Markdown targets resolve. All 18 distinct external Markdown reference URLs were accessed through primary-source pages or authenticated GitHub metadata. This checks reference identity/access, not scientific replication or legal clearance. |
| Preservation and content boundary | Original artifact hashes checked; no tracked historical files deleted. Reviewed eligible files for prohibited paths/extensions, credential-shaped strings and private machine paths. Pattern checks cannot prove exhaustive secret detection. |
| Historical sample verifier and F2 comparison | Help passed. Full verification cannot run: excluded historical per-question result directories are absent. Both commands refused missing inputs; neither is reported as a scientific pass. |
| Full training, GPU inference, original-input integration | Not run: excluded data/model assets and suitable CUDA resources are unavailable. No original score, logit, adapter hash or item-bootstrap reproduction is claimed. |

Machine-readable outcomes are in [INTEGRATION_QC.json](INTEGRATION_QC.json). Locally built archives, virtual environments, caches, test checkpoints and terminal output remain outside version control. Wheels/sdists are code-installation artifacts; the Git tree is the research review package.

## Scientific and publication blockers retained

- Rights/licensing: unresolved HaluEval mirror/upstream scope, mixed-source responses, project ownership/license and exact asset notices. No project license was created.
- Required faculty/institutional/coauthor authorization and separate public-publication authorization are not established. A private branch push does not resolve them.
- Full clean research/CUDA installation and GPU/original-input replay remain unverified. Minimal installation checks do not close that gap.
- Quality gates remain failed: 28/48 supported preferences versus an 80% requirement; 8/48 unusable chosen responses versus a 10% maximum; answer-only estimates 66.01% and 74.69% versus a 65% maximum. There were no independent human audit forms.
- One training seed, nominal item-bootstrap intervals, multiple comparisons, data/recipe/compute confounding, limited lexical leakage checks, shared HaluEval supporting passages and post-hoc analyses limit conclusions. MC2 is candidate-normalized probability mass, not generated-answer factual accuracy. Earlier subgroup claims remain provisional.

See [BLOCKERS.md](BLOCKERS.md), [limitations](../docs/LIMITATIONS.md) and [rights/notices](../docs/RIGHTS_AND_NOTICES.md). Nothing in this integration authorizes a public release, visibility change, license grant, merge into `main` or new model workload.
