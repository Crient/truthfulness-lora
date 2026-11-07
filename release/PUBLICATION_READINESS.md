# Final publication-readiness record — November 7, 2026

This record covers the tree prepared on `research/final-release-review`, following reviewed commit `8ae16e0408e6373e9aa6a9e7668e31a237ba84b0`. It records checks performed before the authorized pull-request merge and visibility change; live completion status belongs to GitHub's repository and pull-request records.

## Authorized publication scope

The owner explicitly authorized a pull request and history-preserving merge into `main`, followed by public visibility once the final safety checks and final `main` verification pass. The project-wide license is deliberately deferred. No `LICENSE`, dataset/weight publication, GitHub Release, force-push, history rewrite or GPU workload is included.

The final README table says **supporting-background prediction**, matching F2-Lite's supervised background-generation target. Existing scientific values, figures, intervals, limitations, historical records, citations and researcher/faculty/institutional attribution are preserved.

## Verification scope

Final check results are completed before the publication commit. They cover all current tracked files and all accessible Git-history trees/blobs, including historical notebook outputs and image metadata. Raw datasets, item-level answer/logit evidence, model weights, checkpoints, private review packets, credentials, operational logs, transfer ZIPs, environments and caches must remain absent from Git. Existing ignored local research/review artifacts are not part of the repository or pull request.

Validation includes current Markdown links/anchors, GitHub Markdown rendering, saved aggregate/figure hash consistency, complete checksum coverage, citation metadata, synthetic CPU tests, CLI inspection, Slurm shell syntax, source/wheel packaging and isolated wheel checks. Optional real-template masking needs a separately supplied local pinned tokenizer. No original-input item-bootstrap reconstruction, CUDA-environment replay, training or GPU inference is claimed.

## Rights and retained limitations

Faculty sharing permission is recorded as reported by the researcher. Third-party source declarations and byte hashes remain in [SOURCE_NOTICE_CHECKS.json](SOURCE_NOTICE_CHECKS.json), with attribution and distribution boundaries in [RIGHTS_AND_NOTICES.md](../docs/RIGHTS_AND_NOTICES.md). Public visibility is not a blanket project-code reuse license.

The failed preference-quality and shortcut gates remain failed. One seed, nominal item-bootstrap intervals, confounded recipes, bounded lexical leakage checks and post-hoc analyses limit the conclusions. MC2 is normalized true-choice likelihood mass, not free-generation factual accuracy. Full clean CUDA/GPU replay remains unverified; these are disclosed scientific/reproduction limitations rather than a requirement to obtain a positive result before sharing this educational study.

## Final check results

**PASS for the reviewed publication tree, with the disclosed optional test/replay gaps below.**

| Check | Outcome |
|---|---|
| Publication content/history | 95 current files and all four pre-existing commit histories reviewed; 127 unique historical blobs including local tool snapshots classified. No credential, personal-filesystem-path or prohibited published-artifact finding. |
| Local metadata classification | An ignored `.DS_Store` exists only in a local Codex tree snapshot. Its blob is unreachable from every branch/tag history and is not uploaded by the branch push; it is not publication content. No local snapshot was deleted. |
| Scientific/source preservation | Existing saved metrics, figure bytes/intervals, training/evaluation code, configs, notebooks, historical research and third-party notice evidence are unchanged. |
| README and links | GitHub Markdown renderer returned HTTP 200; all 177 local Markdown targets/anchors resolve. The four-column table retains the saved metrics. |
| Checksums | Root `SHA256SUMS` covers all 94 other files in the 95-file reviewed tree. All entries match. |
| CPU tests | 6 passed, 1 skipped. The skip requires a supplied local pinned tokenizer for real-template masking; it is not reported as a pass. |
| Isolated wheel | 4 lightweight tests passed; all 9 root/subcommand help invocations, the console entry point and bundled input identities passed. |
| Packaging and dependencies | Editable installation, wheel, source distribution and wheel rebuilt from source distribution passed. Wheel payloads agree, package payloads exclude private artifacts, and `pip check` passed in both test environments. |
| Static/citation checks | 20 Python files parsed, 24 JSON/notebook files parsed, 6 CSVs parsed, 6 Slurm scripts passed shell syntax, and CFF 1.2.0 citation validation passed. |
| Model workloads | No datasets/weights fetched, no scheduler job or GPU workload launched, and no full historical-score/item-bootstrap replay claimed. |

These checks audit the pre-existing history plus the proposed publication tree. The final publication commit and merged `main` are verified against this same tree before visibility changes. The no-project-license decision is intentional; preserved source notices and scientific limitations remain applicable.
