# Publication scope and reproduction boundaries

The owner explicitly authorized merging `research/final-release-review` into `main` and making `Crient/truthfulness-lora` public after final checks pass, on November 7, 2026. This page supersedes preparation-time administrative language in archived candidate/QC records. It distinguishes the owner's publication choices from limitations of the experiment. See [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md) for the final verification scope.

## Recorded publication decisions

1. **Visibility and integration.** Merge through a pull request with preserved history, verify the final `main` tree, then change visibility to public. No force-push, history rewrite or GitHub Release is authorized.
2. **Project-code reuse terms.** A project-wide license is deliberately deferred. No project `LICENSE` or blanket MIT grant is created. Preserve third-party attribution and notices; public visibility does not itself grant broad project-code reuse rights.
3. **Distribution scope.** Publish the reviewed code/documentation/aggregate evidence and retained historical provenance. Keep datasets, answer records, weights, checkpoints, private reviewer packets, credentials, logs, ZIPs, caches and environments outside version control. The [repository guide](../docs/REPOSITORY_GUIDE.md) explains the public/archive boundary and how history remains visible.

## Permission and source notices already recorded

The researcher reports **Professor Salem Othman has given permission to share this educational fine-tuning work**. That account is accepted as reported faculty sharing permission; the same permission is not requested again and no written institutional instrument is asserted.

Exact pinned Qwen, HaluEval mirror, Argilla and TruthfulQA notices/cards, plus the upstream HaluEval license, were retrieved for this review. Declared terms are Apache-2.0 or MIT. See [source notices](../docs/RIGHTS_AND_NOTICES.md) for hashes, the HaluEval mirror/upstream discrepancy and the boundary for mixed-source or future asset redistribution. No model/data assets are bundled here. Source ownership and the scope of any new project-wide reuse grant are not inferred from public availability.

## Disclosed limitations, not publication prerequisites

- Full clean research/CUDA installation, original-input integration and GPU replay remain unverified. Minimal installation, synthetic CPU tests and packaging checks have a narrower scope. A new GPU replay is optional future validation, not a condition for sharing this educational record honestly.
- The preference-quality and answer-only shortcut gates remain failed. One seed, nominal item-bootstrap intervals, data/recipe/compute confounding, bounded leakage checks and post-hoc analyses limit claims. These are legitimate scientific limitations; stronger causal/factual/seed-robust claims would require new evidence.
- MC1 and MC2 rank existing answer choices. Higher MC2 is not generated-answer factual accuracy or proof of broad hallucination reduction.

Historical evidence is preserved. [PRESENTATION_REVIEW.md](PRESENTATION_REVIEW.md) and [INTEGRATION_REVIEW.md](INTEGRATION_REVIEW.md) are earlier snapshots whose private-review status was accurate at the time. The final authorization permits the merge and visibility change only after checks pass; it does not create a license grant, GitHub Release or new model workload.
