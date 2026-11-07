# Source notices, attribution and project licensing

## Sharing permission and project code

The researcher reports that **Professor Salem Othman gave permission to share this educational fine-tuning work**. This records the researcher's account; it does not assert a written institutional license or imply endorsement. Faculty sharing permission is no longer listed as unknown. The repository and accessible history have now been inspected; the earlier statement that the remote could not be read is superseded.

Researcher Gechleng (Leng) Lim, Professor Salem Othman, Wentworth Institute of Technology and UMass Unity HPC are acknowledged in the [README](../README.md). Existing project source and notebook adaptations retain their provenance. No blanket project `LICENSE` has been added, and no third-party ownership is claimed. MIT is a reasonable reuse option for original project-authored code if its owner confirms that scope; the owner still needs to select any license grant. Public source visibility and a grant to reuse it are separate decisions.

## Exact recorded sources

The following primary-source notice files were retrieved on **November 7, 2026**. All five returned HTTP 200. The two cached-card hashes from the original candidate matched the fetched pinned files. [Machine-readable checks](../release/SOURCE_NOTICE_CHECKS.json) record URLs, response status, declared terms and byte hashes. Only notices/cards were read; no weights or dataset records were fetched for this review.

| Resource | Recorded source | Declared terms and scope |
|---|---|---|
| Qwen3-1.7B | [Pinned LICENSE](https://huggingface.co/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/LICENSE), revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Apache-2.0; model assets are not redistributed here |
| HaluEval QA mirror | [Pinned card](https://huggingface.co/datasets/pminervini/HaluEval/resolve/12a856119f03975a94509091e8cada3e6be6ead7/README.md), revision `12a856119f03975a94509091e8cada3e6be6ead7` | Apache-2.0 in the mirror card |
| HaluEval upstream | [Upstream LICENSE](https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/LICENSE) | MIT, copyright RUCAIBox; this live upstream reference is not an immutable execution-time revision |
| Argilla cleaned UltraFeedback | [Pinned card](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned/resolve/770076f077c4c5e298498fa32f804857f46d5134/README.md), revision `770076f077c4c5e298498fa32f804857f46d5134` | MIT declared by this export; mixed-origin source/response terms are not independently settled by that declaration |
| TruthfulQA | [Pinned card](https://huggingface.co/datasets/truthfulqa/truthful_qa/resolve/741b8276f2d1982aa3d5b832d3ee81ed3b896490/README.md), revision `741b8276f2d1982aa3d5b832d3ee81ed3b896490` | Apache-2.0; frozen benchmark/answer records are not redistributed here |

The HaluEval mirror/upstream discrepancy is preserved rather than silently choosing one declaration. Public availability alone does not establish unrestricted redistribution. These permissive declarations support source attribution and acquisition instructions; they do not independently clear every mixed-origin response or decide the terms for a future adapter/data release.

## Distribution boundary

This code repository contains project code, sanitized historical notebooks, methods, references, aggregate tables, figures and numeric provenance. It links to third-party resources instead of bundling source/benchmark text, downloaded responses or model weights. Imported Python dependencies remain separate upstream packages; their respective licenses apply.

Retain applicable upstream copyright, license and NOTICE material if third-party code or assets are included in a later distribution. Any future raw-data, answer-record, checkpoint or adapter release needs its own scope review. The current branch neither redistributes those assets nor grants rights to them. [Source hashes and adaptations](../release/SOURCE_PROVENANCE.json) establish traceability, not ownership or a blanket license grant.

## Research references

HaluEval — Li et al., [HaluEval](https://arxiv.org/abs/2305.11747); TruthfulQA — Lin, Hilton and Evans, [TruthfulQA](https://arxiv.org/abs/2109.07958); DPO — Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290); Qwen team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388); UltraFeedback — Cui et al., [UltraFeedback](https://arxiv.org/abs/2310.01377); LoRA — Hu et al., [LoRA](https://arxiv.org/abs/2106.09685); faithful-finetuning inspiration — Hu et al., [Faithful Finetuning](https://arxiv.org/abs/2406.11267). F2-Lite does not reproduce the full published algorithm.

The [publication notes](../release/BLOCKERS.md) separate decisions about publication and reuse from research limitations. A new GPU replay or a positive scientific result is not required to publish an accurately described educational experiment. This page records declared source terms and unresolved scope; it does not provide legal clearance.
