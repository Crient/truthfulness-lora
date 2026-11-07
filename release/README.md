# Provenance and review records

This folder contains sanitized source hashes, candidate inventories and review evidence. It does not contain raw datasets, model assets, operational logs or private reviewer packets.

- `CANDIDATE_*`, `SOURCE_PROVENANCE.json`, `EXCLUDED_FILES.*`, `LICENSE_EVIDENCE.json` and `QC_REPORT.*` describe the original 50-file transfer candidate. Their statements are preparation-time records, not a live publication decision.
- [INTEGRATION_REVIEW.md](INTEGRATION_REVIEW.md), `INTEGRATION_MANIFEST.json` and `INTEGRATION_QC.json` describe integration commit `5ecf3de`; original paths/hashes in those manifests remain historical identifiers.
- [PRESENTATION_REVIEW.md](PRESENTATION_REVIEW.md) describes the README/figure/layout update and validation. Current publication scope is in [BLOCKERS.md](BLOCKERS.md) and [RIGHTS_AND_NOTICES.md](../docs/RIGHTS_AND_NOTICES.md).
- Root [SHA256SUMS](../SHA256SUMS) covers the current review tree. [CANDIDATE_SHA256SUMS](CANDIDATE_SHA256SUMS) verifies only an untouched extraction of the original ZIP.

The large exclusion ledger contains anonymized aliases and hashes. It is retained for traceability and is optional reading. Its entries are an inventory description, not the excluded item-level evidence itself. See the [public/archive recommendations](../docs/REPOSITORY_GUIDE.md).
