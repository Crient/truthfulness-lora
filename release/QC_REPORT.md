# Candidate quality-control report

**READY FOR PRIVATE INSPECTION; PUBLIC RELEASE BLOCKED.** See [blockers](BLOCKERS.md). Checks were CPU-only; no training-scale workload or GPU job was run.

- All7 portable unit tests passed: stable metrics/tie semantics/pairing, exclusive/disjoint outputs, DPO orientation/reference detach, real-tokenizer masks and tiny random Qwen/PEFT atomic recovery.
- All4 original-input integration checks passed: unchanged AST bodies for retained algorithms, benchmark freezer matches exact recorded SHA256, HaluEval9500/500 split order matches all original rows, and portable aggregation reproduces all817 item metrics and10,000-resample arm/all-six-pair intervals to1e-12.
- Original matched aggregates/intervals and5 aggregate CSVs are preserved. README/results headline tables match them; no six-shot headline scores are mixed in.
- All9 CLI help routes passed without starting a workload. Python AST, TOML, JSON/CSV parsing and46 local references passed at initial finalization; a second scan covers the completed metadata too.
- Credential/email/private-path/account/node/operational-identifier pattern scans found no matches. Candidate has no symlinks, hidden Git metadata, raw record files, large logs, model assets or environment/cache trees. This is a bounded content/allowlist scan, not a guarantee about all possible sensitive semantics.
- All2,033 original research entries in the fresh content/mtime/symlink preservation scope were unchanged. Private before/after ledgers and execution receipts remain outside the ZIP; exported counts omit private source paths and job IDs.

An initial test failed because the synthetic test selected the user EOS instead of the assistant EOS; the test was corrected. No research algorithm changed for that correction. The final suite passed. Output/portability changes are documented in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json). Historical artifacts and private review bundles remain untouched.

## Boundaries

No pretrained model weights were loaded for tests; only a tiny randomly initialized CPU model and the local tokenizer were used. The complete source selection/TF-IDF preparation was not rerun during candidate packaging. Installation in a clean environment, full training, GPU masks/recovery/parity and exact historical adapter re-creation remain unverified. Future scripts deliberately pin/stage recoverable inputs and create new outputs, but are not a bitwise-reproduction guarantee.

Source rights/notices and applicable approvals remain unresolved. The remote repository tree/history could not be inspected; integration on the Mac needs its own diff/history/license review. No Git initialization, upload, commit, push, publication, visibility change, external message or GPU job occurred. A private ZIP may be transferred for inspection; public publication requires later authorization.
