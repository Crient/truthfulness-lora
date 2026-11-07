# Data provenance and bounded audit

## HaluEval

Consumed pminervini/HaluEval QA/data, cached source revision `12a856119f03975a94509091e8cada3e6be6ead7`. Notebook loading was originally unpinned; the new stage command pins it. A seed42 shuffled split reproduces all9,500/500 original rows/order. No missing/blank field rows, exact duplicate rows or normalized question overlap were found. Four normalized knowledge passages cross the split; no document-disjoint claim is justified. HaluEval, introduced for evaluation, was repurposed as training data. F2's three tasks remain within source splits and are correlated.

## UltraFeedback

Argilla cleaned commit `770076f077c4c5e298498fa32f804857f46d5134` contains60,917 pairs; exact parquet SHA256 `27dc59041d73654aecfa7685b6f839a0bbb2014d640ee92eefa047b262d1f1ca`. The archived binarization notebook averages four GPT-4 aspect ratings, selects the highest mean and draws a strictly lower-rated rejection. Its upstream revisions/random draw are not fully pinned; reproduction starts from immutable exported bytes.

Fixed screens require valid matching two-message branches, preserved text, chosen≥4/5 and rating gap≥0.5, no embedded controls, distinct answers, full branches≤2,048, and rejected/chosen word/token ratios in[0.5,2]. Leading-newline template incompatibilities are excluded. Lexical groups use exact normalized prompts, sufficiently long repeated responses and shared20-word prompt spans; spans linking>100 rows are ignored. One representative/group, source quotas and token-length-direction balancing yield18,534 eligible unique groups from19,801 eligible rows, then9,500/500 selected. Selected source labels: sharegpt4,043; evol_instruct2,382; ultrachat2,164; false_qa462; flan_v2_cot372; flan_v2_niv2 323; flan_v2_p3 177; flan_v2_flan2021 77. Keyword domain labels are unvalidated heuristics.

All10,000 selected source rows/content hashes were privately checked against the parquet; group and row IDs are unique, train/validation groups do not intersect, and selected groups avoid recorded benchmark-hit groups. The fresh wrapper preserves the selection algorithms, but omits default private audit packet creation. Neither reconstruction nor new audit judgments were performed during candidate packaging.

## Quality/shortcut failures

The original48 accepted/16 excluded AI audit was frozen before key exposure. Accepted28 supported,8 chosen unusable;20 non-endorsed consist of3 opposite preferences,3 ties,14 unresolved holds. Zero independent human forms were completed. Unweighted supported Wilson95% interval[44.28%,71.15%] is descriptive and omits judge/design uncertainty. Answer-only group-held-out features66.0105%, ngrams74.6895% exceed the65% gate. These can capture legitimate general-quality cues as well as undesirable shortcuts; failure does not prove every source label false. They remain scientific limitations despite exploratory execution authorization.

## Leakage status

Inherited full-source normalized question/long-choice anchors across prompts/responses, question5-shingle containment≥0.8 or Jaccard≥0.65, and word/character TF-IDF≥0.65 flagged149 direct rows. Transitive group exclusions remove9,323 rows; largest component9,102. This is conservative connectivity, not9,323 proven benchmark duplicates. Boilerplate overmerging and semantic false negatives are possible.

No exact normalized whole-question/prompt match was found between selected HaluEval/UltraFeedback splits and the frozen benchmark; no exact whole-prompt match was found across selected HaluEval/UltraFeedback. Such checks do not establish absence of paraphrases, document reuse, short-answer reuse, Qwen upstream exposure or semantic leakage. The benchmark is pinned legacy817 MC1/MC2, not a later updated task. [Input hashes](../configs/input_identities.json) and [rights review](RIGHTS_AND_NOTICES.md) distinguish byte identity from factual/redistribution guarantees.
