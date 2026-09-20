---
title: Phase 2 results, all answers in one call on Gemma 4 12B
date: 2026-09-20 19:57 CEST
author: Claude (runs and review), Codex gpt-6-astra (code)
type: results
status: final, audited (see phase2-audit.md, verdict PASS with caveats)
---

# Phase 2 results: all answers in one call

One prompt with the state and all questions, a grammar that forces one `Qk: <id>` line per question, probabilities read at each id token. Same model as phase 1 (Gemma 4 12B QAT, llama-server on port 8010, MTP draft on), same test set (`data/typed_decisions_test.jsonl`, 400 cases, 2,000 decisions), same metric code, same temperature rule (one value per question type, fitted on the train split only). No weight training; the only fitted values are those three temperatures.

## Headline

| | Jev 1.13 (card) | Phase 1: one call per question, 3 shuffles | Phase 2: one call for all, 3 shuffles | Phase 2: one call, one order |
|---|---|---|---|---|
| Accuracy | 0.727 | 0.7065 | **0.7430** | 0.7370 |
| ECE raw | - | 0.2005 | 0.1546 | - |
| ECE after temperature | 0.144 | 0.0321 | **0.0297** | 0.0444 (a) |
| Brier, our scale (mean over options) | - | 0.0429 | 0.0371 | 0.0449 |
| Brier, card scale (sum over options) (b) | 0.148 | 0.145 | **0.125** | 0.154 |
| Log loss after temperature | - | 1.104 | 1.048 | 4.10 (a) |
| Score MAE after temperature | 0.391 | 0.368 | **0.331** | 0.371 |
| Failed cases (uniform guess) (c) | 0 | 0 | 2 of 400 | 0 |
| Calls per case | 1 (API) | 15 | 3 | 1 |
| Time per case | 710 ms (cloud) | 2,400 ms | 1,380 ms | **508 ms** |

(a) The one order pass reuses the temperatures fitted on the 3 shuffle averages. Its log loss of 4.1 is not a temperature problem: the server returns only the top 20 tokens, so an option missing from that list gets probability zero, and 187 of its 2,000 rows have gold mass on a zero (audit, section 9). Temperature cannot fix a zero. Averaging 3 shuffles leaves one zero in 2,000 rows. A probability floor or a wider top list is the fix for the single call setting.
(b) The audit proved the card's Brier is the sum over options (its uniform control matches 0.238 exactly); the row above is ours on that scale.
(c) The 2 failed test cases (lines 341 and 349, coverage 0.34 and 0.43) got uniform distributions. Under the tie rule 2 of their 10 decisions count as right. All wrong would give 0.7420; dropping them, 0.7457.

Jev 1.13.0 was measured by the benchmark author through the API on 18 Sep 2026 on this same test split, all 400 cases and 2,000 decisions, so the accuracy comparison is like for like: 0.743 against 0.727 on the same decisions. Their tie rule, ECE binning and score conversion are not published, so calibration comparisons stay approximate. Card references: majority 0.520, factor ceiling 0.704, teacher self agreement 0.735, Laya zero shot 0.360, Laya fine tuned 0.766.

## Per workflow and per type (accuracy)

| Group | Phase 1 | Phase 2, 3 shuffles | Phase 2, 1 call |
|---|---|---|---|
| customer_service | 0.736 | 0.786 | 0.794 |
| invoice_processing | 0.726 | 0.792 | 0.768 |
| security_incidents | 0.694 | 0.710 | 0.708 |
| agent_trace_observability | 0.670 | 0.684 | 0.678 |
| choice (600) | 0.698 | 0.737 | 0.720 |
| noul (600) | 0.832 | 0.823 | 0.832 |
| score (800) | 0.619 | 0.688 | 0.679 |

ECE after temperature, 3 shuffles: choice 0.076 (phase 1 0.041, worse), noul 0.030, score 0.044 (phase 1 0.072, better). Raw: choice 0.115, noul 0.132, score 0.203. The pooled ECE is the standard top label ECE with 10 bins; averaged per workflow and question it is 0.120 (phase 1 0.130), so the pooled 0.030 should not be read as uniformly good calibration, and no fivefold claim against Jev's 0.144 is justified without their aggregation rule.

Temperatures, multi (phase 1 in brackets): choice 2.85 (2.58), noul 4.58 (2.94), score 3.29 (3.29). File `calibration_multi.json`.

## What happened on the way

1. **Shared letters failed.** First format gave every question ids A, B, C again. Training case 439 (a two option noul between five and four option questions) put 0.71 of its answer mass on a neighbour's letter, coverage 0.29, and the run aborted. Fix: unique ids across the call (Q1 takes A and B, Q2 the next block, and so on), groups when the alphabet runs out, a format version in the cache key, and failed cases now count as a uniform guess instead of killing the run. On the same 438 training cases the shared letter version had already beaten phase 1 (0.717 vs 0.667 raw), unique letters did slightly better again (0.727 vs 0.680 on 361 cases).
2. **Coverage failures.** 3 of 1,200 train cases and 2 of 400 test cases fell under the 0.5 coverage rule (mass on the question's own ids). They are in the numbers as uniform guesses for all five of their decisions.
3. **llama-server needed a patch.** With the MTP draft, tokens after the first came back with probability 1.0 and no table (upstream `tools/server/server-context.cpp`, "TODO: set result.probs"). Patched on branch `server-spec-probs` in `/home/abhishek/llama-vulkan/llama.cpp-fresh` (commits 611dc03 and 4d45333), rebuilt, serving on 8010 with MTP kept.
4. **One server crash.** 63 minutes into the first full run the server died with a CUDA illegal memory access inside `common_speculative_impl_draft_mtp::process` (`llama_get_embeddings_nextn_ith`), right after a prompt cache eviction. That is the upstream MTP draft path, not the probability patch (which only reads host side logits after sampling). Crash log saved at `/home/abhishek/.claude/jobs/872a19fc/tmp/gemma-server-crash-19h39.log`. The run resumed from the cached training pass and finished without another crash. Watch for it under long generation loads.

## Speed, in plain words

The time per case is reading time, not call count. The state is cached after the first call; each extra call re-reads only what changed. At 5 questions the one call mode saves about 40 percent against phase 1 with shuffles, and reaches 0.5 s per case without shuffles. At 20 questions the saving should grow to 5 to 10 times, because phase 1 needs one call per question; this is expected, not measured (the benchmark has 5 questions per case).

## Audit

Codex gpt-6-astra audited the numbers adversarially (`docs/phase2-audit.md`, scripts in `scratch/`). Verdict: PASS with caveats. All headline values reproduce from the per decision records; no gold leakage in the prompt path (proved by mutating every gold field in a copy of the test set: identical prompts and predictions); no train and test overlap; temperatures refit identically and change no answer; the read is the raw pre sampling distribution. Caveats folded into this document: Jev's split, the Brier scale, the ECE aggregation, the failed case wording, the zero mass explanation for the one order pass, and the speed extrapolation. Two more for a later run: the fixed seed puts the same question first in all three shuffles, and failed cases record zero calls.

## Files

- `report_multi.json` (3 shuffles, fitted on train), `report_multi_p1.json` (1 call), `calibration_multi.json`, `data/preds_train_multi.jsonl` (train cache, format 2), `data/bench_multi.log`.
- Code: `src/system1/core.py` (`build_multi_prompt`, `build_grammar`, `read_multi`, `SystemOne._multi`), `src/system1/backend.py` (`complete_multi`, `probe`), `bench --mode multi`, `ask --mode multi`, API `mode` field. Commits 4c28227 and 7fa18ab.
- Spec `docs/phase2-spec.md`, Codex notes `docs/phase2-notes.md`, audit `docs/phase2-audit.md`.

## Changelog

- 2026-09-20 19:57 CEST: First version with the full test set numbers for 3 shuffles and 1 call.
- 2026-09-20 20:11 CEST: Folded in the Codex audit: Jev measured on this same test split; Brier on the card scale; ECE caveats; failed cases are uniform guesses, not wrong; one order log loss comes from top 20 zeros; speed at 20 questions marked as expected, not measured.
