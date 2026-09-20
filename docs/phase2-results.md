---
title: Phase 2 results, all answers in one call on Gemma 4 12B
date: 2026-09-20 19:57 CEST
author: Claude (runs and review), Codex gpt-6-astra (code)
type: results
status: revision 5 is the product rule and the current numbers; revisions 2 to 4 kept below as history
---

# Phase 2 results: all answers in one call

One prompt with the state and all questions, a grammar that forces one `Qk: <id>` line per question, probabilities read at each id token. Same model as phase 1 (Gemma 4 12B QAT, llama-server on port 8010, MTP draft on), same test set (`data/typed_decisions_test.jsonl`, 400 cases, 2,000 decisions), same metric code, same temperature rule (one value per question type, fitted on the train split only). No weight training; the only fitted values are those three temperatures.

## Current numbers (revision 5, the product rule)

Revision 5 is what the repo ships: the questions stay in the order the caller gives them, only the option letters are shuffled (3 times by default), a probability floor for option ids missing from the server's top 20 list, and a coverage guard of 0.05 in multi mode. Temperatures fitted on the train split only (choice 2.74, noul 5.06, score 3.05). Commits d695071 and 2189987.

| | Jev 1.13 (card, same split) | Phase 1: one call per question | Revision 5: all questions in one call, 3 shuffles | Revision 5: one call, one shuffle |
|---|---|---|---|---|
| Accuracy | 0.727 | 0.7065 | **0.7370** | 0.7315 |
| ECE after temperature | 0.144 | 0.032 | **0.026** | 0.059 |
| Mean per question ECE | - | 0.130 | 0.090 | 0.097 |
| Brier, card scale | 0.148 | 0.145 | **0.136** | 0.171 |
| Log loss after temperature | - | 1.104 | 1.056 | 1.152 |
| Score MAE | 0.391 | 0.368 | **0.332** | 0.363 |
| Failed cases (uniform guess) | 0 | 0 | 0 | 0 |
| Time per case, 5 questions | 710 ms (cloud) | 2,400 ms | 1,351 ms | **500 ms** |

Per type, 3 shuffles: choice 0.705, noul 0.822, score 0.698. Per workflow: agent_trace_observability 0.696, customer_service 0.776, invoice_processing 0.756, security_incidents 0.720. Raw ECE before temperature 0.171.

This is the like for like number against Jev: the same test split, the same question order that Jev receives. The first full run below scored 0.743 with a fixed question order that the shuffle seed picked by luck (a fact check first, urgency late); that order is not something a caller controls, so it is history, not the headline. The revision 3 and 4 sections explain the order effect.

## First full run (revision 2, a lucky fixed question order)

Kept exactly as measured and audited. Its question orders came from `random.Random(42)`; see the revision 3 section for why that mattered.

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

## Revision 3 (run 2026-09-20 21:29 CEST): reseeded shuffles and a probability floor made it worse

Revision 3 applied the two audit suggestions: a different question order per shuffle (`random.Random(1000 + k)` plus a rotation so a different question comes first each time) and a probability floor for option ids missing from the top 20 list. Same model, server, test set and metric code as above. Fresh temperature fit on the train split (`calibration_multi.json` now holds choice 2.48, noul 4.19, score 3.45).

| | Rev 2, 3 shuffles | Rev 3, 3 shuffles | Rev 2, 1 call | Rev 3, 1 call |
|---|---|---|---|---|
| Accuracy | **0.7430** | 0.7230 | 0.7370 | 0.6870 |
| ECE after temperature | 0.030 | 0.040 | 0.044 | 0.103 |
| Brier, card scale | 0.125 | 0.135 | 0.154 | 0.209 |
| Log loss after temperature | 1.048 | 1.066 | 4.10 | 1.24 |
| Score MAE | 0.331 | 0.362 | 0.371 | 0.419 |
| Failed cases (test) | 2 | 2 | 0 | 0 |
| Time per case | 1,380 ms | 1,347 ms | 508 ms | 492 ms |
| Zero probability entries | 1 | 0 | 288 | 0 |

The floor did exactly what the audit predicted: the one call log loss fell from 4.10 to 1.24 because no option gets probability zero any more. It cannot change a top answer (a floored entry never exceeds a real one), so every accuracy change comes from the question order.

**The question order moves accuracy by 2 to 5 points.** The per type and per question breakdown shows where:

| 3 shuffles | Rev 2 | Rev 3 |
|---|---|---|
| choice (600) | 0.737 | 0.733 |
| noul (600) | 0.823 | 0.837 |
| score (800) | 0.688 | 0.630 |
| urgency (400, score) | 0.610 | 0.522 |

Every workflow ends with `urgency`. Revision 2's three orders were `[3,1,2,4,0]`, `[3,2,0,4,1]`, `[3,1,2,0,4]` (question index 3 first every time, urgency 4th, 4th and 5th). Revision 3's orders are `[4,2,1,0,3]`, `[3,4,1,0,2]`, `[0,3,4,1,2]` (urgency 1st, 2nd and 3rd). Asked before the other questions, urgency loses 9 points. The one call pass shows it is not only urgency: with the single revision 3 order `[4,2,1,0,3]`, `matches_order` (invoice_processing, noul) fell from 0.96 to 0.69 because it moved from first to last, after the model had already committed to a `disposition`. Answers written earlier in the reply are context for the later lines, so a question answered after a related judgment follows that judgment, right or wrong.

In simple words: in one call the model fills the form top to bottom and reads its own earlier answers. Facts should come before judgments, and the summary judgment (urgency, risk) last. A random order breaks that, and averaging three random orders only averages the damage. Phase 1 (one question per call) had no such effect, and also no such gain: its 0.7065 is below both revisions.

Decision: revision 4 keeps the question order the caller gives (the caller knows the dependencies), never shuffles questions, keeps shuffling the option letters, and keeps the floor.
### Order experiment (2026-09-20 21:51 CEST)

One call per case, no shuffles, raw accuracy on the full test set (400 cases, 2,000 decisions), seven fixed question orders. Each order is a reordered copy of the test file, so the engine saw the identity order (`scratch/order_experiment.py`). The revision 2 and revision 3 rows reproduce the benchmark's one call numbers to the fourth decimal, which checks the harness. About 500 ms per case in every row.

| Order | Accuracy | choice | noul | score | urgency |
|---|---|---|---|---|---|
| Revision 2 shuffle 0 (question 4 first, urgency 4th) `[3, 1, 2, 4, 0]` | 0.7370 | 0.720 | 0.832 | 0.679 | 0.605 |
| Given order (as the caller wrote it; urgency last) `[0, 1, 2, 3, 4]` | 0.7315 | 0.693 | 0.822 | 0.693 | 0.625 |
| Sorted by type: noul, choice, score `noul,choice,score` | 0.7230 | 0.675 | 0.815 | 0.690 | 0.618 |
| Reversed (urgency first) `[4, 3, 2, 1, 0]` | 0.7055 | 0.685 | 0.820 | 0.635 | 0.552 |
| Urgency first, rest as given `[4, 0, 1, 2, 3]` | 0.7035 | 0.703 | 0.793 | 0.636 | 0.530 |
| Sorted by type: score, choice, noul `score,choice,noul` | 0.6985 | 0.662 | 0.803 | 0.647 | 0.568 |
| Revision 3 shuffle 0 (urgency first, question 4 last) `[4, 2, 1, 0, 3]` | 0.6870 | 0.680 | 0.772 | 0.629 | 0.542 |

Reading: 5 points between the best and the worst order from one model and one prompt format. Every order that puts `urgency` early loses 6 to 10 points on that question. Sorting by type does not beat the caller's order: facts first pushes the choice questions back and they lose. The best fixed order is revision 2's first shuffle, which by luck starts with a fact (a noul in three of four workflows) and keeps urgency late. Revision 4 keeps the caller's order; the README tells callers to write fact checks first and summary judgments last.

## Revision 4 (run 2026-09-20 22:31 CEST): the caller's question order, no question shuffle

Revision 4 keeps the questions in the order the caller gives them, in every shuffle. Only the option letters are shuffled. The floor from revision 3 stays. Fresh temperature fit (choice 2.75, noul 5.00, score 3.04). Commit d695071.

| | Rev 2, 3 shuffles | Rev 4, 3 shuffles | Rev 2, 1 call | Rev 4, 1 call |
|---|---|---|---|---|
| Accuracy | 0.7430 | 0.7345 | 0.7370 | 0.7315 |
| ECE after temperature | 0.030 | **0.025** | 0.044 | 0.060 |
| Mean per question ECE | 0.099 | 0.091 | 0.098 | 0.098 |
| Brier, card scale | 0.125 | 0.136 | 0.154 | 0.171 |
| Log loss after temperature | 1.048 | 1.057 | 4.10 | 1.15 |
| Score MAE | 0.331 | 0.336 | 0.371 | 0.364 |
| Failed cases (test) | 2 | 4 | 0 | 0 |
| Time per case | 1,380 ms | 1,361 ms | 508 ms | 502 ms |

Per type, 3 shuffles: choice 0.703 (rev 2: 0.737), noul 0.822 (0.823), score 0.693 (0.688), urgency 0.623 (0.610). The score questions and urgency are now at their best; the choice questions lost 3 points, almost all of it on `disposition` (0.74 to 0.635, invoice_processing and security_incidents). In the caller's order `disposition` is answered second, right after a score (`discrepancy_severity`) or a noul (`credential_compromise`); in revision 2's orders it came right after the fact check (`matches_order`, `true_positive`). Same lesson as before: a decision answered after the fact it depends on does better.

Revision 4 is 0.85 points under revision 2 with 3 shuffles. Two thirds of that gap is the two extra failed cases: 4 cases of 400 fell under the 0.5 coverage guard and became uniform guesses (20 decisions). See revision 5.

Revision 4 stays the product rule: the caller's order is transparent and the same order that Jev receives, so 0.7345 against Jev's 0.727 is the like for like comparison on this test split. Revision 2's 0.743 was a lucky fixed order (a fact check first, urgency late) and is kept in this document as what it is.

## Revision 5 (run 2026-09-20 23:11 CEST): coverage guard 0.05 in multi mode

Revision 4 gave 4 test cases and 17 train cases a uniform guess because their answer token put less than half of its mass on the question's own ids, the rest on a related question's letters (the `action` letters on the `needs_human` line, ` A` on the `needs_review` line). In every one of them the valid mass still pointed at one id. The 0.5 guard was phase 1's rule for a single question with ids A to E. Revision 5 lowers it to 0.05 in multi mode only; single mode stays byte for byte. Commit 2189987.

| | Rev 4, 3 shuffles | Rev 5, 3 shuffles | Rev 4, 1 call | Rev 5, 1 call |
|---|---|---|---|---|
| Accuracy | 0.7345 | **0.7370** | 0.7315 | 0.7315 |
| ECE after temperature | 0.025 | 0.026 | 0.060 | 0.059 |
| Score MAE | 0.336 | 0.332 | 0.364 | 0.363 |
| Failed cases (test) | 4 | 0 | 0 | 0 |
| Time per case | 1,361 ms | 1,351 ms | 502 ms | 500 ms |

The 20 decisions of the 4 rescued test cases went from a uniform guess to normal reads and lifted the 3 shuffle accuracy by a quarter point; the 1 call pass had no failures and only moves by the refitted temperatures. No warning in 1,600 cases of either pass. Zero server crashes across the revision 3, 4 and 5 runs and the order experiment (about three hours, roughly 12,000 cases on the patched llama-server with the MTP draft).

## Changelog

- 2026-09-20 19:57 CEST: First version with the full test set numbers for 3 shuffles and 1 call.
- 2026-09-20 20:11 CEST: Folded in the Codex audit: Jev measured on this same test split; Brier on the card scale; ECE caveats; failed cases are uniform guesses, not wrong; one order log loss comes from top 20 zeros; speed at 20 questions marked as expected, not measured.
- 2026-09-20 21:29 CEST: Revision 3 measured on the full test set: 0.723 (3 shuffles) and 0.687 (1 call) against 0.743 and 0.737. Cause traced to question order; floor kept; revision 4 decision recorded.
- 2026-09-20 21:51 CEST: Order experiment, seven fixed orders on the full test set, 1 call each; the caller's order is second best at 0.7315, spread 0.687 to 0.737.
- 2026-09-20 22:31 CEST: Revision 4 measured: 0.7345 (3 shuffles, ECE 0.025) and 0.7315 (1 call). Caller's order is the product rule.
- 2026-09-20 23:11 CEST: Revision 5 measured: 0.7370 (3 shuffles, ECE 0.026, no failed cases) and 0.7315 (1 call). Current numbers section added at the top; revision 2 kept as history.
