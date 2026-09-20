---
title: Phase 2 adversarial audit of saved predictions and probability reads
date: 2026-09-20 20:04 CEST
author: Codex gpt-6-astra
type: audit
status: complete with caveats
---

# Phase 2 audit

The five phase 2 headline values reproduce from the decision records. No gold
leakage was found in the audited code path. The strongest concerns are how the
numbers are interpreted: pooled ECE hides question-level error, the Brier scale
differs from the card, missing top-token entries become zeros, and the saved
reports do not prove the identity of the server binary used for inference.

All measurements below were made on 2026-09-20. The front-matter time comes from
`date "+%Y-%m-%d %H:%M %Z"`. I read `AGENTS.md` before the audit. I did not call,
start, or stop a model server. Application code, datasets, original reports,
calibration files, and the llama.cpp checkout were not changed.

The independent implementations are in
[`scratch/audit_phase2.py`](../scratch/audit_phase2.py) and
[`scratch/audit_paths.py`](../scratch/audit_paths.py). Their JSON outputs are
[`audit_phase2_results.json`](../scratch/audit_phase2_results.json) and
[`audit_paths_results.json`](../scratch/audit_paths_results.json). The metric
implementation uses explicit sums and logarithms, then compares its results
with the production functions. The path checks use an offline backend.

## 1. Metric arithmetic and definitions

**Verdict: PASS for arithmetic; FAIL for interpreting these metrics as a common
scale without naming their definitions.**

| Saved run | Correct / total | Accuracy | ECE | Brier as reported | Log loss | Score MAE |
|---|---:|---:|---:|---:|---:|---:|
| Phase 2, three permutations | 1,486 / 2,000 | 0.743000000 | 0.029732389 | 0.037140964 | 1.047999556 | 0.330911374 |
| Phase 1, three permutations | 1,413 / 2,000 | 0.706500000 | 0.032076206 | 0.042873052 | 1.104050923 | 0.368019149 |
| Phase 2, one permutation | 1,474 / 2,000 | 0.737000000 | 0.044395013 | 0.044883469 | 4.099882614 | 0.370936644 |

All headline differences from both the saved totals and `bench.metrics` are at
most 2.23e-16. Score MAE uses the 800 score questions, not all 2,000 questions.

Let `p` be a predicted distribution, `g` the normalized gold distribution,
`y` the stored gold label index, and `K` the number of options:

- Accuracy is the fraction for which the first maximum of `p` has index `y`.
- ECE is `sum_b (n_b / N) * abs(mean(correct_b) - mean(max(p)_b))`.
  There are ten equal-width confidence bins. Bin assignment is
  `min(floor(10 * max(p)), 9)`: intervals are left-closed and right-open,
  except that the last bin includes 1. Empty bins contribute zero. All question
  types and workflows are pooled, with equal weight per question.
- Reported Brier is `mean_questions(sum_k((p_k - g_k)^2) / K)`.
- Reported log loss is `mean_questions(-sum_k(g_k * log(max(p_k, 1e-300))))`,
  in natural-log units. It is soft cross entropy, not hard-label log loss.
- Score MAE is the mean absolute difference between `sum_k(p_k * level_k)`
  and the file's `gold_score`.

This ECE is a standard confidence-binned ECE, often called top-label ECE. It
compares the largest predicted probability with hard-label correctness. It does
not assess all entries of the distribution, agreement with soft gold, or
calibration conditional on the identity of the predicted label. Its value also
depends on binning and pooling. The implementation matches
[`calibrate.py`](../src/system1/calibrate.py) and
[`bench.py`](../src/system1/bench.py).

`calibrate.brier` is a different, unused benchmark alternative: hard one-hot
targets with the squared errors summed over options. `calibrate.log_loss` also
uses hard labels. The benchmark explicitly calls `soft_brier` and
`soft_log_loss` instead. These distinctions materially change the values:

| Alternative definition | Phase 2, three | Phase 1 | Phase 2, one |
|---|---:|---:|---:|
| Soft squared error, summed over options | 0.125030519 | 0.144677048 | 0.153703860 |
| Hard one-hot Brier, summed over options | 0.358150611 | 0.397820477 | 0.369787901 |
| Hard-label log loss | 0.634860323 | 0.705361592 | 0.654827449 |
| KL from gold to prediction | 0.280641000 | 0.336692368 | 3.332524058 |

Mean gold entropy is 0.767358556. Subtracting it from soft cross entropy gives
the KL row. The reported Brier is a valid mean squared distribution error, but
division by the option count makes it smaller and changes the weighting across
questions with different option counts. It must be labeled.

Phase 2 pooled ECE is 0.028743046 with 5 bins, 0.031252077 with 15,
0.035421977 with 20, and 0.047789820 with 50. With the same ten bins calculated
separately for each of the 20 workflow/question pairs, then averaged, it is
**0.120077489**, versus 0.129519195 for phase 1 and 0.116501538 for one
permutation. These smaller groups also have more sampling noise. They show why
the pooled 0.0297 alone does not establish uniformly good calibration.

| Type | Decisions | Phase 2 ECE | Phase 1 ECE | Phase 2 accuracy | Phase 1 accuracy |
|---|---:|---:|---:|---:|---:|
| choice | 600 | 0.076259328 | 0.040858235 | 0.736666667 | 0.698333333 |
| noul | 600 | 0.029717933 | 0.023155493 | 0.823333333 | 0.831666667 |
| score | 800 | 0.044022061 | 0.071612676 | 0.687500000 | 0.618750000 |

Choice calibration gets worse even though pooled ECE improves.

## 2. Record identity, gold integrity, and failed cases

**Verdict: PASS, with explicit normalization and tie conventions.**

All three reports have exactly 2,000 unique `(id, question)` pairs, 400 unique
case IDs, and five questions per case. Each matches the same test file. There
are zero missing or extra keys, zero duplicate decision keys, and zero
mismatches in workflow, question type, option order, label, normalized gold
probabilities, score levels, or gold score. Each workflow has 100 cases.

The gold probability arrays are not always literally equal to the unnormalized
decimal values in the JSONL file. `target_record` normalizes rounded gold.
Original sums range from 0.999999 to 1.0000010000000001; 474 rows have sums
not exactly equal to 1 in floating-point arithmetic. Maximum deviation is
1.00000000014e-6. Independently applying that normalization gives zero
mismatches at tolerance 1e-12. The largest discrepancy between the file's gold
score and the expectation of its normalized distribution is 2.079997920e-6.

Every stored label selects a maximum of its gold distribution. There are 35
gold ties; 23 labels differ from the *first* maximum in the report's option
order. This is a tie convention, not a lower-probability target. If labels are
replaced with the first gold argmax, accuracies become 0.7410, 0.7055, and
0.7355 for multi, phase 1, and one permutation. The reports correctly retain the
dataset's explicit labels.

All probabilities are finite and nonnegative; every array length matches its
options. Maximum prediction sum errors are 4.440892099e-16 for multi,
2.220446049e-16 for phase 1, and 3.330669074e-16 for one permutation. There
are no NaNs. Option counts are 600 binary questions, 1,100 four-option
questions, and 300 five-option questions.

| Failure line | Case ID | Records present | Uniform predictions | Correct under report tie rule |
|---|---|---:|---|---:|
| 341 | security_incidents_000044 | 5 | Yes | 2 |
| 349 | security_incidents_000052 | 5 | Yes | 0 |

Both cases are present before and after calibration and contribute to all
applicable metrics. They are not dropped. `bench_multi.log` reports coverage
0.338175 and 0.425753 for these failures. Uniform replacement gives 2/10
correct answers. Thus the phrase "counted as wrong" in `phase2-results.md`
is false. Treating all ten failed decisions as wrong would give **0.7420**;
dropping them would give 1,484/1,990 = **0.745728643**.

## 3. Gold leakage and option descriptions

**Verdict: PASS for the inspected input boundary and offline mutation checks;
CANNOT CHECK hidden semantic contamination or an external service's requests.**

The code has these data boundaries:

1. `data.convert_row` takes input from the raw `state` and `questions` columns.
   It keeps gold and answers as separate top-level benchmark fields. Latent
   factors, label-agreement columns, and flattened target columns are not
   placed in the request. Re-converting both local parquet files reproduces
   all 1,200 train rows and 400 test rows exactly, with zero mismatches.
2. `bench.run_cases` constructs `Request` with only `state`, `questions`,
   `permutations`, and `mode`. It computes `target_record` for scoring, but
   passes only that restricted request into `engine.decide`, with calibration
   disabled during raw inference. The full case affects a cache hash, not a
   model prompt.
3. Both `build_prompt` and `build_multi_prompt` use state, instructions, option
   labels, and option descriptions. Neither takes gold or the target record.
   The multi prompt also contains other questions, not their gold answers.
4. `read_multi` and `read_probabilities` take generated tokens and assigned
   option IDs. Neither takes labels, gold scores, or gold distributions.

`Request` and `QuestionBase` use `extra="forbid"`. Extras are rejected, not
silently retained. All 12 adversarial checks rejected `gold`, `gold_score`,
`gold_probabilities`, `label`, `answers`, and `majority_answers` at request and
question level. However, `state` deliberately accepts arbitrary JSON. It would
serialize a gold field if someone put one *inside state*. Recursive scans of
both splits found zero explicit target/factor keys from the tested list in
state or questions. This cannot prove that natural-language state contains no
semantic hints; such hints can also be legitimate task evidence.

I changed every gold label and distribution in a scratch copy of all 400 test
cases, changed the scores and answers, and added gold and majority sentinels.
The offline backend captured **6,000 single-mode calls and 1,200 multi-mode
calls**. Original and altered cases produced identical prompts, grammars, and
predictions in both modes. No sentinel reached a prompt.

Available descriptions come from dataset `criteria` in both local methods.
Score list criteria are converted to indexed descriptions; noul true/false
descriptions map to yes/no. There are 200 test noul questions without criteria;
those use plain yes/no. The [dataset card](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
defines criteria as input and describes replaying state plus questions to the
API. That supports their use for Jev too. The card is not a request trace and
does not prove that every external baseline consumed every description.

## 4. Train/test separation, temperature fitting, and caches

**Verdict: PASS for split separation and reconstructing the saved fit;
CANNOT CHECK complete historical execution provenance.**

Train has 1,200 unique IDs and 1,200 unique states; test has 400 unique IDs and
400 unique states. ID overlap and canonical-JSON state overlap are both zero.
All 1,197 entries in `preds_train_multi.jsonl` have unique fingerprints and
match current train cases under multi mode, three permutations, format 2.
There are zero unmatched entries, zero test matches, zero single-mode matches,
and zero one-permutation matches.

Missing train lines are **466, 1018, and 1168**, exactly the three failures in
the run log, whose coverage values are 0.475499, 0.295342, and 0.393181.
I reconstructed their 15 decisions as uniform, as `run_cases` does, and used
all 6,000 train decisions. Uniform rows have constant loss across temperature.

| Type | Train decisions | Saved temperature | Production refit | Independent optimum |
|---|---:|---:|---:|---:|
| choice | 1,800 | 2.845942468414 | 2.845942468414 | 2.845954158309 |
| noul | 1,800 | 4.579596822027 | 4.579596822027 | 4.579593718519 |
| score | 2,400 | 3.289221718550 | 3.289221718550 | 3.289211178604 |

The independent fit uses golden-section minimization over log temperature on
the same interval, 0.2 to 10. Its difference from the production grid search is
at most 1.17e-5. The production refit exactly reproduces every saved value.
The objective is **hard-label train log loss**, not soft-gold cross entropy or
test ECE.

`benchmark` fits and saves temperatures before calling `run_cases` on test.
The log starts at 19:43:29 CEST; the calibration file modification time is
19:43:33.781647517; the report modification time is 19:52:46.961402155.
These times support that order. Reapplying the saved temperatures to the raw
test rows reproduces calibrated phase 2 predictions within 2.22e-16, with zero
argmax changes. The same check passes for phase 1. Calibration cannot explain
the phase 2 accuracy gain.

The fingerprint includes the complete case, model string, and permutation
count. Multi additionally includes `mode="multi"` and `format=2`; single
omits these to preserve its earlier keys. Therefore current single, multi,
format revisions, and different permutation counts do not share keys.

The hash does **not** identify model-weight bytes, server binary, server
settings, or every prompt-code change. Those can change without invalidating
the cache. Calibration files themselves contain only type-to-temperature
values: no split, cache, permutation, or software hash. The three-permutation
temperatures are also reused by the one-permutation report. This is confirmed
by the files; cache separation does not prevent calibration mismatch.

The training cache was last modified at 19:34:16.960496374, before the second
llama.cpp patch at 19:43:18. The patches are discussed below. Logs, modification
times, and a reproducible fit are strong consistency evidence, not an immutable
record proving which executable produced every cached prediction. Nor can
these files exclude earlier test-driven prompt selection or model pretraining
on the dataset.

## 5. Phase 1 comparison

**Verdict: PASS for comparing the saved predictions under the same metrics.**

Phase 1 has exactly the same 2,000 keys and gold targets as phase 2. The
independent results are in section 1. Comparing current code with phase 1
commit `0abf68c` shows no change to `calibrate.py`, `target_record`, or the
`metrics` function. Changes in `bench.py` concern mode routing, caching,
failure handling, and usage summaries.

Phase 2 gains **73 correct decisions**, or **0.0365** accuracy. It fixes 199
phase 1 errors and introduces 126 errors. Both are correct on 1,287 decisions
and both wrong on 388. A paired bootstrap that resamples whole cases, with
10,000 samples and seed 42, gives a percentile 95% interval of **0.0200 to
0.0525** for this difference. This interval describes these saved predictions,
not repeated model runs or stability across other question orders.

The raw phase 2 headline is 0.7430 / 0.154571669 / 0.074219494 /
2.498578828 / 0.449627756. Raw phase 1 is 0.7065 / 0.200496825 /
0.084749139 / 2.279608177 / 0.533217164. In particular, raw soft log loss gets
worse in phase 2, while calibrated soft log loss improves.

This is a method comparison, not an isolated ablation of call batching:
multi changes the prompt, assigns unique ID blocks, exposes other questions,
and conditions later answers on earlier generated answers. The saved files
cannot prove that all external server conditions were identical.

## 6. Probability read and llama.cpp patch

**Verdict: PASS for the source-level pre-sampling distribution read;
CANNOT CHECK exact historical wire responses or full option mass.**

I read `git log -2` and both complete patches in the specified llama.cpp repo:

- `611dc03ceaca9d3c53a2cfb62116488894da71ef`, 2026-09-20 18:37:35 +0200:
  fills probability tables for accepted speculative tokens from the target
  model logits at each token's `spec_i_batch[i]`.
- `4d453334afb552923271d9fa52e5471e3574005f`, 2026-09-20 19:43:18 +0200:
  moves the batch-index vector to a local after sampling, restoring the slot
  member's upstream lifetime while retaining indices for probability reads.

`server_task` defaults `post_sampling_probs` to false. The Python backend asks
for `logprobs=true`, `top_logprobs=20`, greedy generation temperature 0, and
disabled thinking. It does not request post-sampling probabilities.
`server-context.cpp:1730` disables backend sampling when raw probabilities
are required. `populate_token_probs` uses `get_token_probabilities` on that
path, rather than the sampler candidates.

`server-common.cpp:1473` reads `llama_get_logits_ith(ctx, idx)`, computes
softmax across all vocabulary entries, and returns the requested leading
entries. The partial sort selects the leading entries; it does not restrict
the softmax denominator. Without backend sampling,
`llama_context::get_sampled_logits_count` returns vocabulary size. Grammar
and temperature operate on a separate copy in `common/sampling.cpp`, leaving
the target logits used for the probability table unchanged.

`common_sampler_sample_and_accept_n` samples token `i` at `idxs[i]` and
accepts drafts only while they agree with that sampled token. The patch reads
the matching index for each returned token. Although the table is filled
*after* the sampling function returns, it describes the distribution
**before that token was chosen**, conditioned on its preceding sequence. It
is not a post-grammar or post-greedy one-hot distribution. Greedy temperature
0 controls the generated sequence; it does not make these raw tables one-hot.

`read_multi` finds the token completing each `Qk: <id>` line and requires
that stripping that token leaves only the answer ID. It rejects merged
prefix/answer tokens, incomplete output, wrong IDs, and malformed separators.
The probability table belongs to that answer token, not the following token.
Offline checks read `[2/3, 1/3]` from an answer table with masses 0.6 and 0.3,
and rejected a merged `Q1: A` token.

`read_probabilities` strips whitespace, sums token variants matching the
question's assigned IDs, and divides each retained mass by their sum.
Lowercase tokens do not count toward uppercase IDs. A separate check combined
` A` and `A` masses into 0.5, retained 0.1 on `B`, rejected lowercase `a`,
and returned `[5/6, 1/6]` with coverage 0.6.

Coverage is the sum of the **returned raw vocabulary mass matching this
question's IDs**. It is not model accuracy or proof that every option is
represented. Coverage below 0.5 raises `LowCoverageError`; exactly 0.5 passes.
The boundary checks confirmed that 0.499999 fails and 0.5 passes, even when
the returned distribution is `[1, 0]`. Any such failure replaces the entire
case's five predictions with uniform distributions, including earlier answers
or permutations that may have succeeded.

Only the top 20 vocabulary entries are available. A valid option absent from
them receives zero, even if its actual model probability is positive.
Whitespace aggregation can also include token variants beyond the exact
grammar-valid tokenization. Thus the final read is a normalized approximation
to the option distribution, not a certified full-vocabulary extraction of
every option's mass. Later answers are conditional on generated earlier
answers; they are not marginal distributions over all possible answer paths.

The current source supports a raw pre-sampling read. Reports do not retain
raw token streams or per-answer test coverage, so they cannot independently
prove that the running historical binary followed this source on every token.

## 7. Majority and uniform baselines

**Verdict: PASS for independent local baselines; CANNOT CHECK an exact match
to the card's unspecified tie and aggregation conventions.**

I define each class space by `(workflow, question)`. Pooling label indices
across unrelated questions would not be a meaningful majority classifier.

| Baseline on the 2,000 test decisions | Accuracy |
|---|---:|
| Most common test label for each question | 1,045 / 2,000 = 0.5225 |
| Most common train label for each question, evaluated on test | 0.4835 |
| Uniform random draw, expected accuracy | 0.3175 |
| Uniform probabilities, report's first-argmax tie rule | 548 / 2,000 = 0.2740 |

The first baseline uses test labels to establish a descriptive floor, not to
fit the model. The train-majority baseline is a deployable constant predictor.
Uniform random accuracy and uniform argmax accuracy are different: a vector
of equal probabilities becomes the first option under the report's rule.

The [card](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) reports
majority 0.520 on the 1,600-case reference set and test baselines Uniform
0.308 and Prior 0.470. My uniform and prior accuracies do not reproduce those
two test numbers under the local conventions. Their evaluation code and
predictions are not present, so the cause cannot be established.

The local uniform baseline has soft Brier mean 0.071910060, soft Brier sum
**0.238188166**, and KL **0.444463184**. The last two round to the card's
0.238 and 0.444. This is direct numerical evidence that its Brier column uses
the summed scale, rather than the local reported mean-over-options scale.

## 8. Jev comparability

**Verdict: FAIL for the local claim that Jev used a different 1,600-case set;
CANNOT CHECK an exact common-harness metric comparison.**

There is no full dataset card in `docs/`. `hf datasets card
LocalLLaMA/typed-decisions --text` returned the card, saved in
[`scratch/typed-decisions-card.md`](../scratch/typed-decisions-card.md).
The [card](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) says the
baseline table uses test and identifies Jev 1.13.0 as an API measurement on
2026-09-18 covering all 400 cases and 2,000 decisions, with zero errors.
Thus its stated split is the same nominal test split as this audit, not the
1,600-case reference set. The contrary annotations in `bench.py`,
`phase1-results.md`, and `phase2-results.md` conflate the reference statistics
with Jev's test result.

The reported accuracy difference is **0.7430 - 0.727 = 0.0160**. I cannot
verify Jev's exact case hashes, predictions, tie handling, ECE binning or
aggregation, Brier implementation, or score conversion from the card alone.
The local cached dataset ref is
`ea9306458d6e9563628369a3d1e72e362fb381d2`. The separate Hub metadata request
failed with DNS resolution errors, and the web reader could not open the card;
I therefore cannot claim a fresh remote revision verification.

Comparing local Brier 0.0371 directly with Jev 0.148 is invalid. On the scale
supported by the uniform control, local soft Brier is **0.125030519**. This
suggests a smaller advantage, subject to reproducing Jev's metric code.
Likewise, pooled local ECE 0.0297 cannot establish a roughly fivefold
calibration advantage over Jev 0.144. Local question-averaged ECE is 0.1201;
Jev's aggregation rule remains unverified.

The backbone receives no fine-tuning here, but the reported probabilities are
calibrated using labels from these same workflows. Calling the complete
evaluation "zero shot, no training" omits that supervised calibration step.
Its accuracy is unchanged by calibration, so this does not explain the
accuracy difference. Gold is teacher agreement, not independently verified
real-world correctness. No statistical superiority claim over Jev can be
tested without its per-case predictions.

## 9. Usage accounting, order effects, and other concerns

**Verdict: FAIL for several explanatory claims and usage labels; PASS for
reproducing the saved aggregates; CANNOT CHECK unrecorded run conditions.**

| Run | Recorded calls | Calls/case | Reported forward passes | Mean latency ms | Median ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Phase 1 | 6,000 | 15.000 | 6,000 | 2,288.502372 | 2,418.018374 | 2,864.637714 |
| Phase 2, three | 1,194 | 2.985 | 31,044 | 1,380.031017 | 1,374.987014 | 1,530.660165 |
| Phase 2, one | 400 | 1.000 | 10,400 | 507.510368 | 510.479391 | 585.111519 |

These latency values reproduce from 400 unique per-case usage entries in each
report. Recorded prompt tokens total 2,437,620, 851,580, and 285,288,
respectively. Summed latencies are 915,400.948646 ms, 552,012.406928 ms, and
203,004.147326 ms. They are client-measured inference elapsed times, not
training-inclusive wall times or server profiling data.

**Calls are undercounted on failures.** Multi has 398 entries with three calls
and two failed entries with zero. That gives 1,194/400 = 2.985, printed as
2.98. Each coverage failure required at least one completion, so the actual
benchmark completion count is between 1,196 and 1,200, or 2.990 to 3.000 per
case. Setup calls are additional: alphabet tokenization and the initial
multi probe are not included. Failed case latencies of 1,437.692850 and
1,440.817766 ms are included despite zero token and call counters.

**Forward passes is a misleading cross-mode label.** Single mode counts
completion calls. Multi counts returned generated tokens, 78 per successful
three-permutation case and 26 per one-permutation case. With speculative
decoding, returned token count is not a count of target-model forward calls.
The patched sampler changes how tokens are verified in batches. These fields
cannot establish computational work or hardware efficiency.

**The one-permutation loss has a concrete zero-probability problem.** It has
288 zero entries, including 187 decision rows with positive gold mass on a
zero prediction. No hard gold label has probability zero. The 1e-300 log
floor contributes **2.993314569** of the total **4.099882614** soft log loss.
Phase 1 has 28 zero entries, 17 such rows, and a 0.037703917 contribution.
Three-permutation multi has one zero entry, only where gold is also zero.
This explains why one-permutation accuracy can remain good while its soft
cross entropy is poor.

The one-permutation run reuses three-permutation temperatures. A new fit might
improve positive probabilities, but **temperature scaling preserves zeros**.
Therefore the claim in `phase2-results.md` that a one-call temperature fit
would fix this loss is unsupported and cannot fix the dominant zero term.
The raw one-permutation records are not saved separately; its raw headline
and exact missing-token histories cannot be checked.

**Three shuffles do not balance question order.** The fixed seed yields
zero-based index orders `[3,1,2,4,0]`, `[3,2,0,4,1]`, and `[3,1,2,0,4]`.
Question index 3 is always first. Option shuffles use the same seed per
question; score label order stays fixed while assigned IDs rotate. Later
answer distributions depend on earlier generated answers. The one-permutation
run also shuffles once, so calling it "no shuffles" is inaccurate. Its 0.0060
accuracy gap from three permutations is not an isolated estimate of question
order effects: averaging, option IDs, and failure outcomes also change.
Per-permutation test predictions and multiple order seeds were not saved.

**The speed extrapolation is unmeasured.** These cases contain 16, 17, 18, or
20 total options, with 100 cases at each size. All fit the 36-character offline
alphabet and successful recorded cases use one call per permutation. Larger
schemas can be split into several groups by `_multi`. No 20-question timing
is present, so the results note's 5-to-10-times speed claim at that size cannot
be verified. Cloud Jev latency and local GPU latency are also different
deployment conditions.

**Provenance remains incomplete.** The audit began at project commit
`7fa18ab`; an external commit `9cd0728` added results during the audit. I made
no commit. The audited report, calibration, cache, and dataset hashes remained
unchanged. No duplicate IDs, dropped test cases, or metric-code change was
found. These records do not prove the cause of the reported server crash or
exclude unrecorded prompt tuning; neither claim is needed to reproduce the
scores.

Validation commands, run without inference:

```bash
UV_CACHE_DIR=/tmp/system1-uv-cache uv run python scratch/audit_phase2.py
UV_CACHE_DIR=/tmp/system1-uv-cache uv run python scratch/audit_paths.py
UV_CACHE_DIR=/tmp/system1-uv-cache uv run ruff check scratch/audit_phase2.py scratch/audit_paths.py
UV_CACHE_DIR=/tmp/system1-uv-cache uv run pytest -q
```

The audit scripts complete and write the linked evidence. Final lint output:
`All checks passed!`. The existing offline suite reports
`121 passed in 0.71s`. An initial audit-helper run failed because its option
counter assumed every noul had criteria; that scratch-only helper was fixed
to handle absent criteria. Initial scratch lint findings were also fixed.
Neither failure changed application code or the numerical metric results.

Artifact SHA-256 values:

| File | SHA-256 |
|---|---|
| report_multi.json | 60f22bc97c64ae4fb0e73922ee65e460a62176656b8b6e43643b675510a5b7a9 |
| report_full.json | f46fa193a669ec35770cba19d1db4e2084fb937f871ee05451c720c55f854fdc |
| report_multi_p1.json | c0214d536488ee6cc977f3308f7ae95d8c8f7ea3d49286ec54ea8e765f103b21 |
| calibration_multi.json | d794371791a5368be508e653907e8a08a5b9fb9c43f32ddc6565b5cbb250568c |
| data/typed_decisions_test.jsonl | 9b9fec7a982ba5638888923cce6a5171510f58e96670286d57565872479780cd |
| data/typed_decisions_train.jsonl | bd13e1b3963878289acff07f2b51d1abbe1a69130f187f3344a7e3c0ef4f56d9 |
| data/preds_train_multi.jsonl | 4276f305b64fc996e385403ab405616f6db0c758f850fc673f28d1d8620ca696 |

## Final verdict

PASS with caveats: the phase 2 headline scores are legitimate under the implemented definitions, with no gold leakage found, but pooled ECE, Brier scaling, truncated probability reads, and incomplete provenance prevent an unqualified claim of superiority over Jev.

## Changelog

- 2026-09-20 20:04 CEST: Created the audit from independent record calculations,
  gold matching, train-cache reconstruction, offline data-boundary checks,
  dataset-card inspection, and read-only review of both llama.cpp patches.
