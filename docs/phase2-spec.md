---
title: Phase 2 spec, all answers in one call on Gemma 4 12B
date: 2026-09-20 18:34 CEST
author: Claude (spec), Codex gpt-6-astra (code)
type: spec
status: ready for build
---

# Phase 2 spec: all answers in one call

Goal: answer every question of a request in one server call instead of one call per question per permutation, and measure what it costs in accuracy against phase 1.

Phase 1 numbers to beat or match on `data/typed_decisions_test.jsonl` (400 cases, 2,000 decisions): accuracy 0.7065, ECE 0.032 after temperature, 2.4 s per case with 15 calls.

## The mechanism

One prompt holds the state and all the questions. A grammar forces the reply to be exactly one line per question, `Q1: <id>`, nothing else. The server returns the top 20 token probabilities at every generated token, so the token that holds each answer id carries a full distribution over that question's option ids. Read it the same way phase 1 reads its single token.

```
prompt:   State ... Questions: Q1 ... Q2 ... Q3 ...   (one call)
reply:    Q1: C\nQ2: A\nQ3: B\n                       (grammar forced)
read:         ^      ^      ^   top 20 at each id token -> probabilities per question
```

Live proof on the running server, 20 Sep 2026: 5 questions answered in 190 to 500 ms in one call, script `/home/abhishek/.claude/jobs/872a19fc/tmp/multi_read4.py`.

## Server requirement

The server must return `top_logprobs` for every generated token, not only the first. Our llama-server build with the MTP draft dropped them after the first token (upstream `tools/server/server-context.cpp`, comment `// TODO: set result.probs`). A patched build (llama.cpp branch `server-spec-probs` at `/home/abhishek/llama-vulkan/llama.cpp-fresh`) is installed and serving on 8010 since 20 Sep 2026 18:40; every generated token now carries `top_logprobs`, MTP draft kept. The backend must check this at start (see `probe` below) and fail with a clear message instead of returning probabilities of 1.0.

The server stays external on port 8010 as in phase 1. Do not start or stop it. Use `chat_template_kwargs: {"enable_thinking": false}`.

## Deliverables

1. `backend.py`
   - New `TopToken` stays. Add `GeneratedToken = TypedDict(token: str, top_logprobs: list[TopToken])`.
   - `Backend` protocol gains `complete_multi(prompt: str, grammar: str, max_tokens: int) -> tuple[list[GeneratedToken], int]` (tokens with their top 20, prompt token count).
   - `GemmaBackend.complete_multi`: `chat/completions` with `max_tokens`, `temperature 0`, `logprobs true`, `top_logprobs 20`, `grammar`, `enable_thinking false`, under `SERVER_LOCK`. Return `choices[0].logprobs.content` mapped to `GeneratedToken`.
   - `GemmaBackend.probe()`: one call with `max_tokens 4` and no grammar on a short prompt; raise `RuntimeError("server returns no probabilities after the first token; use the patched llama-server build or run without the MTP draft")` if any generated token has an empty `top_logprobs`. Call it once, lazily, before the first `complete_multi`.
   - `FakeBackend.complete_multi`: takes a mapping from question name to a distribution over ids, emits the token stream `Q1`, `:`, ` <id>`, `\n`, ... with the right `top_logprobs` on the id tokens, so the parser is tested for real.

2. `core.py`
   - `build_multi_prompt(state, questions: dict[str, QuestionBase], assigned: dict[str, dict[str, str]], order: list[str]) -> str`. Layout:
     ```
     State:
     <state text or JSON>

     Questions:
     Q1: <instructions>  (Noul: "Is this statement true? " + instructions)
       A. <label>: <description>
       B. ...
     Q2: ...

     Answer every question with its option id only, one per line, in the form "Q1: <id>".
     ```
     `order` lists the question names in prompt order; `Qk` numbering follows `order`. Keep the id assignment rules of phase 1: per question, shuffle the option order for choice and noul, rotate the ids for score, ids come from `backend.alphabet`.
   - `build_grammar(order_ids: list[list[str]]) -> str`: GBNF, `root ::= q0 q1 ...`, `qk ::= "Qk: " ("A" | "B" | ...) "\n"`. Escape nothing fancy; ids are single alphanumerics.
   - `read_multi(tokens: list[GeneratedToken], ids_per_question: list[list[str]]) -> list[tuple[list[float], float]]`: walk the tokens, keep the running text, and when the running text completes `Q<k>: <id>` for the next expected k, the token that completed it is the answer token; feed its `top_logprobs` and that question's ids to `read_probabilities`. Must handle a leading space on the id token (`" C"`) and merged tokens such as `"Q1"` or `": C"`. If the id ends up inside a merged token whose `top_logprobs` are for the merged text, raise `LowCoverageError` with the token shown; do not guess.
   - `SystemOne.decide(request, calibrated=True, mode="single" | "multi")`. In `multi`: for each permutation, shuffle the question order too (`random.Random(42)`), build ids, prompt, grammar, call `complete_multi` with `max_tokens = 8 * len(questions) + 8`, read, and accumulate per question exactly like the single mode (`totals`, `coverages`). Temperature, `confidence`, `expected_index`, `noul` output stay identical. `usage` gets `calls` (1 per permutation) next to `forward_passes` (keep the phase 1 meaning for single mode; in multi mode set `forward_passes` to the count of generated tokens).
   - `system_one(..., mode="single")` passes it through.

3. `calibrate.py`: no change, except `load_calibration` and `save_calibration` take the path they already take. Use `calibration_multi.json` for multi mode fits so `calibration.json` from phase 1 stays untouched.

4. `bench.py` and `cli.py`
   - `bench` gains `--mode single|multi` (default `single`) and passes it to `decide`. The prediction cache key must include the mode and the permutations. Default cache file for multi is `data/preds_train_multi.jsonl`.
   - `--fit-temperature` in multi mode writes `calibration_multi.json`.
   - `summarize` reports mean `calls` and mean `latency_ms` per case as it does now for `forward_passes`.
   - `ask` and the API (`Request.mode`, default `single`) accept the mode.

5. Tests (pytest, no server):
   - prompt layout and numbering follow `order`;
   - grammar string for two questions with 3 and 2 ids;
   - `read_multi` on a FakeBackend stream: leading space ids, a merged `"Q1"` token, a merged `": C"` token raises `LowCoverageError`;
   - `decide(mode="multi")` averages over permutations and matches `decide(mode="single")` on the FakeBackend when the fake returns the same distributions;
   - `probe` raises on empty `top_logprobs`.

## Measure (Claude runs these, the sandbox has no sockets)

```
uv run system1 bench data/typed_decisions_test.jsonl --mode multi --permutations 3 --fit-temperature data/typed_decisions_train.jsonl --out report_multi.json
uv run system1 bench data/typed_decisions_test.jsonl --mode multi --permutations 1 --out report_multi_p1.json
```

Report next to phase 1: accuracy, ECE before and after temperature, soft Brier, score MAE, latency per case, calls per case. Note per type (choice, score, noul) as `report_table` already does.

## Open questions to answer in `docs/phase2-notes.md`

- Does answering all questions in one reply change accuracy against one question per call? Which types move?
- Does the question order matter? Compare permutations 1 against 3.
- Latency per case at 5 questions, and an estimate for 20 questions.

## Not chosen: DiffusionGemma 26B A4B

It exists in one size only (26B A4B, 9 Jun 2026). It would give a true single forward pass for all answers and 3.8B active weights, but its int4 weights take 17.2 GB and its KV cache 40 KB per token, so 262k does not fit on the 24 GB card, and it needs a second runtime (vLLM 0.29.0 or transformers 5.17.0). The one call mode above gives the same interface on the model we already run. Weights are on disk at `/home/abhishek/models/diffusiongemma-awq-int4/weights` if we ever return to it.

## Revision 2 (2026-09-20 19:01 CEST): unique ids per call, and a benchmark that does not die

First full run died at training case 439 (`customer_service`): a two option noul (`needs_human`, ids A and B) sat between five and four option questions, and the raw distribution at its answer token put 0.71 on ` C`, a letter that belongs to a neighbour question. Coverage on its own ids was 0.29, so `read_probabilities` raised and the whole 40 minute run aborted. Every case in this benchmark mixes option counts. No case needs more than 20 ids in total, the alphabet has 36.

Changes, all in multi mode unless stated:

1. **Unique ids across the call.** Assign ids from `backend.alphabet` consecutively in prompt order: Q1 takes the first n1 ids, Q2 the next n2, and so on, so one letter means one question. Choice and noul: shuffle the option order inside the question's own block per permutation (same seeded rules as today). Score: keep the level order and rotate the ids inside the block by the permutation offset (today's rotation, now block local). The grammar lists each line's own ids as before.
2. **Grouping when the alphabet is too small.** If the total number of options exceeds `len(alphabet)`, split the questions (in prompt order) into the fewest groups that each fit, one call per group per permutation. `usage["calls"]` counts every call. Remove the current "use single mode for larger questions" error, except for a single question whose options alone exceed the alphabet: that still raises the same error as today.
3. **Format version in the cache key.** Add `MULTI_FORMAT = 2` in `core.py` and put `"format": MULTI_FORMAT` in the multi fingerprint in `bench.run_cases`, so a prompt format change can never reuse stale cached predictions. Single mode fingerprints stay byte for byte as phase 1.
4. **A failing case must not abort the benchmark (both modes).** In `run_cases`, catch `LowCoverageError` per case: print one warning line to stderr with the file, line number and the message, give every decision of that case a uniform distribution over its options (so it counts as maximally unsure and usually wrong), add the case to `report["failures"] = {"count": n, "lines": [...]}`, and continue. `summarize` and `report_table` show the failure count. Server errors (`httpx.HTTPError`, `RuntimeError` from the probe) still abort.
5. **Old cached usages.** `summarize` must accept usages without a `calls` key: `usage.get("calls", usage["forward_passes"])`.

Tests to add: unique id blocks across three questions (2, 5, 4 options); score rotation stays inside its block; grouping with a 6 letter fake alphabet and questions of 3, 3, 2 options gives two calls; the multi fingerprint contains `"format": 2` and the single one has no format key; a FakeBackend that raises `LowCoverageError` on one case yields uniform records, a failure count of 1, and the run continues; `summarize` on a usage without `calls`.

## Revision 3 (2026-09-20 20:32 CEST): reseeded shuffles and a probability floor, from the audit

Two findings in `docs/phase2-audit.md` section 9.

1. **Reseed the question order per shuffle.** Today `_multi` uses `random.Random(42)` once, and the three default shuffles put the same question first (orders `[3,1,2,4,0]`, `[3,2,0,4,1]`, `[3,1,2,0,4]`). Change: for permutation `k` build the question order with `random.Random(1000 + k)` and, after the shuffle, rotate the list so that the question at position `k mod n` comes first. Option orders keep their current rules. Add a test that for a 5 question request the first question differs across the three default permutations, and that permutations 0, 1, 2 give three different orders.
2. **Floor for options missing from the top 20.** The server returns 20 tokens. A valid option id absent from that list gets probability zero today, which is why the one shuffle run has 288 zero entries and a soft log loss of 4.1. Change `read_probabilities(top_tokens, ids, floor=False)`: when `floor` is true, every id that received no mass gets the probability of the last (smallest) returned token, capped at 0.01, before normalisation; coverage is still computed from the real mass only. Multi mode calls it with `floor=True`; single mode keeps `floor=False` so phase 1 stays byte for byte. Test: an id missing from the list gets the 20th token's probability; the cap applies; coverage unchanged; single mode unaffected.
3. Bump `MULTI_FORMAT` to 3 so the multi cache is not reused.

Run after the change (Claude): the full multi benchmark with 3 shuffles and fit, then 1 shuffle, and compare with revision 2 in `docs/phase2-results.md`.

## Revision 4 (2026-09-20 21:30 CEST): keep the caller's question order

Measured in `docs/phase2-results.md`, revision 3 section: the reseeded question orders of revision 3 cost 2 points with 3 shuffles (0.743 to 0.723) and 5 points with 1 call (0.737 to 0.687), all of it through question position. `urgency` answered before the other questions loses 9 points; `matches_order` answered after `disposition` loses 27 points. In one call the model reads its own earlier answer lines, so facts must come before judgments and the summary judgment last. The caller knows those dependencies. A random order does not.

Changes, multi mode only:

1. `_multi` uses the caller's order for every permutation: `order = list(request.questions)`. Remove the question shuffle and the rotation (the `random.Random(1000 + permutation)` line and the `offset` rotation of `order`). The option rules stay exactly as they are: choice and noul shuffle their option order per permutation with the per question `random.Random(42)`, score rotates its ids inside its block by the permutation offset. Grouping when the alphabet runs out stays; groups follow the given order.
2. `MULTI_FORMAT = 4`.
3. The probability floor from revision 3 stays.
4. Tests: replace the revision 3 order test with one that checks, on a 5 question FakeBackend request with 3 permutations, that every call's prompt lists the questions in the given order (Q1 is the first question given, Q5 the last) while the option order of a choice question differs between at least two permutations. The multi fingerprint contains `"format": 4`. Single mode fingerprints stay unchanged.
5. `docs/phase2-notes.md`: add a short revision 4 note.

Run after the change (Claude): the full benchmark, 3 shuffles with fit, then 1 call; compare with revisions 2 and 3 in `docs/phase2-results.md`.

## Revision 5 (2026-09-20 22:28 CEST): a lower coverage guard in multi mode

The revision 4 run flagged 21 of 1,600 cases as low coverage failures (17 train, 4 test) and gave them uniform guesses. In every one of them the answer token still put most of its valid mass on one id of the question (for example ` P` 0.59 to 0.81 on the `needs_human` line, ` E` 0.45 to 0.61 on the `needs_review` line), and the missing mass sat on the ids of a related question (the `action` letters ` C`, ` D`, ` E` on the `needs_human` line; ` A` on the `needs_review` line). The 0.5 guard comes from phase 1, where the one question in the prompt used ids A to E and low coverage meant a broken read. In multi mode, with unique ids across the call, mass on another question's ids is semantic leakage, and the valid mass still ranks the options.

Changes, multi mode only:

1. `read_probabilities(top_tokens, ids, floor=False, min_coverage=0.5)`: raise `LowCoverageError` when the coverage is below `min_coverage`. `read_multi` passes `min_coverage=0.05` together with `floor=True`. Single mode keeps 0.5 and stays byte for byte.
2. `MULTI_FORMAT = 5`.
3. The `coverage` field in the output already reports the real mass on the question's ids. No change there.
4. Tests: a multi read with coverage 0.3 returns the normalised distribution over the question's own ids and reports coverage 0.3; coverage 0.02 still raises; single mode with coverage 0.3 still raises; the multi fingerprint contains `"format": 5`; formats 2, 3 and 4 are rejected.
5. `docs/phase2-notes.md`: add a short revision 5 note.

Run after the change (Claude): the full benchmark, 3 shuffles with fit, then 1 call; compare with revision 4 in `docs/phase2-results.md`.

## Changelog

- 2026-09-20 18:34 CEST: Rewrote the spec for the one call mode on Gemma 4 12B; DiffusionGemma moved to "not chosen".
- 2026-09-20 19:01 CEST: Revision 2 after the first full run died at case 439: unique ids per call, grouping, format version in the cache key, failing cases counted instead of aborting.
- 2026-09-20 20:32 CEST: Revision 3: reseeded question order per shuffle, probability floor for ids missing from the top 20 (multi mode only), MULTI_FORMAT 3.
- 2026-09-20 21:30 CEST: Revision 4: keep the caller's question order, no question shuffle, MULTI_FORMAT 4.
- 2026-09-20 22:28 CEST: Revision 5: multi mode coverage guard 0.05 instead of 0.5, MULTI_FORMAT 5.
