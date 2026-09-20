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

## Changelog

- 2026-09-20 18:34 CEST: Rewrote the spec for the one call mode on Gemma 4 12B; DiffusionGemma moved to "not chosen".
