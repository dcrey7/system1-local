# Phase 1 spec: a Jev shaped System One server on local Gemma 4

Read AGENTS.md first. Build a small Python package named `system1` in this folder with `uv` (pyproject.toml, `ruff`, `pytest`). Python 3.12. Dependencies: `fastapi`, `uvicorn`, `httpx`, `numpy`, `typer` (CLI), `scipy` only if you need it for the temperature fit (a grid search is fine). Keep the whole package under about 800 lines. Simple code, short functions, type hints.

## What it does

`system_one(state, questions) -> answers`. No text is generated. For every question the model does one forward pass and we read the probability of each allowed answer from the next token distribution.

Backend: llama.cpp server at `http://127.0.0.1:8010/v1` (OpenAI compatible). Model id `gemma-4-12b-qat`. Request: `POST /v1/chat/completions` with `max_tokens: 1`, `temperature: 0`, `logprobs: true`, `top_logprobs: 20`, and `chat_template_kwargs: {"enable_thinking": false}`. The response has `choices[0].logprobs.content[0].top_logprobs`, a list of `{token, logprob}`. The server runs with one slot, so send requests sequentially. Do not start or stop the server. A `FakeBackend` for tests returns canned top_logprobs.

## Request and response shapes (copy TypeSafe's public shape)

Request:
```json
{"state": "string or JSON object or array",
 "questions": {
   "route":   {"type": "choice", "instructions": "Which team handles this?", "criteria": {"billing": "money questions", "tech": "bugs and outages", "sales": null}},
   "urgency": {"type": "score",  "instructions": "How urgent is it?", "levels": ["low", "medium", "high", "critical"]},
   "angry":   {"type": "noul",   "instructions": "The customer is threatening to cancel."}
 }}
```
Response:
```json
{"model": "system1-gemma-4-12b", "answers": {
   "route":   {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.91, "tech": 0.06, "sales": 0.03}, "confidence": 0.865},
   "urgency": {"type": "score",  "score": "high", "probabilities": {"low": 0.02, "medium": 0.2, "high": 0.7, "critical": 0.08}, "confidence": 0.6, "expected_index": 1.84},
   "angry":   {"type": "noul",   "noul": 0.77}},
 "usage": {"input_tokens": 1234, "latency_ms": 812, "forward_passes": 9}}
```
Confidence = `(n * top - 1) / (n - 1)` for n options (1.0 when n is 1). Document that this is a concentration measure, not calibration.

## Prompt

One prompt per question. Format:
```
State:
<state as text, or JSON with indent 2>

Question: <instructions>
Options:
<id>. <label>: <description if any>
...
Answer with the option id only.
```
For noul the options are `yes` and `no` and the question is "Is this statement true? <instructions>".

Option ids: a fixed alphabet of single token strings. On startup (or lazily) verify with the server's `POST /tokenize` endpoint (`{"content": "A"}`) which of `A..Z`, `a..z`, `0..9` are one token; keep only those. If a question has more options than the alphabet, run a tournament: split into groups of alphabet size, pick the top of each group, run a final round on the winners, and combine probabilities by multiplying the group probability by the final probability. Cap at 255 options and raise on more.

Reading probabilities: exp(logprob) for tokens that match an option id after `strip()`. Report `coverage` = sum of that mass before renormalising. Renormalise over the options. If coverage is below 0.5 raise `LowCoverageError` with the top tokens in the message. Never silently return zeros.

Order bias: evaluate `permutations` (default 3) different shuffles of the option order with a fixed seed and average the probabilities. Score questions keep their level order in the prompt but still rotate which ids are used. Expose `permutations=1` for speed tests.

## Calibration module `system1/calibrate.py`

- `fit_temperature(logits_or_probs, labels) -> T` per question type by minimising negative log likelihood over a grid (T in 0.2 to 10, log spaced, then refine). Apply as `softmax(log(p) / T)`.
- `ece(probs, labels, bins=10)`, `brier(probs, labels)`, `log_loss(probs, labels)`.
- Save and load `calibration.json` with one T per question type. The decide path applies it when present.

## Benchmark runner `system1/bench.py`

Input JSONL, one case per line: `{"id": "...", "workflow": "...", "state": ..., "questions": {...same shape as the request...}, "answers": {"route": "billing", "urgency": "high", "angry": true}}`. Runs every case, reports accuracy overall, per workflow, per question type, ECE and Brier per type, and median and p95 latency per decision. `--fit-temperature train.jsonl` fits T on a train file first. Writes a JSON report and prints a table.

## CLI (`uv run system1 ...`)

- `ask --state "..." --choice "route:billing,tech,sales" --noul "angry:The customer is threatening to cancel."` prints the answer JSON.
- `serve --port 8020` runs the FastAPI app with `POST /v1/systemone` and `GET /v1/models`.
- `bench cases.jsonl [--fit-temperature train.jsonl] [--permutations 3] [--out report.json]`.

## Tests (pytest, no network)

- prompt building for the three types, id assignment, tournament grouping at 70 options.
- probability reading: strip and case handling, coverage, LowCoverageError.
- permutation averaging returns the same answer in a different order.
- confidence formula at n = 2, 3, 5 and n = 1.
- temperature fit recovers T on synthetic data, ECE of a perfect predictor is 0.
- FastAPI endpoint round trip with the FakeBackend.

## Done means

`uv run ruff check .` clean, `uv run pytest -q` green, and `uv run system1 ask` works against the live server (you may call it, it is up). Write `docs/phase1-notes.md` with what you built, what you could not verify, and the exact commands.
