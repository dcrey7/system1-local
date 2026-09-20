# Phase 1b spec: the typed decisions benchmark

Read AGENTS.md and docs/phase1-spec.md first. Add the benchmark loader and make `bench` score it properly.

## The dataset

`LocalLLaMA/typed-decisions` on the Hugging Face Hub (Apache 2.0, about 1 MB). Load with the `datasets` library: `load_dataset("LocalLLaMA/typed-decisions", "all", split="train")` (1,200 cases) and `split="test"` (400 cases). Each case has 5 questions over one shared state. Columns: `id`, `workflow`, `split`, `state` (JSON string), `questions` (JSON string), `gold` (JSON string), `factors`, `label_agreement`, `n_questions`.

`state` and `questions` are exactly a `POST /v1/systemone` body. Example question set:
```json
{"action":   {"type": "choice", "instructions": "What should the observability system do with this trace?",
              "criteria": {"continue": "Let the agent proceed without interruption.", "human_review": "Queue this trace for a human to review.", "observe": "Keep running, but flag the trace for later sampling.", "stop": "Halt the agent now."}},
 "needs_review": {"type": "noul", "instructions": "This trace requires human review.",
              "criteria": {"false": "No human attention is warranted.", "true": "A human should inspect this run."}},
 "risk":     {"type": "score", "instructions": "...", "criteria": {"0": "...", "1": "...", "2": "...", "3": "..."}}}
```
Notes: every option carries a description in `criteria`, including noul (`true`, `false`) and score levels (keys are level ids as strings, order them numerically). The descriptions are part of the input, keep them in the prompt. For noul use the two descriptions as the yes and no options.

`gold` per question:
```json
{"action": {"type": "choice", "label": "continue", "confidence": 0.444, "probabilities": {"continue": 0.583, "human_review": 0.29, "observe": 0.09, "stop": 0.037}},
 "needs_review": {"type": "noul", "label": "false", "noul": 0.357, "probabilities": {"false": 0.643, "true": 0.357}},
 "risk": {"type": "score", "label": "1", "score": 0.79, "probabilities": {"0": 0.3, "1": 0.617, "2": 0.077, "3": 0.007}}}
```

## Work

1. `system1/data.py`: `download_typed_decisions(out_dir: Path) -> tuple[Path, Path]` writes `data/typed_decisions_train.jsonl` and `data/typed_decisions_test.jsonl` in the bench format: `{"id", "workflow", "state" (parsed JSON), "questions" (parsed), "answers": {question: label}, "gold": {question: {"probabilities": {...}, "label": ..., "score": ..., "noul": ...}}}`. Add `datasets` as an optional dependency group `bench`. CLI: `uv run system1 fetch-typed-decisions`.
2. `system1/bench.py` scoring, per question and aggregated per workflow, per type, and overall:
   - accuracy: argmax of our probabilities against `label` (for noul: yes if our p_yes >= 0.5, compared with `label` true/false).
   - log loss and Brier score of our full distribution against the gold distribution (soft targets: cross entropy with the gold probabilities, and mean squared difference over options).
   - ECE with 10 bins using our top probability and whether the argmax matched the label.
   - for score questions also mean absolute error of our expected level against gold `score`.
   - latency median and p95 per decision, and forward passes per case.
   Print a table and write `report.json`. Reference lines to print under the table: majority baseline 0.520, factor ceiling 0.704, teacher self agreement 0.735 (from the dataset card, 1,600 case set), Jev published 0.727, Laya zero shot 0.36, Laya fine tuned 0.766.
3. `--limit N` to run the first N cases, `--workflow NAME` to filter, `--permutations K`.
4. Temperature: `--fit-temperature` fits one T per question type on the train JSONL predictions (cache predictions to `data/preds_train.jsonl` so the fit does not rerun the model), then reports test metrics before and after.
5. Tests: loader conversion on two hand written rows, soft log loss and Brier on known values, noul threshold, score expected level.

## Done means

`uv run ruff check .` clean, `uv run pytest -q` green, `uv run system1 fetch-typed-decisions` works, and `uv run system1 bench data/typed_decisions_test.jsonl --limit 5` runs against the live server and prints the table. Append to `docs/phase1-notes.md`.

## Review findings from phase 1 to fix first (Claude, 2026-09-20)

1. Schema: `Noul` must accept an optional `criteria` dict with keys `true` and `false` (descriptions). `Score` must accept either `levels: list[str]` or `criteria: dict[str, str | None]`; when `criteria` is given, order the levels numerically when every key is an integer string, else keep dict order. Keep `extra="forbid"` for unknown keys but allow an optional top level `model: str` in the request (ignored). Add tests.
2. `POST /v1/systemone` is `async def` around a blocking call. Make it a plain `def` so FastAPI runs it in a worker thread.
3. Option ids: use uppercase letters then digits only (36 ids). Drop lowercase. Matching stays exact after `strip()`.
4. `ask`: allow `--choice "route=Which team handles this?:billing,tech,sales"`; the part before `=` is the name, between `=` and `:` the instructions, after `:` the options. Same for `--score`. Keep the old `name:options` form working with the name as instructions.
5. The dataset is already on disk: `data/raw/typed_decisions/all_train.parquet` and `all_test.parquet` (the `all` config). The loader must read these with `pyarrow` or `pandas` first and only fall back to `datasets.load_dataset` when the files are missing. The sandbox has no network, so do not try to download.
6. Noul prompt for the benchmark: show the `true` description as the `yes` option text and the `false` description as the `no` option text.
