# Phase 2 implementation notes

Recorded: 2026-09-20 18:45 CEST. Time comes from the shell.

## Built

- Added `GeneratedToken`, `Backend.complete_multi`, and the Gemma multi request.
  Requests include the grammar, top token probabilities, and disabled thinking.
  The server lock covers the probe and each completion request.
- Added a lazy probe before the first multi completion. It checks every returned
  token for probabilities. Empty probabilities raise the error from the spec.
  A response with fewer than two tokens also fails because it cannot prove that
  probabilities exist after the first token. Concurrent calls share one
  successful probe per backend instance.
- Added multi prompt construction, numbered GBNF rules, and answer parsing.
- Added one completion per permutation. Question order uses a seeded random
  generator. Each question keeps its own seeded option generator, matching the
  single mode assignments. Choice and noul options shuffle. Score labels keep
  their order while ids rotate. The final answer calculation is shared.
- Added `mode` to `decide`, `system_one`, API requests, `ask`, and `bench`.
  The default remains `single`. API probe failures return HTTP 502. CLI probe
  failures print the error and exit with failure.
- Added `usage.calls`. Single mode still counts completion calls as forward
  passes. Multi mode counts returned generated tokens as forward passes.
  Both modes count prompt tokens. The initial probe and alphabet checks are
  setup calls, excluded from usage counts. Their time remains in first-call
  latency.
- Added mean latency and calls per case to benchmark reports. Cache fingerprints
  include mode and permutations. Multi training uses
  `data/preds_train_multi.jsonl` by default.
- Added `calibration_multi.json` with neutral temperatures of 1.0. These values
  are defaults, not a fitted result. Multi fits write this separate file.
  A custom single calibration path uses a multi file in the same directory;
  `multi_calibration_path` can override it. `calibrate.py` needs no change.
- Added offline tests for prompts, grammar, token boundaries, coverage failures,
  permutation averages, probe behavior, request locking, calibration isolation,
  cache isolation, and CLI/API mode routing.

The single prompt, completion payload, option assignment, tournament behavior,
probability reader, and answer calculations retain their phase 1 behavior.
The existing tests and `calibration.json` are unchanged. The requested usage and
report fields are additive changes to the output. Cache keys now include mode,
so old training cache entries do not match the new keys.

## How read_multi finds answer tokens

The parser appends each generated token to a running text string. An offset marks
the next expected line. It waits until that line contains `Qk: <id>`, checks the
question number and allowed id, then reads probabilities from the token that
completed the id. It passes that token's `top_logprobs` to `read_probabilities`.
This preserves phase 1 normalization, coverage checks, and whitespace handling.

Prefix tokens can be split or merged. For example, `Q1` is safe because it
contains no answer. An id token such as ` C` is safe because stripping whitespace
leaves only the id. A token such as `: C` contains other text. Its probabilities
describe that merged text, so the parser raises `LowCoverageError` and shows the
token. It also rejects missing answers, invalid ids, wrong numbering, missing
newlines, extra text, and insufficient probability coverage.

`FakeBackend.multi_responses` maps question names to id distributions. Prompt
instructions omit request names, so `question_names` maps the full prompt
instruction to the request name when needed. Tests can also key distributions by
the instruction itself or by `Q1`, `Q2`, and so on. The fake emits actual numbered
token streams with probabilities on the id tokens.

## Decisions and open questions

The spec does not define a multi tournament. Multi mode rejects any question
with more options than available ids before completion. Single mode retains its
existing tournament support. The parser requires the final newline specified by
the grammar, so a truncated response cannot appear complete.

Live accuracy, per-type changes, and question order effects remain unknown.
Compare the specified runs with one and three permutations to answer these
questions. No live server or benchmark was run for this implementation.

Latency for five questions is not measured here. There is no measured estimate
for twenty questions. A rough projection could multiply a measured five-question
latency by four, but that assumes linear scaling and ignores shared prompt work,
fixed request costs, and decode effects. The token budget is `8 * questions + 8`;
it is a limit, not a latency estimate.

## Validation

The default uv cache is read-only in this sandbox. Validation uses the writable
cache `/tmp/system1-uv-cache`. No dependencies were added.

```bash
export UV_CACHE_DIR=/tmp/system1-uv-cache
uv run ruff format . && uv run ruff check . && uv run pytest -q
```

Validation results recorded at the time above:

- First offline test pass: `100 passed in 0.70s`.
- Final format and lint checks passed.
- Final pytest summary: `108 passed in 0.73s`.

All tests use fake responses or in-process HTTP transports. The network guard
remains active. No server was started or stopped. No commit was made.

Revision 2 validation, 2026-09-20 19:05 CEST: `uv run ruff format .`,
`uv run ruff check .`, and `uv run pytest -q` passed with
`UV_CACHE_DIR=/tmp/system1-uv-cache`. Pytest reported `121 passed in 0.75s`.
The single mode tests and cache fingerprint check pass unchanged.

Low coverage failures produce uniform records and a warning with the source
line number. Reports count these failed cases, and failed predictions are not
cached. Failed decisions return no usage counters, so their call and token
counts are recorded as zero; their elapsed time is recorded. Server errors
still stop the run. No live server or benchmark was run.

## Changelog

- 2026-09-20 18:49 CEST: Restored phase 1 cache fingerprints for single mode by
  omitting `mode`. Multi fingerprints keep `"mode": "multi"`. Added two regression tests.
  Validation: format and lint passed; `110 passed in 0.69s`.
- 2026-09-20 19:05 CEST: Multi calls assign consecutive, separate id blocks in prompt order and start a new call when the next question does not fit the alphabet. Choice and noul options shuffle within their blocks, while score labels keep their order and their ids rotate only within their own block.
- 2026-09-20 20:21 CEST: Added live integration tests for the four-token probe, five-question multi decisions, single and multi argmax agreement, and grammar output. Registered the `live` marker and added `just test-live`. The health check uses a two-second timeout and skips the module when the server is unavailable. Each live test has a nine-second deadline. Validation with `UV_CACHE_DIR=/tmp/system1-uv-cache`: format and lint passed; `121 passed, 1 skipped in 0.71s`. Only the live module skip path ran because the server is unreachable in the sandbox; live assertions and runtime remain unverified.
