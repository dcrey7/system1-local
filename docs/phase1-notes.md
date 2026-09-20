# Phase 1 notes

Recorded: 2026-09-20 15:47 CEST. Time comes from
`date "+%Y-%m-%d %H:%M %Z"`.

## Built

The Python 3.12 package is in `src/system1/`. It has 740 source lines.
Runtime and test dependencies have exact version pins in `pyproject.toml`.
`uv.lock` contains the registry URLs and artifact hashes. No files are committed.

- `system_one(state, questions)` and `SystemOne.decide` answer choice, score,
  and noul questions. Requests validate option counts and score level uniqueness.
- `GemmaBackend` checks the candidate IDs with `/tokenize` on first use.
  It caches the verified alphabet for that backend instance. HTTP calls use a
  process-wide lock. The API runs each complete decision sequentially.
- Each inference request asks for one token and its top log probabilities.
  The response contains decisions and probabilities, without generated text.
  Thinking is disabled on every inference request.
- Choice and noul options use deterministic shuffles. Canonical label order
  makes choice results, including ties, independent of input dictionary order.
  Consecutive shuffles differ when possible. Small option sets can repeat an
  order when permutations exceed the available orders. Score prompts retain
  level order and rotate the verified IDs.
- Tournaments split options by alphabet size. Each group winner represents its
  group in the final round. Every option retains its group probability,
  multiplied by its representative's final probability. Winners recurse if
  they also exceed the alphabet size. This is a hierarchical estimate of the
  distribution. Questions cannot exceed 255 options.
- Tokens match IDs after whitespace removal. Case stays significant because
  `A` and `a` can identify different options. Whitespace variants contribute
  their combined mass. Coverage below 0.5 raises `LowCoverageError` with the
  returned top tokens. Each answer reports the mean raw coverage across its
  rounds and permutations. The threshold applies to each round separately.
- Confidence is `(n * max(p) - 1) / (n - 1)`, or 1 for one option.
  **Confidence measures concentration, not calibration or correctness.**
  Scores also report a zero-based expected level index. Noul reports P(yes).
- `calibrate.py` fits temperatures on a log-spaced grid with refinement,
  applies temperature scaling, and computes top-label ECE, multiclass Brier,
  and log loss. Metrics accept questions with different option counts.
  Brier is the mean sum of squared class errors. ECE uses equal-width bins.
- `calibration.json`, when present in the current directory, maps question
  types to positive temperatures. Invalid files raise an error. Training fits
  use raw probabilities. Evaluation loads the resulting temperatures.
- `bench.py` reads JSONL cases and reports accuracy overall, per workflow,
  and per question type, with ECE and Brier. It also reports log loss and
  median/p95 latency per decision. Accuracy counts individual questions.
  `--fit-temperature` fits and saves calibration before evaluation.
- The CLI implements `ask`, `serve`, and `bench`. `ask` also accepts `--score`
  and JSON object/array states. `serve` exposes `POST /v1/systemone` and
  `GET /v1/models` on localhost. The request supports `permutations`, default 3.
  The `justfile` contains format, check, and live recipes.

## Verification

Tests block socket connections and DNS lookups. API tests use an in-memory
ASGI transport. Backend protocol tests use HTTPX MockTransport. No test needs
the model server or a network connection.

The final required checks produce:

```text
$ uv run ruff check .
All checks passed!
$ uv run pytest -q
..............................................                           [100%]
46 passed in 0.33s
$ uv lock --check --offline
Resolved 32 packages in 0.41ms
```

These checks cover prompts, ID assignment, case and whitespace handling,
coverage failures, confidence, permutation averaging and ties, score order,
tournament probability composition, calibration fit and persistence,
API round trips, backend request serialization, benchmark aggregation, and
CLI commands. The synthetic temperature test recovers a known temperature
from exact class frequencies, using both logits and probabilities.

Earlier diagnostic test runs report `40 passed in 0.22s`,
`45 passed in 0.38s`, `45 passed in 0.36s`, and `45 passed in 0.33s`.
The first API test client attempt hangs on cross-thread event-loop wakeup in
the restricted environment. The final tests use ASGITransport on one loop.

## Environment and exact commands

PyPI DNS resolution is unavailable in this sandbox. The default uv cache is
read-only. A writable cache is at `/tmp/system1-uv-cache`. The initial online
`uv add` attempt fails with a DNS error. An offline attempt also fails because
the cache lacks some required packages.

Project creation and dependency declarations use uv:

```bash
export UV_CACHE_DIR=/tmp/system1-uv-cache
uv init --package --name system1 --python 3.12 --build-backend uv --no-readme --vcs none
uv add --frozen fastapi==0.136.3 uvicorn==0.49.0 httpx==0.28.1 numpy==2.4.6 typer==0.26.7
uv add --frozen --dev ruff==0.15.16 pytest==9.0.3
```

The lockfile is recovered from the dependency entries and artifact hashes in
the existing local `pipecat_buddy/uv.lock`. Only the required dependency closure
and this project's metadata are retained. `uv lock --check --offline` validates
the result. There are no references to that other project in the resulting
lockfile. The local package itself builds and installs with uv's build backend.
The following installs only that editable package while the dependencies remain
available in the existing environment:

```bash
export UV_CACHE_DIR=/tmp/system1-uv-cache
UV_NO_SYNC=1 uv run python - <<'PY'
import subprocess
import tomllib
from pathlib import Path

command = ['uv', 'sync', '--offline', '--locked']
for package in tomllib.loads(Path('uv.lock').read_text())['package']:
    if package['name'] != 'system1':
        command.extend(['--no-install-package', package['name']])
raise SystemExit(subprocess.call(command))
PY
```

Final verification uses the existing Python 3.12 packages. Their versions match
all seven direct dependency pins above. These environment overrides avoid
downloads; they are specific to this sandbox:

```bash
export UV_CACHE_DIR=/tmp/system1-uv-cache UV_NO_SYNC=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH=/home/abhishek/Downloads/work/system1-local/src:/home/abhishek/Downloads/work/pipecat_buddy/.venv/lib/python3.12/site-packages
export PATH=/home/abhishek/Downloads/work/pipecat_buddy/.venv/bin:$PATH
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .
uv run pytest -q
uv lock --check --offline
```

A normal environment with PyPI access can use `uv sync --locked` followed by
the commands in `justfile`, without these overrides. A fresh dependency install
could not be verified in this sandbox.

## Live check

The only live model attempt uses the same environment overrides above:

```bash
uv run system1 ask --state "I was charged twice. Refund me or I will cancel." --choice "route:billing,tech,sales" --noul "angry:The customer is threatening to cancel."
```

Actual output, exit status 1:

```text
Error: [Errno 1] Operation not permitted
```

The sandbox denies the socket connection. This does not establish whether the
external server is healthy. No server is started or stopped. The live token
alphabet, coverage, answers, model latency, and real-data calibration remain
unverified. No model or dataset is downloaded.

## Phase 1b

Recorded: 2026-09-20 16:00 CEST. Time comes from
`date "+%Y-%m-%d %H:%M %Z"`.
This section supersedes the phase 1 behavior where it differs.

### Review fixes

- Noul accepts optional `true` and `false` criteria. Prompts show their
  descriptions beside `yes` and `no`.
- Score accepts levels or criteria. Integer-string criteria keys sort
  numerically. Other keys retain their input order. Descriptions remain in
  prompts. If both forms are supplied, their ordered levels must match.
- Requests accept an ignored `model` string. Unknown fields still fail.
- The decision API uses a synchronous route. FastAPI runs inference in a
  worker thread. Backend HTTP requests remain serialized by the shared lock.
- The ID alphabet is uppercase letters followed by digits, with no lowercase
  IDs. Token matching remains exact after stripping whitespace.
- Choice and score CLI options accept `name=instructions:options` as well as
  the old `name:options` form.
- The loader reads local parquet files first. Only a missing split uses
  `datasets.load_dataset`. No Hub request runs during local conversion.

### Benchmark behavior

`src/system1/data.py` converts local rows to JSONL with parsed state,
questions, labels, and full gold data. The actual parquet files use lists
for score criteria, unlike the dictionary example in the spec. The loader
converts each list to string index keys and retains every description.

The generated files are:

- `data/typed_decisions_train.jsonl`: 1,200 cases and 6,000 questions.
- `data/typed_decisions_test.jsonl`: 400 cases and 2,000 questions.

Every converted request and gold target passes validation. Gold probability
sums differ from one by at most approximately 0.000001 in the local data.
Scoring normalizes rounded gold distributions. It rejects negative or
nonfinite values, mismatched option keys, and sum errors greater than 0.005.

Benchmark records align predicted and gold probabilities by label. Noul uses
P(yes) >= 0.5, including ties, against the true/false gold label. Accuracy and
10-bin ECE use hard labels. Log loss uses cross entropy against soft gold
probabilities. Brier averages squared differences over options, then over
questions. This differs from the phase 1 hard-label sum of squared errors.
Legacy JSONL cases without gold distributions use one-hot targets.

Score MAE compares the predicted expected numeric level with gold `score`.
Named levels in legacy cases use their zero-based indices. Reports include
individual question results, aggregates by question name, workflow and type,
overall metrics, decision latency median/p95, and per-case forward passes.
The printed table includes log loss, score MAE, and all six reference values
from the spec. These reference values are not measurements of this run.

`--workflow` filters test cases before `--limit` selects the first matching
cases. `--permutations` controls both train and test inference.
`--fit-temperature train.jsonl` fits hard-label negative log likelihood per
question type on all training cases. It saves raw training predictions in
`data/preds_train.jsonl`. Cache fingerprints include case content, dictionary
order, model name, permutations, and a cache version. Changed inputs or
settings cause fresh inference. Completed cases survive an interrupted run.
Calibration fits always ignore an existing temperature file.

Test inference runs once. Before and after metrics use the same predictions
and latency samples. The JSON report uses the top-level metrics for the
calibrated results and `before_calibration` for raw results. Tests check cache
reuse, input/order/setting changes, and the absence of repeated test inference.
No real training cache, calibration fit, or live report is produced here,
because the sandbox blocks the model connection.

### Dependencies and environment

The optional `bench` extra pins `datasets==5.0.0` and `pyarrow==25.0.1`.
The declarations use:

```bash
UV_CACHE_DIR=/tmp/system1-uv-cache uv add --offline --frozen --optional bench datasets==5.0.0 pyarrow==25.0.1
```

An earlier offline resolution attempt fails because the registry cache lacks
FastAPI metadata. The new lock entries are recovered from the existing local
`nanoeval/uv.lock`, with registry URLs and artifact hashes retained. Existing
project package versions remain unchanged. `uv lock --check --offline` passes.

A full offline sync cannot install from the incomplete cache. A selective
sync also fails on the copied parquet cache layout and removes environment
packages. The local environment is restored from matching installed package
versions in `pipecat_buddy` and `nanoeval`, using their wheel RECORD entries.
The restored project environment includes the benchmark dependencies.
Automatic uv sync attempts during recovery fail DNS resolution; no package
is downloaded. Subsequent commands force uv offline. No dataset or model is
downloaded. A fresh install remains unverified in this sandbox.

In an environment with package access, install with `uv sync --locked --extra
bench`. The final checks here use the project environment without PYTHONPATH
or UV_NO_SYNC overrides:

```bash
export UV_CACHE_DIR=/tmp/system1-uv-cache UV_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .
uv run pytest -q
uv run system1 fetch-typed-decisions
uv lock --check --offline
git diff --check
```

### Verification results

```text
$ uv run ruff check .
All checks passed!
$ uv run pytest -q
.....................................................................    [100%]
69 passed in 0.54s
$ uv run system1 fetch-typed-decisions
data/typed_decisions_train.jsonl
data/typed_decisions_test.jsonl
$ uv lock --check --offline
Resolved 57 packages in 0.39ms
$ git diff --check
(no output)
```

Tests block socket connections and DNS. Parquet tests write small local
fixtures. The Hub fallback uses a mock. API tests use ASGITransport and assert
that inference runs on a worker thread. Timer wakeups keep the test event loop
moving when the sandbox blocks its worker-completion wakeup socket.

Earlier checks report `55 passed in 0.38s`, `67 passed in 0.55s`,
and `69 passed in 0.55s`.
An intermediate run reports `1 failed, 66 passed in 1.36s`: the new cache test
supplies A/B tokens for a score permutation that uses B/C. Correcting the
fixture resolves that failure. An earlier API check hangs on the sandbox
wakeup limitation and is interrupted before the timer workaround is added.

The requested live check is attempted once:

```text
$ uv run system1 bench data/typed_decisions_test.jsonl --limit 5
Error: [Errno 1] Operation not permitted
```

It exits with status 1. The sandbox denies the connection to port 8010.
This does not indicate a model-server bug. Live scores, latency, and the live
printed table remain unverified. The table and JSON report paths pass with
FakeBackend in unit tests. No server is started or stopped. Nothing is committed.
The justfile adds `fetch`, `bench-smoke`, and `bench-calibrate` recipes.
