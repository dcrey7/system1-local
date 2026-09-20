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
