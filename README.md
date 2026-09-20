# system1-local

Calibrated decisions from a local language model. You send a state (any text or JSON) and typed questions. You get back a probability for every option, not text. Runs on one consumer GPU with Gemma 4 12B through llama-server. No training.

This is the same shape of service as TypeSafe's Jev, built in the open on a model you run yourself. It is not affiliated with TypeSafe.

## Numbers

Benchmark: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), test split, 400 cases, 2,000 decisions, four workflows. Temperatures fitted on the train split only. Model weights untouched.

| | Jev 1.13 (from the dataset card) | This repo, one call per question | This repo, all questions in one call |
|---|---|---|---|
| Accuracy | 0.727 | 0.7065 | 0.743 |
| Brier, card scale | 0.148 | 0.145 | 0.125 |
| Calibration error (top label ECE, 10 bins) | 0.144 | 0.032 | 0.030 |
| Time per case, 5 questions | 710 ms (cloud) | 2.4 s | 1.4 s, or 0.5 s with one shuffle |

Measured on an RTX 3090 with a 4 bit Gemma 4 12B and its MTP draft. Details, per type and per workflow numbers, and the caveats are in `docs/phase2-results.md`. An independent adversarial audit of the numbers is in `docs/phase2-audit.md`.

## How it works

1. The prompt holds the state and all the questions. Every option gets a one character id, unique across the call.
2. A grammar forces the model to write exactly one line per question, `Q1: C`, `Q2: A`, and so on.
3. At each answer token the server returns the model's top 20 token probabilities. The probabilities of that question's ids are read and normalised. That is the answer distribution.
4. The call is repeated with the options and questions shuffled (3 times by default) and the distributions are averaged.
5. One temperature per question type, fitted once on the benchmark's train split, softens the output so that 0.8 means right about 80 percent of the time.

Question types: `choice` (pick one option, each with an optional description), `score` (ordered levels, returns the distribution and the expected level), `noul` (a statement, returns the probability that it is true).

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- llama-server from llama.cpp, serving Gemma 4 12B on port 8010 with `--jinja`. If you use a draft model for speed (`--spec-type draft-mtp`), the server needs the fix from [llama.cpp PR #27196](https://github.com/ggml-org/llama.cpp/pull/27196), otherwise only the first generated token carries probabilities. The backend checks this at start and stops with a clear error.

Any model works in principle if the server returns `top_logprobs` for every generated token and supports GBNF grammars. The numbers above are for Gemma 4 12B QAT (Q4_K_XL).

## Use

```bash
uv sync
uv run pytest            # offline tests; live tests run too when the server is up

uv run system1 ask --state "Candidate: 5 years data science in Paris. Job: senior data scientist, Paris, needs Python and SQL." \
  --choice "fit=How well does the candidate fit the job:weak,medium,strong" \
  --noul "apply=We should apply to this job" \
  --score "seniority=Seniority of the candidate for this role:junior,mid,senior" \
  --mode multi

uv run system1 serve --port 8020
```

The server exposes `POST /v1/systemone`:

```json
{
  "state": "text or JSON",
  "questions": {
    "fit": {"type": "choice", "instructions": "How well does the candidate fit the job",
            "criteria": {"weak": "missing core skills", "medium": "most skills", "strong": "all skills and seniority"}},
    "apply": {"type": "noul", "instructions": "We should apply to this job"},
    "urgency": {"type": "score", "instructions": "How urgent is this", "levels": ["low", "mid", "high"]}
  },
  "permutations": 3,
  "mode": "multi"
}
```

Each answer carries `probabilities`, `confidence` (how far the top option is above chance), `coverage` (how much of the model's mass landed on valid ids), and for scores `expected_index`.

## Benchmark

```bash
uv run system1 fetch-typed-decisions
uv run system1 bench data/typed_decisions_test.jsonl --mode multi --permutations 3 \
  --fit-temperature data/typed_decisions_train.jsonl --out report.json
```

The train and test JSONL files and the raw parquet files are included under `data/`. They come from the LocalLLaMA/typed-decisions dataset, Apache 2.0, and keep that licence.

## Repo map

- `src/system1/`: `core.py` (prompts, grammar, reads, decisions), `backend.py` (llama-server client and probe), `calibrate.py` (temperature, ECE, Brier, log loss), `bench.py`, `cli.py`, `api.py`, `schema.py`.
- `docs/`: phase specs, results, notes, the audit, and a facts sheet for upstream llama.cpp contributions.
- `tests/`: offline tests with a fake backend, and `test_live.py` against a running server.

## Licence

MIT for the code in this repo. The bundled benchmark data is Apache 2.0 from its authors. Model weights are not included; Gemma comes with its own terms from Google.
