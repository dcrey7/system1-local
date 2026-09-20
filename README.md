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

## Demo, step by step

A real run on the RTX 3090, job fit for a candidate, five questions in one call. Script: the same calls the CLI makes, printed stage by stage.

**1. The prompt.** State first, then every question with its options. Every option gets its own letter, unique across the whole call.

```
State:
Job posting: Senior Data Scientist, fintech, Paris. Build credit risk models ... Salary 70 to 85k EUR.
Candidate: 5 years data science across Amazon, EXL, AXA. Python, SQL, PyTorch ... French A2 ...

Questions:
Q1: How well does the candidate match this job
  A. weak: missing core requirements
  B. medium: most requirements, a real gap
  C. strong: all core requirements
Q2: Is this statement true? We should apply to this job
  D. yes
  E. no
Q3: Is this statement true? The French level is a real risk for this role
  F. yes
  G. no
Q4: Seniority of the candidate for this role
  H. junior
  I. mid
  J. senior
Q5: How well the salary band matches this profile
  K. low
  L. fair
  M. good

Answer every question with its option id only, one per line, in the form "Q1: <id>".
```

**2. The grammar.** The model is allowed to write exactly this shape and nothing else.

```
root ::= q0 q1 q2 q3 q4
q0 ::= "Q1: " ("A" | "B" | "C") "\n"
q1 ::= "Q2: " ("D" | "E") "\n"
q2 ::= "Q3: " ("F" | "G") "\n"
q3 ::= "Q4: " ("H" | "I" | "J") "\n"
q4 ::= "Q5: " ("K" | "L" | "M") "\n"
```

**3. The reply.** 313 prompt tokens in, 26 tokens out, 428 ms, one call.

```
Q1: C
Q2: D
Q3: G
Q4: J
Q5: L
```

**4. What the server hands back at each answer letter.** The top of the model's own distribution before that letter was chosen. This is where the confidence comes from.

```
fit          ' C'  C 1.000  B 0.000  A 0.000
apply        ' D'  D 1.000  E 0.000
french_risk  ' G'  G 0.630  F 0.369
seniority    ' J'  J 1.000  I 0.000
salary_fit   ' L'  L 0.985  M 0.014  K 0.000
```

**5. Read over the question's own letters and normalise.**

```
fit          weak 0.000  medium 0.000  strong 1.000
apply        yes 1.000   no 0.000
french_risk  yes 0.369   no 0.631
seniority    junior 0.000  mid 0.000  senior 1.000
salary_fit   low 0.000  fair 0.985  good 0.014
```

**6. The product answer.** Three shuffled calls averaged, then the per type temperature, 1.0 s in total. The raw 1.000s become honest numbers.

```
fit          strong   confidence 0.70   weak 0.02  medium 0.18  strong 0.80
apply        true with probability 0.85
french_risk  true with probability 0.43
seniority    senior   confidence 0.88   junior 0.02  mid 0.06  senior 0.92
salary_fit   fair     confidence 0.15   low 0.33  fair 0.44  good 0.23
```

Repeated runs move these by a few points: the server's speculative decoding and prompt cache are not bit exact between runs.

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
