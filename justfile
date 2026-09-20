check:
    uv run ruff check .
    uv run pytest -q

format:
    uv run ruff check --fix .
    uv run ruff format .

live:
    uv run system1 ask --state "I was charged twice. Refund me or I will cancel." --choice "route:billing,tech,sales" --noul "angry:The customer is threatening to cancel."

fetch:
    uv run system1 fetch-typed-decisions

bench-smoke:
    uv run system1 bench data/typed_decisions_test.jsonl --limit 5

bench-calibrate:
    uv run system1 bench data/typed_decisions_test.jsonl --fit-temperature data/typed_decisions_train.jsonl --out report.json
