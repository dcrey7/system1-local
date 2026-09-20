check:
    uv run ruff check .
    uv run pytest -q

format:
    uv run ruff check --fix .
    uv run ruff format .

live:
    uv run system1 ask --state "I was charged twice. Refund me or I will cancel." --choice "route:billing,tech,sales" --noul "angry:The customer is threatening to cancel."
