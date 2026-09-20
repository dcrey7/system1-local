"""Convert typed decisions parquet rows to benchmark JSONL."""

import json
from pathlib import Path

from system1.schema import Request

RAW_DIR = Path("data/raw/typed_decisions")


def convert_row(row: dict) -> dict:
    """Keep question descriptions and gold targets from one dataset row."""
    state = json.loads(row["state"])
    questions = json.loads(row["questions"])
    for question in questions.values():
        if question["type"] == "score" and isinstance(question.get("criteria"), list):
            question["criteria"] = {
                str(index): description
                for index, description in enumerate(question["criteria"])
            }
    gold = json.loads(row["gold"])
    Request(state=state, questions=questions)
    if set(gold) != set(questions):
        raise ValueError("Gold questions must match input questions")
    return {
        "id": row["id"],
        "workflow": row["workflow"],
        "state": state,
        "questions": questions,
        "answers": {name: target["label"] for name, target in gold.items()},
        "gold": gold,
    }


def download_typed_decisions(out_dir: Path) -> tuple[Path, Path]:
    """Use local parquet first, and fetch missing splits from the Hub."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for split in ("train", "test"):
        local = RAW_DIR / f"all_{split}.parquet"
        if local.exists():
            import pyarrow.parquet as parquet

            rows = parquet.read_table(local).to_pylist()
        else:
            from datasets import load_dataset

            rows = load_dataset("LocalLLaMA/typed-decisions", "all", split=split)
        path = out_dir / f"typed_decisions_{split}.jsonl"
        converted = []
        for index, row in enumerate(rows, 1):
            try:
                converted.append(json.dumps(convert_row(row), allow_nan=False))
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{split} row {index}: {error}") from error
        path.write_text("\n".join(converted) + "\n")
        paths.append(path)
    return paths[0], paths[1]
