import json
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from system1.cli import app
from system1.data import convert_row, download_typed_decisions


def handwritten_rows():
    first = {
        "id": "one",
        "workflow": "review",
        "state": json.dumps({"trace": "read"}),
        "questions": json.dumps(
            {
                "risk": {
                    "type": "score",
                    "instructions": "Risk?",
                    "criteria": ["safe", "risky"],
                },
                "review": {
                    "type": "noul",
                    "instructions": "Review?",
                    "criteria": {"true": "Inspect", "false": "Continue"},
                },
            }
        ),
        "gold": json.dumps(
            {
                "risk": {
                    "label": "1",
                    "probabilities": {"0": 0.25, "1": 0.75},
                    "score": 0.75,
                },
                "review": {
                    "label": "false",
                    "probabilities": {"true": 0.4, "false": 0.6},
                    "noul": 0.4,
                },
            }
        ),
    }
    second = {
        "id": "two",
        "workflow": "route",
        "state": json.dumps(["refund"]),
        "questions": json.dumps(
            {
                "team": {
                    "type": "choice",
                    "instructions": "Team?",
                    "criteria": {"billing": "money", "tech": None},
                }
            }
        ),
        "gold": json.dumps(
            {
                "team": {
                    "label": "billing",
                    "probabilities": {"billing": 0.8, "tech": 0.2},
                }
            }
        ),
    }
    return [first, second]


def test_conversion_of_two_handwritten_rows():
    first, second = map(convert_row, handwritten_rows())
    assert first["state"] == {"trace": "read"}
    assert first["questions"]["risk"]["criteria"] == {"0": "safe", "1": "risky"}
    assert first["questions"]["review"]["criteria"] == {
        "true": "Inspect",
        "false": "Continue",
    }
    assert first["answers"] == {"risk": "1", "review": "false"}
    assert first["gold"]["risk"]["score"] == 0.75
    assert first["gold"]["review"]["noul"] == 0.4
    assert second["state"] == ["refund"]
    assert second["answers"] == {"team": "billing"}
    assert second["gold"] == json.loads(handwritten_rows()[1]["gold"])


def test_local_parquet_and_fetch_cli_never_load_hub(monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw = tmp_path / "raw"
    raw.mkdir()
    for split in ("train", "test"):
        pq.write_table(
            pa.Table.from_pylist(handwritten_rows()), raw / f"all_{split}.parquet"
        )
    monkeypatch.setattr("system1.data.RAW_DIR", raw)
    monkeypatch.setitem(sys.modules, "datasets", None)
    result = CliRunner().invoke(
        app, ["fetch-typed-decisions", "--out-dir", str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.output
    for split in ("train", "test"):
        path = tmp_path / "out" / f"typed_decisions_{split}.jsonl"
        assert str(path) in result.output
        assert [json.loads(line) for line in path.read_text().splitlines()] == list(
            map(convert_row, handwritten_rows())
        )


def test_missing_parquet_falls_back_to_mocked_dataset(monkeypatch, tmp_path):
    calls = []

    def load_dataset(repo, config, split):
        calls.append((repo, config, split))
        return handwritten_rows()

    monkeypatch.setattr("system1.data.RAW_DIR", tmp_path / "missing")
    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset)
    )
    paths = download_typed_decisions(tmp_path / "out")
    assert all(path.exists() for path in paths)
    assert calls == [
        ("LocalLLaMA/typed-decisions", "all", split) for split in ("train", "test")
    ]


def test_loader_rejects_mismatched_gold():
    row = handwritten_rows()[0]
    row["gold"] = "{}"
    with pytest.raises(ValueError, match="Gold questions"):
        convert_row(row)
