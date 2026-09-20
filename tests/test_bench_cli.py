import json
import math

import pytest
from typer.testing import CliRunner

from system1.backend import FakeBackend
from system1.bench import benchmark, report_table
from system1.calibrate import load_calibration, save_calibration
from system1.cli import app
from system1.core import SystemOne


def cases_file(path):
    cases = [
        {
            "id": "1",
            "workflow": "support",
            "state": "refund",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Team?",
                    "criteria": {"billing": None, "tech": None},
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Urgency?",
                    "levels": ["low", "high"],
                },
                "angry": {"type": "noul", "instructions": "Angry?"},
            },
            "answers": {"route": "tech", "urgency": "low", "angry": True},
        },
        {
            "id": "2",
            "workflow": "sales",
            "state": ["quote"],
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Team?",
                    "criteria": {"billing": None, "tech": None, "sales": None},
                },
            },
            "answers": {"route": "sales"},
        },
    ]
    path.write_text("\n".join(json.dumps(case) for case in cases) + "\n")
    return path


def canned_backend(count=4):
    return FakeBackend(
        [
            [
                {"token": "A", "logprob": math.log(0.8)},
                {"token": "B", "logprob": math.log(0.2)},
            ]
        ]
        * count
    )


def test_benchmark_groups_and_latency(tmp_path):
    report = benchmark(
        cases_file(tmp_path / "cases.jsonl"),
        SystemOne(canned_backend()),
        permutations=1,
    )
    assert report["overall"]["count"] == 4
    assert report["overall"]["accuracy"] == 1
    assert report["overall"]["ece"] == pytest.approx(0.2)
    assert report["overall"]["brier"] == pytest.approx(0.08)
    assert report["per_workflow"]["support"]["count"] == 3
    assert report["per_type"]["choice"]["count"] == 2
    assert report["decisions"] == 2
    assert report["latency_ms"]["p95"] >= report["latency_ms"]["median"] >= 0
    assert "type/noul" in report_table(report)


def test_training_ignores_existing_temperature_and_saves_per_type(tmp_path):
    path = cases_file(tmp_path / "train.jsonl")
    calibration = tmp_path / "calibration.json"
    save_calibration({"choice": 9.0, "score": 9.0, "noul": 9.0}, calibration)
    backend = canned_backend(8)
    report = benchmark(
        path, SystemOne(backend, calibration), permutations=1, train=path
    )
    assert load_calibration(calibration) == pytest.approx(
        {"choice": 0.2, "score": 0.2, "noul": 0.2}
    )
    assert report["overall"]["ece"] < 0.01
    assert len(backend.prompts) == 8


def test_bad_dataset_has_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        benchmark(path, SystemOne(FakeBackend([])))


def test_cli_ask_and_bench(monkeypatch, tmp_path):
    backend = canned_backend(8)
    backend.close = lambda: None
    monkeypatch.setattr("system1.cli.GemmaBackend", lambda: backend)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ask",
            "--state",
            '{"text": "refund"}',
            "--choice",
            "route:billing,tech",
            "--score",
            "urgency:low,high",
            "--noul",
            "angry:Angry?",
            "--permutations",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["usage"]["forward_passes"] == 3
    path = cases_file(tmp_path / "cases.jsonl")
    out = tmp_path / "report.json"
    result = runner.invoke(
        app, ["bench", str(path), "--permutations", "1", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["overall"]["count"] == 4
    assert "Accuracy" in result.output
    invalid = runner.invoke(app, ["ask", "--state", "s", "--choice", "route:a,a"])
    assert invalid.exit_code == 1
    assert "unique" in invalid.output
