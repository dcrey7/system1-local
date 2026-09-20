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
    assert report["overall"]["brier"] == pytest.approx((3 * 0.04 + 0.08 / 3) / 4)
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


def test_cli_named_instructions(monkeypatch):
    backend = canned_backend(2)
    backend.close = lambda: None
    monkeypatch.setattr("system1.cli.GemmaBackend", lambda: backend)
    result = CliRunner().invoke(
        app,
        [
            "ask",
            "--state",
            "s",
            "--choice",
            "route=Which team handles this?:billing,tech",
            "--score",
            "risk=How risky?:low,high",
            "--permutations",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Question: Which team handles this?" in backend.prompts[0]
    assert "Question: How risky?" in backend.prompts[1]


@pytest.mark.parametrize("p_yes,correct", [(0.49, 0.0), (0.5, 1.0), (0.51, 1.0)])
def test_noul_threshold_against_string_gold(p_yes, correct, tmp_path):
    case = {
        "id": "n",
        "workflow": "w",
        "state": "",
        "questions": {"q": {"type": "noul", "instructions": "?"}},
        "answers": {"q": "true"},
        "gold": {"q": {"label": "true", "probabilities": {"false": 0.4, "true": 0.6}}},
    }
    path = tmp_path / "noul.jsonl"
    path.write_text(json.dumps(case) + "\n")
    backend = FakeBackend(
        [
            [
                {"token": "A", "logprob": math.log(p_yes)},
                {"token": "B", "logprob": math.log(1 - p_yes)},
            ]
        ]
    )
    report = benchmark(path, SystemOne(backend), permutations=1)
    assert report["overall"]["accuracy"] == correct
    assert report["overall"]["brier"] == pytest.approx((p_yes - 0.6) ** 2)
    assert report["overall"]["ece"] == pytest.approx(
        abs(correct - max(p_yes, 1 - p_yes))
    )


def test_score_expected_numeric_level_and_alignment(tmp_path):
    case = {
        "id": "s",
        "workflow": "w",
        "state": "",
        "questions": {
            "risk": {
                "type": "score",
                "instructions": "?",
                "criteria": {"10": "high", "2": "low"},
            }
        },
        "answers": {"risk": "2"},
        "gold": {
            "risk": {
                "label": "2",
                "score": 3.0,
                "probabilities": {"10": 0.25, "2": 0.75},
            }
        },
    }
    path = tmp_path / "score.jsonl"
    path.write_text(json.dumps(case) + "\n")
    report = benchmark(path, SystemOne(canned_backend(1)), permutations=1)
    assert report["overall"]["accuracy"] == 1
    assert report["overall"]["score_mae"] == pytest.approx(0.6)
    assert report["overall"]["brier"] == pytest.approx(0.0025)
    assert report["per_question"]["risk"]["score_mae"] == pytest.approx(0.6)
    assert report["question_results"][0]["gold_probabilities"] == [0.75, 0.25]


def test_filter_before_limit_and_forward_passes(tmp_path):
    backend = canned_backend(1)
    report = benchmark(
        cases_file(tmp_path / "cases.jsonl"),
        SystemOne(backend),
        permutations=1,
        workflow="sales",
        limit=1,
    )
    assert report["overall"]["count"] == 1
    assert report["decisions"] == 1
    assert report["forward_passes"] == {"total": 1, "per_case": 1.0}
    assert report["cases"][0]["id"] == "2"
    assert len(backend.prompts) == 1
    for reference in (
        "majority baseline: 0.520",
        "factor ceiling: 0.704",
        "teacher self agreement: 0.735",
        "Jev published: 0.727",
        "Laya zero shot: 0.360",
        "Laya fine tuned: 0.766",
    ):
        assert reference in report_table(report)


def test_temperature_cache_reuse_and_setting_invalidation(tmp_path):
    path = cases_file(tmp_path / "cases.jsonl")
    engine = SystemOne(canned_backend(8), tmp_path / "calibration.json")
    first = benchmark(path, engine, permutations=1, train=path)
    assert len(engine.backend.prompts) == 8
    assert first["before_calibration"]["overall"]["ece"] == pytest.approx(0.2)
    assert "Before calibration" in report_table(first)
    assert "After calibration" in report_table(first)
    assert first["latency_ms"] == first["before_calibration"]["latency_ms"]
    assert len((tmp_path / "data/preds_train.jsonl").read_text().splitlines()) == 2
    engine.backend = canned_backend(4)
    second = benchmark(path, engine, permutations=1, train=path)
    assert len(engine.backend.prompts) == 4
    assert second["overall"] == first["overall"]
    engine.backend = FakeBackend(
        [[{"token": token, "logprob": math.log(1 / 3)} for token in "ABC"]] * 16
    )
    benchmark(path, engine, permutations=2, train=path)
    assert len(engine.backend.prompts) == 16


def test_cli_benchmark_filters(monkeypatch, tmp_path):
    backend = canned_backend(1)
    backend.close = lambda: None
    monkeypatch.setattr("system1.cli.GemmaBackend", lambda: backend)
    path = cases_file(tmp_path / "cases.jsonl")
    result = CliRunner().invoke(
        app,
        [
            "bench",
            str(path),
            "--limit",
            "1",
            "--workflow",
            "sales",
            "--permutations",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "report.json").read_text())["decisions"] == 1


def test_training_cache_respects_case_content_and_score_order(tmp_path):
    from system1.bench import run_cases

    path = cases_file(tmp_path / "train.jsonl")
    cache = tmp_path / "predictions.jsonl"
    engine = SystemOne(canned_backend(8))
    run_cases(path, engine, 1, cache_path=cache)
    cases = [json.loads(line) for line in path.read_text().splitlines()]
    cases[0]["state"] = "different request"
    path.write_text("\n".join(json.dumps(case) for case in cases))
    run_cases(path, engine, 1, cache_path=cache)
    assert len(engine.backend.prompts) == 7
    question = cases[0]["questions"]["urgency"]
    question.pop("levels")
    question["criteria"] = {"low": "routine", "high": "urgent"}
    path.write_text("\n".join(json.dumps(case) for case in cases))
    engine.backend = canned_backend(6)
    run_cases(path, engine, 1, cache_path=cache)
    question["criteria"] = {"high": "urgent", "low": "routine"}
    path.write_text("\n".join(json.dumps(case) for case in cases))
    run_cases(path, engine, 1, cache_path=cache)
    assert len(engine.backend.prompts) == 6


def test_unknown_workflow_fails_without_model_call(tmp_path):
    backend = FakeBackend([])
    with pytest.raises(ValueError, match="no benchmark questions"):
        benchmark(
            cases_file(tmp_path / "cases.jsonl"), SystemOne(backend), workflow="absent"
        )
    assert backend.prompts == []
