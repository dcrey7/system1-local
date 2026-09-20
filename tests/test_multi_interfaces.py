import asyncio
import hashlib
import json
import math

import httpx
import pytest
from typer.testing import CliRunner

from system1.api import create_app
from system1.backend import FakeBackend
from system1.bench import benchmark, report_table, run_cases, summarize
from system1.calibrate import load_calibration, save_calibration
from system1.cli import app
from system1.core import MODEL, LowCoverageError, SystemOne
from system1.schema import Request


def cases_file(path):
    case = {
        "id": "1",
        "workflow": "support",
        "state": "refund",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Team?",
                "criteria": {"billing": None, "tech": None},
            },
            "risk": {
                "type": "score",
                "instructions": "Risk?",
                "levels": ["low", "high"],
            },
            "angry": {"type": "noul", "instructions": "Angry?"},
        },
        "answers": {"route": "tech", "risk": "low", "angry": True},
    }
    path.write_text(json.dumps(case) + "\n")
    return path


def fake_backend():
    top = [
        {"token": token, "logprob": math.log(mass)}
        for token, mass in {"A": 0.8, "B": 0.2}.items()
    ]
    return FakeBackend(
        [top] * 12,
        alphabet="AB",
        multi_responses={name: {"A": 0.8, "B": 0.2} for name in ["Q1", "Q2", "Q3"]},
    )


def test_multi_fit_and_default_cache_leave_single_calibration_untouched(tmp_path):
    path = cases_file(tmp_path / "cases.jsonl")
    single_path = tmp_path / "calibration.json"
    save_calibration({"choice": 7, "score": 8, "noul": 9}, single_path)
    original = single_path.read_bytes()
    backend = fake_backend()
    engine = SystemOne(backend, single_path)
    report = benchmark(path, engine, permutations=1, train=path, mode="multi")
    assert single_path.read_bytes() == original
    assert load_calibration(tmp_path / "calibration_multi.json") == pytest.approx(
        {"choice": 0.2, "score": 0.2, "noul": 0.2}
    )
    assert (tmp_path / "data/preds_train_multi.jsonl").exists()
    assert not (tmp_path / "data/preds_train.jsonl").exists()
    assert len(backend.prompts) == 6
    assert report["overall"]["ece"] < report["before_calibration"]["overall"]["ece"]
    assert report["calls"] == {"total": 3, "per_case": 3.0}
    assert report["forward_passes"] == {"total": 12, "per_case": 12.0}
    assert report["latency_ms"]["mean"] == report["cases"][0]["latency_ms"]
    table = report_table(report)
    assert "Calls: total=3, per case=3.00" in table
    assert (
        "Reference accuracies (dataset card, their harness on the same test split; not this run):"
        in table
    )
    assert "Latency (ms): mean=" in table
    assert "type/noul" in table
    benchmark(path, engine, permutations=1, train=path, mode="multi")
    assert len(backend.prompts) == 9


def test_shared_cache_separates_modes_and_permutations(tmp_path):
    path = cases_file(tmp_path / "cases.jsonl")
    cache = tmp_path / "cache.jsonl"
    backend = fake_backend()
    engine = SystemOne(backend)
    run_cases(path, engine, 1, cache_path=cache)
    assert len(backend.prompts) == 3
    run_cases(path, engine, 1, cache_path=cache, mode="multi")
    assert len(backend.prompts) == 6
    run_cases(path, engine, 1, cache_path=cache, mode="multi")
    assert len(backend.prompts) == 6
    run_cases(path, engine, 2, cache_path=cache, mode="multi")
    assert len(backend.prompts) == 12
    assert len(cache.read_text().splitlines()) == 3


def test_single_cache_fingerprint_matches_phase1(tmp_path, monkeypatch):
    path = cases_file(tmp_path / "cases.jsonl")
    cache = tmp_path / "cache.jsonl"
    backend = fake_backend()
    engine = SystemOne(backend)
    phase1 = {
        "version": 1,
        "model": MODEL,
        "case": json.loads(path.read_text()),
        "permutations": 1,
    }
    hashed_bytes = []
    sha256 = hashlib.sha256

    def capture_fingerprint(data: bytes):
        hashed_bytes.append(data)
        return sha256(data)

    monkeypatch.setattr("system1.bench.hashlib.sha256", capture_fingerprint)
    run_cases(path, engine, 1, cache_path=cache, mode="single")
    assert "mode" not in json.loads(hashed_bytes[0])
    assert hashed_bytes == [json.dumps(phase1).encode()]
    entry = json.loads(cache.read_text())
    assert entry["fingerprint"] == sha256(json.dumps(phase1).encode()).hexdigest()
    prompts = len(backend.prompts)
    run_cases(path, engine, 1, cache_path=cache, mode="single")
    assert len(backend.prompts) == prompts


def test_multi_cache_fingerprint_differs_from_phase1(tmp_path):
    path = cases_file(tmp_path / "cases.jsonl")
    cache = tmp_path / "cache.jsonl"
    phase1 = {
        "version": 1,
        "model": MODEL,
        "case": json.loads(path.read_text()),
        "permutations": 1,
    }
    run_cases(path, SystemOne(fake_backend()), 1, cache_path=cache, mode="multi")
    entry = json.loads(cache.read_text())
    assert (
        entry["fingerprint"] != hashlib.sha256(json.dumps(phase1).encode()).hexdigest()
    )
    assert (
        entry["fingerprint"]
        == hashlib.sha256(
            json.dumps({**phase1, "mode": "multi", "format": 3}).encode()
        ).hexdigest()
    )


def test_custom_multi_calibration_path(tmp_path):
    path = tmp_path / "custom.json"
    engine = SystemOne(fake_backend(), multi_calibration_path=path)
    cases = cases_file(tmp_path / "cases.jsonl")
    benchmark(cases, engine, permutations=1, train=cases, mode="multi")
    assert path.exists()
    assert not (tmp_path / "calibration.json").exists()
    assert not (tmp_path / "calibration_multi.json").exists()


@pytest.mark.parametrize("old_format", [None, 2])
def test_multi_format_invalidates_old_cache(tmp_path, old_format):
    path = cases_file(tmp_path / "cases.jsonl")
    cache = tmp_path / "cache.jsonl"
    old_data = {
        "version": 1,
        "model": MODEL,
        "case": json.loads(path.read_text()),
        "permutations": 1,
        "mode": "multi",
    }
    if old_format is not None:
        old_data["format"] = old_format
    cache.write_text(
        json.dumps(
            {
                "fingerprint": hashlib.sha256(
                    json.dumps(old_data).encode()
                ).hexdigest(),
                "result": {},
            }
        )
        + "\n"
    )
    backend = fake_backend()
    run_cases(path, SystemOne(backend), 1, cache_path=cache, mode="multi")
    assert len(backend.prompts) == 3
    assert len(cache.read_text().splitlines()) == 2


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_low_coverage_case_is_uniform_and_run_continues(tmp_path, capsys, mode):
    path = cases_file(tmp_path / "cases.jsonl")
    case = json.loads(path.read_text())
    case["state"] = "bad"
    case["questions"]["route"]["criteria"]["sales"] = None
    good = {**case, "id": "2", "state": "good"}
    path.write_text("\n" + json.dumps(case) + "\n\n" + json.dumps(good) + "\n")

    class FailingBackend(FakeBackend):
        def complete(self, prompt):
            if "State:\nbad\n" in prompt:
                raise LowCoverageError("missing mass\nanswer token")
            return super().complete(prompt)

        def complete_multi(self, prompt, grammar, max_tokens):
            if "State:\nbad\n" in prompt:
                raise LowCoverageError("missing mass\nanswer token")
            return super().complete_multi(prompt, grammar, max_tokens)

    backend = FailingBackend(
        responses=[
            [
                {"token": "A", "logprob": math.log(0.8)},
                {"token": "B", "logprob": math.log(0.2)},
            ]
        ]
        * 3,
        alphabet="ABC",
        multi_responses={"Q1": {"A": 0.8, "B": 0.2}},
    )
    engine = SystemOne(backend)
    cache = tmp_path / "cache.jsonl"
    records, usages = run_cases(path, engine, 1, cache_path=cache, mode=mode)
    report = summarize(records, usages)
    assert report["failures"] == {"count": 1, "lines": [2]}
    assert usages[0]["calls"] == 1
    assert report["calls"]["total"] == 1 + usages[1]["calls"]
    assert report["decisions"] == 2
    assert report["overall"]["count"] == 6
    assert "Failures: 1" in report_table(report)
    assert [record["id"] for record in records] == ["1"] * 3 + ["2"] * 3
    for record in records[:3]:
        size = len(record["options"])
        assert record["probabilities"] == [1 / size] * size
    assert records[3]["probabilities"] != [1 / 3] * 3
    assert len(cache.read_text().splitlines()) == 1
    assert capsys.readouterr().err.splitlines() == [
        f"Warning: {path}:2: missing mass answer token"
    ]
    limited, usages = run_cases(path, engine, 1, limit=1, mode=mode)
    assert len(limited) == 3
    assert summarize(limited, usages)["failures"] == {"count": 1, "lines": [2]}


@pytest.mark.parametrize("mode", ["single", "multi"])
@pytest.mark.parametrize(
    "error", [httpx.HTTPError("server error"), RuntimeError("probe error")]
)
def test_benchmark_server_errors_still_abort(tmp_path, monkeypatch, mode, error):
    path = cases_file(tmp_path / "cases.jsonl")
    backend = fake_backend()

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(
        backend, "complete" if mode == "single" else "complete_multi", fail
    )
    with pytest.raises(type(error), match=str(error)):
        run_cases(path, SystemOne(backend), 1, mode=mode)


def test_summarize_accepts_cached_usage_without_calls(tmp_path):
    path = cases_file(tmp_path / "cases.jsonl")
    records, usages = run_cases(path, SystemOne(fake_backend()), 1)
    del usages[0]["calls"]
    report = summarize(records, usages)
    assert report["calls"] == report["forward_passes"] == {"total": 3, "per_case": 3.0}
    assert report["failures"] == {"count": 0, "lines": []}


def test_cli_multi_ask_and_bench(monkeypatch, tmp_path):
    backend = fake_backend()
    backend.close = lambda: None
    monkeypatch.setattr("system1.cli.GemmaBackend", lambda: backend)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ask",
            "--state",
            "refund",
            "--choice",
            "route:billing,tech",
            "--noul",
            "angry:Angry?",
            "--mode",
            "multi",
            "--permutations",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["usage"]["calls"] == 2
    assert len(backend.prompts) == 2
    path = cases_file(tmp_path / "cases.jsonl")
    out = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "bench",
            str(path),
            "--mode",
            "multi",
            "--permutations",
            "1",
            "--fit-temperature",
            str(path),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["calls"] == {"total": 3, "per_case": 3.0}
    assert len(backend.prompts) == 8
    assert (tmp_path / "calibration_multi.json").exists()


@pytest.mark.parametrize(
    "command",
    [["ask", "--state", "refund", "--noul", "q:True?"], ["bench", "cases.jsonl"]],
)
def test_cli_rejects_unknown_mode(command, monkeypatch):
    def forbidden():
        raise AssertionError("Invalid mode must fail before backend creation")

    monkeypatch.setattr("system1.cli.GemmaBackend", forbidden)
    result = CliRunner().invoke(app, [*command, "--mode", "unknown"])
    assert result.exit_code == 2
    assert "Invalid value" in result.output


def test_cli_reports_probe_failure(monkeypatch):
    backend = fake_backend()
    backend.close = lambda: None

    def fail_probe(*args, **kwargs):
        raise RuntimeError("server returns no probabilities after the first token")

    backend.complete_multi = fail_probe
    monkeypatch.setattr("system1.cli.GemmaBackend", lambda: backend)
    result = CliRunner().invoke(
        app, ["ask", "--state", "", "--noul", "q:True?", "--mode", "multi"]
    )
    assert result.exit_code == 1
    assert "Error: server returns no probabilities" in result.output


def test_api_multi_default_mode_and_probe_error():
    backend = fake_backend()
    application = create_app(SystemOne(backend))
    payload = {
        "state": "",
        "permutations": 1,
        "questions": {"q": {"type": "noul", "instructions": "True?"}},
    }
    assert Request(**payload).mode == "single"

    def fail_probe(*args, **kwargs):
        raise RuntimeError("server returns no probabilities after the first token")

    async def round_trip():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(application), base_url="http://test"
        ) as client:
            result = await client.post(
                "/v1/systemone", json={**payload, "mode": "multi"}
            )
            assert result.status_code == 200
            assert result.json()["usage"]["calls"] == 1
            assert result.json()["answers"]["q"]["noul"] == pytest.approx(0.8)
            assert len(backend.grammars) == 1
            result = await client.post(
                "/v1/systemone", json={**payload, "mode": "unknown"}
            )
            assert result.status_code == 422
            backend.complete_multi = fail_probe
            result = await client.post(
                "/v1/systemone", json={**payload, "mode": "multi"}
            )
            assert result.status_code == 502
            assert "server returns no probabilities" in result.json()["detail"]

    async def run():
        task = asyncio.create_task(round_trip())
        # Timer wakeups also work when the sandbox blocks the loop's wakeup socket.
        for _ in range(500):
            if task.done():
                return await task
            await asyncio.sleep(0.01)
        task.cancel()
        raise AssertionError("API request timed out")

    asyncio.run(run())
