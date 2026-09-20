"""JSONL evaluation and temperature fitting."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from system1.calibrate import brier, ece, fit_temperature, log_loss, save_calibration
from system1.core import SystemOne
from system1.schema import Request, options


def run_cases(
    path: Path, engine: SystemOne, permutations: int, calibrated: bool = True
) -> tuple[list[dict], list[float]]:
    records, latencies = [], []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
            request = Request(
                state=case["state"],
                questions=case["questions"],
                permutations=permutations,
            )
            targets = {}
            for name, question in request.questions.items():
                target = case["answers"][name]
                if question.type == "noul":
                    if not isinstance(target, bool):
                        raise ValueError("A noul answer must be a JSON boolean")
                    target = "yes" if target else "no"
                targets[name] = list(options(question)).index(target)
            workflow, case_id = case["workflow"], case["id"]
            result = engine.decide(request, calibrated=calibrated)
            latencies.append(result["usage"]["latency_ms"])
            for name, answer in result["answers"].items():
                probabilities = (
                    [answer["noul"], 1 - answer["noul"]]
                    if answer["type"] == "noul"
                    else list(answer["probabilities"].values())
                )
                records.append(
                    {
                        "id": case_id,
                        "workflow": workflow,
                        "type": answer["type"],
                        "probabilities": probabilities,
                        "label": targets[name],
                    }
                )
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    if not records:
        raise ValueError(f"{path}: no benchmark questions")
    return records, latencies


def metrics(records: list[dict]) -> dict:
    probs = [record["probabilities"] for record in records]
    labels = [record["label"] for record in records]
    return {
        "count": len(records),
        "accuracy": float(
            np.mean([np.argmax(p) == label for p, label in zip(probs, labels)])
        ),
        "ece": ece(probs, labels),
        "brier": brier(probs, labels),
        "log_loss": log_loss(probs, labels),
    }


def benchmark(
    path: Path, engine: SystemOne, permutations: int = 3, train: Path | None = None
) -> dict:
    if train is not None:
        training, _ = run_cases(train, engine, permutations, calibrated=False)
        grouped = defaultdict(list)
        for record in training:
            grouped[record["type"]].append(record)
        temperatures = {
            kind: fit_temperature(
                [r["probabilities"] for r in rows], [r["label"] for r in rows]
            )
            for kind, rows in grouped.items()
        }
        save_calibration(temperatures, engine.calibration_path)
    records, latencies = run_cases(path, engine, permutations)
    report = {
        "overall": metrics(records),
        "decisions": len(latencies),
        "latency_ms": {
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
        },
    }
    for field in ("workflow", "type"):
        report["per_" + field] = {
            key: metrics([r for r in records if r[field] == key])
            for key in sorted({r[field] for r in records})
        }
    return report


def report_table(report: dict) -> str:
    lines = [f"{'Group':28} {'Count':>7} {'Accuracy':>10} {'ECE':>10} {'Brier':>10}"]
    groups = {"overall": report["overall"]}
    for field in ("workflow", "type"):
        groups.update(
            {f"{field}/{key}": value for key, value in report["per_" + field].items()}
        )
    for name, values in groups.items():
        lines.append(
            f"{name:28} {values['count']:7} {values['accuracy']:10.4f} {values['ece']:10.4f} {values['brier']:10.4f}"
        )
    lines.append(
        f"Latency (ms): median={report['latency_ms']['median']:.2f}, p95={report['latency_ms']['p95']:.2f}"
    )
    return "\n".join(lines)
