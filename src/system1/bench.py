"""JSONL evaluation, soft targets, and cached temperature fitting."""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from system1.calibrate import (
    apply_temperature,
    ece,
    fit_temperature,
    load_calibration,
    save_calibration,
    soft_brier,
    soft_log_loss,
    soft_targets,
)
from system1.core import MODEL, MULTI_FORMAT, LowCoverageError, SystemOne
from system1.schema import Mode, Request, options

REFERENCES = {
    "majority baseline": 0.520,
    "factor ceiling": 0.704,
    "teacher self agreement": 0.735,
    "Jev published": 0.727,
    "Laya zero shot": 0.36,
    "Laya fine tuned": 0.766,
}


def target_record(case: dict, name: str, request: Request) -> dict:
    """Align labels and gold probabilities to a fixed option order."""
    question = request.questions[name]
    labels = list(options(question))
    if question.type == "choice":
        labels.sort()
    gold = case.get("gold", {}).get(name)
    target = gold["label"] if gold is not None else case["answers"][name]
    keys = labels
    if question.type == "noul":
        keys = ["true", "false"]
        if isinstance(target, bool):
            target = "true" if target else "false"
        if target not in keys:
            raise ValueError("A noul label must be true or false")
        label = keys.index(target)
    else:
        label = labels.index(target)
    if gold is None:
        distribution = [float(i == label) for i in range(len(labels))]
    else:
        if set(gold["probabilities"]) != set(keys):
            raise ValueError("Gold probability keys must match question options")
        distribution = [gold["probabilities"][key] for key in keys]
    _, normalized = soft_targets([[1 / len(labels)] * len(labels)], [distribution])
    record = {
        "id": case["id"],
        "workflow": case["workflow"],
        "question": name,
        "type": question.type,
        "options": labels,
        "label": label,
        "gold_probabilities": normalized[0].tolist(),
    }
    if question.type == "score":
        try:
            levels = [float(level) for level in labels]
        except ValueError:
            levels = list(range(len(labels)))
        score = float(gold["score"]) if gold is not None else levels[label]
        if not np.isfinite(levels).all() or not np.isfinite(score):
            raise ValueError("Score levels and gold score must be finite")
        record.update(levels=levels, gold_score=score)
    return record


def run_cases(
    path: Path,
    engine: SystemOne,
    permutations: int,
    *,
    limit: int | None = None,
    workflow: str | None = None,
    cache_path: Path | None = None,
    mode: Mode = "single",
) -> tuple[list[dict], list[dict]]:
    """Run raw predictions, reusing matching cached training cases."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    cached = {}
    if cache_path is not None and cache_path.exists():
        for line in cache_path.read_text().splitlines():
            entry = json.loads(line)
            cached[entry["fingerprint"]] = entry["result"]
    records, usages = [], []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        start = perf_counter()
        try:
            case = json.loads(line)
            if workflow is not None and case["workflow"] != workflow:
                continue
            request = Request(
                state=case["state"],
                questions=case["questions"],
                permutations=permutations,
                mode=mode,
            )
            targets = [target_record(case, name, request) for name in request.questions]
            fingerprint_data = {
                "version": 1,
                "model": MODEL,
                "case": case,
                "permutations": permutations,
            }
            if mode == "multi":
                fingerprint_data["mode"] = mode
                fingerprint_data["format"] = MULTI_FORMAT
            fingerprint = hashlib.sha256(
                json.dumps(fingerprint_data).encode()
            ).hexdigest()
            if fingerprint in cached:
                result = cached[fingerprint]
            else:
                result = engine.decide(request, calibrated=False, mode=mode)
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with cache_path.open("a") as cache:
                        cache.write(
                            json.dumps(
                                {"fingerprint": fingerprint, "result": result},
                                allow_nan=False,
                            )
                            + "\n"
                        )
            usages.append({"id": case["id"], **result["usage"]})
            for record in targets:
                answer = result["answers"][record["question"]]
                record["probabilities"] = (
                    [answer["noul"], 1 - answer["noul"]]
                    if record["type"] == "noul"
                    else [answer["probabilities"][label] for label in record["options"]]
                )
                records.append(record)
        except LowCoverageError as error:
            message = " ".join(str(error).splitlines())
            print(f"Warning: {path}:{line_number}: {message}", file=sys.stderr)
            for record in targets:
                size = len(record["options"])
                records.append({**record, "probabilities": [1 / size] * size})
            # One call is a lower bound in both modes; failed decisions expose no counts.
            usages.append(
                {
                    "id": case["id"],
                    "input_tokens": 0,
                    "forward_passes": 0,
                    "calls": 1,
                    "latency_ms": (perf_counter() - start) * 1000,
                    "failure_line": line_number,
                }
            )
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        if limit is not None and len(usages) >= limit:
            break
    if not records:
        raise ValueError(f"{path}: no benchmark questions")
    return records, usages


def metrics(records: list[dict]) -> dict:
    """Score each question equally, including questions of different sizes."""
    probs = [record["probabilities"] for record in records]
    labels = [record["label"] for record in records]
    gold = [record["gold_probabilities"] for record in records]
    errors = [
        abs(np.dot(r["probabilities"], r["levels"]) - r["gold_score"])
        for r in records
        if r["type"] == "score"
    ]
    return {
        "count": len(records),
        "accuracy": float(
            np.mean([np.argmax(p) == label for p, label in zip(probs, labels)])
        ),
        "ece": ece(probs, labels),
        "brier": soft_brier(probs, gold),
        "log_loss": soft_log_loss(probs, gold),
        "score_mae": float(np.mean(errors)) if errors else None,
    }


def summarize(records: list[dict], usages: list[dict]) -> dict:
    latencies = [usage["latency_ms"] for usage in usages]
    passes = [usage["forward_passes"] for usage in usages]
    calls = [usage.get("calls", usage["forward_passes"]) for usage in usages]
    failures = [usage["failure_line"] for usage in usages if "failure_line" in usage]
    report = {
        "overall": metrics(records),
        "decisions": len(usages),
        "failures": {"count": len(failures), "lines": failures},
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "forward_passes": {"total": sum(passes), "per_case": float(np.mean(passes))},
        "calls": {"total": sum(calls), "per_case": float(np.mean(calls))},
        "cases": usages,
        "question_results": [{**record, **metrics([record])} for record in records],
    }
    for field in ("workflow", "type", "question"):
        report["per_" + field] = {
            key: metrics([r for r in records if r[field] == key])
            for key in sorted({r[field] for r in records})
        }
    return report


def benchmark(
    path: Path,
    engine: SystemOne,
    permutations: int = 3,
    train: Path | None = None,
    *,
    limit: int | None = None,
    workflow: str | None = None,
    cache_path: Path | None = None,
    mode: Mode = "single",
) -> dict:
    """Evaluate once and apply fitted temperatures to the saved distributions."""
    calibration_path = engine.calibration_for(mode)
    if cache_path is None:
        cache_path = Path(
            "data/preds_train_multi.jsonl"
            if mode == "multi"
            else "data/preds_train.jsonl"
        )
    if train is not None:
        training, _ = run_cases(
            train, engine, permutations, cache_path=cache_path, mode=mode
        )
        grouped = defaultdict(list)
        for record in training:
            grouped[record["type"]].append(record)
        temperatures = {
            kind: fit_temperature(
                [r["probabilities"] for r in rows], [r["label"] for r in rows]
            )
            for kind, rows in grouped.items()
        }
        save_calibration(temperatures, calibration_path)
    else:
        temperatures = load_calibration(calibration_path)
    records, usages = run_cases(
        path, engine, permutations, limit=limit, workflow=workflow, mode=mode
    )
    before = summarize(records, usages)
    calibrated = [
        {
            **record,
            "probabilities": apply_temperature(
                record["probabilities"], temperatures.get(record["type"], 1)
            ).tolist(),
        }
        for record in records
    ]
    report = summarize(calibrated, usages)
    if train is not None:
        report["before_calibration"] = before
    report["temperatures"] = temperatures
    report["references"] = REFERENCES
    return report


def metric_table(report: dict) -> list[str]:
    lines = [
        f"{'Group':28} {'Count':>7} {'Accuracy':>10} {'ECE':>10} {'Brier':>10} {'Log loss':>10} {'Score MAE':>10}"
    ]
    groups = {"overall": report["overall"]}
    for field in ("workflow", "type"):
        groups.update(
            {f"{field}/{key}": value for key, value in report["per_" + field].items()}
        )
    for name, values in groups.items():
        mae = "-" if values["score_mae"] is None else f"{values['score_mae']:.4f}"
        lines.append(
            f"{name:28} {values['count']:7} {values['accuracy']:10.4f} {values['ece']:10.4f} {values['brier']:10.4f} {values['log_loss']:10.4f} {mae:>10}"
        )
    return lines


def report_table(report: dict) -> str:
    lines = []
    if "before_calibration" in report:
        lines.extend(
            [
                "Before calibration",
                *metric_table(report["before_calibration"]),
                "After calibration",
            ]
        )
    lines.extend(metric_table(report))
    lines.append(f"Failures: {report['failures']['count']}")
    lines.append(
        f"Latency (ms): mean={report['latency_ms']['mean']:.2f}, median={report['latency_ms']['median']:.2f}, p95={report['latency_ms']['p95']:.2f}"
    )
    lines.append(
        f"Forward passes: total={report['forward_passes']['total']}, per case={report['forward_passes']['per_case']:.2f}"
    )
    lines.append(
        f"Calls: total={report['calls']['total']}, per case={report['calls']['per_case']:.2f}"
    )
    lines.append(
        "Reference accuracies (dataset card, their harness on the same test split; not this run):"
    )
    lines.extend(f"  {name}: {value:.3f}" for name, value in REFERENCES.items())
    return "\n".join(lines)
