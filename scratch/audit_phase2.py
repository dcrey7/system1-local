"""Audit saved benchmark records without contacting a model server."""

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from system1 import bench, calibrate
from system1.core import MODEL, MULTI_FORMAT

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines() if line]


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def normalized(values: list[float]) -> list[float]:
    return [value / sum(values) for value in values]


def independent_metrics(rows: list[dict], bins: int = 10) -> dict:
    """Compute metrics with explicit sums and no production metric helpers."""
    buckets = [[] for _ in range(bins)]
    correct, brier, loss, scores, hard_brier, hard_loss = [], [], [], [], [], []
    soft_sum, entropy = [], []
    for row in rows:
        p = row["probabilities"]
        g = normalized(row["gold_probabilities"])
        y = row["label"]
        hit = int(argmax(p) == y)
        correct.append(hit)
        buckets[min(int(max(p) * bins), bins - 1)].append((max(p), hit))
        squared = sum((a - b) ** 2 for a, b in zip(p, g))
        brier.append(squared / len(p))
        soft_sum.append(squared)
        loss.append(-sum(b * math.log(max(a, 1e-300)) for a, b in zip(p, g)))
        entropy.append(-sum(b * math.log(b) for b in g if b))
        hard_brier.append(sum((a - int(i == y)) ** 2 for i, a in enumerate(p)))
        hard_loss.append(-math.log(max(p[y], 1e-300)))
        if row["type"] == "score":
            scores.append(
                abs(sum(a * b for a, b in zip(p, row["levels"])) - row["gold_score"])
            )
    details = [
        {
            "bin": i,
            "n": len(bucket),
            "confidence": mean(p for p, _ in bucket),
            "accuracy": mean(y for _, y in bucket),
        }
        for i, bucket in enumerate(buckets)
        if bucket
    ]
    return {
        "count": len(rows),
        "correct": sum(correct),
        "accuracy": mean(correct),
        "ece": sum(d["n"] * abs(d["confidence"] - d["accuracy"]) for d in details)
        / len(rows),
        "brier": mean(brier),
        "log_loss": mean(loss),
        "score_mae": mean(scores) if scores else None,
        "score_count": len(scores),
        "soft_brier_sum": mean(soft_sum),
        "hard_brier_sum": mean(hard_brier),
        "hard_log_loss": mean(hard_loss),
        "gold_entropy": mean(entropy),
        "kl_gold_pred": mean(loss) - mean(entropy),
        "ece_bins": details,
    }


def targets(case: dict) -> list[dict]:
    """Align gold without target_record or schema helpers."""
    rows = []
    for name, question in case["questions"].items():
        kind = question["type"]
        keys = (
            ["true", "false"]
            if kind == "noul"
            else sorted(question["criteria"], key=int)
            if kind == "score"
            else sorted(question["criteria"])
        )
        gold = case["gold"][name]
        label = str(gold["label"]).lower() if kind == "noul" else gold["label"]
        row = {
            "id": case["id"],
            "workflow": case["workflow"],
            "question": name,
            "type": kind,
            "options": ["yes", "no"] if kind == "noul" else keys,
            "label": keys.index(label),
            "gold_probabilities": normalized(
                [gold["probabilities"][key] for key in keys]
            ),
        }
        if kind == "score":
            row.update(levels=[float(key) for key in keys], gold_score=gold["score"])
        rows.append(row)
    return rows


def key(row: dict) -> tuple[str, str]:
    return row["id"], row["question"]


def fingerprint(case: dict, mode: str, permutations: int) -> str:
    value = {"version": 1, "model": MODEL, "case": case, "permutations": permutations}
    if mode == "multi":
        value.update(mode=mode, format=MULTI_FORMAT)
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def temperature(p: list[float], t: float) -> list[float]:
    logs = [math.log(x) / t if x else -math.inf for x in p]
    return normalized([math.exp(x - max(logs)) for x in logs])


def independent_fit(rows: list[dict]) -> float:
    """Minimize hard-label loss with golden-section search in log temperature."""

    def loss(log_t: float) -> float:
        t = math.exp(log_t)
        total = 0.0
        for row in rows:
            logits = [math.log(max(p, 1e-300)) / t for p in row["probabilities"]]
            maximum = max(logits)
            total += (
                maximum
                + math.log(sum(math.exp(x - maximum) for x in logits))
                - logits[row["label"]]
            )
        return total / len(rows)

    a, b = math.log(0.2), math.log(10)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(55):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = loss(d)
    return math.exp((a + b) / 2)


def audit_report(name: str, expected: dict, cases: list[dict]) -> dict:
    report = json.loads((ROOT / name).read_text())
    rows = report["question_results"]
    calculated = independent_metrics(rows)
    production = bench.metrics(rows)
    mismatches = []
    for row in rows:
        gold = expected.get(key(row))
        if gold is None:
            mismatches.append([key(row), "unknown key"])
            continue
        for field, value in gold.items():
            if isinstance(value, list) and value and isinstance(value[0], float):
                same = (
                    len(value) == len(row[field])
                    and max(abs(a - b) for a, b in zip(value, row[field])) < 1e-12
                )
            else:
                same = value == row[field]
            if not same:
                mismatches.append([key(row), field])
    failed = []
    for line in report.get("failures", {}).get("lines", []):
        case = cases[line - 1]
        selected = [r for r in rows if r["id"] == case["id"]]
        failed.append(
            {
                "line": line,
                "id": case["id"],
                "count": len(selected),
                "uniform": all(
                    all(
                        abs(p - 1 / len(r["options"])) < 1e-12
                        for p in r["probabilities"]
                    )
                    for r in selected
                ),
                "correct": sum(
                    argmax(r["probabilities"]) == r["label"] for r in selected
                ),
            }
        )
    usage = report["cases"]
    latencies = [u["latency_ms"] for u in usage]
    zeros = [
        r
        for r in rows
        if any(
            p == 0 and g > 0
            for p, g in zip(r["probabilities"], r["gold_probabilities"])
        )
    ]
    result = {
        "metrics": calculated,
        "headline_deltas": {
            k: calculated[k] - report["overall"][k]
            for k in production
            if calculated[k] is not None
        },
        "production_deltas": {
            k: calculated[k] - v for k, v in production.items() if v is not None
        },
        "unique_records": len({key(r) for r in rows}),
        "unique_cases": len({r["id"] for r in rows}),
        "case_question_counts": dict(Counter(Counter(r["id"] for r in rows).values())),
        "missing": len(set(expected) - {key(r) for r in rows}),
        "gold_mismatches": mismatches,
        "label_not_gold_argmax": sum(
            r["gold_probabilities"][r["label"]] < max(r["gold_probabilities"])
            for r in rows
        ),
        "label_diff_first_argmax": sum(
            r["label"] != argmax(r["gold_probabilities"]) for r in rows
        ),
        "prediction_ties": sum(
            p.count(max(p)) > 1 for p in [r["probabilities"] for r in rows]
        ),
        "nonfinite": sum(
            not math.isfinite(p) for r in rows for p in r["probabilities"]
        ),
        "negative": sum(p < 0 for r in rows for p in r["probabilities"]),
        "length_mismatches": sum(
            len(r["probabilities"]) != len(r["options"]) for r in rows
        ),
        "max_sum_error": max(abs(sum(r["probabilities"]) - 1) for r in rows),
        "failures": failed,
        "temperatures": report["temperatures"],
        "usage": {
            "unique_cases": len({u["id"] for u in usage}),
            "calls": sum(u.get("calls", u["forward_passes"]) for u in usage),
            "call_histogram": dict(
                Counter(u.get("calls", u["forward_passes"]) for u in usage)
            ),
            "forward_passes": sum(u["forward_passes"] for u in usage),
            "input_tokens": sum(u["input_tokens"] for u in usage),
            "latency_mean": mean(latencies),
            "latency_median": float(np.median(latencies)),
            "latency_p95": float(np.percentile(latencies, 95)),
            "latency_sum_ms": sum(latencies),
        },
        "zero_entries": sum(p == 0 for r in rows for p in r["probabilities"]),
        "zero_rows_positive_gold": len(zeros),
        "zero_hard_gold": sum(r["probabilities"][r["label"]] == 0 for r in rows),
        "zero_log_loss_contribution": sum(
            -g * math.log(1e-300)
            for r in rows
            for p, g in zip(r["probabilities"], normalized(r["gold_probabilities"]))
            if p == 0
        )
        / len(rows),
        "ece_sensitivity": {
            str(b): independent_metrics(rows, b)["ece"] for b in (5, 15, 20, 50)
        },
        "per_type": {
            t: independent_metrics([r for r in rows if r["type"] == t])
            for t in ("choice", "noul", "score")
        },
    }
    if "before_calibration" in report:
        raw = report["before_calibration"]["question_results"]
        raw_by_key = {key(r): r for r in raw}
        result["before"] = independent_metrics(raw)
        result["temperature_max_error"] = max(
            abs(p - q)
            for r in rows
            for p, q in zip(
                r["probabilities"],
                temperature(
                    raw_by_key[key(r)]["probabilities"],
                    report["temperatures"][r["type"]],
                ),
            )
        )
        result["temperature_changed_argmax"] = sum(
            argmax(r["probabilities"]) != argmax(raw_by_key[key(r)]["probabilities"])
            for r in rows
        )
    return result


def main() -> None:
    test = read_jsonl("data/typed_decisions_test.jsonl")
    train = read_jsonl("data/typed_decisions_train.jsonl")
    gold = [r for case in test for r in targets(case)]
    train_gold = [r for case in train for r in targets(case)]
    expected = {key(r): r for r in gold}
    out = {
        "reports": {
            name: audit_report(name, expected, test)
            for name in (
                "report_multi.json",
                "report_full.json",
                "report_multi_p1.json",
            )
        }
    }

    def canonical(case: dict) -> str:
        return json.dumps(case["state"], sort_keys=True)

    out["splits"] = {
        "train_cases": len(train),
        "test_cases": len(test),
        "train_unique_ids": len({c["id"] for c in train}),
        "test_unique_ids": len({c["id"] for c in test}),
        "id_overlap": len({c["id"] for c in train} & {c["id"] for c in test}),
        "state_overlap": len(
            {canonical(c) for c in train} & {canonical(c) for c in test}
        ),
        "train_unique_states": len({canonical(c) for c in train}),
        "test_unique_states": len({canonical(c) for c in test}),
        "workflows": dict(Counter(c["workflow"] for c in test)),
    }

    def group_key(row: dict) -> tuple[str, str]:
        return row["workflow"], row["question"]

    test_counts, train_counts = defaultdict(Counter), defaultdict(Counter)
    for rows, counts in ((gold, test_counts), (train_gold, train_counts)):
        for r in rows:
            counts[group_key(r)][r["label"]] += 1
    majority = {
        k: min(v, key=lambda label: (-v[label], label)) for k, v in test_counts.items()
    }
    prior = {
        k: min(v, key=lambda label: (-v[label], label)) for k, v in train_counts.items()
    }
    uniform = [
        {**r, "probabilities": [1 / len(r["options"])] * len(r["options"])}
        for r in gold
    ]
    out["baselines"] = {
        "test_majority_correct": sum(
            r["label"] == majority[group_key(r)] for r in gold
        ),
        "test_majority_accuracy": mean(
            r["label"] == majority[group_key(r)] for r in gold
        ),
        "train_majority_accuracy": mean(
            r["label"] == prior[group_key(r)] for r in gold
        ),
        "uniform_random_expected_accuracy": mean(1 / len(r["options"]) for r in gold),
        "uniform_first_argmax": independent_metrics(uniform),
        "test_majorities": {
            "/".join(k): {
                "index": majority[k],
                "count": v[majority[k]],
                "labels": dict(v),
            }
            for k, v in test_counts.items()
        },
    }
    cache_lines = read_jsonl("data/preds_train_multi.jsonl")
    cache = {c["fingerprint"]: c["result"] for c in cache_lines}
    training, missing = [], []
    used = set()
    for line, case in enumerate(train, 1):
        fp = fingerprint(case, "multi", 3)
        result = cache.get(fp)
        if result is None:
            missing.append(line)
        else:
            used.add(fp)
        for row in targets(case):
            if result is None:
                p = [1 / len(row["options"])] * len(row["options"])
            else:
                answer = result["answers"][row["question"]]
                p = (
                    [answer["noul"], 1 - answer["noul"]]
                    if row["type"] == "noul"
                    else [answer["probabilities"][k] for k in row["options"]]
                )
            training.append({**row, "probabilities": p})
    saved = json.loads((ROOT / "calibration_multi.json").read_text())
    out["calibration"] = {
        "cache_lines": len(cache_lines),
        "cache_unique": len(cache),
        "matched": len(used),
        "unmatched": len(set(cache) - used),
        "missing_train_lines": missing,
        "test_cache_matches": sum(fingerprint(c, "multi", 3) in cache for c in test),
        "single_cache_matches": sum(
            fingerprint(c, "single", 3) in cache for c in train
        ),
        "p1_cache_matches": sum(fingerprint(c, "multi", 1) in cache for c in train),
        "saved": saved,
        "fitted": {},
    }
    for kind in saved:
        rows = [r for r in training if r["type"] == kind]
        out["calibration"]["fitted"][kind] = {
            "count": len(rows),
            "independent": independent_fit(rows),
            "production": calibrate.fit_temperature(
                [r["probabilities"] for r in rows], [r["label"] for r in rows]
            ),
        }
    raw_sums = [
        sum(g["probabilities"].values()) for c in test for g in c["gold"].values()
    ]
    out["gold_rounding"] = {
        "min_sum": min(raw_sums),
        "max_sum": max(raw_sums),
        "nonunit": sum(s != 1 for s in raw_sums),
        "max_error": max(abs(s - 1) for s in raw_sums),
    }
    score_errors = [
        abs(
            sum(p * level for p, level in zip(r["gold_probabilities"], r["levels"]))
            - r["gold_score"]
        )
        for r in gold
        if r["type"] == "score"
    ]
    out["gold_score_vs_distribution_max_error"] = max(score_errors)
    full = json.loads((ROOT / "report_full.json").read_text())["question_results"]
    multi = json.loads((ROOT / "report_multi.json").read_text())["question_results"]
    old = {key(r): r for r in full}
    changes = Counter()
    case_differences = defaultdict(int)
    for r in multi:
        a = int(argmax(old[key(r)]["probabilities"]) == r["label"])
        b = int(argmax(r["probabilities"]) == r["label"])
        changes[f"{a}->{b}"] += 1
        case_differences[r["id"]] += b - a
    rng = random.Random(42)
    differences = list(case_differences.values())
    bootstrap = sorted(
        sum(rng.choices(differences, k=400)) / 2000 for _ in range(10000)
    )
    out["paired"] = {
        "correctness_changes": dict(changes),
        "accuracy_difference": sum(differences) / 2000,
        "case_bootstrap_95_percentile": [bootstrap[250], bootstrap[9749]],
    }
    out["artifact_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "report_multi.json",
            "report_full.json",
            "report_multi_p1.json",
            "calibration_multi.json",
            "data/typed_decisions_test.jsonl",
            "data/typed_decisions_train.jsonl",
            "data/preds_train_multi.jsonl",
        )
    }
    (ROOT / "scratch/audit_phase2_results.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    for name, result in out["reports"].items():
        print(
            name,
            json.dumps({k: v for k, v in result["metrics"].items() if k != "ece_bins"}),
        )
        print(
            "integrity",
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "unique_records",
                        "gold_mismatches",
                        "missing",
                        "label_not_gold_argmax",
                        "label_diff_first_argmax",
                        "max_sum_error",
                        "failures",
                        "zero_entries",
                        "zero_rows_positive_gold",
                        "zero_hard_gold",
                        "zero_log_loss_contribution",
                    )
                }
            ),
        )
    print("splits", out["splits"])
    print(
        "baselines",
        {
            k: v
            for k, v in out["baselines"].items()
            if k not in ("test_majorities", "uniform_first_argmax")
        },
    )
    print("calibration", out["calibration"])
    print("paired", out["paired"])


if __name__ == "__main__":
    main()
