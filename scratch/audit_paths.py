"""Check data boundaries and parser behavior using only local fake responses."""

import copy
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import pyarrow.parquet as parquet
from pydantic import ValidationError

from audit_phase2 import ROOT, argmax, independent_metrics, read_jsonl
from system1.bench import run_cases
from system1.core import LowCoverageError, SystemOne, read_multi, read_probabilities
from system1.data import convert_row
from system1.schema import Request


class OfflineBackend:
    """Capture prompts and return uniform option distributions."""

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, prompt: str) -> tuple[list[dict], int]:
        self.calls.append((prompt, ""))
        ids = re.findall(r"^([A-Z0-9])\. ", prompt, re.M)
        return [{"token": i, "logprob": -math.log(len(ids))} for i in ids], 0

    def complete_multi(
        self, prompt: str, grammar: str, max_tokens: int
    ) -> tuple[list[dict], int]:
        self.calls.append((prompt, grammar))
        tokens = []
        for number, alternatives in re.findall(
            r'^q\d+ ::= "(Q\d+): " \((.*?)\)', grammar, re.M
        ):
            ids = re.findall(r'"([A-Za-z0-9])"', alternatives)
            top = [{"token": i, "logprob": -math.log(len(ids))} for i in ids]
            tokens.extend(
                [
                    {"token": number + ": ", "top_logprobs": []},
                    {"token": ids[0], "top_logprobs": top},
                    {"token": "\n", "top_logprobs": []},
                ]
            )
        return tokens, 0


def suspicious_keys(value: object, path: str = "") -> list[str]:
    hits = []
    if isinstance(value, dict):
        for k, v in value.items():
            current = path + "/" + k
            if k in {
                "gold",
                "gold_score",
                "gold_probabilities",
                "answers",
                "label",
                "factors",
                "label_agreement",
            }:
                hits.append(current)
            hits.extend(suspicious_keys(v, current))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            hits.extend(suspicious_keys(v, path + "/" + str(i)))
    return hits


def main() -> None:
    test = read_jsonl("data/typed_decisions_test.jsonl")
    train = read_jsonl("data/typed_decisions_train.jsonl")
    out = {"conversion": {}, "prompt_gold_mutation": {}}
    for split, cases in (("train", train), ("test", test)):
        raw = parquet.read_table(
            ROOT / f"data/raw/typed_decisions/all_{split}.parquet"
        ).to_pylist()
        out["conversion"][split] = {
            "rows": len(raw),
            "convert_row_mismatches": sum(
                convert_row(a) != b for a, b in zip(raw, cases)
            ),
            "length_match": len(raw) == len(cases),
        }
    altered = copy.deepcopy(test)
    for case in altered:
        for name, gold in case["gold"].items():
            keys = list(gold["probabilities"])
            winner = next(k for k in keys if k != str(gold["label"]).lower())
            gold["probabilities"] = {k: float(k == winner) for k in keys}
            gold["label"] = winner
            gold["confidence"] = 1.0
            if "score" in gold:
                gold["score"] = float(winner)
            if "noul" in gold:
                gold["noul"] = float(winner == "true")
            case["answers"][name] = winner
        case["gold_score"] = "AUDIT_GOLD_SENTINEL"
        case["majority_answers"] = "AUDIT_MAJORITY_SENTINEL"
    altered_path = ROOT / "scratch/altered_gold.jsonl"
    altered_path.write_text("\n".join(json.dumps(c) for c in altered) + "\n")
    for mode in ("single", "multi"):
        first, second = OfflineBackend(), OfflineBackend()
        original, _ = run_cases(
            ROOT / "data/typed_decisions_test.jsonl", SystemOne(first), 3, mode=mode
        )
        changed, _ = run_cases(altered_path, SystemOne(second), 3, mode=mode)
        out["prompt_gold_mutation"][mode] = {
            "cases": len(test),
            "captured_calls": len(first.calls),
            "identical_prompts_and_grammars": first.calls == second.calls,
            "identical_predictions": all(
                a["probabilities"] == b["probabilities"]
                for a, b in zip(original, changed)
            ),
            "sentinel_in_prompts": any("AUDIT_" in p for p, _ in second.calls),
        }
    rejected = []
    for field in (
        "gold",
        "gold_score",
        "gold_probabilities",
        "label",
        "answers",
        "majority_answers",
    ):
        payload = {
            "state": test[0]["state"],
            "questions": copy.deepcopy(test[0]["questions"]),
        }
        for where in ("request", "question"):
            candidate = copy.deepcopy(payload)
            target = (
                candidate
                if where == "request"
                else next(iter(candidate["questions"].values()))
            )
            target[field] = "AUDIT_SENTINEL"
            try:
                Request.model_validate(candidate)
            except ValidationError:
                rejected.append(where + "/" + field)
    out["schema_rejected_extra_keys"] = rejected
    out["input_target_key_hits"] = [
        c["id"] + p
        for c in train + test
        for part in ("state", "questions")
        for p in suspicious_keys(c[part], part)
    ]
    out["parser"] = {}
    p, coverage = read_probabilities(
        [
            {"token": " A", "logprob": math.log(0.3)},
            {"token": "A", "logprob": math.log(0.2)},
            {"token": "B", "logprob": math.log(0.1)},
            {"token": "a", "logprob": math.log(0.4)},
        ],
        ["A", "B"],
    )
    out["parser"]["whitespace_aggregate"] = {"p": p, "coverage": coverage}
    for value in (0.499999, 0.5):
        try:
            p, coverage = read_probabilities(
                [{"token": "A", "logprob": math.log(value)}], ["A", "B"]
            )
            out["parser"][str(value)] = {"accepted": True, "p": p, "coverage": coverage}
        except LowCoverageError:
            out["parser"][str(value)] = {"accepted": False}
    tokens = [
        {"token": "Q1: ", "top_logprobs": []},
        {
            "token": "A",
            "top_logprobs": [
                {"token": "A", "logprob": math.log(0.6)},
                {"token": "B", "logprob": math.log(0.3)},
            ],
        },
        {"token": "\n", "top_logprobs": []},
    ]
    out["parser"]["answer_token_read"] = read_multi(tokens, [["A", "B"]])
    try:
        read_multi(
            [{"token": "Q1: A\n", "top_logprobs": tokens[1]["top_logprobs"]}],
            [["A", "B"]],
        )
        out["parser"]["merged_rejected"] = False
    except LowCoverageError:
        out["parser"]["merged_rejected"] = True
    out["reports"] = {}
    for name in ("report_multi.json", "report_full.json", "report_multi_p1.json"):
        report = json.loads((ROOT / name).read_text())
        rows = report["question_results"]
        groups = defaultdict(list)
        for r in rows:
            groups[(r["workflow"], r["question"])].append(r)
        first_gold = [{**r, "label": argmax(r["gold_probabilities"])} for r in rows]
        out["reports"][name] = {
            "macro_question_ece": mean(
                independent_metrics(g)["ece"] for g in groups.values()
            ),
            "first_gold_argmax_accuracy": independent_metrics(first_gold)["accuracy"],
            "ties_in_gold": sum(
                r["gold_probabilities"].count(max(r["gold_probabilities"])) > 1
                for r in rows
            ),
            "per_workflow": {
                w: independent_metrics([r for r in rows if r["workflow"] == w])[
                    "accuracy"
                ]
                for w in sorted({r["workflow"] for r in rows})
            },
        }
    out["option_counts_per_case"] = dict(
        Counter(
            sum(
                len(q.get("criteria", {"true": None, "false": None}))
                for q in c["questions"].values()
            )
            for c in test
        )
    )
    out["questions_without_criteria"] = dict(
        Counter(
            q["type"]
            for c in test
            for q in c["questions"].values()
            if "criteria" not in q
        )
    )
    rng = random.Random(42)
    orders = []
    for _ in range(3):
        order = list(range(5))
        rng.shuffle(order)
        orders.append(order)
    out["question_index_orders"] = orders
    Path(ROOT / "scratch/audit_paths_results.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
