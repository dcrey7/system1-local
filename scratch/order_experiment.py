"""Question order experiment: 1 call per case, fixed question orders, raw accuracy.

Each order is written as a reordered copy of the test set, so the result does not
depend on how the engine orders questions (the engine gets the identity order).
Run with the server up: uv run python scratch/order_experiment.py [--limit N] [--orders a,b]
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from system1 import bench, core
from system1.backend import GemmaBackend
from system1.core import SystemOne

ORDERS: dict[str, list[int] | str] = {
    "given": [0, 1, 2, 3, 4],
    "reversed": [4, 3, 2, 1, 0],
    "urgency_first": [4, 0, 1, 2, 3],
    "rev2_shuffle0": [3, 1, 2, 4, 0],
    "rev3_shuffle0": [4, 2, 1, 0, 3],
    "facts_first": "noul,choice,score",
    "score_first": "score,choice,noul",
}
OUT = Path("/home/abhishek/.claude/jobs/872a19fc/tmp/order_data")


class IdentityRandom(random.Random):
    """Question order seeds (1000 and up) keep the given order; other seeds shuffle for real."""

    def __init__(self, seed=None):
        super().__init__(seed)
        self._seed_value = seed

    def shuffle(self, x):
        seed = self._seed_value
        if isinstance(seed, int) and seed >= 1000:
            return
        super().shuffle(x)


def reorder(questions: dict, perm: list[int] | str) -> dict:
    names = list(questions)
    if isinstance(perm, str):
        rank = perm.split(",")
        names = sorted(names, key=lambda name: rank.index(questions[name]["type"]))
    else:
        names = [names[i] for i in perm]
    return {name: questions[name] for name in names}


def write_order(source: Path, name: str, perm: list[int] | str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{name}.jsonl"
    with target.open("w") as out:
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            case["questions"] = reorder(case["questions"], perm)
            out.write(json.dumps(case) + "\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--orders", default=",".join(ORDERS))
    args = parser.parse_args()
    core.random.Random = IdentityRandom
    source = Path("data/typed_decisions_test.jsonl")
    engine = SystemOne(GemmaBackend())
    for name in args.orders.split(","):
        path = write_order(source, name, ORDERS[name])
        records, usages = bench.run_cases(path, engine, 1, limit=args.limit, mode="multi")
        overall = bench.metrics(records)
        by_type = defaultdict(list)
        by_question = defaultdict(list)
        for r in records:
            by_type[r["type"]].append(r)
            by_question[r["question"]].append(r)
        latency = sum(u["latency_ms"] for u in usages) / len(usages)
        per_question = {q: round(bench.metrics(rs)["accuracy"], 3) for q, rs in sorted(by_question.items())}
        print(
            f"{name:14} {ORDERS[name]}  acc {overall['accuracy']:.4f}  "
            f"choice {bench.metrics(by_type['choice'])['accuracy']:.4f}  "
            f"noul {bench.metrics(by_type['noul'])['accuracy']:.4f}  "
            f"score {bench.metrics(by_type['score'])['accuracy']:.4f}  "
            f"urgency {bench.metrics(by_question['urgency'])['accuracy']:.4f}  "
            f"{latency:.0f} ms/case",
            flush=True,
        )
        print(f"  per question {per_question}", flush=True)
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
