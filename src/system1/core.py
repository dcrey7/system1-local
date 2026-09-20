"""Prompt construction and probability decisions."""

import json
import math
import random
from pathlib import Path
from time import perf_counter

import numpy as np

from system1.backend import Backend, GemmaBackend, TopToken
from system1.calibrate import apply_temperature, load_calibration
from system1.schema import Noul, QuestionBase, Request, Score, State, options

MODEL = "system1-gemma-4-12b"


class LowCoverageError(ValueError):
    """The returned token distribution omits too much option mass."""


def build_prompt(state: State, question: QuestionBase, assigned: dict[str, str]) -> str:
    text = (
        state
        if isinstance(state, str)
        else json.dumps(state, indent=2, ensure_ascii=False)
    )
    instruction = question.instructions
    if isinstance(question, Noul):
        instruction = "Is this statement true? " + instruction
    descriptions = options(question)
    lines = [
        f"{token}. {label}"
        + (f": {descriptions[label]}" if descriptions[label] else "")
        for token, label in assigned.items()
    ]
    return (
        f"State:\n{text}\n\nQuestion: {instruction}\nOptions:\n"
        + "\n".join(lines)
        + "\nAnswer with the option id only."
    )


def read_probabilities(
    top_tokens: list[TopToken], ids: list[str]
) -> tuple[list[float], float]:
    mass = dict.fromkeys(ids, 0.0)
    for item in top_tokens:
        logprob = item["logprob"]
        if math.isnan(logprob) or logprob > 0:
            raise ValueError("Invalid token log probability")
        token = item["token"].strip()
        # Lowercase tokens must not contribute mass to uppercase IDs.
        if token in mass:
            mass[token] += math.exp(logprob)
    coverage = sum(mass.values())
    if coverage < 0.5:
        raise LowCoverageError(
            f"Option coverage {coverage:.6f} is below 0.5; top tokens: {top_tokens!r}"
        )
    return [value / coverage for value in mass.values()], coverage


def confidence(probabilities: list[float]) -> float:
    """Return concentration, not a calibrated probability of correctness."""
    n = len(probabilities)
    return (n * max(probabilities) - 1) / (n - 1) if n > 1 else 1.0


class SystemOne:
    def __init__(
        self, backend: Backend, calibration_path: Path = Path("calibration.json")
    ) -> None:
        self.backend = backend
        self.calibration_path = calibration_path

    def _round(
        self,
        state: State,
        question: QuestionBase,
        labels: list[str],
        alphabet: str,
        usage: dict[str, int],
        coverages: list[float],
    ) -> dict[str, float]:
        if len(labels) > len(alphabet):
            groups = [
                labels[i : i + len(alphabet)]
                for i in range(0, len(labels), len(alphabet))
            ]
            distributions = [
                self._round(state, question, group, alphabet, usage, coverages)
                for group in groups
            ]
            winners = [max(group, key=group.get) for group in distributions]
            final = self._round(state, question, winners, alphabet, usage, coverages)
            return {
                label: probability * final[winner]
                for group, winner in zip(distributions, winners)
                for label, probability in group.items()
            }
        assigned = dict(zip(alphabet, labels))
        tokens, count = self.backend.complete(build_prompt(state, question, assigned))
        usage["input_tokens"] += count
        usage["forward_passes"] += 1
        probabilities, coverage = read_probabilities(tokens, list(assigned))
        coverages.append(coverage)
        return dict(zip(labels, probabilities))

    def decide(self, request: Request, calibrated: bool = True) -> dict:
        start = perf_counter()
        temperatures = load_calibration(self.calibration_path) if calibrated else {}
        alphabet = self.backend.alphabet
        if len(alphabet) < 2 or len(set(alphabet)) != len(alphabet):
            raise ValueError("The alphabet must contain at least two unique ids")
        usage = {"input_tokens": 0, "forward_passes": 0}
        answers = {}
        for name, question in request.questions.items():
            labels = list(options(question))
            totals = dict.fromkeys(labels, 0.0)
            coverages: list[float] = []
            rng = random.Random(42)
            previous: list[str] = []
            for permutation in range(request.permutations):
                order = labels.copy() if isinstance(question, Score) else sorted(labels)
                ids = alphabet
                if isinstance(question, Score):
                    offset = permutation % len(alphabet)
                    ids = alphabet[offset:] + alphabet[:offset]
                else:
                    rng.shuffle(order)
                    if order == previous and len(order) > 1:
                        order = order[1:] + order[:1]
                previous = order
                probabilities = self._round(
                    request.state, question, order, ids, usage, coverages
                )
                for label, probability in probabilities.items():
                    totals[label] += probability / request.permutations
            values = apply_temperature(
                list(totals.values()), temperatures.get(question.type, 1)
            )
            probabilities = dict(zip(labels, values.tolist()))
            answer = {"type": question.type, "coverage": float(np.mean(coverages))}
            if isinstance(question, Noul):
                answer["noul"] = probabilities["yes"]
            else:
                candidates = labels if isinstance(question, Score) else sorted(labels)
                answer.update(
                    {
                        question.type: max(candidates, key=probabilities.__getitem__),
                        "probabilities": probabilities,
                        "confidence": confidence(values.tolist()),
                    }
                )
                if isinstance(question, Score):
                    answer["expected_index"] = float(
                        np.dot(np.arange(len(labels)), values)
                    )
            answers[name] = answer
        return {
            "model": MODEL,
            "answers": answers,
            "usage": {**usage, "latency_ms": (perf_counter() - start) * 1000},
        }


def system_one(
    state: State,
    questions: dict,
    *,
    permutations: int = 3,
    backend: Backend | None = None,
) -> dict:
    """Answer typed questions about a state."""
    request = Request(state=state, questions=questions, permutations=permutations)
    if backend is not None:
        return SystemOne(backend).decide(request)
    live = GemmaBackend()
    try:
        return SystemOne(live).decide(request)
    finally:
        live.close()
