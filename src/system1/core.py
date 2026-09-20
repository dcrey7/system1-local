"""Prompt construction and probability decisions."""

import json
import math
import random
from pathlib import Path
from time import perf_counter

import numpy as np

from system1.backend import Backend, GemmaBackend, GeneratedToken, TopToken
from system1.calibrate import apply_temperature, load_calibration
from system1.schema import Mode, Noul, QuestionBase, Request, Score, State, options

MODEL = "system1-gemma-4-12b"
MULTI_FORMAT = 5


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


def build_multi_prompt(
    state: State,
    questions: dict[str, QuestionBase],
    assigned: dict[str, dict[str, str]],
    order: list[str],
) -> str:
    """Number questions in prompt order and list their assigned option ids."""
    text = (
        state
        if isinstance(state, str)
        else json.dumps(state, indent=2, ensure_ascii=False)
    )
    lines = [f"State:\n{text}\n\nQuestions:"]
    for index, name in enumerate(order, 1):
        question = questions[name]
        instruction = question.instructions
        if isinstance(question, Noul):
            instruction = "Is this statement true? " + instruction
        lines.append(f"Q{index}: {instruction}")
        descriptions = options(question)
        lines.extend(
            f"  {token}. {label}"
            + (f": {descriptions[label]}" if descriptions[label] else "")
            for token, label in assigned[name].items()
        )
    lines.append(
        "\nAnswer every question with its option id only, one per line, "
        'in the form "Q1: <id>".'
    )
    return "\n".join(lines)


def build_grammar(order_ids: list[list[str]]) -> str:
    """Require one numbered answer line for each question."""
    if not order_ids or any(not ids for ids in order_ids):
        raise ValueError("Grammar requires at least one id per question")
    if any(
        len(token) != 1 or not token.isascii() or not token.isalnum()
        for ids in order_ids
        for token in ids
    ):
        raise ValueError("Grammar ids must be single ASCII alphanumerics")
    rules = ["root ::= " + " ".join(f"q{index}" for index in range(len(order_ids)))]
    for index, ids in enumerate(order_ids):
        alternatives = " | ".join(f'"{token}"' for token in ids)
        rules.append(f'q{index} ::= "Q{index + 1}: " ({alternatives}) "\\n"')
    return "\n".join(rules)


def read_multi(
    tokens: list[GeneratedToken], ids_per_question: list[list[str]]
) -> list[tuple[list[float], float]]:
    """Read the distribution on the token that completes each answer id."""
    text = ""
    offset = 0
    results = []
    needs_newline = False
    for item in tokens:
        text += item["token"]
        while offset < len(text):
            remaining = text[offset:]
            if needs_newline:
                if not remaining.startswith("\n"):
                    raise LowCoverageError(f"Expected an answer newline: {text!r}")
                offset += 1
                needs_newline = False
                continue
            if len(results) == len(ids_per_question):
                raise LowCoverageError(f"Unexpected text after the answers: {text!r}")
            ids = ids_per_question[len(results)]
            prefix = f"Q{len(results) + 1}: "
            if len(remaining) < len(prefix) + 1:
                if not any((prefix + token).startswith(remaining) for token in ids):
                    raise LowCoverageError(f"Invalid answer prefix: {text!r}")
                break
            answer = remaining[len(prefix)]
            if not remaining.startswith(prefix) or answer not in ids:
                raise LowCoverageError(f"Invalid answer line: {text!r}")
            if item["token"].strip() != answer:
                raise LowCoverageError(
                    f"Answer id is in a merged token: {item['token']!r}"
                )
            results.append(
                read_probabilities(
                    item["top_logprobs"], ids, floor=True, min_coverage=0.05
                )
            )
            offset += len(prefix) + 1
            needs_newline = True
    if len(results) != len(ids_per_question) or needs_newline:
        raise LowCoverageError(f"Incomplete multi answer: {text!r}")
    return results


def read_probabilities(
    top_tokens: list[TopToken],
    ids: list[str],
    floor: bool = False,
    min_coverage: float = 0.5,
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
    if coverage < min_coverage:
        raise LowCoverageError(
            f"Option coverage {coverage:.6f} is below {min_coverage}; "
            f"top tokens: {top_tokens!r}"
        )
    total = coverage
    if floor:
        minimum = min(math.exp(top_tokens[-1]["logprob"]), 0.01)
        mass = {token: value if value > 0 else minimum for token, value in mass.items()}
        total = sum(mass.values())
    return [value / total for value in mass.values()], coverage


def confidence(probabilities: list[float]) -> float:
    """Return concentration, not a calibrated probability of correctness."""
    n = len(probabilities)
    return (n * max(probabilities) - 1) / (n - 1) if n > 1 else 1.0


class SystemOne:
    def __init__(
        self,
        backend: Backend,
        calibration_path: Path = Path("calibration.json"),
        multi_calibration_path: Path | None = None,
    ) -> None:
        self.backend = backend
        self.calibration_path = calibration_path
        self.multi_calibration_path = (
            multi_calibration_path
            or calibration_path.with_name("calibration_multi.json")
        )

    def calibration_for(self, mode: Mode) -> Path:
        if mode == "single":
            return self.calibration_path
        if mode == "multi":
            return self.multi_calibration_path
        raise ValueError("Mode must be single or multi")

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

    def decide(
        self, request: Request, calibrated: bool = True, mode: Mode = "single"
    ) -> dict:
        start = perf_counter()
        calibration_path = self.calibration_for(mode)
        temperatures = load_calibration(calibration_path) if calibrated else {}
        alphabet = self.backend.alphabet
        if len(alphabet) < 2 or len(set(alphabet)) != len(alphabet):
            raise ValueError("The alphabet must contain at least two unique ids")
        usage = {"input_tokens": 0, "forward_passes": 0}
        multi_totals, multi_coverages = (
            self._multi(request, alphabet, usage) if mode == "multi" else ({}, {})
        )
        answers = {}
        for name, question in request.questions.items():
            labels = list(options(question))
            if mode == "multi":
                totals = multi_totals[name]
                coverages = multi_coverages[name]
            else:
                totals = dict.fromkeys(labels, 0.0)
                coverages: list[float] = []
                rng = random.Random(42)
                previous: list[str] = []
                for permutation in range(request.permutations):
                    order = (
                        labels.copy() if isinstance(question, Score) else sorted(labels)
                    )
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
            "usage": {
                **usage,
                "calls": usage["calls"] if mode == "multi" else usage["forward_passes"],
                "latency_ms": (perf_counter() - start) * 1000,
            },
        }

    def _multi(
        self, request: Request, alphabet: str, usage: dict[str, int]
    ) -> tuple[dict[str, dict[str, float]], dict[str, list[float]]]:
        """Average grouped answer streams with unique ids per call."""
        labels = {name: list(options(q)) for name, q in request.questions.items()}
        if any(len(values) > len(alphabet) for values in labels.values()):
            raise ValueError(
                "Multi mode requires each question to fit the option alphabet; "
                "use single mode for larger questions"
            )
        totals = {name: dict.fromkeys(values, 0.0) for name, values in labels.items()}
        coverages: dict[str, list[float]] = {name: [] for name in labels}
        option_rngs = {name: random.Random(42) for name in labels}
        previous: dict[str, list[str]] = {}
        usage["calls"] = 0
        for permutation in range(request.permutations):
            order = list(request.questions)
            groups: list[list[str]] = [[]]
            size = 0
            for name in order:
                if size + len(labels[name]) > len(alphabet):
                    groups.append([])
                    size = 0
                groups[-1].append(name)
                size += len(labels[name])
            for group in groups:
                assigned = {}
                start = 0
                for name in group:
                    ids = alphabet[start : start + len(labels[name])]
                    start += len(labels[name])
                    if isinstance(request.questions[name], Score):
                        option_order = labels[name].copy()
                        offset = permutation % len(ids)
                        ids = ids[offset:] + ids[:offset]
                    else:
                        option_order = sorted(labels[name])
                        option_rngs[name].shuffle(option_order)
                        if option_order == previous.get(name) and len(option_order) > 1:
                            option_order = option_order[1:] + option_order[:1]
                    previous[name] = option_order
                    assigned[name] = dict(zip(ids, option_order))
                ids_per_question = [list(assigned[name]) for name in group]
                tokens, count = self.backend.complete_multi(
                    build_multi_prompt(
                        request.state, request.questions, assigned, group
                    ),
                    build_grammar(ids_per_question),
                    max_tokens=8 * len(group) + 8,
                )
                usage["calls"] += 1
                usage["input_tokens"] += count
                usage["forward_passes"] += len(tokens)
                for name, (values, coverage) in zip(
                    group, read_multi(tokens, ids_per_question)
                ):
                    coverages[name].append(coverage)
                    for label, probability in zip(assigned[name].values(), values):
                        totals[name][label] += probability / request.permutations
        return totals, coverages


def system_one(
    state: State,
    questions: dict,
    *,
    permutations: int = 3,
    backend: Backend | None = None,
    mode: Mode = "single",
) -> dict:
    """Answer typed questions about a state."""
    request = Request(
        state=state, questions=questions, permutations=permutations, mode=mode
    )
    if backend is not None:
        return SystemOne(backend).decide(request, mode=mode)
    live = GemmaBackend()
    try:
        return SystemOne(live).decide(request, mode=mode)
    finally:
        live.close()
