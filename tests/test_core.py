import math

import pytest
from pydantic import ValidationError

from system1 import SystemOne, system_one
from system1.backend import ALPHABET, FakeBackend
from system1.core import LowCoverageError, build_prompt, confidence, read_probabilities
from system1.schema import Choice, Noul, Request, Score


def tokens(masses):
    return [
        {"token": token, "logprob": math.log(mass)} for token, mass in masses.items()
    ]


def test_choice_prompt():
    question = Choice(
        type="choice",
        instructions="Which team?",
        criteria={"billing": "money", "sales": None},
    )
    assert build_prompt(
        {"text": "refund"}, question, {"A": "billing", "a": "sales"}
    ) == (
        'State:\n{\n  "text": "refund"\n}\n\nQuestion: Which team?\nOptions:\n'
        "A. billing: money\na. sales\nAnswer with the option id only."
    )


def test_score_and_noul_prompts():
    score = Score(type="score", instructions="Urgency?", levels=["low", "high"])
    assert "Options:\nB. low\nC. high" in build_prompt(
        [1, 2], score, {"B": "low", "C": "high"}
    )
    noul = Noul(type="noul", instructions="It is urgent.")
    assert build_prompt("hello", noul, {"A": "yes", "B": "no"}) == (
        "State:\nhello\n\nQuestion: Is this statement true? It is urgent.\n"
        "Options:\nA. yes\nB. no\nAnswer with the option id only."
    )


def test_reading_strips_spaces_preserves_case_and_sums_variants():
    probabilities, coverage = read_probabilities(
        tokens({" A": 0.3, "A\n": 0.2, "a": 0.1, "B": 0.1, "x": 0.3}), ["A", "a", "B"]
    )
    assert coverage == pytest.approx(0.7)
    assert probabilities == pytest.approx([5 / 7, 1 / 7, 1 / 7])


@pytest.mark.parametrize("masses", [{"a": 0.9}, {"A": 0.49, "x": 0.51}, {}])
def test_low_coverage_is_explicit(masses):
    with pytest.raises(LowCoverageError, match="top tokens"):
        read_probabilities(tokens(masses), ["A", "B"])


def test_coverage_boundary_and_missing_option():
    assert read_probabilities(tokens({"A": 0.5}), ["A", "B"]) == ([1.0, 0.0], 0.5)


@pytest.mark.parametrize(
    "n,top,expected", [(1, 1.0, 1.0), (2, 0.8, 0.6), (3, 0.8, 0.7), (5, 0.8, 0.75)]
)
def test_confidence(n, top, expected):
    values = [top] + [(1 - top) / (n - 1)] * (n - 1) if n > 1 else [1.0]
    assert confidence(values) == pytest.approx(expected)
    assert confidence([1 / n] * n) == pytest.approx(0.0 if n > 1 else 1.0)


def test_permutations_map_back_to_labels_and_ignore_input_order():
    responses = [
        tokens({"A": 0.8, "B": 0.1, "C": 0.1}),
        tokens({"A": 0.1, "B": 0.1, "C": 0.8}),
        tokens({"A": 0.8, "B": 0.1, "C": 0.1}),
    ]
    backends, answers = [], []
    for labels in (["billing", "tech", "sales"], ["sales", "billing", "tech"]):
        backend = FakeBackend(responses)
        result = system_one(
            "refund",
            {
                "route": {
                    "type": "choice",
                    "instructions": "Route?",
                    "criteria": dict.fromkeys(labels),
                }
            },
            backend=backend,
        )
        answers.append(result["answers"]["route"])
        backends.append(backend)
        assert result["usage"]["forward_passes"] == 3
        assert result["usage"]["input_tokens"] == 30
    assert answers[0] == answers[1]
    assert backends[0].prompts == backends[1].prompts
    assert len(set(backends[0].prompts)) == 3
    expected = dict.fromkeys(["billing", "tech", "sales"], 0.0)
    for prompt, response in zip(backends[0].prompts, responses):
        mapping = {
            line[0]: line[3:]
            for line in prompt.splitlines()
            if line[:3] in ["A. ", "B. ", "C. "]
        }
        for item in response:
            expected[mapping[item["token"]]] += math.exp(item["logprob"]) / 3
    assert answers[0]["probabilities"] == pytest.approx(expected)


def test_score_order_rotates_ids_and_expected_index():
    backend = FakeBackend(
        [
            tokens({"A": 0.2, "B": 0.8}),
            tokens({"B": 0.2, "C": 0.8}),
            tokens({"C": 0.2, "D": 0.8}),
        ]
    )
    result = system_one(
        "urgent",
        {
            "urgency": {
                "type": "score",
                "instructions": "Urgency?",
                "levels": ["low", "high"],
            }
        },
        backend=backend,
    )
    assert result["answers"]["urgency"]["score"] == "high"
    assert result["answers"]["urgency"]["expected_index"] == pytest.approx(0.8)
    for i, prompt in enumerate(backend.prompts):
        assert f"{ALPHABET[i]}. low\n{ALPHABET[i + 1]}. high" in prompt


def test_choice_ties_do_not_depend_on_input_order():
    answers = []
    for labels in (["billing", "tech"], ["tech", "billing"]):
        backend = FakeBackend([tokens({"A": 0.5, "B": 0.5})] * 3)
        result = system_one(
            "",
            {
                "q": {
                    "type": "choice",
                    "instructions": "Team?",
                    "criteria": dict.fromkeys(labels),
                }
            },
            backend=backend,
        )
        answers.append(result["answers"]["q"]["choice"])
    assert answers == ["billing", "billing"]


def test_tournament_70_options_preserves_group_mass():
    first = {token: 0.8 / 35 for token in ALPHABET}
    first["A"] = 0.2
    second = {token: 0.4 / 33 for token in ALPHABET[:34]}
    second["A"] = 0.6
    backend = FakeBackend([tokens(first), tokens(second), tokens({"A": 0.7, "B": 0.3})])
    levels = [f"level{i}" for i in range(70)]
    result = system_one(
        "state",
        {"q": {"type": "score", "instructions": "Rank?", "levels": levels}},
        permutations=1,
        backend=backend,
    )
    probabilities = result["answers"]["q"]["probabilities"]
    assert len(backend.prompts) == 3
    assert "9. level35" in backend.prompts[0]
    assert "7. level69" in backend.prompts[1]
    assert "A. level0\nB. level36" in backend.prompts[2]
    assert probabilities["level0"] == pytest.approx(0.14)
    assert probabilities["level36"] == pytest.approx(0.18)
    assert sum(probabilities.values()) == pytest.approx(1)
    assert result["usage"]["forward_passes"] == 3


def test_recursive_tournament_with_small_alphabet():
    backend = FakeBackend([tokens({"A": 0.6, "B": 0.4})] * 6, alphabet="AB")
    request = Request(
        state="",
        questions={
            "q": {
                "type": "choice",
                "instructions": "",
                "criteria": dict.fromkeys("abcde"),
            }
        },
        permutations=1,
    )
    result = SystemOne(backend).decide(request)
    assert result["usage"]["forward_passes"] == 6
    assert sum(result["answers"]["q"]["probabilities"].values()) == pytest.approx(1)


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "", "criteria": {}},
        {
            "type": "choice",
            "instructions": "",
            "criteria": dict.fromkeys(map(str, range(256))),
        },
        {"type": "score", "instructions": "", "levels": ["a", "a"]},
        {"type": "score", "instructions": "", "levels": []},
        {"type": "noul", "instructions": "", "levels": ["yes"]},
    ],
)
def test_invalid_questions_fail_before_backend(question):
    with pytest.raises(ValidationError):
        system_one("", {"q": question}, backend=FakeBackend([]))


def test_criteria_descriptions_and_level_order():
    score = Score(
        type="score", instructions="Risk?", criteria={"10": "high", "2": "low"}
    )
    assert score.levels == ["2", "10"]
    assert "A. 2: low\nB. 10: high" in build_prompt("", score, {"A": "2", "B": "10"})
    named = Score(
        type="score", instructions="Risk?", criteria={"low": None, "high": "danger"}
    )
    assert named.levels == ["low", "high"]
    noul = Noul(
        type="noul",
        instructions="Review?",
        criteria={"false": "No review", "true": "Inspect"},
    )
    assert "A. yes: Inspect\nB. no: No review" in build_prompt(
        "", noul, {"A": "yes", "B": "no"}
    )


def test_request_accepts_model_but_rejects_unknown_fields():
    payload = {
        "state": "",
        "model": "ignored",
        "questions": {"q": {"type": "noul", "instructions": "?"}},
    }
    assert Request(**payload).model == "ignored"
    with pytest.raises(ValidationError):
        Request(**payload, unknown=True)


@pytest.mark.parametrize(
    "question",
    [
        {"type": "score", "instructions": "?"},
        {"type": "score", "instructions": "?", "criteria": {}},
        {
            "type": "score",
            "instructions": "?",
            "levels": ["a"],
            "criteria": {"b": None},
        },
        {"type": "noul", "instructions": "?", "criteria": {"yes": "yes", "no": "no"}},
        {"type": "noul", "instructions": "?", "criteria": {"true": "yes"}},
    ],
)
def test_invalid_review_schema(question):
    with pytest.raises(ValidationError):
        Request(state="", questions={"q": question})


def test_option_alphabet_is_uppercase_then_digits():
    assert ALPHABET == "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
