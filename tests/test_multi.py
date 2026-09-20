import json
import math
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from system1.backend import FakeBackend, GemmaBackend, SERVER_LOCK
from system1.calibrate import save_calibration
from system1.core import (
    LowCoverageError,
    SystemOne,
    build_grammar,
    build_multi_prompt,
    read_multi,
    system_one,
)
from system1.schema import Choice, Noul, Request


def top_tokens(distribution):
    return [
        {"token": token, "logprob": math.log(mass)}
        for token, mass in distribution.items()
    ]


def generated(token, distribution=None):
    return {"token": token, "top_logprobs": top_tokens(distribution or {})}


def test_multi_prompt_numbering_and_descriptions():
    questions = {
        "route": Choice(
            type="choice",
            instructions="Team?",
            criteria={"billing": "money", "tech": None},
        ),
        "angry": Noul(
            type="noul",
            instructions="Angry?",
            criteria={"true": "upset", "false": None},
        ),
    }
    assigned = {
        "route": {"C": "tech", "D": "billing"},
        "angry": {"B": "no", "A": "yes"},
    }
    assert build_multi_prompt(
        {"text": "refund"}, questions, assigned, ["angry", "route"]
    ) == (
        'State:\n{\n  "text": "refund"\n}\n\nQuestions:\n'
        "Q1: Is this statement true? Angry?\n  B. no\n  A. yes: upset\n"
        "Q2: Team?\n  C. tech\n  D. billing: money\n\n"
        'Answer every question with its option id only, one per line, in the form "Q1: <id>".'
    )


def test_multi_grammar():
    assert build_grammar([list("ABC"), list("DE")]) == (
        "root ::= q0 q1\n"
        'q0 ::= "Q1: " ("A" | "B" | "C") "\\n"\n'
        'q1 ::= "Q2: " ("D" | "E") "\\n"'
    )


@pytest.mark.parametrize("ids", [[], [[]], [["AA"]], [['"']], [["é"]]])
def test_grammar_rejects_invalid_ids(ids):
    with pytest.raises(ValueError):
        build_grammar(ids)


def test_read_multi_fake_stream():
    backend = FakeBackend(
        multi_responses={"Q1": {"A": 0.2, "C": 0.8}, "Q2": {"D": 0.7, "E": 0.3}}
    )
    stream, count = backend.complete_multi(
        "Questions:\nQ1: Team?\nQ2: Risk?", build_grammar([list("ABC"), list("DE")]), 24
    )
    assert [item["token"] for item in stream] == [
        "Q1",
        ":",
        " C",
        "\n",
        "Q2",
        ":",
        " D",
        "\n",
    ]
    results = read_multi(stream, [list("ABC"), list("DE")])
    assert results[0][0] == pytest.approx([0.2, 0, 0.8])
    assert results[1][0] == pytest.approx([0.7, 0.3])
    assert [coverage for _, coverage in results] == pytest.approx([1, 1])
    assert count == 10


@pytest.mark.parametrize(
    "prefix,answer",
    [(["Q", "1", ":", " "], "C"), (["Q1: "], "C\n"), (["Q1", ":"], " C")],
)
def test_read_multi_split_prefix_and_id_whitespace(prefix, answer):
    stream = [generated(token) for token in prefix]
    stream.append(generated(answer, {" C": 0.6, "A": 0.2, "other": 0.2}))
    if not answer.endswith("\n"):
        stream.append(generated("\n"))
    values, coverage = read_multi(stream, [list("ABC")])[0]
    assert values == pytest.approx([0.25, 0, 0.75])
    assert coverage == pytest.approx(0.8)


@pytest.mark.parametrize("answer", [": C", "Q1: C", " C\nQ2: A"])
def test_read_multi_rejects_merged_answer_tokens(answer):
    prefix = (
        []
        if answer.startswith("Q1")
        else [generated("Q1" if answer.startswith(":") else "Q1:")]
    )
    stream = prefix + [generated(answer, {answer: 1.0}), generated("\n")]
    with pytest.raises(LowCoverageError, match="merged token") as error:
        read_multi(stream, [list("ABC")])
    assert repr(answer) in str(error.value)


@pytest.mark.parametrize(
    "text", ["", "Q1: ", "Q1: A", "Q2: A\n", "Q1: Z\n", "Q1: A\nextra"]
)
def test_read_multi_rejects_incomplete_or_invalid_streams(text):
    stream = [generated(token, {"A": 0.7, "B": 0.3}) for token in text]
    with pytest.raises(LowCoverageError):
        read_multi(stream, [list("AB")])


def test_read_multi_does_not_replace_missing_probabilities_with_certainty():
    stream = [generated(token) for token in ["Q1", ":", " A", "\n"]]
    with pytest.raises(LowCoverageError, match="coverage"):
        read_multi(stream, [list("AB")])


@pytest.mark.parametrize("calibrated", [False, True])
def test_multi_matches_single_and_averages_permutations(tmp_path, calibrated):
    request = Request(
        state="refund",
        questions={
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
    )
    distributions = {
        "route": {"A": 0.8, "B": 0.2},
        "risk": {"A": 0.3, "B": 0.7},
        "angry": {"A": 0.65, "B": 0.35},
    }
    single = FakeBackend(
        [
            top_tokens(distribution)
            for distribution in distributions.values()
            for _ in range(3)
        ],
        alphabet="AB",
    )
    multi = FakeBackend(
        alphabet="AB",
        multi_responses=distributions,
        question_names={
            "Team?": "route",
            "Risk?": "risk",
            "Is this statement true? Angry?": "angry",
        },
    )
    temperatures = {"choice": 2.0, "score": 3.0, "noul": 1.5}
    save_calibration(temperatures, tmp_path / "calibration.json")
    save_calibration(temperatures, tmp_path / "calibration_multi.json")
    first = SystemOne(single).decide(request, calibrated=calibrated)
    second = SystemOne(multi).decide(request, calibrated=calibrated, mode="multi")
    assert first["answers"] == second["answers"]
    assert first["usage"]["calls"] == first["usage"]["forward_passes"] == 9
    assert second["usage"]["calls"] == 3
    assert second["usage"]["forward_passes"] == 36
    assert second["usage"]["input_tokens"] == 30
    assert multi.max_tokens == [32, 32, 32]
    assert len(set(multi.prompts)) == 3
    first_questions = [
        next(line for line in prompt.splitlines() if line.startswith("Q1:"))
        for prompt in multi.prompts
    ]
    assert len(set(first_questions)) > 1
    for index, prompt in enumerate(multi.prompts):
        assert (
            "  A. low\n  B. high" if index % 2 == 0 else "  B. low\n  A. high"
        ) in prompt
    if not calibrated:
        assert second["answers"]["risk"]["expected_index"] == pytest.approx(
            (0.7 + 0.3 + 0.7) / 3
        )


def test_system_one_multi_and_separate_calibration(tmp_path):
    save_calibration({"score": 9}, tmp_path / "calibration.json")
    save_calibration({"score": 1}, tmp_path / "calibration_multi.json")
    result = system_one(
        "",
        {"risk": {"type": "score", "instructions": "Risk?", "levels": ["low", "high"]}},
        permutations=1,
        mode="multi",
        backend=FakeBackend(multi_responses={"Q1": {"A": 0.2, "B": 0.8}}),
    )
    assert result["answers"]["risk"]["expected_index"] == pytest.approx(0.8)
    assert result["usage"]["calls"] == 1


def test_multi_rejects_large_question_before_completion():
    backend = FakeBackend(alphabet="AB")
    with pytest.raises(ValueError, match="option alphabet"):
        system_one(
            "",
            {"q": {"type": "score", "instructions": "?", "levels": ["a", "b", "c"]}},
            backend=backend,
            mode="multi",
        )
    assert backend.prompts == []


def mock_backend(handle):
    backend = GemmaBackend()
    backend.client.close()
    backend.client = httpx.Client(
        base_url="http://fake/v1/", transport=httpx.MockTransport(handle)
    )
    return backend


def response(tokens):
    return httpx.Response(
        200,
        json={
            "choices": [{"logprobs": {"content": tokens}}],
            "usage": {"prompt_tokens": 17},
        },
    )


def test_multi_backend_contract_and_probe_once_under_concurrent_calls():
    requests = []
    stream = [
        generated("Q1", {"Q1": 1}),
        generated(":", {":": 1}),
        generated(" A", {" A": 0.8, " B": 0.2}),
        generated("\n", {"\n": 1}),
    ]
    grammar = build_grammar([list("AB")])

    def handle(request):
        assert SERVER_LOCK.locked()
        assert request.url.path == "/v1/chat/completions"
        data = json.loads(request.content)
        requests.append(data)
        assert data["model"] == "gemma-4-12b-qat"
        assert data["temperature"] == 0
        assert data["logprobs"] is True
        assert data["top_logprobs"] == 20
        assert data["chat_template_kwargs"] == {"enable_thinking": False}
        if "grammar" in data:
            assert data["grammar"] == grammar
            assert data["max_tokens"] == 16
            assert data["messages"] == [{"role": "user", "content": "prompt"}]
        else:
            assert data["max_tokens"] == 4
        return response(stream)

    backend = mock_backend(handle)
    try:
        assert requests == []
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda _: backend.complete_multi("prompt", grammar, 16), range(2)
                )
            )
        assert results == [(stream, 17), (stream, 17)]
        assert len(requests) == 3
        assert "grammar" not in requests[0]
        assert all("grammar" in data for data in requests[1:])
    finally:
        backend.close()


@pytest.mark.parametrize(
    "bad",
    [
        {"token": "two", "top_logprobs": []},
        {"token": "two"},
        {"token": "two", "top_logprobs": None},
    ],
)
def test_probe_rejects_missing_probabilities_and_blocks_multi(bad):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return response([generated("one", {"one": 1}), bad])

    backend = mock_backend(handle)
    try:
        with pytest.raises(RuntimeError) as error:
            backend.complete_multi("prompt", build_grammar([list("AB")]), 16)
        assert str(error.value) == (
            "server returns no probabilities after the first token; use the patched "
            "llama-server build or run without the MTP draft"
        )
        assert len(requests) == 1
        assert "grammar" not in requests[0]
    finally:
        backend.close()


@pytest.mark.parametrize("stream", [[], [generated("one", {"one": 1})]])
def test_probe_requires_enough_tokens_to_check(stream):
    backend = mock_backend(lambda request: response(stream))
    try:
        with pytest.raises(RuntimeError, match="fewer than two"):
            backend.probe()
    finally:
        backend.close()
