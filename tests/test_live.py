"""Integration checks for the patched local llama-server."""

import re
import signal
from collections.abc import Iterator
from types import FrameType

import httpx
import pytest

from system1.backend import GemmaBackend
from system1.core import SystemOne, build_grammar, build_multi_prompt
from system1.schema import Request, options

SERVER_URL = "http://127.0.0.1:8010"
pytestmark = pytest.mark.live

try:
    health = httpx.get(f"{SERVER_URL}/health", timeout=2.0, trust_env=False)
    health.raise_for_status()
except httpx.HTTPError as error:
    pytest.skip(
        f"Live llama-server unavailable: GET {SERVER_URL}/health must succeed "
        f"within 2 seconds ({error}).",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def no_network() -> None:
    """Override the unit test socket guard only in this live module."""


@pytest.fixture(autouse=True)
def deadline() -> Iterator[None]:
    """Bound the whole test, including setup and all sequential HTTP calls."""

    def expired(signum: int, frame: FrameType | None) -> None:
        pytest.fail("Live test exceeded its 9 second budget")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 9.0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.fixture
def backend(deadline: None) -> Iterator[GemmaBackend]:
    live = GemmaBackend(f"{SERVER_URL}/v1")
    live.client.close()
    live.client = httpx.Client(
        base_url=f"{SERVER_URL}/v1/", timeout=9.0, trust_env=False
    )
    try:
        yield live
    finally:
        live.close()


@pytest.fixture
def decision_request() -> Request:
    return Request(
        state="I was charged twice. Please refund the duplicate charge. I am angry.",
        permutations=1,
        questions={
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": None, "tech": None},
            },
            "action": {
                "type": "choice",
                "instructions": "What does the customer want?",
                "criteria": {"refund": None, "password reset": None},
            },
            "duplicate": {
                "type": "noul",
                "instructions": "The customer was charged twice.",
            },
            "angry": {
                "type": "noul",
                "instructions": "The customer is angry.",
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is the request?",
                "levels": ["low", "medium", "high"],
            },
        },
    )


def probabilities(answer: dict) -> dict[str, float]:
    """Expand the public noul scalar into its two option probabilities."""
    if answer["type"] == "noul":
        return {"yes": answer["noul"], "no": 1 - answer["noul"]}
    return answer["probabilities"]


def test_probe(backend: GemmaBackend) -> None:
    # A fresh backend makes the four-token probe run instead of using its cache.
    # Let its error propagate so an unpatched server shows the repair guidance.
    backend.probe()


def test_multi_decide(backend: GemmaBackend, decision_request: Request) -> None:
    result = SystemOne(backend).decide(decision_request, calibrated=False, mode="multi")
    assert len(result["answers"]) == 5
    assert set(result["answers"]) == set(decision_request.questions)
    for name, answer in result["answers"].items():
        assert answer["coverage"] >= 0.05, (name, answer)
        distribution = probabilities(answer)
        assert set(distribution) == set(options(decision_request.questions[name]))
        assert all(0 <= value <= 1 for value in distribution.values()), (name, answer)
        assert sum(distribution.values()) == pytest.approx(1.0), (name, answer)
    assert result["usage"]["calls"] == decision_request.permutations
    assert result["usage"]["forward_passes"] > 5


def test_single_multi_argmax(backend: GemmaBackend, decision_request: Request) -> None:
    engine = SystemOne(backend)
    multi = engine.decide(decision_request, calibrated=False, mode="multi")["answers"]
    single = engine.decide(decision_request, calibrated=False, mode="single")["answers"]
    matches = []
    for name in decision_request.questions:
        first = probabilities(single[name])
        second = probabilities(multi[name])
        matches.append(max(first, key=first.get) == max(second, key=second.get))
    assert sum(matches) >= 3, {"single": single, "multi": multi}


def test_multi_grammar(backend: GemmaBackend, decision_request: Request) -> None:
    order = ["route", "duplicate"]
    questions = {name: decision_request.questions[name] for name in order}
    alphabet = backend.alphabet
    assert len(alphabet) >= 4
    assigned = {
        name: dict(zip(alphabet[index * 2 : index * 2 + 2], options(questions[name])))
        for index, name in enumerate(order)
    }
    prompt = build_multi_prompt(decision_request.state, questions, assigned, order)
    grammar = build_grammar([list(assigned[name]) for name in order])
    tokens, _ = backend.complete_multi(prompt, grammar, max_tokens=24)
    text = "".join(token["token"] for token in tokens)
    assert re.fullmatch(r"^Q1: [A-Z0-9]\nQ2: [A-Z0-9]\n$", text), repr(text)
    for name, line in zip(order, text.splitlines()):
        assert line[-1] in assigned[name], (name, text)
