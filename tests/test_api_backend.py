import asyncio
import json
import math
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

import httpx
import pytest

from system1.api import create_app
from system1.backend import ALPHABET, FakeBackend, GemmaBackend
from system1.core import MODEL, SystemOne


def test_fastapi_round_trip():
    response = [
        {"token": "A", "logprob": math.log(0.8)},
        {"token": "B", "logprob": math.log(0.2)},
    ]
    backend = FakeBackend([response] * 3)
    app = create_app(SystemOne(backend))

    async def round_trip():
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client,
        ):
            result = await client.post(
                "/v1/systemone",
                json={
                    "state": {"text": "refund"},
                    "permutations": 1,
                    "questions": {
                        "route": {
                            "type": "choice",
                            "instructions": "Team?",
                            "criteria": {"billing": None, "tech": None},
                        },
                        "urgency": {
                            "type": "score",
                            "instructions": "Urgency?",
                            "levels": ["low", "high"],
                        },
                        "angry": {
                            "type": "noul",
                            "instructions": "The customer is angry.",
                        },
                    },
                },
            )
            assert result.status_code == 200
            body = result.json()
            assert body["model"] == MODEL
            assert body["answers"]["urgency"]["expected_index"] == pytest.approx(0.2)
            assert body["answers"]["angry"]["noul"] == pytest.approx(0.8)
            assert body["usage"]["forward_passes"] == 3
            assert (await client.get("/v1/models")).json()["data"][0]["id"] == MODEL
            invalid = await client.post(
                "/v1/systemone", json={"state": "", "questions": {}}
            )
            assert invalid.status_code == 422

    asyncio.run(round_trip())


def test_fastapi_reports_low_coverage():
    app = create_app(SystemOne(FakeBackend([[{"token": "wrong", "logprob": 0.0}]])))

    async def round_trip():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/systemone",
                json={
                    "state": "",
                    "questions": {"q": {"type": "noul", "instructions": "True?"}},
                },
            )
            assert response.status_code == 502
            assert "wrong" in response.json()["detail"]

    asyncio.run(round_trip())


def test_live_backend_request_contract_and_lazy_token_verification():
    requests = []

    def handle(request):
        requests.append(request)
        data = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(
                200, json={"tokens": [1] if data["content"] in "ABa0" else [1, 2]}
            )
        assert request.url.path == "/v1/chat/completions"
        assert data == {
            "model": "gemma-4-12b-qat",
            "messages": [{"role": "user", "content": "prompt"}],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {"top_logprobs": [{"token": "A", "logprob": 0.0}]}
                            ]
                        }
                    }
                ],
                "usage": {"prompt_tokens": 42},
            },
        )

    backend = GemmaBackend()
    backend.client.close()
    backend.client = httpx.Client(
        base_url="http://fake/v1/", transport=httpx.MockTransport(handle)
    )
    assert requests == []
    assert backend.alphabet == "ABa0"
    assert backend.alphabet == "ABa0"
    assert len(requests) == len(ALPHABET)
    assert backend.complete("prompt") == ([{"token": "A", "logprob": 0.0}], 42)
    backend.close()


def test_requests_are_sequential_even_across_backend_instances():
    active = 0
    lock = Lock()

    def handle(request):
        nonlocal active
        with lock:
            active += 1
            assert active == 1
        sleep(0.01)
        with lock:
            active -= 1
        return httpx.Response(
            200,
            json={
                "choices": [{"logprobs": {"content": [{"top_logprobs": []}]}}],
                "usage": {"prompt_tokens": 1},
            },
        )

    backends = [GemmaBackend(), GemmaBackend()]
    for backend in backends:
        backend.client.close()
        backend.client = httpx.Client(
            base_url="http://fake/v1/", transport=httpx.MockTransport(handle)
        )
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(
                pool.map(lambda backend: backend.complete("prompt"), backends)
            ) == [([], 1), ([], 1)]
    finally:
        for backend in backends:
            backend.close()


def test_backend_http_errors_propagate():
    backend = GemmaBackend()
    backend.client.close()
    backend.client = httpx.Client(
        base_url="http://fake/v1/",
        transport=httpx.MockTransport(lambda request: httpx.Response(503)),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            backend.complete("prompt")
    finally:
        backend.close()
