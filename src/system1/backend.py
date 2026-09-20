"""Sequential HTTP inference and canned test responses."""

from collections.abc import Iterable
from string import ascii_uppercase, digits
from threading import Lock
from typing import Protocol, TypedDict

import httpx

ALPHABET = ascii_uppercase + digits
SERVER_LOCK = Lock()


class TopToken(TypedDict):
    token: str
    logprob: float


class Backend(Protocol):
    @property
    def alphabet(self) -> str: ...

    def complete(self, prompt: str) -> tuple[list[TopToken], int]: ...


class GemmaBackend:
    def __init__(self, base_url: str = "http://127.0.0.1:8010/v1") -> None:
        self.client = httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=120)
        self.tokenize_url = base_url.removesuffix("/").removesuffix("/v1") + "/tokenize"
        self._alphabet: str | None = None

    @property
    def alphabet(self) -> str:
        with SERVER_LOCK:
            if self._alphabet is None:
                ids = []
                for token in ALPHABET:
                    response = self.client.post(
                        self.tokenize_url, json={"content": token}
                    )
                    response.raise_for_status()
                    if len(response.json()["tokens"]) == 1:
                        ids.append(token)
                if len(ids) < 2:
                    raise ValueError(
                        "The server must provide at least two single-token ids"
                    )
                self._alphabet = "".join(ids)
            return self._alphabet

    def complete(self, prompt: str) -> tuple[list[TopToken], int]:
        with SERVER_LOCK:
            response = self.client.post(
                "chat/completions",
                json={
                    "model": "gemma-4-12b-qat",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1,
                    "temperature": 0,
                    "logprobs": True,
                    "top_logprobs": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            response.raise_for_status()
            data = response.json()
            return (
                data["choices"][0]["logprobs"]["content"][0]["top_logprobs"],
                data["usage"]["prompt_tokens"],
            )

    def close(self) -> None:
        self.client.close()


class FakeBackend:
    def __init__(
        self,
        responses: Iterable[list[TopToken]],
        alphabet: str = ALPHABET,
        input_tokens: int = 10,
    ) -> None:
        self.alphabet = alphabet
        self.responses = iter(responses)
        self.prompts: list[str] = []
        self.input_tokens = input_tokens

    def complete(self, prompt: str) -> tuple[list[TopToken], int]:
        self.prompts.append(prompt)
        try:
            return next(self.responses), self.input_tokens
        except StopIteration as error:
            raise RuntimeError("FakeBackend has no response left") from error
