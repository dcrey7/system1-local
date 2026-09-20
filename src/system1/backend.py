"""Sequential HTTP inference and canned test responses."""

import math
import re
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


class GeneratedToken(TypedDict):
    token: str
    top_logprobs: list[TopToken]


class Backend(Protocol):
    @property
    def alphabet(self) -> str: ...

    def complete(self, prompt: str) -> tuple[list[TopToken], int]: ...

    def complete_multi(
        self, prompt: str, grammar: str, max_tokens: int
    ) -> tuple[list[GeneratedToken], int]: ...


class GemmaBackend:
    def __init__(self, base_url: str = "http://127.0.0.1:8010/v1") -> None:
        self.client = httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=120)
        self.tokenize_url = base_url.removesuffix("/").removesuffix("/v1") + "/tokenize"
        self._alphabet: str | None = None
        self._multi_probed = False

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

    def _generate(
        self, prompt: str, max_tokens: int, grammar: str | None = None
    ) -> tuple[list[GeneratedToken], int]:
        """Generate tokens while the caller holds SERVER_LOCK."""
        payload = {
            "model": "gemma-4-12b-qat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if grammar is not None:
            payload["grammar"] = grammar
        response = self.client.post("chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        tokens = [
            GeneratedToken(
                token=item["token"], top_logprobs=item.get("top_logprobs") or []
            )
            for item in data["choices"][0]["logprobs"]["content"]
        ]
        return tokens, data["usage"]["prompt_tokens"]

    def probe(self) -> None:
        """Check once that probabilities survive past the first token."""
        with SERVER_LOCK:
            if self._multi_probed:
                return
            tokens, _ = self._generate("Count from one to ten.", max_tokens=4)
            if any(not token["top_logprobs"] for token in tokens):
                raise RuntimeError(
                    "server returns no probabilities after the first token; "
                    "use the patched llama-server build or run without the MTP draft"
                )
            if len(tokens) < 2:
                raise RuntimeError("server probe returned fewer than two tokens")
            self._multi_probed = True

    def complete_multi(
        self, prompt: str, grammar: str, max_tokens: int
    ) -> tuple[list[GeneratedToken], int]:
        self.probe()
        with SERVER_LOCK:
            return self._generate(prompt, max_tokens, grammar)

    def close(self) -> None:
        self.client.close()


class FakeBackend:
    def __init__(
        self,
        responses: Iterable[list[TopToken]] = (),
        alphabet: str = ALPHABET,
        input_tokens: int = 10,
        *,
        multi_responses: dict[str, dict[str, float]] | None = None,
        question_names: dict[str, str] | None = None,
    ) -> None:
        self.alphabet = alphabet
        self.responses = iter(responses)
        self.prompts: list[str] = []
        self.input_tokens = input_tokens
        self.multi_responses = multi_responses or {}
        self.question_names = question_names or {}
        self.grammars: list[str] = []
        self.max_tokens: list[int] = []

    def complete(self, prompt: str) -> tuple[list[TopToken], int]:
        self.prompts.append(prompt)
        try:
            return next(self.responses), self.input_tokens
        except StopIteration as error:
            raise RuntimeError("FakeBackend has no response left") from error

    def complete_multi(
        self, prompt: str, grammar: str, max_tokens: int
    ) -> tuple[list[GeneratedToken], int]:
        """Emit distributions keyed by question name, instruction, or Q number.

        question_names maps prompt instructions to request question names.
        """
        self.prompts.append(prompt)
        self.grammars.append(grammar)
        self.max_tokens.append(max_tokens)
        stream: list[GeneratedToken] = []
        rules = re.findall(r'^q\d+ ::= "(Q\d+): " \((.*?)\)', grammar, re.M)
        for number, alternatives in rules:
            match = re.search(rf"^{number}: (.*)$", prompt, re.M)
            if match is None:
                raise ValueError(f"FakeBackend cannot find {number} in the prompt")
            instruction = match[1]
            name = self.question_names.get(instruction, instruction)
            key = name if name in self.multi_responses else number
            if key not in self.multi_responses:
                raise RuntimeError(f"FakeBackend has no multi response for {name!r}")
            distribution = self.multi_responses[key]
            ids = re.findall(r'"([A-Za-z0-9])"', alternatives)
            selected = max(ids, key=lambda token: distribution.get(token, 0.0))
            top = [
                TopToken(
                    token=" " + token, logprob=math.log(mass) if mass else -math.inf
                )
                for token, mass in distribution.items()
            ]
            stream.extend(
                [
                    GeneratedToken(token=number, top_logprobs=[]),
                    GeneratedToken(token=":", top_logprobs=[]),
                    GeneratedToken(token=" " + selected, top_logprobs=top),
                    GeneratedToken(token="\n", top_logprobs=[]),
                ]
            )
        return stream, self.input_tokens
