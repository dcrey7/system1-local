# system1-local

A local "System One" decision model: state plus typed questions in, calibrated probabilities out, no text generation. Jev shaped API on top of open models on one RTX 3090.

## Rules for every agent

1. Python with `uv`. Never pip, never conda. Run with `uv run`. Lint with `ruff`. Tests with `pytest`.
2. Simple code. A junior engineer must read it once and understand it. Short functions. Type hints. Docstrings that say what, not how.
3. No em dashes in code comments, docs, or commit messages. Plain English. Short sentences.
4. Never commit tokens, keys, or passwords. Never add Co-Authored-By or Generated with lines.
5. Every measured number goes in `docs/` with the date from `date "+%Y-%m-%d %H:%M %Z"`, never guessed.
6. The model servers are external. Gemma 4 12B: `http://127.0.0.1:8010/v1` (llama.cpp, OpenAI compatible, `logprobs` and `top_logprobs` supported, model id `gemma-4-12b-qat`, always send `chat_template_kwargs: {"enable_thinking": false}`). Do not start or stop servers.
7. Do not download models or datasets over 1 GB without asking.
