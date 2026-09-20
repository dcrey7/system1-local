---
title: Facts for upstream llama.cpp contributions (test report and crash report)
date: 2026-09-20 20:23 CEST
author: Claude (facts only; Abhishek writes any post himself)
type: reference
status: current
---

# Facts for upstream llama.cpp contributions

llama.cpp rules (CONTRIBUTING.md and AGENTS.md on master, read 2026-09-20 20:23 CEST): check existing issues and PRs first; AI generated code must be disclosed; posts (issues, PR text, comments) must be written by the human; bug fix PRs need a reproducible issue and a regression test; the contributor must be able to explain every line. This file holds data only.

## 1. Test report for PR #27196 (server: Support logprobs with speculative decoding)

- PR: https://github.com/ggml-org/llama.cpp/pull/27196 by pnb, opened 2026-08-16, rebased 2026-09-01, unreviewed. Original request: issue #24271 "logprobs working for gemma MTP" (2026-06-07, auto closed 2026-07-23).
- What we observed before the fix: with `--spec-type draft-mtp`, a chat completion with `logprobs: true, top_logprobs: 20` returned a probability table for the first generated token only; every later token came back with logprob 0.0 and an empty `top_logprobs` list. Confirmed with a 26 token answer: 1 token with a table, 25 without.
- Our patch (same idea as the PR: read target logits at each accepted token's batch index): branch `server-spec-probs` in /home/abhishek/llama-vulkan/llama.cpp-fresh, base commit 666f889, commits 611dc03, 4d45333, test in 9140408.
- After the fix: all 26 tokens carry 20 alternatives; the MTP draft stays active (acceptance 0.75 to 0.89 in the server log).
- Load tested: about 6,000 chat completions of 26 tokens each with a GBNF grammar, plus 6,000 single token completions, over roughly two hours.
- The PR's own regression test `test_draft_token_probs` passes on our build: `2 passed in 49.51s` (run from tools/server/tests with LLAMA_SERVER_BIN_PATH set to build-cuda/bin/llama-server).
- Hardware and build: RTX 3090 24 GB, driver 580.173.02, CUDA 13 toolkit at /home/abhishek/cuda-13, GCC 16.1.1, Linux 6.18.42 CachyOS, CMake Release, GGML_CUDA=ON.
- Model: Gemma 4 12B QAT, `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`, draft `mtp-gemma-4-12B-it.gguf`, mmproj F16, context 262144, KV f16, `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0`, `--parallel 1`, `--jinja`.
- Server command (from the launch script): ``
- Possible review remark, only if you understand and agree with it: the PR always fills the pre sampling table (it passes `false` for post_sampling), even when the client asked for `post_sampling_probs`. With post sampling requested, only the last accepted token has valid sampler candidates; our version fills only that one in that mode.

## 2. Crash report candidate (not reproduced yet)

- Symptom: after 63 minutes of the load above, `CUDA error: an illegal memory access was encountered` in `ggml_backend_cuda_synchronize`, raised from `llama_get_embeddings_nextn_ith` inside `common_speculative_impl_draft_mtp::process`, right after the log line "making room for prompt cache entry, removing oldest entry (size = 610 MiB)" and a slot selected by LRU. 4,172 earlier evictions did not crash.
- Related open issue: #27306 "draft-mtp DeviceLost during prompt on AMD RADV: common_speculative_process runs llama_decode(ctx_dft) after every prefill ubatch".
- Full log with backtrace: /home/abhishek/.claude/jobs/872a19fc/tmp/gemma-server-crash-19h39.log
- Backtrace head:
```
53326:/home/abhishek/llama-vulkan/llama.cpp-fresh/ggml/src/ggml-cuda/ggml-cuda.cu:106: CUDA error
53327:63.13.925.291 E CUDA error: an illegal memory access was encountered
53362:#0  0x00007fe04ecb61f2 in ?? () from /usr/lib/libc.so.6
53363:#1  0x00007fe04ed2ead2 in wait4 () from /usr/lib/libc.so.6
53364:#2  0x00007fe05a97183b in ggml_print_backtrace () from /home/abhishek/llama-vulkan/llama.cpp-fresh/build-cuda/bin/libggml-base.so.0
53365:#3  0x00007fe05a9719ce in ggml_abort () from /home/abhishek/llama-vulkan/llama.cpp-fresh/build-cuda/bin/libggml-base.so.0
53366:#4  0x00007fe055ce4e51 in ggml_cuda_error(char const*, char const*, char const*, int, char const*) () from /home/abhishek/llama-vulkan/llama.cpp-fresh/build-cuda/bi
53367:#5  0x00007fe055cec468 in ggml_backend_cuda_synchronize(ggml_backend*) () from /home/abhishek/llama-vulkan/llama.cpp-fresh/build-cuda/bin/libggml-cuda.so.0
53368:#6  0x00007fe05a98c51e in ggml_backend_sched_synchronize () from /home/abhishek/llama-vulkan/llama.cpp-fresh/build-cuda/bin/libggml-base.so.0
53369:#7  0x00007fe0596ec718 in llama_context::synchronize() () from /home/abhishek/llama-vulkan/llama.cpp-fresh/build-cuda/bin/libllama.so.0
53370:#8  0x00007fe0596f2901 in llama_get_embeddings_nextn_ith(llama_context*, int) () from /home/abhishek/llama-vulkan/llama.cpp-fresh/build-cuda/bin/libllama.so.0
53371:#9  0x00007fe059f32ac8 in common_speculative_impl_draft_mtp::process(llama_batch const&) () from /home/abhishek/llama-vulkan/llama.cpp-fresh/build-cuda/bin/libllama
53372:#10 0x00007fe059f27ee0 in common_speculative_process(common_speculative*, llama_batch const&) () from /home/abhishek/llama-vulkan/llama.cpp-fresh/build-cuda/bin/lib
53373:#11 0x00007fe05a3b398b in std::_Function_handler<void (), server_context_impl::decode(int&, int, llama_batch&)::{lambda()#2}>::_M_invoke(std::_Any_data const&) () f
```
- Rule from CONTRIBUTING.md: a bug report needs a reproduction. Plan: rerun the same benchmark load (`uv run system1 bench data/typed_decisions_test.jsonl --mode multi --permutations 3` in system1-local) until it crashes or for one hour, then report the exact request that died, with the server log. Needs Abhishek's go because it can take the server down.

## Changelog

- 2026-09-20 20:23 CEST: Created.
