---
title: Phase 1 result, zero shot Gemma 4 12B on the typed decisions benchmark
date: 2026-09-20 17:06 CEST
author: Abhishek Thomas (Codex gpt-6-astra wrote the code, Claude Code reviewed and ran it)
type: research
status: frozen
---

# Phase 1 result, zero shot Gemma 4 12B on the typed decisions benchmark

## One line

A local Jev clone on Gemma 4 12B scores 0.7065 on the 2,000 decision LocalLLaMA/typed-decisions test set with no training, two points under Jev's published 0.727, and reaches ECE 0.032 after one temperature per question type fitted on the train split.

## Setup

+ Model - Gemma 4 12B QAT UD Q4_K_XL on llama.cpp, port 8010, thinking off, one slot.
+ Method - one forward pass per question, max_tokens 1, read the probability of each option id from top_logprobs, three shuffled option orders averaged, criteria descriptions kept in the prompt.
+ Data - 400 test cases, 5 questions each, 4 workflows. Temperature fitted on the 1,200 train cases (6,000 decisions), one T per type.
+ Command - `uv run system1 bench data/typed_decisions_test.jsonl --fit-temperature data/typed_decisions_train.jsonl --out report_full.json`. Run 16:03 to 17:04 CEST.

## Numbers

| | accuracy | ECE | Brier | log loss | score MAE |
|---|---|---|---|---|---|
| raw | 0.7065 | 0.2005 | 0.0847 | 2.2796 | 0.5332 |
| after temperature | 0.7065 | 0.0321 | 0.0429 | 1.1041 | 0.3680 |

Temperatures: choice 2.58, noul 2.94, score 3.29. All above 1, so the raw model was overconfident on every type.

| type | count | accuracy | ECE raw | ECE after |
|---|---|---|---|---|
| choice | 600 | 0.698 | 0.170 | 0.041 |
| noul | 600 | 0.832 | 0.114 | 0.023 |
| score | 800 | 0.619 | 0.292 | 0.072 |

| workflow | accuracy |
|---|---|
| customer_service | 0.736 |
| invoice_processing | 0.726 |
| security_incidents | 0.694 |
| agent_trace_observability | 0.670 |

Best questions: category 1.00, matches_order 0.96, duplicate 0.94. Worst: urgency 0.52 (dataset ceiling for agent_trace/urgency is 0.56), severity 0.56, action 0.58.

Latency: median 2,418 ms per case, p95 2,865 ms, 15 forward passes per case (5 questions x 3 permutations). One permutation would be about 0.8 s per case.

## Reference points (dataset card, 1,600 case set)

majority 0.520, factor ceiling 0.704, teacher self agreement 0.735, Jev published 0.727 (third party), Laya zero shot 0.36, Laya fine tuned on train 0.766.

## What it means

1. The generalist backbone is the intelligence. A 12B model with no training lands where Jev lands. Laya's 400M encoder needed fine tuning on this exact benchmark to pass it.
2. Calibration is cheap when the backbone is good. One number per question type took ECE from 0.20 to 0.03. This is in distribution calibration (train and test come from the same generator); expect worse on a new domain.
3. Score questions are the weak type (0.619). Ordinal rubrics need the head, not just the prompt. Phase 3.
4. Speed is the price: 2.4 s per case against Jev's 70 to 500 ms. Phase 2 (one pass for all questions) and one permutation attack that.

## Changelog

- 2026-09-20 17:06 CEST - Created from report_full.json.
