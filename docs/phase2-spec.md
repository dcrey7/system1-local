# Phase 2 spec: one pass parallel decisions on DiffusionGemma (draft)

Goal: replace "one forward pass per question" with "one pass for all questions", the way Jev describes its parallel sampler, and measure what it costs in accuracy against the phase 1 number.

## Backbone

`google/diffusiongemma-26B-A4B-it` (Apache 2.0). 25.2B total, 3.8B active, 256K context, text plus image plus video input. It has an autoregressive encoder that processes and caches the prompt, and a 256 token generation canvas with bidirectional attention that is denoised in parallel. Transformers class `DiffusionGemmaForBlockDiffusion`, processor via `AutoProcessor`.

Runtime options, try in this order:
1. Transformers with 4 bit weights (`bitsandbytes`), `unsloth/diffusiongemma-26B-A4B-it` or the Google repo. About 15 GB. The 3090 is Ampere, so FP8 and NVFP4 builds are out.
2. vLLM with the open PR vllm-project/vllm#57250 ("structured reads" for DiffusionGemma: `diffusion_seed_canvas`, `diffusion_max_steps`, `diffusion_read_only`, logprobs at the slots). Only if 1 is too slow. It needs a rebase and a quantized MoE build that Ampere can run (AWQ or GPTQ int4, or bitsandbytes).

Gemma 4 12B on port 8010 must be stopped first (`local-model down gemma`). Never run both.

## The mechanism to replicate

```
prompt (encoder, cached once):   State: ... plus the list of questions with their option ids
canvas (bidirectional):          "Q1: [MASK]  Q2: [MASK]  Q3: [MASK] ..."  one slot per question
one forward pass:                logits at every [MASK] slot at the same time
read:                            softmax over each question's option ids at its own slot
```

Deliverables:
1. `system1/backend_diffusion.py`: a `DiffusionBackend` that takes state plus a list of questions and returns per question option probabilities from a single forward pass with `diffusion_max_steps=1` and no sampling. Also support `steps=k` for a small number of denoising steps, to test if it helps.
2. A `--backend diffusion` switch in `bench` so the same JSONL and the same metrics run against it.
3. Measure on the typed decisions test set: accuracy, ECE, Brier, latency per case, forward passes per case (should be 1 plus the encoder pass). Compare with the phase 1 report.
4. Vision smoke test: one image in the state, three noul questions.

## Open questions to answer in the notes

- Does reading the slots at step 1 (no denoising) give usable distributions, or does it need a few steps?
- Does one shared canvas hurt independence between questions (an answer at slot 2 attending to slot 1)? Test with the questions in two orders.
- How much slower is prefill than Gemma 12B on this card, for 1k and 8k token states?
