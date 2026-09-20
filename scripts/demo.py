"""Step by step demo: five questions in one call against the live server on 8010.

Run: uv run python scripts/demo.py
"""

import math
import time

from system1.backend import GemmaBackend
from system1.core import SystemOne, build_grammar, build_multi_prompt, read_multi
from system1.schema import Request, options

STATE = (
    "Job posting: Senior Data Scientist, fintech, Paris. Build credit risk models, own the "
    "feature pipeline, mentor two juniors. Needs 5+ years, Python, SQL, PyTorch or similar, "
    "production ML, French B2 preferred, hybrid 3 days on site. Salary 70 to 85k EUR.\n"
    "Candidate: 5 years data science across Amazon, EXL, AXA. Python, SQL, PyTorch, LLM fine "
    "tuning, shipped models to production. Based in Paris, French A2, MSc Data Science and AI."
)
QUESTIONS = {
    "fit": {
        "type": "choice",
        "instructions": "How well does the candidate match this job",
        "criteria": {
            "weak": "missing core requirements",
            "medium": "most requirements, a real gap",
            "strong": "all core requirements",
        },
    },
    "apply": {"type": "noul", "instructions": "We should apply to this job"},
    "french_risk": {
        "type": "noul",
        "instructions": "The French level is a real risk for this role",
    },
    "seniority": {
        "type": "score",
        "instructions": "Seniority of the candidate for this role",
        "levels": ["junior", "mid", "senior"],
    },
    "salary_fit": {
        "type": "score",
        "instructions": "How well the salary band matches this profile",
        "levels": ["low", "fair", "good"],
    },
}


def main() -> None:
    req = Request(state=STATE, questions=QUESTIONS, permutations=1, mode="multi")
    backend = GemmaBackend()
    order = list(req.questions)

    # Unique ids across the whole call: Q1 takes A, B, C; Q2 takes D, E; and so on.
    assigned: dict[str, dict[str, str]] = {}
    start = 0
    for name in order:
        labels = list(options(req.questions[name]))
        ids = backend.alphabet[start : start + len(labels)]
        start += len(labels)
        assigned[name] = dict(zip(ids, labels))
    ids_per_question = [list(assigned[name]) for name in order]

    prompt = build_multi_prompt(STATE, req.questions, assigned, order)
    grammar = build_grammar(ids_per_question)
    print("=== 1. THE PROMPT (state shortened) ===")
    print(prompt.replace(STATE, STATE[:90] + " ..."))
    print()
    print("=== 2. THE GRAMMAR (the model may only write this shape) ===")
    print(grammar)
    print()

    t = time.perf_counter()
    tokens, n_prompt = backend.complete_multi(prompt, grammar, max_tokens=8 * len(order) + 8)
    ms = (time.perf_counter() - t) * 1000
    print(f"=== 3. THE MODEL'S REPLY ({n_prompt} prompt tokens, {len(tokens)} generated, {ms:.0f} ms, one call) ===")
    print(repr("".join(tk["token"] for tk in tokens)))
    print()

    print("=== 4. WHAT THE SERVER GIVES US AT EACH ANSWER LETTER (top 4 of 20) ===")
    all_ids = {id_ for ids in ids_per_question for id_ in ids}
    answer_tokens = [tk for tk in tokens if tk["token"].strip() in all_ids]
    for name, tk in zip(order, answer_tokens):
        top: dict[str, float] = {}
        for entry in tk["top_logprobs"]:
            key = entry["token"].strip() or repr(entry["token"])
            top[key] = max(top.get(key, 0.0), round(math.exp(entry["logprob"]), 3))
        top = dict(sorted(top.items(), key=lambda kv: -kv[1])[:4])
        print(f"  {name:12} token {tk['token']!r:5} top {top}   ids: {assigned[name]}")
    print()

    print("=== 5. READ AND NORMALISE OVER THE QUESTION'S OWN IDS ===")
    for name, (probs, coverage) in zip(order, read_multi(tokens, ids_per_question)):
        labelled = ", ".join(f"{label} {p:.3f}" for label, p in zip(assigned[name].values(), probs))
        print(f"  {name:12} coverage {coverage:.3f}  {labelled}")
    print()

    print("=== 6. FINAL ANSWER (3 shuffles averaged, temperature applied), the product output ===")
    t = time.perf_counter()
    final = Request(state=STATE, questions=QUESTIONS, permutations=3, mode="multi")
    out = SystemOne(backend).decide(final, mode="multi")
    ms = (time.perf_counter() - t) * 1000
    for name, answer in out["answers"].items():
        if "noul" in answer:
            print(f"  {name:12} true with probability {answer['noul']:.2f}")
        else:
            probs = ", ".join(f"{k} {v:.2f}" for k, v in answer["probabilities"].items())
            print(f"  {name:12} {answer[answer['type']]:8} confidence {answer['confidence']:.2f}   {probs}")
    print(f"  ({out['usage']['calls']} calls, {ms:.0f} ms)")
    backend.close()


if __name__ == "__main__":
    main()
