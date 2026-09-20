"""Step by step demo of one call with five questions against the live server on 8010. Run: uv run python scripts/demo.py"""
import json, math, time
from system1.backend import GemmaBackend
from system1.core import SystemOne, build_grammar, build_multi_prompt, read_multi
from system1.schema import Request, options

state = ("Job posting: Senior Data Scientist, fintech, Paris. Build credit risk models, own the feature pipeline, "
         "mentor two juniors. Needs 5+ years, Python, SQL, PyTorch or similar, production ML, French B2 preferred, "
         "hybrid 3 days on site. Salary 70 to 85k EUR.\n"
         "Candidate: 5 years data science across Amazon, EXL, AXA. Python, SQL, PyTorch, LLM fine tuning, shipped "
         "models to production. Based in Paris, French A2, MSc Data Science and AI.")
questions = {
    "fit": {"type": "choice", "instructions": "How well does the candidate match this job",
            "criteria": {"weak": "missing core requirements", "medium": "most requirements, a real gap", "strong": "all core requirements"}},
    "apply": {"type": "noul", "instructions": "We should apply to this job"},
    "french_risk": {"type": "noul", "instructions": "The French level is a real risk for this role"},
    "seniority": {"type": "score", "instructions": "Seniority of the candidate for this role", "levels": ["junior", "mid", "senior"]},
    "salary_fit": {"type": "score", "instructions": "How well the salary band matches this profile", "levels": ["low", "fair", "good"]},
}
req = Request(state=state, questions=questions, permutations=1, mode="multi")
backend = GemmaBackend()
alphabet = backend.alphabet
order = list(req.questions)
assigned, start = {}, 0
for name in order:
    labels = list(options(req.questions[name]))
    ids = alphabet[start:start + len(labels)]; start += len(labels)
    assigned[name] = dict(zip(ids, labels))
prompt = build_multi_prompt(state, req.questions, assigned, order)
grammar = build_grammar([list(assigned[n]) for n in order])
print("=== 1. THE PROMPT (state shortened) ===")
print(prompt.replace(state, state[:90] + " ...")); print()
print("=== 2. THE GRAMMAR (the model may only write this shape) ===")
print(grammar); print()
t = time.perf_counter()
tokens, n_prompt = backend.complete_multi(prompt, grammar, max_tokens=8 * len(order) + 8)
dt = time.perf_counter() - t
print(f"=== 3. THE MODEL'S REPLY ({n_prompt} prompt tokens, {len(tokens)} generated, {dt*1000:.0f} ms, one call) ===")
print(repr("".join(tk["token"] for tk in tokens))); print()
print("=== 4. WHAT THE SERVER GIVES US AT EACH ANSWER LETTER (top 5 of 20) ===")
for name, tk in zip(order, [tk for tk in tokens if tk["token"].strip() and tk["token"].strip() in "".join(assigned[n] and "".join(assigned[n].keys()) for n in order)]):
    top = {}
    for x in tk["top_logprobs"]:
        key = x["token"].strip() or repr(x["token"])
        top[key] = max(top.get(key, 0.0), round(math.exp(x["logprob"]), 3))
    top = dict(sorted(top.items(), key=lambda kv: -kv[1])[:4])
    print(f"  {name:12} token {tk['token']!r:5} top5 {top}   ids for this question: {assigned[name]}")
print()
print("=== 5. READ AND NORMALISE OVER THE QUESTION'S OWN IDS ===")
for name, (probs, coverage) in zip(order, read_multi(tokens, [list(assigned[n]) for n in order])):
    print(f"  {name:12} coverage {coverage:.3f}  " + ", ".join(f"{label} {p:.3f}" for label, p in zip(assigned[name].values(), probs)))
print()
print("=== 6. FINAL ANSWER (3 shuffles averaged, temperature applied), the product output ===")
t = time.perf_counter()
out = SystemOne(backend).decide(Request(state=state, questions=questions, permutations=3, mode="multi"), mode="multi")
dt = time.perf_counter() - t
for name, a in out["answers"].items():
    if "noul" in a: print(f"  {name:12} true with probability {a['noul']:.2f}")
    else: print(f"  {name:12} {a[a['type']]:8} confidence {a['confidence']:.2f}   " + ", ".join(f"{k} {v:.2f}" for k, v in a["probabilities"].items()))
print(f"  ({out['usage']['calls']} calls, {dt*1000:.0f} ms)")
backend.close()
