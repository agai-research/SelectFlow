"""Citizen formulations (Claude Sonnet 5.5) verified by a second model (Claude Opus 5.5), run without Colab.
Resumable: results accumulate in data/formulations.json and data/formulation_checks.json.

  python scripts/formulations_local.py write | verify | summary
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
from selectflow.agent.claude import ClaudeConfig
from selectflow.agent.claude_generator import _forced

INP = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "data/selectflow_requests_input.json"))
F, C, P = "data/formulations.json", "data/formulation_checks.json", "data/procedure_checks.json"
load = lambda p: json.load(open(p)) if os.path.exists(p) else {}
W = ClaudeConfig(model="claude-sonnet-5-5"); W.thinking = {"type": "between_tools"}
V = ClaudeConfig(model="claude-opus-5-5"); V.thinking = None

SYS_W = ("You write requests that citizens or business owners would type into a public-service assistant. Given an "
         "administrative goal, write: vague_en, a short informal English request stating the goal in everyday words "
         "without the official procedure name; fr, the same need in natural French; ar, the same need in natural Modern "
         "Standard Arabic. Do not mention documents that are not needed. Do not add any other goal.")
S_W = {"type": "object", "properties": {"vague_en": {"type": "string"}, "fr": {"type": "string"}, "ar": {"type": "string"}},
       "required": ["vague_en", "fr", "ar"]}
SYS_V = "You are a strict verifier of annotations for an administrative-services dataset. Judge only what is asked."
S_F = {"type": "object", "properties": {"faithful": {"type": "boolean"}, "reason": {"type": "string"}}, "required": ["faithful", "reason"]}
S_P = {"type": "object", "properties": {"purpose_ok": {"type": "boolean"}, "categories_ok": {"type": "boolean"},
       "missing_categories": {"type": "array", "items": {"type": "string"}},
       "wrong_categories": {"type": "array", "items": {"type": "string"}}},
       "required": ["purpose_ok", "categories_ok", "missing_categories", "wrong_categories"]}


def retry(fn, *a):
    for k in range(4):
        try:
            return fn(*a)
        except Exception:
            time.sleep(3 * (k + 1))
    return None


def write():
    out = load(F)
    todo = [r for r in INP["requests"] if r["id"] not in out]
    res = ThreadPoolExecutor(8).map(lambda r: (r["id"], retry(_forced, W, SYS_W, f"Administrative goal: {r['goal']}\nCountry: {r['country']}",
                                                               "requests", S_W)), todo)
    for k, v in res:
        if v and all(v.get(x) for x in ("vague_en", "fr", "ar")):
            out[k] = v
    json.dump(out, open(F, "w"), ensure_ascii=False, indent=1)
    print(len(out), "of", len(INP["requests"]), "requests formulated")


def verify():
    forms, chk, pchk = load(F), load(C), load(P)
    goal = {r["id"]: r["goal"] for r in INP["requests"]}
    jobs = [("f", rid, k) for rid in forms for k in ("vague_en", "fr", "ar") if f"{rid}|{k}" not in chk]
    jobs += [("p", pid, None) for pid in INP["procedures"] if pid not in pchk]
    jobs = jobs[:160]                      # one chunk per call (time limit of the sandbox)
    def one(j):
        t, a, k = j
        if t == "f":
            return j, retry(_forced, V, SYS_V, f"Goal: {goal[a]}\nRequest written by a citizen: {forms[a][k]}\nDoes the request express "
                            "exactly this goal, with no other goal added? A different language is fine.", "verdict", S_F)
        p = INP["procedures"][a]
        return j, retry(_forced, V, SYS_V, f"Administrative procedure: {p['title']}\nDescription: {p['text'][:800]}\nAnnotated purpose: "
                        f"{p['purpose']}\nAnnotated personal-data categories the applicant provides: {p['categories']} (ID identity, BUS "
                        "business, LOC premises or location, FIN financial or tax, HLT health). Are the purpose and the categories correct?",
                        "verdict", S_P)
    for (t, a, k), v in ThreadPoolExecutor(8).map(one, jobs):
        if v:
            (chk.__setitem__(f"{a}|{k}", v) if t == "f" else pchk.__setitem__(a, v))
    json.dump(chk, open(C, "w"), ensure_ascii=False, indent=1); json.dump(pchk, open(P, "w"), ensure_ascii=False, indent=1)
    print(len(chk), "formulation checks,", len(pchk), "procedure checks")


def summary():
    chk, pchk = load(C), load(P)
    by = {}
    for key, v in chk.items():
        by.setdefault(key.split("|")[1], []).append(bool(v.get("faithful")))
    S = {"writer": W.model, "verifier": V.model,
         "faithful_by_language": {k: sum(x) / len(x) for k, x in by.items()},
         "faithful_overall": sum(sum(x) for x in by.values()) / max(1, sum(len(x) for x in by.values())),
         "procedures_checked": len(pchk),
         "purpose_ok": sum(bool(v.get("purpose_ok")) for v in pchk.values()) / max(1, len(pchk)),
         "categories_ok": sum(bool(v.get("categories_ok")) for v in pchk.values()) / max(1, len(pchk))}
    from collections import Counter
    S["most_missing_categories"] = Counter(c for v in pchk.values() for c in v.get("missing_categories", [])).most_common(5)
    S["most_wrong_categories"] = Counter(c for v in pchk.values() for c in v.get("wrong_categories", [])).most_common(5)
    json.dump(S, open("results/formulations_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(S, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"write": write, "verify": verify, "summary": summary}[sys.argv[1]]()
