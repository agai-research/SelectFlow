"""E2 on the real federations (data version 2, meaningful document names), resumable in chunks.

For each citizen request (vague English formulation): interpretation accuracy; SelectFlow's derivation
(grammar-constrained generation, critics, revision, forward chaining) with the requester's confirmation of the
targets; direct generation (one unconstrained call with the same confirmed targets, no critics); the planner alone
equals the reference by construction (the references are its output) and is reported as such.

  python scripts/e2_chunks.py run [--k 10]   |   python scripts/e2_chunks.py summary
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
import numpy as np

from selectflow import io, stats
from selectflow.agent.claude import ClaudeConfig, Ledger
from selectflow.agent import claude_generator as CG
from selectflow.derivation import check, derive
from selectflow.experiments.runners_llm import _edges_f1, _jacc
from selectflow.model import Context, Params

REC = "results/e2_real_records.jsonl"
PARAMS = Params(tau_chi=0.6)
cfg = ClaudeConfig(model="claude-sonnet-5-5"); cfg.thinking = {"type": "between_tools"}


def score(fed, order, rq):
    G, H = set(rq["targets"]), set(rq["held"])
    rep, wf = check(order, G, H, fed.descriptors, PARAMS.tau_chi)
    ref = rq["references"][0]
    return {"closed": not (rep.missing or rep.targets or rep.companions), "valid": rep.empty(),
            "exact": sorted(order) == sorted(ref), "jaccard": _jacc(order, ref),
            "dep_f1": _edges_f1(wf.ho, ref, fed, G, H, PARAMS.tau_chi), "n_tasks": len(order)}


def run_one(name, rq, text):
    fed = io.load(f"data/v2/{name}.json")
    arts = sorted({a for d in fed.descriptors.values() for a in d.In | d.Out})
    G, H = set(rq["targets"]), set(rq["held"])
    rec = {"fed": name, "id": rq["id"], "n_ref": len(rq["references"][0])}
    t0 = time.time()
    g, h = CG.ClaudeInterpreter(cfg, arts)(text, Context(H))
    rec["interpretation_exact"] = g == G; rec["interpretation_found"] = bool(g & G)
    res = derive(text, Context(H), fed, PARAMS, lambda q, c: (g, h), CG.ClaudeGenerator(cfg), confirm=lambda a, b: (G, H), k=3, n_rev=2)
    rec["selectflow"] = score(fed, res.workflows[0].order, rq) if res.workflows else {"closed": False, "valid": False, "exact": False,
                                                                                       "jaccard": 0.0, "dep_f1": 0.0, "n_tasks": 0}
    rec["llm_candidates"] = res.log["candidates"]; rec["llm_accepted"] = res.log["accepted"]
    rec["llm_accepted_after_revision"] = res.log["accepted_after_revision"]
    rec["llm_rejected_first_pass"] = res.log["rejected_by_critics"]
    rec["llm_first_candidate"] = score(fed, [t for t in res.log["llm_orders"][0] if t in fed.descriptors], rq) if res.log["llm_orders"] else None
    rec["selectflow_time_s"] = round(time.time() - t0, 1)
    t1 = time.time()
    direct = CG.ClaudeGenerator(cfg, unconstrained=True).generate(text, G, H, fed.descriptors, 1)
    order = direct[0] if direct else []
    rec["direct_invalid_ids"] = sum(t not in fed.descriptors for t in order)
    rec["direct"] = score(fed, [t for t in order if t in fed.descriptors], rq)
    rec["direct_time_s"] = round(time.time() - t1, 1)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["run", "summary"]); ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    done = {json.loads(l)["id"] for l in open(REC)} if os.path.exists(REC) else set()
    if a.mode == "run":
        texts = json.load(open("data/vague_requests.json"))
        jobs = [(n, rq, texts[rq["id"]]) for n in ("saudi", "greece") for rq in json.load(open(f"data/v2/requests_{n}.json"))
                if rq["id"] not in done][: a.k]
        def safe(j):
            for attempt in range(3):
                try:
                    return run_one(*j)
                except Exception as e:
                    err = type(e).__name__; time.sleep(5)
            return None
        with ThreadPoolExecutor(max(1, len(jobs))) as ex:
            for rec in ex.map(safe, jobs):
                if rec:
                    open(REC, "a").write(json.dumps(rec) + "\n"); print(rec["id"], rec["selectflow"]["exact"], rec["direct"]["exact"], flush=True)
        print(len(done) + len(jobs), "of 82 attempted")
        return
    R = [json.loads(l) for l in open(REC)]
    out = {"n": len(R)}
    for m in ("selectflow", "direct"):
        out[m] = {k: float(np.mean([r[m][k] for r in R])) for k in ("closed", "valid", "exact", "jaccard", "dep_f1")}
    fc = [r["llm_first_candidate"] for r in R if r["llm_first_candidate"]]
    out["llm_first_candidate"] = {k: float(np.mean([x[k] for x in fc])) for k in ("closed", "exact", "jaccard")}
    out["llm_candidate_accepted_by_critics"] = float(np.mean([r["llm_accepted"] > 0 for r in R]))
    out["llm_accepted_only_after_revision"] = float(np.mean([r["llm_accepted_after_revision"] > 0 and r["llm_accepted"] == r["llm_accepted_after_revision"] for r in R]))
    out["first_pass_rejection_rate"] = float(np.sum([r["llm_rejected_first_pass"] for r in R]) / max(1, np.sum([r["llm_candidates"] for r in R])))
    out["direct_invalid_ids_mean"] = float(np.mean([r["direct_invalid_ids"] for r in R]))
    out["interpretation_exact"] = float(np.mean([r["interpretation_exact"] for r in R]))
    out["interpretation_found"] = float(np.mean([r["interpretation_found"] for r in R]))
    out["closed_selectflow_vs_direct"] = {"diff_ci": stats.paired_bootstrap([r["selectflow"]["closed"] for r in R], [r["direct"]["closed"] for r in R]),
                                          "mcnemar_p": stats.mcnemar([r["selectflow"]["closed"] for r in R], [r["direct"]["closed"] for r in R])}
    out["exact_selectflow_vs_direct"] = {"diff_ci": stats.paired_bootstrap([r["selectflow"]["exact"] for r in R], [r["direct"]["exact"] for r in R]),
                                         "mcnemar_p": stats.mcnemar([r["selectflow"]["exact"] for r in R], [r["direct"]["exact"] for r in R])}
    out["by_reference_length"] = {str(L): {m: float(np.mean([r[m]["exact"] for r in R if r["n_ref"] == L])) for m in ("selectflow", "direct")}
                                  for L in sorted({r["n_ref"] for r in R})}
    out["time_s"] = {"selectflow": float(np.mean([r["selectflow_time_s"] for r in R])), "direct": float(np.mean([r["direct_time_s"] for r in R]))}
    json.dump(out, open("results/e2_real.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
