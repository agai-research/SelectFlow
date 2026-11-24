"""E6 in resumable chunks (the sandbox stops background jobs): each call processes the next pending
requests in parallel and appends one record per request to results/e6_real_records.jsonl.

  python scripts/e6_chunks.py run [--k 8]      # repeat until nothing is pending
  python scripts/e6_chunks.py summary
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
from selectflow.agent.claude import ClaudeConfig, Ledger, _client, _to_dict
from selectflow.agent.claude_generator import ClaudeGenerator, ClaudeInterpreter
from selectflow.agent.enact import Enactor, Episode, RegistryEnvironment
from selectflow.model import Context, Params
from selectflow.repair import solve_request
from selectflow.selection import Featurizer, SelectionModel

DATA = os.environ.get("SF_DATA", "data")
REC = os.environ.get("E6_REC", "results/e6_real_records.jsonl")
SUBSET = os.environ.get("E6_SUBSET")          # JSON list of request ids
SEARCH_TOOL = {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}
SYS_TASK = ("You carry out one task of an administrative procedure for a citizen by calling tools. Complete every "
            "step of the task, in the given order, calling one tool after another until the document is obtained; then stop.")
SYS_REQ = ("You help a citizen complete an administrative procedure by calling tools until the requested document is "
           "obtained. Apply, pay when required, and retrieve the documents, in the right order.")
ORDER = {"submit": 0, "pay": 1, "retrieve": 2}
PARAMS = Params(k_max=5, N=12, n_min=2, rep_max=2, t_max=30)
cfg = ClaudeConfig(model="claude-sonnet-5-5", cache=True); cfg.thinking = {"type": "between_tools"}


def call_backend(ledger):
    client = _client()
    def run(params_list):
        out = []
        for p in params_list:
            for attempt in range(5):
                try:
                    m = _to_dict(client.messages.create(**p)); ledger.add(m.get("usage", {}), batch=False); out.append(m); break
                except Exception:
                    time.sleep(2 ** attempt)
            else:
                out.append({"content": []})
        return out
    return run


def run_request(name, rq, text, fed, M):
    ledger = Ledger(cfg); back = call_backend(ledger)
    arts = sorted({a for d in fed.descriptors.values() for a in d.In | d.Out})
    G, H = set(rq["targets"]), set(rq["held"])
    got = {}
    def interp(q, ctx):
        g, h = ClaudeInterpreter(cfg, arts)(q, ctx); got["G"] = g; return g, h
    rec = {"fed": name, "id": rq["id"]}
    t0 = time.time()
    try:
        nd, best, info = solve_request(text, Context(H), fed, PARAMS, M, interp, ClaudeGenerator(cfg), confirm=lambda g, h: (G, H))
    except Exception as e:
        best, rec["plan_error"] = None, type(e).__name__
    rec["plan_time_s"] = round(time.time() - t0, 1)
    rec["interpretation_correct"] = got.get("G") == G
    env = RegistryEnvironment(fed, set(H))
    if best is not None:
        rec.update({"derived": True, "tasks": len(best.W.order), "left_to_requester": len(best.gap),
                    "workflow_matches_reference": sorted(best.W.order) == sorted(rq["references"][0]),
                    "toolkit_sizes": [len(best.plan[v].K) for v in best.plan]})
        for v in best.gap:
            env.held |= fed.descriptors[v].Out
        en = Enactor(fed, cfg, max_turns=6)
        for v in best.W.order:
            if v in best.gap or v not in best.plan:
                continue
            d = fed.descriptors[v]
            caps = sorted(d.R, key=lambda r: ORDER.get(r.split(":")[0], 3))
            ep = Episode(SYS_TASK, best.plan[v].K, env,
                         [{"role": "user", "content": f"Procedure: {d.title}. Steps: {'; then '.join(d.cap_text[r] for r in caps)}."}])
            en.run([ep], back)
    else:
        rec["derived"] = False
    rec["selectflow_success"] = G <= env.held
    rec["selectflow_calls"] = len(env.calls)
    goal = fed.descriptors[rq["id"].replace("req_", "")].title
    for meth in ([] if os.environ.get("E6_ONLY_SF") else ["all_tools", "tool_search"]):
        e = RegistryEnvironment(fed, set(H))
        ep = Episode(SYS_REQ, list(fed.tools) if meth == "all_tools" else [], e, [{"role": "user", "content": text}],
                     meta={"native_search": meth == "tool_search"})
        en = Enactor(fed, cfg, max_turns=10, native_tool_search=SEARCH_TOOL)
        en.run([ep], back)
        confirmed = False
        if not G <= e.held:            # same goal confirmation as SelectFlow's requester
            ep.messages.append({"role": "user", "content": f"Yes. I need: {goal}. Please do everything needed to obtain it, without asking me again."})
            ep.done, ep.turns = False, 0; confirmed = True
            en.run([ep], back)
        rec[f"{meth}_success"] = G <= e.held; rec[f"{meth}_calls"] = len(e.calls); rec[f"{meth}_needed_confirmation"] = confirmed
    rec["cost_usd"] = round(ledger.cost(), 4)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["run", "summary"]); ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    done = {json.loads(l)["id"] for l in open(REC)} if os.path.exists(REC) else set()
    if a.mode == "run":
        texts = json.load(open("data/vague_requests.json"))
        fit = json.load(open("results/e1_signed_gamma.json"))["fit_all_signed"]
        jobs = []
        for name in ("saudi", "greece"):
            fed = io.load(f"{DATA}/{name}.json")
            m = SelectionModel(np.array(fit["beta"]), fit["gamma"], Featurizer(fed)); M = {h: m for h in fed.hosts}
            keep = set(json.load(open(SUBSET))) if SUBSET else None
            jobs += [(name, rq, texts[rq["id"]], fed, M) for rq in json.load(open(f"{DATA}/requests_{name}.json"))
                     if rq["id"] not in done and (keep is None or rq["id"] in keep)]
        jobs = jobs[: a.k]
        with ThreadPoolExecutor(len(jobs) or 1) as ex:
            for rec in ex.map(lambda j: run_request(*j), jobs):
                with open(REC, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(rec["id"], rec["selectflow_success"], rec.get("left_to_requester"), flush=True)
        print(len(done) + len(jobs), "done of 82")
        return
    R = [json.loads(l) for l in open(REC)]
    S = {m: [r[f"{m}_success"] for r in R] for m in ("selectflow", "all_tools", "tool_search")}
    out = {m: {"success": float(np.mean(v)), "n": len(v), "tool_calls": float(np.mean([r[f"{m}_calls"] for r in R]))} for m, v in S.items()}
    for m in ("all_tools", "tool_search"):
        out[f"selectflow_vs_{m}"] = {"diff_ci": stats.paired_bootstrap(S["selectflow"], S[m]), "mcnemar_p": stats.mcnemar(S["selectflow"], S[m])}
        out[m]["needed_confirmation"] = float(np.mean([r[f"{m}_needed_confirmation"] for r in R]))
    for k in ("derived", "interpretation_correct", "workflow_matches_reference"):
        out[k] = float(np.mean([bool(r.get(k)) for r in R]))
    out["left_to_requester_mean"] = float(np.mean([r.get("left_to_requester", 0) for r in R]))
    out["by_federation"] = {f: {m: float(np.mean([r[f"{m}_success"] for r in R if r["fed"] == f])) for m in S} for f in ("greece", "saudi")}
    out["cost_usd"] = round(sum(r["cost_usd"] for r in R), 2)
    json.dump(out, open(REC.replace("_records.jsonl", ".json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
