"""E6 on the real federations: end to end, from a citizen request to the delivered document.

  SelectFlow   : interpretation of the request, requester confirmation of the targets, verified derivation,
                 joint composition (E1-complete model), then one agent episode per task with its toolkit and
                 the precise step text; tasks left to the requester are completed through the usual channel.
  all_tools    : one agent episode with every tool of the federation and the citizen request.
  tool_search  : one agent episode with every tool deferred and the native Tool Search.
Success = the target documents are held at the end (procedure semantics: apply, pay, obtain).
Claude Sonnet 5.5, synchronous calls in parallel threads.   python scripts/e6_real.py
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
import numpy as np

from selectflow import io, stats
from selectflow.agent.claude import ClaudeConfig, Ledger
from selectflow.agent.claude_generator import ClaudeGenerator, ClaudeInterpreter
from selectflow.agent.enact import Enactor, Episode, RegistryEnvironment, threaded_backend
from selectflow.model import Context, Params
from selectflow.repair import solve_request
from selectflow.selection import Featurizer, SelectionModel

SEARCH_TOOL = {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}
SYS_TASK = ("You carry out one task of an administrative procedure for a citizen by calling tools. Complete every "
            "step of the task, in the given order, calling one tool after another until the document is obtained; "
            "then stop.")
ORDER = {"submit": 0, "pay": 1, "retrieve": 2}
SYS_REQ = ("You help a citizen complete an administrative procedure by calling tools until the requested document is "
           "obtained. Apply, pay when required, and retrieve the documents, in the right order.")
PARAMS = Params(k_max=5, N=12, n_min=2, rep_max=2, t_max=30)


def main():
    cfg = ClaudeConfig(model="claude-sonnet-5-5", cache=True); cfg.thinking = {"type": "between_tools"}
    ledger = Ledger(cfg); backend = threaded_backend(ledger, workers=8)
    texts = json.load(open("data/vague_requests.json"))
    fit = json.load(open("results/e1_signed_gamma.json"))["fit_all_signed"]
    out, per_req = {}, []
    for name in ("saudi", "greece"):
        fed = io.load(f"data/{name}.json")
        reqs = json.load(open(f"data/requests_{name}.json"))
        import os
        if os.environ.get("E6_LIMIT"):
            reqs = reqs[: int(os.environ["E6_LIMIT"])]
        m = SelectionModel(np.array(fit["beta"]), fit["gamma"], Featurizer(fed)); M = {h: m for h in fed.hosts}
        arts = sorted({a for d in fed.descriptors.values() for a in d.In | d.Out})
        interp = ClaudeInterpreter(cfg, arts); gen = ClaudeGenerator(cfg)

        def plan(rq):
            G, H = set(rq["targets"]), set(rq["held"])
            got = {}
            def interp_logged(q, ctx):
                g, h = interp(q, ctx); got["G"] = g; return g, h
            t0 = time.time()
            try:
                nd, best, info = solve_request(texts[rq["id"]], Context(H), fed, PARAMS, M, interp_logged, gen,
                                               confirm=lambda g, h: (G, H))
            except Exception as e:
                return rq, None, {"error": type(e).__name__}, got, time.time() - t0
            return rq, best, info, got, time.time() - t0

        with ThreadPoolExecutor(6) as ex:
            plans = list(ex.map(plan, reqs))
        envs, sf_eps = {}, []
        for rq, best, info, got, dt in plans:
            env = RegistryEnvironment(fed, set(rq["held"])); envs[("selectflow", rq["id"])] = env
            if best is not None:
                for v in best.gap:                       # completed by the requester through the usual channel
                    env.held |= fed.descriptors[v].Out
                pos = 0
                for v in best.W.order:
                    if v in best.gap or v not in best.plan:
                        continue
                    d = fed.descriptors[v]
                    caps = sorted(d.R, key=lambda r: ORDER.get(r.split(":")[0], 3))   # apply, pay, then obtain
                    step = "; then ".join(d.cap_text[r] for r in caps)
                    sf_eps.append((pos, Episode(SYS_TASK, best.plan[v].K, env,
                                                [{"role": "user", "content": f"Procedure: {d.title}. Steps: {step}."}],
                                                meta={"rid": rq["id"]})))
                    pos += 1
            per_req.append({"fed": name, "id": rq["id"], "derived": best is not None,
                            "interpretation_correct": got.get("G") == set(rq["targets"]),
                            "workflow_matches_reference": bool(best) and sorted(best.W.order) == sorted(rq["references"][0]),
                            "tasks": len(best.W.order) if best else 0, "left_to_requester": len(best.gap) if best else 0,
                            "plan_time_s": round(dt, 2)})
        base_eps = []
        for rq in reqs:
            e2 = RegistryEnvironment(fed, set(rq["held"])); envs[("all_tools", rq["id"])] = e2
            base_eps.append(Episode(SYS_REQ, list(fed.tools), e2, [{"role": "user", "content": texts[rq["id"]]}], meta={"rid": rq["id"]}))
            e3 = RegistryEnvironment(fed, set(rq["held"])); envs[("tool_search", rq["id"])] = e3
            base_eps.append(Episode(SYS_REQ, [], e3, [{"role": "user", "content": texts[rq["id"]]}],
                                    meta={"rid": rq["id"], "native_search": True}))
        en_task = Enactor(fed, cfg, max_turns=6)
        en_req = Enactor(fed, cfg, max_turns=14, native_tool_search=SEARCH_TOOL)
        for p in range(max([q for q, _ in sf_eps] + [-1]) + 1):   # workflow positions in order, requests in parallel
            en_task.run([e for q, e in sf_eps if q == p], backend)
        en_req.run(base_eps, backend)
        # fairness: SelectFlow's requester confirms the goal; the baseline agents get the same confirmation once,
        # as the citizen's reply, if they stopped without the document
        goal = {rq["id"]: fed.descriptors[rq["id"].replace("req_", "")].title for rq in reqs}
        tgt = {rq["id"]: set(rq["targets"]) for rq in reqs}
        again = []
        for e in base_eps:
            if not tgt[e.meta["rid"]] <= e.env.held:
                e.messages.append({"role": "user", "content": f"Yes. I need: {goal[e.meta['rid']]}. Please do everything "
                                                              "needed to obtain it, without asking me again."})
                e.done, e.turns = False, 0
                again.append(e)
        en_req.run(again, backend)
        for meth in ("selectflow", "all_tools", "tool_search"):
            succ = [set(rq["targets"]) <= envs[(meth, rq["id"])].held for rq in reqs]
            calls = [len(envs[(meth, rq["id"])].calls) for rq in reqs]
            out.setdefault(meth, {"succ": [], "calls": []})
            out[meth]["succ"] += succ; out[meth]["calls"] += calls
            print(name, meth, "success", round(float(np.mean(succ)), 3), flush=True)
    R = {m: {"success": float(np.mean(v["succ"])), "tool_calls": float(np.mean(v["calls"])), "n": len(v["succ"])} for m, v in out.items()}
    for m in ("all_tools", "tool_search"):
        R[f"selectflow_vs_{m}"] = {"diff_ci": stats.paired_bootstrap(out["selectflow"]["succ"], out[m]["succ"]),
                                   "mcnemar_p": stats.mcnemar(out["selectflow"]["succ"], out[m]["succ"])}
    R["interpretation_accuracy"] = float(np.mean([r["interpretation_correct"] for r in per_req]))
    R["workflow_matches_reference"] = float(np.mean([r["workflow_matches_reference"] for r in per_req]))
    R["left_to_requester_mean"] = float(np.mean([r["left_to_requester"] for r in per_req]))
    R["cost_usd"] = round(ledger.cost(), 2)
    import os
    tag = "_smoke" if os.environ.get("E6_LIMIT") else ""
    json.dump({"summary": R, "requests": per_req}, open(f"results/e6_real{tag}.json", "w"), indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))


if __name__ == "__main__":
    main()
