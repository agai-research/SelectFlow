"""E2 (derivation) and E6 (end to end): these need the LLM.

A request file is a JSON list of
  {"id", "text", "targets": [...], "held": [...], "references": [[task ids], ...], "jurisdiction"}.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..agent.claude import ClaudeConfig, Ledger
from ..agent.claude_generator import ClaudeGenerator, ClaudeInterpreter
from ..agent.enact import SEARCH_REGISTRY, SEARCH_TOOLS, Enactor, Episode, MockEnvironment
from ..derivation import check, derive, forward_chain
from ..model import Context, Federation, Params
from ..repair import solve_request
from .. import stats


def _jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def _edges_f1(wf_ho, ref_order, fed, G, held, tau):
    _, ref = check(ref_order, G, held, fed.descriptors, tau)
    E, R = set(wf_ho), set(ref.ho)
    if not E and not R:
        return 1.0
    p = len(E & R) / len(E) if E else 0.0; r = len(E & R) / len(R) if R else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def e2(fed: Federation, requests: List[dict], cfg: ClaudeConfig, params: Params, k: int = 3) -> dict:
    """SelectFlow derivation vs direct generation vs planner alone. Targets come from the interpretation
    (or from the confirmed targets in the request file, to separate interpretation errors)."""
    arts = sorted({a for d in fed.descriptors.values() for a in d.In | d.Out})
    interp = ClaudeInterpreter(cfg, arts)
    rows = []
    for rq in requests:
        ctx = Context(set(rq.get("held", [])), jurisdiction=rq.get("jurisdiction"))
        G_i, H_i = interp(rq["text"], ctx)
        G_c, H_c = set(rq["targets"]), set(rq.get("held", []))
        rows.append({"id": rq["id"], "interpretation_correct": G_i == G_c})
        confirm = lambda G, H: (G_c, H_c)                    # the requester confirms the targets
        refs = rq["references"]
        def score(order, name, ho=None):
            rep, wf = check(order, G_c, H_c, fed.descriptors, params.tau_chi)
            best = max(refs, key=lambda r: _jacc(order, r)) if refs else []
            rows[-1][f"{name}_closed"] = rep.empty() or (not rep.missing and not rep.targets)
            rows[-1][f"{name}_task_jaccard"] = _jacc(order, best)
            rows[-1][f"{name}_dep_f1"] = _edges_f1(wf.ho, best, fed, G_c, H_c, params.tau_chi) if best else 0.0
        res = derive(rq["text"], ctx, fed, params, interp, ClaudeGenerator(cfg), confirm=confirm, k=k)
        score(res.workflows[0].order if res.workflows else [], "selectflow")
        rows[-1]["rejected_by_critics"] = res.log["rejected_by_critics"] / max(1, res.log["candidates"])
        direct = ClaudeGenerator(cfg, unconstrained=True).generate(rq["text"], G_c, H_c, fed.descriptors, 1)
        score([t for t in (direct[0] if direct else []) if t in fed.descriptors], "direct")
        order, _ = forward_chain(fed.descriptors, G_c, H_c, params.tau_chi)
        score(order or [], "planner")
    out = {}
    for m in ("selectflow", "direct", "planner"):
        out[m] = {k_: float(np.mean([r[f"{m}_{k_}"] for r in rows])) for k_ in ("closed", "task_jaccard", "dep_f1")}
    out["interpretation_accuracy"] = float(np.mean([r["interpretation_correct"] for r in rows]))
    out["critics_rejection_rate"] = float(np.mean([r["rejected_by_critics"] for r in rows]))
    for m in ("direct", "planner"):
        out[f"closed_vs_{m}"] = stats.mcnemar([r["selectflow_closed"] for r in rows], [r[f"{m}_closed"] for r in rows])
    out["rows"] = rows
    return out


def e6(fed: Federation, requests: List[dict], cfg: ClaudeConfig, params: Params, models: Dict,
       backend, native_tool_search=None) -> dict:
    """End to end on citizen requests. Success = targets held at the end (postconditions)."""
    arts = sorted({a for d in fed.descriptors.values() for a in d.In | d.Out})
    interp = ClaudeInterpreter(cfg, arts)
    en = Enactor(fed, cfg, native_tool_search=native_tool_search)
    episodes = {"selectflow": [], "all_tools": [], "search_agent": []}
    envs = {}
    sys_task = "You carry out one step of an administrative procedure for a citizen by calling tools."
    sys_req = "You help a citizen complete an administrative procedure by calling tools until it is done."
    for rq in requests:
        G_c, H_c = set(rq["targets"]), set(rq.get("held", []))
        ctx = Context(H_c, jurisdiction=rq.get("jurisdiction"))
        nd, best, info = solve_request(rq["text"], ctx, fed, params, models, interp, ClaudeGenerator(cfg),
                                       confirm=lambda G, H: (G_c, H_c))
        env = MockEnvironment(fed, H_c); envs[("selectflow", rq["id"])] = env
        if best is not None:
            gap_outputs = set().union(*[fed.descriptors[v].Out for v in best.gap]) if best.gap else set()
            env.held |= gap_outputs                      # completed by the requester through the usual channel
            for v in best.W.order:
                if v in best.gap or v not in best.plan:
                    continue
                d = fed.descriptors[v]
                episodes["selectflow"].append(Episode(sys_task, best.plan[v].K, env,
                    [{"role": "user", "content": f"Step: {d.title}. {' '.join(d.cap_text.values())}"}],
                    meta={"rid": rq["id"]}))
        env2 = MockEnvironment(fed, H_c); envs[("all_tools", rq["id"])] = env2
        episodes["all_tools"].append(Episode(sys_req, list(fed.tools), env2,
                                             [{"role": "user", "content": rq["text"]}], meta={"rid": rq["id"]}))
        env3 = MockEnvironment(fed, H_c); envs[("search_agent", rq["id"])] = env3
        episodes["search_agent"].append(Episode(sys_req, [], env3, [{"role": "user", "content": rq["text"]}],
                                                meta={"rid": rq["id"], "native_search": bool(native_tool_search)},
                                                meta_tools=[] if native_tool_search else [SEARCH_TOOLS, SEARCH_REGISTRY]))
    # SelectFlow tasks run in workflow order: one round of episodes per position
    en.run(episodes["all_tools"] + episodes["search_agent"], backend)
    for ep in episodes["selectflow"]:
        en.run([ep], backend)
    out = {}
    for m in episodes:
        succ = [G <= envs[(m, rq["id"])].held for rq in requests for G in [set(rq["targets"])]]
        calls = [len(envs[(m, rq["id"])].calls) for rq in requests]
        out[m] = {"success": float(np.mean(succ)), "tool_calls": float(np.mean(calls))}
        out[m]["_succ"] = succ
    for m in ("all_tools", "search_agent"):
        out[f"selectflow_vs_{m}"] = stats.mcnemar(out["selectflow"]["_succ"], out[m]["_succ"])
    return out
