"""E3 on the real administrative federations: toolkit composition, intended-selection rate.

Methods (same task, same capability, same seeds): SelectFlow (exact composition with the E1-complete
model), dense/lexical top-k (BM25), QoS single service, attraction ranking, HYSET (if rankings exist),
and the native Tool Search of the Claude API (all tools deferred). Message Batches, Claude Sonnet 5.5.

  python scripts/e3_real.py submit [--reps 3]
  python scripts/e3_real.py collect
"""
import argparse
import json
import os
import sys

sys.path.insert(0, ".")
import numpy as np

from selectflow import io, stats
from selectflow.agent.claude import (ClaudeConfig, Ledger, ProbeSpec, _client, _to_dict, parse_selection, probe_params, tool_definition)
from selectflow.baselines import toolkits as B
from selectflow.compose import compose
from selectflow.flows import admissible_hosts
from selectflow.model import Context, Params, Workflow
from selectflow.selection import Featurizer, Quality, SelectionModel

STATE = "results/e3_real_state.json"
SYSTEM = ("You are the assistant of a public-service portal. You carry out one step of an administrative "
          "procedure for a citizen by calling the single tool that performs it. Do not explain: call the tool "
          "directly. If no available tool performs the step, answer in one short sentence without calling a tool.")
SEARCH_TOOL = {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}
PARAMS = Params(k_max=5, N=12, n_min=2, t_max=30)


def cfg():
    c = ClaudeConfig(model="claude-sonnet-5-5", system_probe=SYSTEM, max_tokens=1024, store="results/api_cache.jsonl")
    c.thinking = {"type": "between_tools"}
    return c


def toolkits(fed, name):
    fit = json.load(open("results/e1_signed_gamma.json"))["fit_all_signed"]   # E1 complete, vague texts, signed load effect
    m = SelectionModel(np.array(fit["beta"]), fit["gamma"], Featurizer(fed))
    M = {h: m for h in fed.hosts}
    sim = B.bm25_sim(fed)
    hy = json.load(open("data/hyset_rankings_admin.json")) if os.path.exists("data/hyset_rankings_admin.json") else None
    rows = []
    for v, d in sorted(fed.descriptors.items()):
        W = Workflow([v], {})
        hosts = admissible_hosts(fed, W, v)
        if not hosts:
            continue
        res = compose(fed, W, set(), {h: M[h] for h in hosts}, PARAMS, Context(set()))
        h = res.plan[v].host if res.plan else ("hP" if "hP" in hosts else hosts[0])
        q = Quality(fed, PARAMS, M[h])
        kits = {"selectflow": res.plan[v].K if res.plan else None,
                "bm25_topk": B.dense_topk(fed, W, v, h, PARAMS, PARAMS.k_max, sim=sim),
                "qos_single": B.qos_single(fed, W, v, h, PARAMS, q),
                "attraction": B.attraction_rank(fed, W, v, h, PARAMS, q)}
        if hy:
            kits["hyset"] = B.hyset_precomputed(fed, W, v, h, PARAMS, {v: {"predicted_set": hy.get(f"{v}|{sorted(d.R)[0]}", {}).get("predicted_set", []),
                                                                         "ranking": []}})
        rows.append((v, h, kits))
    return rows


TEXTS = json.load(open("data/vague_step_texts.json"))   # citizen phrasing of each step (same for every method)


def build(fed, name, reps):
    specs = []
    for v, h, kits in toolkits(fed, name):
        d = fed.descriptors[v]
        for r in sorted(d.R):
            for rep in range(reps):
                for meth, K in kits.items():
                    if K is None:
                        continue
                    specs.append(ProbeSpec(list(K), r, v, seed=rep, text=TEXTS[f"{v}|{r}"], meta={"method": meth, "fed": name}))
                specs.append(ProbeSpec([], r, v, seed=rep, text=TEXTS[f"{v}|{r}"], meta={"method": "tool_search", "fed": name}))
    return specs


def params_for(fed, s, c):
    if s.meta["method"] != "tool_search":
        return probe_params(fed, s, c)
    p = probe_params(fed, ProbeSpec([sorted(fed.tools)[0]], s.r, s.v, s.seed, text=s.text), c)
    p["tools"] = [{**tool_definition(fed, t), "defer_loading": True} for t in sorted(fed.tools)] + [SEARCH_TOOL]
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["submit", "collect"]); ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    feds = {n: io.load(f"data/{n}.json") for n in ("greece", "saudi")}
    if a.mode == "submit":
        reqs, index = [], []
        for n, fed in feds.items():
            for s in build(fed, n, a.reps):
                index.append({"fed": n, "v": s.v, "r": s.r, "K": s.K, "method": s.meta["method"], "seed": s.seed})
                reqs.append({"custom_id": f"e{len(reqs)}", "params": params_for(fed, s, cfg())})
        b = _client().messages.batches.create(requests=reqs)
        json.dump({"batch_id": b.id, "index": index}, open(STATE, "w"))
        print("submitted", b.id, len(reqs), "probes")
        return
    st = json.load(open(STATE)); client = _client()
    b = client.messages.batches.retrieve(st["batch_id"]); print(b.processing_status, b.request_counts)
    if b.processing_status != "ended":
        return
    led = Ledger(cfg()); msgs = {}
    for r in client.messages.batches.results(b.id):
        if r.result.type == "succeeded":
            m = _to_dict(r.result.message); msgs[int(r.custom_id[1:])] = m; led.add(m.get("usage", {}), batch=True)
    rows = []
    for i, x in enumerate(st["index"]):
        if i not in msgs:
            continue
        fed = feds[x["fed"]]
        K = x["K"] if x["method"] != "tool_search" else sorted(fed.tools)
        c = parse_selection(msgs[i], K)
        ad = fed.tools[c].adequacy.get(x["r"], 0.0) if c else 0.0
        rows.append({**x, "chosen": c, "intended": bool(c and ad >= 0.5), "wrong": bool(c and ad < 0.5), "none": c is None,
                     "load": fed.load(K) if x["method"] != "tool_search" else None})
    json.dump(rows, open("results/e3_real_observations.json", "w"))
    out = {"cost_usd": round(led.cost(), 2), "n_probes": len(rows)}
    methods = sorted({r["method"] for r in rows})
    for m in methods:
        R = [r for r in rows if r["method"] == m]
        out[m] = {"intended": float(np.mean([r["intended"] for r in R])), "wrong": float(np.mean([r["wrong"] for r in R])),
                  "none": float(np.mean([r["none"] for r in R])), "n": len(R),
                  "mean_load": float(np.mean([r["load"] for r in R if r["load"] is not None] or [0]))}
    key = lambda r: (r["fed"], r["v"], r["r"], r["seed"])
    sf = {key(r): r for r in rows if r["method"] == "selectflow"}
    pv = {}
    for m in methods:
        if m == "selectflow":
            continue
        o = {key(r): r for r in rows if r["method"] == m}; ks = sorted(set(sf) & set(o))
        a1 = [sf[k]["intended"] for k in ks]; b1 = [o[k]["intended"] for k in ks]
        out[f"selectflow_vs_{m}"] = {"diff_ci": stats.paired_bootstrap(a1, b1), "n_pairs": len(ks)}
        pv[m] = stats.mcnemar(a1, b1)
    out["mcnemar_holm"] = stats.holm(pv)
    json.dump(out, open("results/e3_real.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
