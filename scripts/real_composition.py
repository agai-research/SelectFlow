"""E4 (flows, second part) and E5 (joint composition and repair) on the REAL administrative federations.
No LLM call: targets come from the requests, workflows from the registry dependencies (planner), and the
tool-selection model is the one fitted in E1 complete (results/e1_full.json).

  python scripts/real_composition.py
"""
import copy
import os
import json
import sys
import time

sys.path.insert(0, ".")
import numpy as np

from selectflow import flows, io, stats
from selectflow.baselines.toolkits import declared_view
from selectflow.compose import compose
from selectflow.derivation import check
from selectflow.experiments.runners import broken_handoffs
from selectflow.model import Context, Params
from selectflow.registry_build import build_federation, mitos_descriptors, saudi_descriptors
from selectflow.repair import solve_request
from selectflow.selection import Featurizer, SelectionModel

FIT = json.load(open("results/e1_signed_gamma.json"))["fit_all_signed"]
PARAMS = Params(k_max=5, N=10, n_min=2, rep_max=2, t_max=30)


def models_for(fed):
    m = SelectionModel(np.array(FIT["beta"]), FIT["gamma"], Featurizer(fed))
    return {h: m for h in fed.hosts}


def federations(density, budget=6000):
    for name, raw, fn in (("greece", "data/registries/raw_mitos.json", mitos_descriptors),
                          ("saudi", "data/registries/raw_saudi_manual.json", saudi_descriptors)):
        D = fn(json.load(open(raw)))
        if os.environ.get("CORRECTED"):        # categories corrected after the second model's verification
            chk = json.load(open("data/procedure_checks.json"))
            for k, d in D.items():
                c = chk.get(k)
                if c and not c.get("categories_ok"):
                    ok = {"ID", "BUS", "LOC", "FIN", "HLT"}
                    d.A = ((d.A - set(c.get("wrong_categories", []))) | (set(c.get("missing_categories", [])) & ok)) or {"ID"}
        yield name, build_federation(D, seed=0, sharing_density=density, budget=budget), json.load(open(f"data/requests_{name}.json"))


def e5():
    out = {}
    for density in (1.0, 0.75, 0.5):
        for budget in (6000, 1500):
            key = f"density_{density}_budget_{budget}"
            agg = {m: {"equipped": [], "gap": [], "broken": [], "repairs": [], "time": [], "Qbar": [], "solved": 0, "n": 0}
                   for m in ("selectflow", "task_by_task", "no_repair")}
            for name, fed, reqs in federations(density, budget):
                M = models_for(fed)
                for rq in reqs:
                    G, H = set(rq["targets"]), set(rq["held"])
                    interp = (lambda G_, H_: (lambda q, ctx: (G_, H_)))(G, H)
                    for m, kw in (("selectflow", {}), ("task_by_task", {"task_by_task": True}), ("no_repair", {"repair": False})):
                        t0 = time.time()
                        nd, best, info = solve_request(rq["text"], Context(H), fed, PARAMS, M, interp, None, **kw)
                        a = agg[m]; a["n"] += 1; a["time"].append(time.time() - t0)
                        if best is None:
                            continue
                        nT = len([v for v in best.W.order if fed.descriptors[v].R])
                        br = broken_handoffs(fed, best.W, best.plan, best.gap)
                        a["solved"] += 1
                        a["equipped"].append((nT - len(best.gap) - len(br)) / nT)
                        a["gap"].append(len(best.gap) / nT); a["broken"].append(len(br) / nT)
                        a["repairs"].append(best.repairs); a["Qbar"].append(best.Qbar)
            out[key] = {m: {"requests": a["n"], "solved": a["solved"], "equipped": float(np.mean(a["equipped"] or [0])),
                            "left_to_requester": float(np.mean(a["gap"] or [0])), "broken_handoffs": float(np.mean(a["broken"] or [0])),
                            "repairs": float(np.mean(a["repairs"] or [0])), "mean_quality": float(np.mean(a["Qbar"] or [0])),
                            "time_s": float(np.mean(a["time"]))} for m, a in agg.items()}
            print(key, json.dumps({m: {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()} for m, r in out[key].items()}))
    return out


def e4_real():
    """Plans composed with certified labels vs with declared annotations (all tools trusted):
    how many plans open a denied flow under the TRUE behaviours of the tools?"""
    res = {}
    for name, fed, reqs in federations(1.0):
        M = models_for(fed)
        dec = declared_view(fed)
        for label, F in (("certified", fed), ("declared", dec)):
            risky, tasks = 0, 0
            third = 0
            for rq in reqs:
                order = rq["references"][0]
                rep, W = check(order, set(rq["targets"]), set(rq["held"]), fed.descriptors, PARAMS.tau_chi)
                r = compose(F, W, set(), M, PARAMS, Context(set(rq["held"])))
                if r.plan is None:
                    continue
                for v, p in r.plan.items():
                    tasks += 1
                    risky += flows.risk(fed, W, v, p.K, p.host, true_labels=True) > 0
                    third += any(fed.tools[t].org == "ThirdParty" for t in p.K)
            res[f"{name}_{label}"] = {"tasks": tasks, "tasks_with_denied_flow": risky, "rate": risky / max(1, tasks),
                                      "tasks_with_uncertified_third_party_tool": third}
            print(name, label, res[f"{name}_{label}"])
    return res


if __name__ == "__main__":
    R = {"e4_real": e4_real(), "e5_real": e5()}
    json.dump(R, open("results/real_composition_corrected.json" if os.environ.get("CORRECTED") else "results/real_composition.json", "w"), indent=1)
