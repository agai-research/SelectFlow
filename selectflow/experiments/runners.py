"""Experiment runners E1, E3, E4, E5 (E2 and E6 need the LLM: see scripts/).

Each runner takes `select_fn(specs) -> choices`, so the same code runs with the
simulated agent (dry run) or with Claude probes (real campaign).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..agent.claude import ProbeSpec
from ..baselines import toolkits as B
from ..certification import benjamini_hochberg, conformal_pvalues
from ..compose import compose
from ..flows import admissible_hosts
from ..model import Context, Federation, Params, Workflow
from ..repair import solve_request
from ..selection import (Featurizer, Observation, Quality, RelevanceModel, fit, predictive_scores)
from .. import stats

SelectFn = Callable[[List[ProbeSpec]], List[Optional[str]]]


def _caps(fed):
    return [(r, v) for v, d in sorted(fed.descriptors.items()) for r in sorted(d.R)]


# ------------------------------------------------------------------ E1
def e1(fed: Federation, feat: Featurizer, select_fn: SelectFn, n_obs: int, seed: int = 0,
       sizes=(2, 3, 5, 8, 12, 20)) -> dict:
    """Prediction of the selection: MNL vs MNL with gamma = 0 vs relevance model, on unseen capabilities."""
    rng = np.random.default_rng(seed)
    caps = _caps(fed)
    specs = []
    for i in range(n_obs):
        r, v = caps[int(rng.integers(len(caps)))]
        provs = [t for t, T in fed.tools.items() if r in T.caps]
        k = int(rng.choice(sizes))
        K = list(rng.choice(provs, min(len(provs), int(rng.integers(1, 4))), replace=False)) if rng.random() < 0.8 else []
        others = [t for t in fed.tools if t not in K]
        K += list(rng.choice(others, max(0, k - len(K)), replace=False))
        specs.append(ProbeSpec(K, r, v, seed=i))
    return e1_analyse(feat, specs, select_fn(specs), seed)


def e1_analyse(feat: Featurizer, specs, choices, seed: int = 0) -> dict:
    """Fit on 70% of the (task, capability) groups, compare on the 15% test groups."""
    rng = np.random.default_rng(seed)
    obs = [Observation(s.K, s.r, s.v, c, group=f"{s.v}|{s.r}") for s, c in zip(specs, choices) if c != "__error__"]
    groups = sorted({o.group for o in obs}); rng.shuffle(groups)
    n = len(groups); tr = set(groups[: int(0.7 * n)]); va = set(groups[int(0.7 * n): int(0.85 * n)])
    train = [o for o in obs if o.group in tr]; test = [o for o in obs if o.group not in tr and o.group not in va]
    full, _ = fit(train, feat)
    g0, _ = fit(train, feat, gamma_fixed=0.0)
    rel = RelevanceModel.fit(train, feat)
    out = {"n_train": len(train), "n_test": len(test), "gamma_hat": full.gamma, "beta_hat": full.beta.round(3).tolist()}
    ll = {}
    for name, m in [("mnl", full), ("mnl_gamma0", g0), ("relevance", rel)]:
        out[name] = predictive_scores(m, test)
        ll[name] = [predictive_scores(m, [o])["log_loss"] for o in test]
    out["diff_vs_relevance"] = stats.paired_bootstrap(ll["relevance"], ll["mnl"])
    out["diff_vs_gamma0"] = stats.paired_bootstrap(ll["mnl_gamma0"], ll["mnl"])
    out["models"] = {"mnl": full}
    return out


# ------------------------------------------------------------------ E3
def e3(fed: Federation, models: Dict, select_fn: SelectFn, params: Params, n_rep: int = 5,
       dense_sim=None) -> dict:
    """Toolkit composition: SelectFlow vs relevance, QoS single-service and attraction ranking."""
    rows, infeasible = [], []
    specs, meta = [], []
    for v, d in sorted(fed.descriptors.items()):
        if not d.R:
            continue
        W = Workflow([v], {})
        hosts = admissible_hosts(fed, W, v)
        if not hosts:
            continue
        h = "hP" if "hP" in hosts else hosts[0]
        q = Quality(fed, params, models[h])
        res = compose(fed, W, set(), {hh: models[hh] for hh in hosts}, params, Context(set()))
        kits = {}
        if not res.plan:
            infeasible.append(v)
        if res.plan:
            kits["selectflow"] = res.plan[v].K
            h = res.plan[v].host
        kits["dense_topk"] = B.dense_topk(fed, W, v, h, params, params.k_max, sim=dense_sim)
        kits["qos_single"] = B.qos_single(fed, W, v, h, params, q)
        kits["attraction"] = B.attraction_rank(fed, W, v, h, params, q)
        for name, K in kits.items():
            for r in sorted(d.R):
                for rep in range(n_rep):
                    specs.append(ProbeSpec(list(K), r, v, seed=rep))
                    meta.append((name, v, r, rep, fed.load(K), len(K)))
    choices = select_fn(specs)
    for (name, v, r, rep, load, size), c in zip(meta, choices):
        ad = fed.tools[c].adequacy.get(r, 0.0) if (c and r in fed.tools[c].caps) else 0.0
        rows.append({"method": name, "v": v, "r": r, "rep": rep, "intended": c is not None and ad >= 0.5,
                     "wrong": c is not None and ad < 0.5, "none": c is None, "load": load, "size": size})
    out = {}
    methods = sorted({r["method"] for r in rows})
    for m in methods:
        R = [r for r in rows if r["method"] == m]
        out[m] = {k: float(np.mean([r[k] for r in R])) for k in ("intended", "wrong", "none", "load", "size")}
        out[m]["n"] = len(R)
    # paired comparisons on the units (v, r, rep) where every method was evaluated
    key = lambda r: (r["v"], r["r"], r["rep"])
    if "selectflow" in methods:
        sf = {key(r): r for r in rows if r["method"] == "selectflow"}
        pv = {}
        for m in methods:
            if m == "selectflow":
                continue
            other = {key(r): r for r in rows if r["method"] == m}
            ks = sorted(set(sf) & set(other))
            a = [sf[k]["intended"] for k in ks]; b = [other[k]["intended"] for k in ks]
            out[f"selectflow_vs_{m}"] = {"diff_ci": stats.paired_bootstrap(a, b), "n_pairs": len(ks)}
            pv[m] = stats.mcnemar(a, b)
        out["mcnemar_holm"] = stats.holm(pv)
    out["selectflow_infeasible_tasks"] = infeasible
    out["rows"] = rows
    return out


# ------------------------------------------------------------------ E4
def e4(n_tools: int = 3000, eps: float = 0.1, audit_frac: float = 0.2, reps: int = 50, seed: int = 0) -> dict:
    """Certification: conformal selection vs threshold vs declared annotations (ground truth known)."""
    rng = np.random.default_rng(seed)
    strata = {("institution", "read"): 0.05, ("institution", "write"): 0.15,
              ("third_party", "read"): 0.3, ("third_party", "write"): 0.5}
    res = {m: {"fdr": [], "power": []} for m in ("conformal", "threshold", "declared")}
    for _ in range(reps):
        sel_all, unsafe_all = {m: [] for m in res}, []
        for (cls, typ), rate in strata.items():
            n = n_tools // len(strata)
            unsafe = rng.random(n) < rate
            score = np.clip(np.where(unsafe, rng.normal(0.35, 0.2, n), rng.normal(0.7, 0.2, n)), 0, 1)
            aud = np.zeros(n, bool); aud[rng.choice(n, int(audit_frac * n), replace=False)] = True
            p = conformal_pvalues(score[~aud], score[aud], unsafe[aud])
            sel_all["conformal"].append(benjamini_hochberg(p, eps))
            sel_all["threshold"].append(score[~aud] >= 0.5)
            declared_safe = np.ones((~aud).sum(), bool) if cls == "third_party" else ~unsafe[~aud]
            sel_all["declared"].append(declared_safe)
            unsafe_all.append(unsafe[~aud])
        u = np.concatenate(unsafe_all)
        for m in res:
            s = np.concatenate(sel_all[m])
            res[m]["fdr"].append(float(u[s].mean()) if s.any() else 0.0)
            res[m]["power"].append(float(s[~u].mean()))
    return {m: {"fdr": float(np.mean(v["fdr"])), "power": float(np.mean(v["power"]))} for m, v in res.items()} | {"eps": eps}


# ------------------------------------------------------------------ E5
def broken_handoffs(fed: Federation, W: Workflow, plan: Dict, gap) -> set:
    """Equipped tasks whose incoming hand-off has no producer upstream or no consumer in their toolkit."""
    bad = set()
    for (u, v), arts in W.ho.items():
        if v in gap or v not in plan:
            continue
        for a in arts:
            if not any(a in fed.tools[t].cons for t in plan[v].K):
                bad.add(v)
            if u in plan and u not in gap and not any(a in fed.tools[t].prod for t in plan[u].K):
                bad.add(v)
    return bad


def e5(fed: Federation, models: Dict, params: Params, n_req: int = 20, seed: int = 0) -> dict:
    """Joint composition with repair vs task-by-task vs without repair, on derived workflows."""
    rng = np.random.default_rng(seed)
    targets = [a for d in fed.descriptors.values() for a in d.Out]
    configs = {"selectflow": dict(), "task_by_task": dict(task_by_task=True), "no_repair": dict(repair=False)}
    out = {k: {"equipped": [], "gap": [], "repairs": [], "time": [], "solved": 0} for k in configs}
    for i in range(n_req):
        G = {targets[int(rng.integers(len(targets)))]}
        interp = (lambda G_: (lambda q, ctx: (G_, set())))(G)
        for name, kw in configs.items():
            t0 = time.time()
            nd, best, info = solve_request(f"request {i}", Context(set(), lat_max=40.0), fed, params, models,
                                           interp, None, **kw)
            out[name]["time"].append(time.time() - t0)
            if best is None:
                continue
            nT = len([v for v in best.W.order if fed.descriptors[v].R])
            broken = broken_handoffs(fed, best.W, best.plan, best.gap)
            out[name]["solved"] += 1
            out[name]["equipped"].append((nT - len(best.gap) - len(broken)) / nT if nT else 1.0)
            out[name].setdefault("broken", []).append(len(broken) / nT if nT else 0.0)
            out[name]["gap"].append(len(best.gap) / nT if nT else 0.0)
            out[name]["repairs"].append(best.repairs)
    return {k: {"solved": v["solved"], "equipped": float(np.mean(v["equipped"] or [0])),
                "broken_handoffs": float(np.mean(v.get("broken", [0]))),
                "gap": float(np.mean(v["gap"] or [0])), "repairs": float(np.mean(v["repairs"] or [0])),
                "time_s": float(np.mean(v["time"]))} for k, v in out.items()}
