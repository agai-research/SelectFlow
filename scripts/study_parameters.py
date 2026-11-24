"""Study S5: key parameters of the composer, on the real federations with tool cards mapped from independent
descriptions (data version 3m). Outcomes predicted by the fitted tool-selection model (no API call).
  python scripts/study_parameters.py params | grid DENSITIES"""
import json, os, sys, time
sys.path.insert(0, ".")
import numpy as np
from selectflow import io
from selectflow.compose import compose
from selectflow.flows import admissible_hosts
from selectflow.model import Context, Params, Workflow
from selectflow.selection import Featurizer, Quality, SelectionModel

FIT = json.load(open("results/e1_signed_gamma.json"))["fit_all_signed"]
OUT = "results/studies/s5_parameters.json"
out = json.load(open(OUT)) if os.path.exists(OUT) else {}


def run(fed, M, p, budget=None):
    if budget:
        for h in fed.hosts.values(): h.budget = budget
    rows = []
    for v, d in sorted(fed.descriptors.items()):
        W = Workflow([v], {}); hosts = admissible_hosts(fed, W, v)
        if not hosts:
            continue
        r = compose(fed, W, set(), {h: M[h] for h in hosts}, p, Context(set()))
        if not r.plan:
            rows.append({"feasible": False}); continue
        P = r.plan[v]; m = M[P.host]; q = Quality(fed, p, m)
        star, none = [], []
        for cap in d.R:
            pr, p0 = m.probs(P.K, cap, v)
            star.append(sum(x for t, x in pr.items() if fed.tools[t].adequacy.get(cap, 0) >= 0.5)); none.append(p0)
        rows.append({"feasible": True, "size": len(P.K), "load": P.load, "Q": P.Q / len(d.R), "p_intended": float(np.mean(star)), "p_none": float(np.mean(none))})
    F = [x for x in rows if x["feasible"]]
    return {"feasible": len(F) / len(rows), **{k: float(np.mean([x[k] for x in F])) for k in ("size", "load", "Q", "p_intended", "p_none")}}


def feds():
    for n in ("greece", "saudi"):
        fed = io.load(f"data/v3m/{n}.json"); m = SelectionModel(np.array(FIT["beta"]), FIT["gamma"], Featurizer(fed))
        yield n, fed, {h: m for h in fed.hosts}


def merge(a, b, wa, wb):
    return {k: (a[k] * wa + b[k] * wb) / (wa + wb) for k in a}


if sys.argv[1] == "params":
    grid = [("k_max", k, Params(k_max=k, N=12, n_min=min(2, k), tau_mu=0.5, t_max=10), None) for k in (1, 2, 3, 5, 8)]
    grid += [("budget", b, Params(k_max=5, N=12, n_min=2, tau_mu=0.5, t_max=10), b) for b in (150, 300, 600, 1200, 6000)]
    grid += [("tau_mu", t, Params(k_max=5, N=12, n_min=2, tau_mu=t, t_max=10), None) for t in (0.2, 0.4, 0.6, 0.8, 1.0)]
    grid += [("N", n, Params(k_max=5, N=n, n_min=min(2, n), tau_mu=0.5, t_max=10), None) for n in (2, 4, 8, 16)]
    todo = [g for g in grid if f"{g[0]}={g[1]}" not in out]
    t0 = time.time()
    for name, val, p, b in todo:
        if time.time() - t0 > 230:
            break
        res = {n: run(fed, M, p, b) for n, fed, M in feds()}
        out[f"{name}={val}"] = merge(res["greece"], res["saudi"], 93, 24) | {"param": name, "value": val}
        print(name, val, {k: round(v, 3) for k, v in out[f"{name}={val}"].items() if isinstance(v, float)}, flush=True)
json.dump(out, open(OUT, "w"), indent=1)
print(len(out), "configurations done")
