"""Study S2: number of offered tools. For every capability of both federations (data version 1, citizen-phrased
steps, as in E3), BM25 top-k with k in {1,2,3,5,8,12} and SelectFlow with k_max in {1,2,3,5,8}; one repetition.
  python scripts/e3_sizes.py submit | collect"""
import json, sys
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np
from selectflow import io
from selectflow.agent.claude import Ledger, ProbeSpec, _client, _to_dict, parse_selection, probe_params
from selectflow.baselines import toolkits as B
from selectflow.compose import compose
from selectflow.flows import admissible_hosts
from selectflow.model import Context, Params, Workflow
from selectflow.selection import Featurizer, SelectionModel
from e3_real import cfg, TEXTS

STATE = "results/e3_sizes_state.json"
KS_BM25, KS_SF = (1, 2, 3, 5, 8, 12), (1, 2, 3, 5, 8)


def build():
    fit = json.load(open("results/e1_signed_gamma.json"))["fit_all_signed"]
    specs = []
    for name in ("greece", "saudi"):
        fed = io.load(f"data/{name}.json")
        m = SelectionModel(np.array(fit["beta"]), fit["gamma"], Featurizer(fed)); M = {h: m for h in fed.hosts}
        sim = B.bm25_sim(fed)
        for v, d in sorted(fed.descriptors.items()):
            W = Workflow([v], {}); hosts = admissible_hosts(fed, W, v)
            if not hosts:
                continue
            h = "hP" if "hP" in hosts else hosts[0]
            kits = {}
            for k in KS_BM25:
                kits[("bm25", k)] = B.dense_topk(fed, W, v, h, Params(k_max=k, N=12, n_min=2), k, sim=sim)
            for k in KS_SF:
                r = compose(fed, W, set(), {hh: M[hh] for hh in hosts}, Params(k_max=k, N=12, n_min=min(2, k), t_max=20), Context(set()))
                if r.plan:
                    kits[("selectflow", k)] = r.plan[v].K
            for r_ in sorted(d.R):
                for (meth, k), K in kits.items():
                    specs.append((name, ProbeSpec(list(K), r_, v, seed=0, text=TEXTS[f"{v}|{r_}"], meta={"method": meth, "k": k})))
    return specs


def main():
    mode = sys.argv[1]
    feds = {n: io.load(f"data/{n}.json") for n in ("greece", "saudi")}
    if mode == "submit":
        specs = build()
        idx = [{"fed": n, "v": s.v, "r": s.r, "K": s.K, "method": s.meta["method"], "k": s.meta["k"]} for n, s in specs]
        reqs = [{"custom_id": f"z{i}", "params": probe_params(feds[n], s, cfg())} for i, (n, s) in enumerate(specs)]
        b = _client().messages.batches.create(requests=reqs)
        json.dump({"batch_id": b.id, "index": idx}, open(STATE, "w")); print("submitted", b.id, len(reqs)); return
    st = json.load(open(STATE)); client = _client()
    b = client.messages.batches.retrieve(st["batch_id"]); print(b.processing_status, b.request_counts)
    if b.processing_status != "ended":
        return
    rows, led = [], Ledger(cfg())
    for r in client.messages.batches.results(b.id):
        if r.result.type != "succeeded":
            continue
        i = int(r.custom_id[1:]); x = st["index"][i]; m = _to_dict(r.result.message); led.add(m.get("usage", {}), batch=True)
        c = parse_selection(m, x["K"]); fed = feds[x["fed"]]
        ad = fed.tools[c].adequacy.get(x["r"], 0.0) if c else 0.0
        u = m.get("usage", {})
        rows.append({**x, "chosen": c, "intended": bool(c and ad >= 0.5), "wrong": bool(c and ad < 0.5), "none": c is None,
                     "size": len(x["K"]), "load": fed.load(x["K"]),
                     "in_tokens": (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)})
    json.dump(rows, open("results/e3_sizes_observations.json", "w"))
    out = {"cost_usd": round(led.cost(), 2)}
    for meth in ("bm25", "selectflow"):
        out[meth] = {}
        for k in sorted({r["k"] for r in rows if r["method"] == meth}):
            R = [r for r in rows if r["method"] == meth and r["k"] == k]
            out[meth][str(k)] = {q: float(np.mean([r[q] for r in R])) for q in ("intended", "wrong", "none", "size", "load", "in_tokens")} | {"n": len(R)}
    json.dump(out, open("results/e3_sizes.json", "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
