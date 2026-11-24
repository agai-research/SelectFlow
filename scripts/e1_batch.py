"""Submit / collect the E1 pilot as a Message Batch (50% cheaper), without blocking.

  python scripts/e1_batch.py submit --model claude-sonnet-5-5 --n 2000
  python scripts/e1_batch.py collect          # repeat until the batch has ended; then analyses
"""
import argparse
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np

from selectflow import io
from selectflow.agent.claude import ClaudeConfig, Ledger, Store, _client, _to_dict, parse_selection, probe_params
from selectflow.experiments.runners import e1_analyse
from selectflow.selection import Featurizer
from e1_pilot import SYSTEM, make_specs

STATE = "results/e1_batch_state.json"


def cfg_for(model):
    c = ClaudeConfig(model=model, system_probe=SYSTEM, max_tokens=1024, store="results/api_cache.jsonl")
    c.thinking = {"type": "disabled"} if "haiku" in model else {"type": "between_tools"}
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["submit", "collect"])
    ap.add_argument("--model", default="claude-sonnet-5-5"); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    fed = io.load("data/e1_mcpbench.json")
    client = _client()
    if a.mode == "submit":
        cfg = cfg_for(a.model)
        specs = make_specs(fed, a.n, a.seed)
        reqs = [{"custom_id": f"p{i}", "params": probe_params(fed, s, cfg)} for i, s in enumerate(specs)]
        b = client.messages.batches.create(requests=reqs)
        json.dump({"batch_id": b.id, "model": a.model, "n": a.n, "seed": a.seed}, open(STATE, "w"))
        print("submitted", b.id, b.processing_status)
        return
    st = json.load(open(STATE))
    b = client.messages.batches.retrieve(st["batch_id"])
    print(b.id, b.processing_status, b.request_counts)
    if b.processing_status != "ended":
        return
    cfg = cfg_for(st["model"])
    specs = make_specs(fed, st["n"], st["seed"])
    ledger = Ledger(cfg)
    msgs = {}
    for r in client.messages.batches.results(b.id):
        if r.result.type == "succeeded":
            m = _to_dict(r.result.message); msgs[int(r.custom_id[1:])] = m; ledger.add(m.get("usage", {}), batch=True)
    choices = [parse_selection(msgs[i], s.K) if i in msgs else "__error__" for i, s in enumerate(specs)]
    stops = {}
    for m in msgs.values():
        stops[m.get("stop_reason")] = stops.get(m.get("stop_reason"), 0) + 1
    json.dump([{"v": s.v, "K": s.K, "chosen": c} for s, c in zip(specs, choices)],
              open("results/e1_observations_sonnet.json", "w"))
    with open("results/e1_sonnet_messages.jsonl", "w") as f:
        for i in sorted(msgs):
            f.write(json.dumps({"i": i, "message": msgs[i]}) + "\n")
    feat = Featurizer(fed)
    r = e1_analyse(feat, specs, choices, st["seed"])
    m = r.pop("models")["mnl"]
    r.update({"model": st["model"], "n_probes": st["n"], "succeeded": len(msgs), "stop_reasons": stops,
              "no_call_rate": float(np.mean([c is None for c in choices if c != "__error__"])),
              "cost_usd": round(ledger.cost(), 3)})
    json.dump(r, open("results/e1_sonnet.json", "w"), indent=1, default=str)
    json.dump({"beta": m.beta.tolist(), "gamma": m.gamma, "features": list(feat.names)}, open("results/e1_fit_sonnet.json", "w"))
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()
