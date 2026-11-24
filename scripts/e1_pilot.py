"""E1 pilot on real tools (MCP-Bench) and real requests (MCP-Bench fuzzy task descriptions).

  python scripts/e1_pilot.py --model claude-haiku-5-5 --n 200 --sync        # plumbing check
  python scripts/e1_pilot.py --model claude-sonnet-5-5 --n 2000             # pilot, batch
"""
import argparse
import json
import sys

sys.path.insert(0, ".")
import numpy as np

from selectflow import io
from selectflow.agent.claude import ClaudeConfig, Ledger, ProbeSpec, run_probes
from selectflow.experiments.runners import e1_analyse
from selectflow.selection import Featurizer

SYSTEM = ("You are an AI assistant connected to MCP servers. Help the user by calling the single most appropriate "
          "tool for the first step of their request. Do not explain or plan in text: call the tool directly. "
          "If no available tool fits, answer in one short sentence without calling a tool.")


def make_specs(fed, n, seed, sizes=(2, 3, 5, 8, 12, 20)):
    rng = np.random.default_rng(seed)
    tasks = sorted(fed.descriptors)
    by_org = {}
    for t, T in fed.tools.items():
        by_org.setdefault(T.org, []).append(t)
    specs = []
    for i in range(n):
        v = tasks[int(rng.integers(len(tasks)))]
        d = fed.descriptors[v]
        own = [t for o in {d.org} | d.orgs for t in by_org.get(o, [])]
        distract_orgs = [o for o in (d.record or "").split(",") if o in by_org]
        k = int(rng.choice(sizes))
        K = list(rng.choice(own, min(len(own), int(rng.integers(1, 4))), replace=False)) if own and rng.random() < 0.8 else []
        pool = [t for o in distract_orgs for t in by_org[o]] if rng.random() < 0.7 and distract_orgs else list(fed.tools)
        pool = [t for t in pool if t not in K]
        K += list(rng.choice(pool, min(len(pool), max(0, k - len(K))), replace=False))
        specs.append(ProbeSpec(K, "step", v, seed=i, text=d.text))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5-5"); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--sync", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    fed = io.load("data/e1_mcpbench.json")
    cfg = ClaudeConfig(model=a.model, system_probe=SYSTEM, max_tokens=1024, store="results/api_cache.jsonl")
    cfg.thinking = {"type": "disabled"} if "haiku" in a.model else {"type": "between_tools"}
    if "haiku" in a.model:
        cfg.price_in, cfg.price_out, cfg.price_cache_read, cfg.price_cache_write = 0.10, 0.50, 0.01, 0.125
    ledger = Ledger(cfg)
    specs = make_specs(fed, a.n, a.seed)
    choices = run_probes(fed, specs, cfg, batch=not a.sync, ledger=ledger)
    json.dump([{"v": s.v, "K": s.K, "chosen": c} for s, c in zip(specs, choices)],
              open(f"results/e1_observations{a.tag}.json", "w"), indent=0)
    feat = Featurizer(fed)
    r = e1_analyse(feat, specs, choices, a.seed)
    m = r.pop("models")["mnl"]
    r["model"] = a.model; r["n_probes"] = a.n
    r["no_call_rate"] = float(np.mean([c is None for c in choices]))
    r["cost_usd_this_run"] = round(ledger.cost(), 4)
    json.dump(r, open(f"results/e1{a.tag}.json", "w"), indent=1, default=str)
    json.dump({"beta": m.beta.tolist(), "gamma": m.gamma, "features": list(feat.names)}, open(f"results/e1_fit{a.tag}.json", "w"))
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()
