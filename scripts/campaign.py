"""Real campaign with the Claude API. Requires ANTHROPIC_API_KEY.

Usage:
  python scripts/campaign.py e1 --fed data/greece.json --n 2000          # pilot, go/no-go
  python scripts/campaign.py e3 --fed data/greece.json --fit results/e1_fit.json
  python scripts/campaign.py e2 --fed data/greece.json --requests data/requests.json
  python scripts/campaign.py e6 --fed data/greece.json --requests data/requests.json --fit results/e1_fit.json
  add --debug to use the cheap model (all debugging must be done with it)
"""
import argparse
import json
import sys

sys.path.insert(0, ".")
import numpy as np

from selectflow import io
from selectflow.agent.claude import ClaudeConfig, Ledger, run_probes
from selectflow.agent.enact import batch_backend, sync_backend
from selectflow.experiments import runners as X
from selectflow.experiments import runners_llm as XL
from selectflow.model import Params
from selectflow.selection import Featurizer, SelectionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["e1", "e3", "e2", "e6"])
    ap.add_argument("--fed", required=True); ap.add_argument("--requests"); ap.add_argument("--fit")
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--debug", action="store_true")
    ap.add_argument("--sync", action="store_true", help="no batch (faster, twice the price)")
    a = ap.parse_args()
    conf = json.load(open("config.json"))
    cfg = ClaudeConfig(model=conf["debug_model"] if a.debug else conf["model"], debug_model=conf["debug_model"])
    params = Params(**conf["params"])
    fed = io.load(a.fed)
    ledger = Ledger(cfg)
    feat = Featurizer(fed)
    select = lambda specs: run_probes(fed, specs, cfg, batch=not a.sync, ledger=ledger)

    def fitted():
        f = json.load(open(a.fit))
        m = SelectionModel(np.array(f["beta"]), f["gamma"], feat)
        return {h: m for h in fed.hosts}

    if a.exp == "e1":
        r = X.e1(fed, feat, select, a.n, sizes=tuple(conf["e1"]["toolkit_sizes"]))
        m = r.pop("models")["mnl"]
        json.dump({"beta": m.beta.tolist(), "gamma": m.gamma}, open("results/e1_fit.json", "w"))
    elif a.exp == "e3":
        r = X.e3(fed, fitted(), select, params, n_rep=conf["e3"]["repetitions"]); r.pop("rows")
    elif a.exp == "e2":
        r = XL.e2(fed, json.load(open(a.requests)), cfg, params); r.pop("rows")
    else:
        back = sync_backend(ledger) if a.sync else batch_backend(ledger)
        r = XL.e6(fed, json.load(open(a.requests)), cfg, params, fitted(), back, conf.get("native_tool_search"))
    r["cost_usd"] = ledger.cost()
    json.dump(r, open(f"results/{a.exp}.json", "w"), indent=1, default=str)
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()
