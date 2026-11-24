"""Dry run of the whole pipeline with a SIMULATED agent (no API calls).

It checks that the code runs end to end and that the measurements are computed
correctly. The numbers it prints are NOT experimental results: the simulated
agent follows the model that SelectFlow assumes.
"""
import json
import sys
import time

sys.path.insert(0, ".")
import numpy as np

from selectflow import scenario_greece as GR, synth
from selectflow.derivation import ScriptedGenerator
from selectflow.experiments import runners as X
from selectflow.model import Params
from selectflow.repair import solve_request


def main(out="results/dryrun.json"):
    t0 = time.time(); R = {}
    # Running scenario
    fed = GR.build()
    nd, best, info = solve_request(GR.REQUEST, GR.CTX, fed, GR.PARAMS, GR.models(fed), GR.interpreter,
                                   ScriptedGenerator(GR.CANDIDATES))
    R["greece"] = {"gap": sorted(best.gap), "duration_days": best.dur,
                   "plan": {v: {"host": p.host, "toolkit": p.K, "Q": round(p.Q, 3), "load": p.load}
                            for v, p in best.plan.items()},
                   "ledger": info["ledger"]}
    # Synthetic federation, simulated agent
    fed, true_models = synth.generate(synth.SynthConfig(seed=7, n_procedures=16, sharing_density=0.75))
    agent = synth.SimulatedAgent(true_models["hP"], seed=1)
    sel = lambda specs: [agent.select(s.K, s.r, s.v) for s in specs]
    e1 = X.e1(fed, true_models["hP"].featurizer, sel, n_obs=6000)
    fitted = e1.pop("models")["mnl"]
    R["E1"] = e1
    params = Params(k_max=4, N=10, n_min=2)
    R["E3"] = {k: v for k, v in X.e3(fed, {"hP": fitted, "hO": fitted}, sel, params, n_rep=10).items() if k != "rows"}
    R["E4"] = X.e4(reps=30)
    R["E5"] = {}
    for dens in (0.5, 0.75, 0.9):           # the three sharing densities of the protocol
        fed5, m5 = synth.generate(synth.SynthConfig(seed=11, n_procedures=14, sharing_density=dens))
        R["E5"][f"density_{dens}"] = X.e5(fed5, m5, Params(k_max=4, N=8, n_min=1, rep_max=2), n_req=12)
    R["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(R, open(out, "w"), indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps(R, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
