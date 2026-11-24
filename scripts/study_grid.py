"""Study S5b: density of sharing bases x context budget, joint composition with repair on the 82 real requests
(data version 1). No API call.   python scripts/study_grid.py DENS1,DENS2,..."""
import json, os, sys, time
sys.path.insert(0, ".")
import numpy as np
sys.argv_saved = list(sys.argv); sys.argv = [sys.argv[0]]
sys.path.insert(0, "scripts")
from real_composition import federations, models_for, PARAMS
from selectflow.experiments.runners import broken_handoffs
from selectflow.model import Context
from selectflow.repair import solve_request
OUT = "results/studies/s5_grid.json"
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
for dens in (float(x) for x in sys.argv_saved[1].split(",")):
    for budget in (260, 320, 400, 6000):
        key = f"{dens}|{budget}"
        if key in out: continue
        eq, gap, sizes = [], [], []
        for name, fed, reqs in federations(dens, budget):
            M = models_for(fed)
            for rq in reqs:
                G, H = set(rq["targets"]), set(rq["held"])
                nd, best, info = solve_request(rq["text"], Context(H), fed, PARAMS, M, (lambda G_, H_: (lambda q, c: (G_, H_)))(G, H), None)
                if best is None: continue
                nT = len([v for v in best.W.order if fed.descriptors[v].R])
                eq.append((nT - len(best.gap)) / nT); gap.append(len(best.gap) / nT)
                sizes += [len(p.K) for p in best.plan.values()]
        out[key] = {"density": dens, "budget": budget, "equipped": float(np.mean(eq)), "left": float(np.mean(gap)), "toolkit_size": float(np.mean(sizes)) if sizes else 0.0}
        print(key, round(out[key]["equipped"], 3), round(out[key]["toolkit_size"], 2), flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
