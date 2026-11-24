"""Study S4: scalability. (a) exact joint composition: solve time vs number of candidate tools N and number of
tasks |V| on synthetic workflows (time limit 10 s; a run that reaches it returns its best plan);
(b) forward chaining: time vs catalogue size. No API call.   python scripts/study_scalability.py milp SEED | fc"""
import json, os, random, sys, time
sys.path.insert(0, ".")
import numpy as np
from selectflow import synth
from selectflow.compose import compose
from selectflow.derivation import forward_chain
from selectflow.model import Context, Descriptor, Params, Workflow

OUT = "results/studies/s4_scalability.json"
out = json.load(open(OUT)) if os.path.exists(OUT) else {"milp": [], "fc": []}
if sys.argv[1] == "milp":
    seed = int(sys.argv[2]); TL = 10.0
    fed, M = synth.generate(synth.SynthConfig(seed=100 + seed, n_procedures=24, caps_per_task=(2, 3), providers_per_cap=(3, 5),
                                              n_distractors=60, sharing_density=0.9, budget=6000, p_alternative=0.0))
    ids = sorted(fed.descriptors)
    for nV in (int(a) for a in sys.argv[3].split(",")):
        for N in (4, 6, 8, 10, 15):
            p = Params(k_max=5, N=N, n_min=2, t_max=TL)
            t0 = time.time(); r = compose(fed, Workflow(ids[:nV], {}), set(), M, p, Context(set())); dt = time.time() - t0
            t1 = time.time(); rt = compose(fed, Workflow(ids[:nV], {}), set(), M, p, Context(set()), task_by_task=True); dt2 = time.time() - t1
            q = lambda res: sum(x.Q for x in res.plan.values()) if res.plan else None
            out["milp"].append({"seed": seed, "tasks": nV, "N": N, "time_s": dt, "time_limit_reached": dt >= 0.95 * TL,
                                "plan": bool(r.plan), "Q_joint": q(r), "time_decomposed_s": dt2, "Q_decomposed": q(rt)})
            print(nV, N, round(dt, 2), round(dt2, 2), q(r), q(rt), flush=True)
else:
    rng = random.Random(0)
    for n in (100, 300, 1000, 3000, 6000):
        for rep in range(3):
            D = {}
            for i in range(n):
                In = {f"a{rng.randrange(i)}"} if i and rng.random() < 0.7 else set()
                D[f"d{i}"] = Descriptor(f"d{i}", f"O{i % 40}", "p", In, {f"a{i}"}, {"r"})
            G = {f"a{n - 1 - rng.randrange(max(1, n // 10))}"}
            t0 = time.time(); order, U = forward_chain(D, G, set(), 0.5); dt = time.time() - t0
            out["fc"].append({"catalogue": n, "rep": rep, "time_s": dt, "found": order is not None, "tasks": len(order or [])})
        print("fc", n, round(dt, 3), flush=True)
json.dump(out, open(OUT, "w"), indent=1)
