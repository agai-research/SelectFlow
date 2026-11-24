"""Correction-driven repair and choice among non-dominated solutions (Sect. 4.6, Algorithm 3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .compose import CompositionResult, compose, correction_by_task
from .derivation import derive
from .model import Context, Federation, Params, Workflow
from .selection import SelectionModel


@dataclass
class Solution:
    W: Workflow
    plan: Dict
    gap: Set[str]
    Qbar: float
    n_gap: int
    fee: float
    dur: float
    repairs: int = 0
    log: List[dict] = field(default_factory=list)

    def objectives(self):
        return (-self.Qbar, self.n_gap, self.fee, self.dur)   # all to be minimized


def critical_path(W: Workflow, dur: Dict[str, float]) -> float:
    f: Dict[str, float] = {}
    for v in W.order:                      # order is topological by construction
        f[v] = dur[v] + max([f[u] for u in W.preds(v)] + [0.0])
    return max(f.values()) if f else 0.0


def evaluate(fed: Federation, W: Workflow, plan: Dict, gap: Set[str]) -> Tuple[float, float, float]:
    nW = sum(len(fed.descriptors[v].R) for v in W.order)
    Qbar = sum(p.Q for v, p in plan.items() if v not in gap) / nW if nW else 0.0
    fee = sum(fed.descriptors[v].fee for v in W.order)
    dur = critical_path(W, {v: fed.descriptors[v].dur for v in W.order})
    return Qbar, fee, dur


def _qos_task(fed: Federation, res: CompositionResult) -> Optional[str]:
    best, arg = -1.0, None
    for v, cands in res.candidates.items():
        lat = sum(max([fed.tools[t].qos.lat for t in cands if r in fed.tools[t].caps] + [0.0])
                  for r in fed.descriptors[v].R)
        if lat > best:
            best, arg = lat, v
    return arg


def solve_request(q: str, ctx: Context, fed: Federation, params: Params, models: Dict[str, SelectionModel],
                  interpreter, generator, confirm=None, repair: bool = True,
                  task_by_task: bool = False) -> Tuple[List[Solution], Optional[Solution], dict]:
    """Algorithm 3. Returns (non-dominated solutions, default choice, ledger)."""
    der = derive(q, ctx, fed, params, interpreter, generator, confirm=confirm)
    ledger: List[dict] = []
    sols: List[Solution] = []
    for W0 in der.workflows:
        W, X, gap, nrep = W0, set(), set(), 0
        while True:
            res = compose(fed, W, gap, models, params, ctx, task_by_task=task_by_task)
            if res.plan is not None:
                break
            corr = res.correction or set()
            weights = correction_by_task(corr, params.weights, fed)
            if ("qos",) in corr:
                vq = _qos_task(fed, res)
                if vq:
                    weights[vq] = weights.get(vq, 0.0) + params.weights["qos"]
            cand = {v: w for v, w in weights.items() if v in W.order and v not in gap}
            if not cand:   # nothing attributable: leave every equipped task to the requester
                gap |= set(res.T); continue
            v = max(cand, key=cand.get)
            ledger.append({"request": q, "descriptor": v, "correction": sorted(map(str, corr))})
            Wn = []
            if repair and nrep < params.rep_max:
                Wn = derive(q, ctx, fed, params, interpreter, generator, X=X | {v},
                            G_held=(der.G, der.held)).workflows
            if Wn:
                D_gap = set(gap)
                X |= {v}; nrep += 1
                W = max(Wn, key=lambda w: len(set(w.order) & set(W.order)))
                gap = {u for u in W.order if u in D_gap}
            else:
                gap |= {v}
        Qbar, fee, dur = evaluate(fed, W, res.plan, gap)
        sols.append(Solution(W, res.plan, set(gap), Qbar, len(gap), fee, dur, nrep))
    nd = [s for s in sols if not any(_dominates(o, s) for o in sols if o is not s)]
    default = min(nd, key=lambda s: (s.n_gap, -s.Qbar)) if nd else None
    return nd, default, {"ledger": ledger, "unmet": der.unmet, "derivation": der.log}


def _dominates(a: Solution, b: Solution) -> bool:
    oa, ob = a.objectives(), b.objectives()
    return all(x <= y for x, y in zip(oa, ob)) and any(x < y for x, y in zip(oa, ob))
