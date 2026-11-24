"""Joint exact composition of model hosts and toolkits (Sect. 4.5, Eqs. 19-20, Algorithm 2)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import pulp

from . import flows
from .model import Federation, Params, Workflow
from .selection import KTOK, Quality, SelectionModel


@dataclass
class TaskPlan:
    host: str
    K: List[str]
    Q: float = 0.0
    loss: float = 0.0
    load: int = 0
    occ: Dict[str, float] = field(default_factory=dict)


@dataclass
class CompositionResult:
    plan: Optional[Dict[str, TaskPlan]]
    correction: Optional[Set[Tuple]]
    status: str
    T: List[str] = field(default_factory=list)
    candidates: Dict[str, List[str]] = field(default_factory=dict)
    blocked_weights: Dict[str, float] = field(default_factory=dict)


def _providers(fed, cands, r, tau=None):
    return [t for t in cands if r in fed.tools[t].caps and (tau is None or fed.tools[t].caps[r] >= tau)]


def candidates(fed: Federation, W: Workflow, v: str, hosts: List[str], q_by_host: Dict[str, Quality],
               params: Params) -> List[str]:
    """CANDIDATES: providers of R(v) and of the hand-off artefacts, ranked by diluted value."""
    d = fed.descriptors[v]
    ho_in = set().union(*[a for (u, w), a in W.ho.items() if w == v]) if W.ho else set()
    ho_out = set().union(*[a for (u, w), a in W.ho.items() if u == v]) if W.ho else set()
    C = [t for t, T in fed.tools.items()
         if any(T.caps.get(r, 0) >= params.tau_mu for r in d.R) or (T.prod & ho_out) or (T.cons & ho_in)]

    def score(t, r):
        return max(q_by_host[h].diluted(t, r, v) for h in hosts)

    keep: List[str] = []
    for r in sorted(d.R):
        prov = sorted(_providers(fed, C, r, params.tau_mu), key=lambda t: -score(t, r))
        keep += [t for t in prov[: params.n_min] if t not in keep]
    for a in sorted(ho_out):
        pr = [t for t in C if a in fed.tools[t].prod and t not in keep][: params.n_min]
        keep += pr
    for a in sorted(ho_in):
        cs = [t for t in C if a in fed.tools[t].cons and t not in keep][: params.n_min]
        keep += cs
    rest = sorted([t for t in C if t not in keep],
                  key=lambda t: -max([score(t, r) for r in d.R if r in fed.tools[t].caps] + [-9]))
    keep += rest[: max(0, params.N - len(keep))]
    return keep


def compose(fed: Federation, W: Workflow, gap: Iterable[str], models: Dict[str, SelectionModel],
            params: Params, ctx, task_by_task: bool = False) -> CompositionResult:
    """Algorithm 2. With task_by_task=True, the coupling constraints (hand-offs and
    workflow-level QoS) are dropped and each task is solved alone (E5 baseline)."""
    gap = set(gap)
    q_by_host = {h: Quality(fed, params, m) for h, m in models.items()}
    T = [v for v in W.order if v not in gap and fed.descriptors[v].R]
    H, N, P = {}, {}, {}
    for v in T:
        H[v] = flows.admissible_hosts(fed, W, v, gap)
        if not H[v]:
            return CompositionResult(None, {("host", v)}, "no_host", T, blocked_weights={v: params.weights["host"]})
        N[v] = candidates(fed, W, v, H[v], q_by_host, params)
        P[v] = flows.prohibitions(fed, W, v, N[v], H[v], gap)

    if task_by_task:
        plan, corr = {}, set()
        for v in T:
            sub = Workflow([v], {})
            r = _solve(fed, sub, [v], {v: H[v]}, {v: N[v]}, {v: P[v]}, q_by_host, params, ctx, gap,
                       relax=False, coupled=False)
            if r[0] is None:
                rc = _solve(fed, sub, [v], {v: H[v]}, {v: N[v]}, {v: P[v]}, q_by_host, params, ctx, gap,
                            relax=True, coupled=False)
                corr |= rc[1] or set()
            else:
                plan.update(r[0])
        if corr:
            return CompositionResult(None, corr, "infeasible", T, N)
        return CompositionResult(plan, None, "optimal", T, N)

    plan, _, status = _solve(fed, W, T, H, N, P, q_by_host, params, ctx, gap, relax=False, coupled=True)
    if plan is not None:
        return CompositionResult(plan, None, status, T, N)
    _, corr, st2 = _solve(fed, W, T, H, N, P, q_by_host, params, ctx, gap, relax=True, coupled=True)
    return CompositionResult(None, corr, "infeasible", T, N)


def _solve(fed, W, T, H, N, P, q_by_host, params, ctx, gap, relax: bool, coupled: bool):
    prob = pulp.LpProblem("selectflow", pulp.LpMinimize if relax else pulp.LpMaximize)
    BIG = 1e6
    e: Dict[Tuple, pulp.LpVariable] = {}

    def grp(key):
        if not relax:
            return 0
        if key not in e:
            e[key] = pulp.LpVariable(f"e_{len(e)}", cat="Binary")
        return e[key]

    def var(name, **kw):
        return pulp.LpVariable(name.replace(".", "_").replace("-", "_").replace(":", "_"), **kw)

    y, x, z, rho, xi, zeta = {}, {}, {}, {}, {}, {}
    obj = []
    for v in T:
        d = fed.descriptors[v]
        servers = sorted({fed.tools[t].server for t in N[v]})
        for h in H[v]:
            m = q_by_host[h].model
            y[v, h] = var(f"y_{v}_{h}", cat="Binary")
            for t in N[v]:
                x[v, h, t] = var(f"x_{v}_{h}_{t}", cat="Binary")
                prob += x[v, h, t] <= y[v, h]
            for s in servers:
                z[v, h, s] = var(f"z_{v}_{h}_{s}", cat="Binary")
                ts = [t for t in N[v] if fed.tools[t].server == s]
                for t in ts:
                    prob += x[v, h, t] <= z[v, h, s]
                prob += z[v, h, s] <= pulp.lpSum(x[v, h, t] for t in ts)
            RMAX = 1.0 / max(1e-3, 1.0 + min(0.0, m.gamma) * fed.hosts[h].budget / KTOK)   # bound of rho
            for r in sorted(d.R):
                rho[v, h, r] = var(f"rho_{v}_{h}_{r}", lowBound=0, upBound=RMAX)
                norm = [rho[v, h, r]]
                for t in N[v]:
                    a = m.attraction(t, r, v)
                    xi[v, h, r, t] = var(f"xi_{v}_{h}_{r}_{t}", lowBound=0, upBound=RMAX)
                    prob += xi[v, h, r, t] <= RMAX * x[v, h, t]
                    prob += xi[v, h, r, t] <= rho[v, h, r]
                    prob += xi[v, h, r, t] >= rho[v, h, r] - RMAX * (1 - x[v, h, t])
                    norm.append((a + m.gamma * fed.tools[t].cost / KTOK) * xi[v, h, r, t])
                    obj.append(a * q_by_host[h].w(t, r) * xi[v, h, r, t])
                for s in servers:
                    zeta[v, h, r, s] = var(f"zeta_{v}_{h}_{r}_{s}", lowBound=0, upBound=RMAX)
                    prob += zeta[v, h, r, s] <= RMAX * z[v, h, s]
                    prob += zeta[v, h, r, s] <= rho[v, h, r]
                    prob += zeta[v, h, r, s] >= rho[v, h, r] - RMAX * (1 - z[v, h, s])
                    norm.append(m.gamma * fed.servers[s].preamble / KTOK * zeta[v, h, r, s])
                prob += pulp.lpSum(norm) == 1
            # context budget
            prob += (pulp.lpSum(fed.servers[s].preamble * z[v, h, s] for s in servers)
                     + pulp.lpSum(fed.tools[t].cost * x[v, h, t] for t in N[v])
                     <= fed.hosts[h].budget + BIG * grp(("budget", v, h)))
            # forbidden tools under h
            for t, causes in {**P[v].F, **P[v].F_h.get(h, {})}.items():
                if t in N[v]:
                    gs = set().union(*causes)
                    if relax:
                        for g in gs:
                            prob += x[v, h, t] <= grp(g)
                    else:
                        prob += x[v, h, t] == 0
        prob += pulp.lpSum(y[v, h] for h in H[v]) == 1
        allx = lambda ts: pulp.lpSum(x[v, h, t] for h in H[v] for t in ts)
        for r in sorted(d.R):
            prob += allx(_providers(fed, N[v], r, params.tau_mu)) >= 1 - grp(("cov", v, r))
        prob += allx(N[v]) <= params.k_max + BIG * grp(("size", v))
        for pair, causes in P[v].F2.items():
            t1, t2 = tuple(pair)
            if t1 in N[v] and t2 in N[v]:
                gs = set().union(*causes)
                lhs = pulp.lpSum(x[v, h, t1] + x[v, h, t2] for h in H[v])
                if relax:
                    for g in gs:
                        prob += lhs <= 1 + grp(g)
                else:
                    prob += lhs <= 1

    if coupled:
        Tset = set(T)
        for (u, v), arts in W.ho.items():
            if v not in Tset:
                continue
            for a in sorted(arts):
                g = grp(("ho", u, v, a))
                if u in Tset:   # producer side only when the supplier is equipped
                    prob += pulp.lpSum(x[u, h, t] for h in H[u] for t in N[u] if a in fed.tools[t].prod) >= 1 - g
                prob += pulp.lpSum(x[v, h, t] for h in H[v] for t in N[v] if a in fed.tools[t].cons) >= 1 - g

    # robust QoS along the critical path (Eq. 7)
    if coupled or len(T) == 1:
        blat, bpr, bdep, f = {}, {}, {}, {}
        for v in T:
            for r in sorted(fed.descriptors[v].R):
                blat[v, r] = var(f"blat_{v}_{r}", lowBound=0)
                bpr[v, r] = var(f"bpr_{v}_{r}", lowBound=0)
                bdep[v, r] = var(f"bdep_{v}_{r}", lowBound=0)
                for h in H[v]:
                    for t in _providers(fed, N[v], r):
                        q = fed.tools[t].qos
                        prob += blat[v, r] >= q.lat * x[v, h, t]
                        prob += bpr[v, r] >= q.pr * x[v, h, t]
                        prob += bdep[v, r] >= -math.log(max(q.av * q.rel, 1e-9)) * x[v, h, t]
        gq = grp(("qos",))
        for v in W.order:
            f[v] = var(f"f_{v}", lowBound=0)
            own = pulp.lpSum(blat[v, r] for r in fed.descriptors[v].R if (v, r) in blat)
            prob += f[v] >= own
            for u in W.preds(v):
                prob += f[v] >= f[u] + own
            if math.isfinite(ctx.lat_max):
                prob += f[v] <= ctx.lat_max + BIG * gq
        if math.isfinite(ctx.pr_max):
            prob += pulp.lpSum(bpr.values()) <= ctx.pr_max + BIG * gq
        if ctx.dep_min > 0:
            prob += pulp.lpSum(bdep.values()) <= -math.log(ctx.dep_min) + BIG * gq

    if relax:
        prob += pulp.lpSum(params.weights[k[0]] * var_ for k, var_ in e.items())
    else:
        prob += pulp.lpSum(obj)
    solver = pulp.HiGHS(msg=False, timeLimit=params.t_max)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if relax:
        corr = {k for k, var_ in e.items() if var_.value() is not None and var_.value() > 0.5}
        return None, corr, status
    if status not in ("Optimal",):
        return None, None, status
    plan = {}
    for v in T:
        h = max(H[v], key=lambda h_: y[v, h_].value())
        K = [t for t in N[v] if x[v, h, t].value() > 0.5]
        qh = q_by_host[h]
        plan[v] = TaskPlan(h, K, qh.Q(K, v), qh.selection_loss(K, v), fed.load(K),
                           {r: qh.occasions(K, r, v) for r in fed.descriptors[v].R})
    return plan, None, "optimal"


def correction_by_task(corr: Set[Tuple], weights: Dict[str, float], fed: Federation) -> Dict[str, float]:
    """Weight of the correction attributable to each task (used to find the blocked task)."""
    out: Dict[str, float] = {}
    for g in corr:
        kind = g[0]
        if kind in ("cov", "budget", "size", "flow", "host"):
            v = g[1]
        elif kind == "ho":
            v = g[2]
        elif kind == "label":
            continue
        else:
            continue
        out[v] = out.get(v, 0.0) + weights[kind]
    return out
