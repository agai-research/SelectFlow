"""Synthetic federations with controlled difficulty (used by dry runs and tests).

The synthetic part of the protocol: organizations, servers, tools with
look-alikes and distractors, sharing bases at a chosen density, model hosts,
and a ground-truth tool-selection model that the simulated agent samples from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .model import (REQ, Context, Descriptor, Federation, ModelHost, Params, QoS, Server, Tool)
from .selection import Featurizer, SelectionModel

CATS = ["ID", "BUS", "LOC", "FIN", "HLT"]


@dataclass
class SynthConfig:
    n_orgs: int = 6
    n_procedures: int = 12
    caps_per_task: Tuple[int, int] = (1, 3)
    providers_per_cap: Tuple[int, int] = (1, 3)
    p_lookalike: float = 0.6
    p_alternative: float = 0.4
    n_distractors: int = 20
    sharing_density: float = 0.7
    p_free_text: float = 0.3
    p_forwarding: float = 0.15
    budget: int = 4000
    seed: int = 0


TRUE_BETA = np.array([-0.5, 2.0, 0.5, 0.5, 1.2, 0.3])
TRUE_GAMMA = 0.4


def generate(cfg: SynthConfig) -> Tuple[Federation, Dict[str, SelectionModel]]:
    rng = np.random.default_rng(cfg.seed)
    orgs = [f"O{i}" for i in range(cfg.n_orgs)]
    servers = {f"s_{o}": Server(f"s_{o}", o, int(rng.integers(50, 250))) for o in orgs}
    servers["s_X"] = Server("s_X", "ThirdParty", 100)
    D: Dict[str, Descriptor] = {}
    tools: Dict[str, Tool] = {}
    cat: Dict[str, set] = {}
    purposes = ["registration", "authorization", "notification", "payment", "verification"]
    VOC = ("permit licence register certificate notify declare request attest renew verify pay fee "
           "residence business health tax premises vehicle insurance identity address employer").split()
    phrase = lambda: " ".join(rng.choice(VOC, 3, replace=False))

    desc_of: Dict[str, str] = {}

    def mk_tool(tid, org, caps, prod=(), cons=(), sim=0.8, ad=0.9, psi=0.0, assertive=0.0, fam=None, inp=None):
        srv = f"s_{org}" if org in orgs else "s_X"
        free = rng.random() < cfg.p_free_text
        inp = set(CATS) if free else (inp or set(rng.choice(CATS, 2, replace=False)))
        ret = set(rng.choice(CATS, int(rng.integers(0, 3)), replace=False))
        t = Tool(tid, srv, org, desc_of.get(tid, f"tool {tid}"), caps, int(rng.integers(200, 700)),
                 QoS(float(rng.uniform(0.2, 5)), float(rng.uniform(0, 1)), float(rng.uniform(0.95, 1)),
                     float(rng.uniform(0.95, 1))), set(prod), set(cons), inp,
                 adequacy={r: ad for r in caps}, psi=psi,
                 attrs={"sim": sim, "assertive": assertive, "name_sim": float(rng.uniform(0, 1)), "params": float(rng.uniform(0.5, 1))},
                 true_ret=ret, true_fwd=bool(rng.random() < cfg.p_forwarding), family=fam or tid)
        tools[tid] = t
        return t

    cap_phrase: Dict[str, str] = {}

    def add_procedure(pid, o, In, Out, purpose):
        k = int(rng.integers(cfg.caps_per_task[0], cfg.caps_per_task[1] + 1))
        R = {f"c_{pid}_{j}" for j in range(k)}
        for r in R:
            cap_phrase[r] = phrase()
        D[pid] = Descriptor(pid, o, purpose, set(In), set(Out), R,
                            set(rng.choice(CATS, 2, replace=False)), fee=float(rng.integers(0, 50)),
                            dur=float(rng.integers(1, 15)), title=f"procedure {pid}",
                            text=" ".join(cap_phrase[r] for r in sorted(R)),
                            cap_text={r: cap_phrase[r] for r in R})
        for r in sorted(R):
            n_prov = int(rng.integers(cfg.providers_per_cap[0], cfg.providers_per_cap[1] + 1))
            for j in range(n_prov):
                desc_of[f"t_{r}_{j}"] = f"{cap_phrase[r]} {rng.choice(VOC)}"
                mk_tool(f"t_{r}_{j}", o, {r: float(rng.uniform(0.6, 1))}, prod=Out if j == 0 else (),
                        cons=In if j == 0 else (), sim=float(rng.uniform(0.5, 0.9)),
                        ad=float(rng.uniform(0.6, 0.95)), psi=float(rng.uniform(0, 0.2)), fam=f"f_{r}")
            if rng.random() < cfg.p_lookalike:     # a neighbouring procedure, attractive but wrong
                desc_of[f"t_{r}_look"] = f"{cap_phrase[r]} change existing"
                mk_tool(f"t_{r}_look", o, {r: float(rng.uniform(0.6, 0.9))}, sim=float(rng.uniform(0.8, 0.95)),
                        ad=0.0, psi=float(rng.uniform(0.1, 0.3)), assertive=1.0, fam=f"f_{r}")

    for i in range(cfg.n_procedures):
        o = orgs[i % cfg.n_orgs]
        In = {f"a{int(rng.integers(0, i))}"} if i > 0 and rng.random() < 0.7 else set()
        Out = {f"a{i}"}
        cat[f"a{i}"] = set(rng.choice(CATS, int(rng.integers(1, 3)), replace=False))
        add_procedure(f"p{i}", o, In, Out, purposes[i % len(purposes)])
        if rng.random() < cfg.p_alternative:   # another organization delivers the same artefact
            add_procedure(f"p{i}alt", orgs[(i + 1) % cfg.n_orgs], In, Out, purposes[i % len(purposes)])
    for j in range(cfg.n_distractors):
        all_caps = sorted({r for d in D.values() for r in d.R})
        r = all_caps[int(rng.integers(0, len(all_caps)))]
        desc_of[f"t_dis_{j}"] = f"{cap_phrase[r].split()[0]} {phrase()}"
        org = orgs[int(rng.integers(0, cfg.n_orgs))] if rng.random() < 0.7 else "ThirdParty"
        mk_tool(f"t_dis_{j}", org, {r: float(rng.uniform(0.3, 0.7))}, sim=float(rng.uniform(0.2, 0.6)),
                ad=float(rng.uniform(0, 0.4)), fam=f"dis_{j}")

    hosts = {"hP": ModelHost("hP", "Contractor", cfg.budget), "hO": ModelHost("hO", orgs[0], cfg.budget)}
    sharing = set()
    for p in purposes:
        for a in CATS:
            for o in [REQ] + orgs + ["ThirdParty"]:
                for o2 in orgs + ["ThirdParty", "Contractor"]:
                    if o != o2 and rng.random() < cfg.sharing_density:
                        sharing.add((p, a, o, o2))
    fed = Federation(D, tools, servers, hosts, sharing, cat, set(CATS))
    fed.add_competence_bases()
    fed.certified = {"fwd": {t.id for t in tools.values() if not t.true_fwd and t.org in orgs}}
    for a in CATS:
        fed.certified[f"ret:{a}"] = {t.id for t in tools.values() if a not in t.true_ret and t.org in orgs}
    feat = Featurizer(fed)
    m = SelectionModel(TRUE_BETA.copy(), TRUE_GAMMA, feat)
    return fed, {"hP": m, "hO": m}


class SimulatedAgent:
    """Samples selections from a ground-truth MNL. Stands in for the LLM in dry runs.

    Its output has the same form as the Claude probe harness: a tool id or None."""

    def __init__(self, model: SelectionModel, seed: int = 0):
        self.model, self.rng = model, np.random.default_rng(seed)

    def select(self, K, r, v):
        p, p0 = self.model.probs(list(K), r, v)
        keys = list(p) + [None]
        probs = np.array([p[k] for k in p] + [p0])
        return keys[int(self.rng.choice(len(keys), p=probs / probs.sum()))]
