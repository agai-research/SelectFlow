"""Cross-organizational flow admissibility (Sect. 4.4, Eqs. 13-18).

All functions use the pessimistic labels of Sect. 4.1: a tool that is not
certified safe for a label is treated as unsafe for it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from .model import EXT, PUB, REQ, Federation, Workflow

Flow = Tuple[str, str, str]          # (category, origin, receiver)
Cause = FrozenSet[Tuple]             # set of group keys that jointly cause a prohibition


def pess_fwd(fed: Federation, t: str) -> bool:
    return t not in fed.certified.get("fwd", set())


def pess_ret(fed: Federation, t: str) -> Set[str]:
    return {a for a in fed.categories if t not in fed.certified.get(f"ret:{a}", set())}


def dest(fed: Federation, t: str) -> Set[str]:
    o = fed.tools[t].org
    return {o, EXT, PUB} if pess_fwd(fed, t) else {o}


def denied(fed: Federation, purpose: str, a: str, o: str, o2: str) -> bool:
    return o2 != o and (purpose, a, o, o2) not in fed.sharing


def pres0(fed: Federation, W: Workflow, v: str, gap: Iterable[str] = ()) -> Set[Tuple[str, str]]:
    """Category-origin pairs present in the context of v whatever the toolkit (Eq. 13)."""
    gap = set(gap)
    d = fed.descriptors[v]
    out = {(a, REQ) for a in d.A}
    for (u, w), arts in W.ho.items():
        if w != v:
            continue
        origin = REQ if u in gap else fed.descriptors[u].org
        for art in arts:
            for a in fed.cat.get(art, set()):
                out.add((a, origin))
    return out


def own(fed: Federation, t: str) -> Set[Tuple[str, str]]:
    return {(a, fed.tools[t].org) for a in pess_ret(fed, t)}


def admissible_hosts(fed: Federation, W: Workflow, v: str, gap=()) -> List[str]:
    p = fed.descriptors[v].purpose
    P0 = pres0(fed, W, v, gap)
    return [h for h, H in fed.hosts.items() if not any(denied(fed, p, a, o, H.org) for (a, o) in P0)]


@dataclass
class Prohibitions:
    """Forbidden tools and pairs of one task, with their causes (group keys)."""
    F: Dict[str, List[Cause]] = field(default_factory=dict)                  # F_v
    F_h: Dict[str, Dict[str, List[Cause]]] = field(default_factory=dict)     # extra under host h
    F2: Dict[FrozenSet[str], List[Cause]] = field(default_factory=dict)      # forbidden pairs

    def forbidden_under(self, h: str) -> Set[str]:
        return set(self.F) | set(self.F_h.get(h, {}))


def _label_groups(fed, t, a=None, via_fwd=False, via_ret=False):
    g = []
    if via_fwd and pess_fwd(fed, t):
        g.append(("label", t, "fwd"))
    if via_ret and a is not None and t not in fed.certified.get(f"ret:{a}", set()):
        g.append(("label", t, f"ret:{a}"))
    return g


def prohibitions(fed: Federation, W: Workflow, v: str, cands: Iterable[str], hosts: Iterable[str],
                 gap=()) -> Prohibitions:
    """F_v, F_vh and F^2_v (Eqs. 16-18), each with the groups that cause it."""
    p = fed.descriptors[v].purpose
    P0 = pres0(fed, W, v, gap)
    cands = list(cands)
    res = Prohibitions()

    def add(dct, key, cause):
        dct.setdefault(key, []).append(frozenset(cause))

    for t in cands:
        T = fed.tools[t]
        D = dest(fed, t)
        # (alpha, o) present whatever the toolkit, or returned by t itself
        sources = [(a, o, False) for (a, o) in P0] + [(a, o, True) for (a, o) in own(fed, t)]
        for (a, o, is_own) in sources:
            if a not in T.inp:
                continue
            for o2 in D:
                if denied(fed, p, a, o, o2):
                    cause = [("flow", v, a, o, o2)]
                    cause += _label_groups(fed, t, a, via_fwd=o2 in (EXT, PUB) and o2 != T.org,
                                           via_ret=is_own)
                    add(res.F, t, cause)
        for h in hosts:
            oh = fed.hosts[h].org
            for (a, o) in own(fed, t):
                if denied(fed, p, a, o, oh):
                    add(res.F_h.setdefault(h, {}), t, [("flow", v, a, o, oh)] + _label_groups(fed, t, a, via_ret=True))
    for t in cands:
        for t2 in cands:
            if t == t2:
                continue
            T, T2 = fed.tools[t], fed.tools[t2]
            for a in pess_ret(fed, t) & T2.inp:
                for o2 in dest(fed, t2):
                    if denied(fed, p, a, T.org, o2):
                        cause = [("flow", v, a, T.org, o2)] + _label_groups(fed, t, a, via_ret=True)
                        cause += _label_groups(fed, t2, via_fwd=o2 in (EXT, PUB) and o2 != T2.org)
                        add(res.F2, frozenset((t, t2)), cause)
    return res


def reach(fed: Federation, W: Workflow, v: str, K: Iterable[str], h: str, gap=(),
          true_labels: bool = False) -> Set[Flow]:
    """Flows that host h and toolkit K make possible (Eq. 15).

    With true_labels=True, the ground-truth behaviours are used (for evaluation)."""
    K = list(K)
    P = set(pres0(fed, W, v, gap))
    for t in K:
        rets = fed.tools[t].true_ret if true_labels else pess_ret(fed, t)
        P |= {(a, fed.tools[t].org) for a in rets}
    out = set()
    oh = fed.hosts[h].org
    for (a, o) in P:
        if oh != o:
            out.add((a, o, oh))
        for t in K:
            T = fed.tools[t]
            if a not in T.inp:
                continue
            D = ({T.org, EXT, PUB} if T.true_fwd else {T.org}) if true_labels else dest(fed, t)
            for o2 in D:
                if o2 != o:
                    out.add((a, o, o2))
    return out


def risk(fed: Federation, W: Workflow, v: str, K, h: str, gap=(), true_labels=False) -> int:
    p = fed.descriptors[v].purpose
    return sum(1 for (a, o, o2) in reach(fed, W, v, K, h, gap, true_labels) if denied(fed, p, a, o, o2))
