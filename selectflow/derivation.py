"""Verified workflow derivation (Sect. 4.2, Algorithm 1, Proposition 1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

from .model import Context, Descriptor, Federation, Params, Workflow


# ---------------------------------------------------------------- critics
@dataclass
class Report:
    missing: Dict[str, Set[str]]          # task -> artefacts not available
    weak: List[Tuple[str, str, str]]      # (supplier, consumer, artefact) below tau_chi
    targets: Set[str]                     # targets not delivered
    companions: List[Tuple[str, str]]     # (task, missing companion)
    superfluous: List[str]

    def empty(self) -> bool:
        return not (self.missing or self.weak or self.targets or self.companions or self.superfluous)

    def missing_artefacts(self) -> Set[str]:
        s = set(self.targets)
        for v in self.missing.values():
            s |= v
        return s


def check(order: Sequence[str], G: Set[str], held: Set[str], D: Dict[str, Descriptor],
          tau_chi: float) -> Tuple[Report, Workflow]:
    """The four deterministic critics: closure, alignment, companions, minimality."""
    rep = Report({}, [], set(), [], [])
    ho: Dict[Tuple[str, str], Set[str]] = {}
    av = set(held); sup: Dict[str, str] = {}
    for v in order:
        d = D[v]
        miss = d.In - av
        if miss:
            rep.missing[v] = set(miss)
        for a in (d.In & av) - held:
            u = sup[a]
            ho.setdefault((u, v), set()).add(a)
            if min(D[u].chi_of(a), d.chi_of(a)) < tau_chi:
                rep.weak.append((u, v, a))
        for a in d.Out:
            sup.setdefault(a, v)      # the first producer supplies (consistent with ForwardChain)
        av |= d.Out
    rep.targets = set(G) - av
    tasks = set(order)
    for v in order:
        for c in D[v].comp - tasks:
            rep.companions.append((v, c))
    comps = set().union(*[D[v].comp for v in order]) if order else set()
    used = set().union(*ho.values()) if ho else set()
    for v in order:
        if v in comps:
            continue
        if not (D[v].Out & (set(G) | {a for (u, w), arts in ho.items() if u == v for a in arts})):
            rep.superfluous.append(v)
    return rep, Workflow(list(order), ho)


# ---------------------------------------------------------------- forward chaining
def forward_chain(D: Dict[str, Descriptor], G: Set[str], held: Set[str], tau_chi: float
                  ) -> Tuple[Optional[List[str]], Set[str]]:
    """Symbolic completion: returns (task order, unmet artefacts)."""
    D = dict(D)
    while True:
        av = set(held); fired: List[str] = []; first: Dict[str, str] = {}
        progress = True
        while progress:
            progress = False
            for k, d in D.items():
                if k in fired or not d.In <= av:
                    continue
                if any(min(D[first[a]].chi_of(a), d.chi_of(a)) < tau_chi for a in d.In - held):
                    continue
                fired.append(k)
                out_plus = {a for a in d.Out if d.chi_of(a) >= tau_chi}
                for a in out_plus - av:
                    first[a] = k
                av |= out_plus
                progress = True
        bad = [k for k in fired if not D[k].comp <= set(fired)]
        if not bad:
            break
        for k in bad:
            del D[k]
    if not set(G) <= av:
        return None, set(G) - av

    def comp_star(k, seen=None):
        seen = seen or set()
        for c in D[k].comp:
            if c not in seen:
                seen.add(c); comp_star(c, seen)
        return seen

    S: Set[str] = set(); need = set(G) - held; seen: Set[str] = set()
    while need - seen:
        a = sorted(need - seen)[0]; seen.add(a)
        new = ({first[a]} | comp_star(first[a])) - S
        S |= new
        for k in new:
            need |= D[k].In - held
    return [k for k in fired if k in S], set()


# ---------------------------------------------------------------- generation interfaces
class Interpreter(Protocol):
    def __call__(self, q: str, ctx: Context) -> Tuple[Set[str], Set[str]]: ...


class Generator(Protocol):
    def generate(self, q: str, G: Set[str], held: Set[str], Dq: Dict[str, Descriptor], k: int) -> List[List[str]]: ...
    def revise(self, order: List[str], rep: Report, Dq: Dict[str, Descriptor]) -> List[str]: ...


def retrieve(D: Dict[str, Descriptor], q: str, G: Set[str], n: int) -> Dict[str, Descriptor]:
    """Keep the n descriptors closest to the request and targets (token overlap)."""
    qt = set((q + " " + " ".join(G)).lower().replace("-", " ").split())
    def score(d):
        txt = set((d.title + " " + d.text + " " + " ".join(d.Out)).lower().replace("-", " ").split())
        return len(qt & txt) + 3 * len(d.Out & G)
    keys = sorted(D, key=lambda k: -score(D[k]))[:n]
    return {k: D[k] for k in keys}


@dataclass
class DerivationResult:
    workflows: List[Workflow]
    unmet: Set[str]
    G: Set[str]
    held: Set[str]
    log: dict


def derive(q: str, ctx: Context, fed: Federation, params: Params, interpreter: Interpreter,
           generator: Optional[Generator], X: Iterable[str] = (), confirm: Optional[Callable] = None,
           G_held: Optional[Tuple[Set[str], Set[str]]] = None, k: int = 3, n_ret: int = 30,
           n_rev: int = 2) -> DerivationResult:
    """Algorithm 1. `G_held` skips interpretation (used by repair: no new confirmation)."""
    if G_held is None:
        G, held = interpreter(q, ctx)
        if confirm is not None:
            G, held = confirm(G, held)
    else:
        G, held = G_held
    held = set(held) | set(ctx.held)
    X = set(X)
    D = {k_: d for k_, d in fed.descriptors.items()
         if k_ not in X and (d.jurisdiction is None or d.jurisdiction == ctx.jurisdiction)}
    Dq = retrieve(D, q, G, n_ret)
    W: List[Workflow] = []
    log = {"candidates": 0, "rejected_by_critics": 0, "revisions": 0, "accepted": 0, "accepted_after_revision": 0,
           "llm_orders": []}
    if generator is not None:
        for cand in generator.generate(q, G, held, Dq, k):
            log["candidates"] += 1
            log["llm_orders"].append(list(cand))
            cand = [c for c in cand if c in Dq]         # grammar: only retrieved identifiers
            for i in range(n_rev + 1):
                rep, wf = check(cand, G, held, Dq, params.tau_chi)
                if rep.empty():
                    W.append(wf); log["accepted"] += 1; log["accepted_after_revision"] += int(i > 0); break
                if i == 0:
                    log["rejected_by_critics"] += 1
                if i == n_rev:
                    break
                log["revisions"] += 1
                miss = rep.missing_artefacts()
                Dq.update({k_: d for k_, d in D.items() if d.Out & miss})
                cand = [c for c in generator.revise(cand, rep, Dq) if c in Dq]
    order, U = forward_chain(D, G, held, params.tau_chi)
    if order is not None:
        rep, wf = check(order, G, held, D, params.tau_chi)
        if rep.empty():
            W.append(wf)
    uniq, seen = [], set()
    for w in W:
        key = tuple(sorted(w.order))
        if key not in seen:
            seen.add(key); uniq.append(w)
    return DerivationResult(uniq, U, set(G), held, log)


# ---------------------------------------------------------------- scripted generator (tests, dry runs)
class ScriptedGenerator:
    """Returns fixed candidates; revision adds the producers of missing artefacts.

    Stands in for the LLM in tests. The LLM generator is in agent/claude_generator.py."""

    def __init__(self, candidates: List[List[str]]):
        self.candidates = candidates

    def generate(self, q, G, held, Dq, k):
        return [list(c) for c in self.candidates[:k]]

    def revise(self, order, rep, Dq):
        out = list(order)
        for a in sorted(rep.missing_artefacts()):
            prod = [k for k, d in Dq.items() if a in d.Out and k not in out]
            if prod:
                # insert the producer before its first consumer
                cons = [i for i, v in enumerate(out) if a in Dq[v].In]
                out.insert(cons[0] if cons else len(out), prod[0])
        return out
