"""Core abstractions of SelectFlow (Section 3 of the paper).

Symbols follow Table 3 of the paper: tools t, servers s, model hosts h,
service descriptors d, workflows W = (V, E, partial), toolkits K_v,
sharing bases S.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

REQ = "REQ"   # the requester (bottom in the paper)
EXT = "EXT"   # any party outside the federation
PUB = "PUB"   # the public web


@dataclass(frozen=True)
class QoS:
    lat: float          # latency (seconds)
    pr: float           # price per call
    av: float = 1.0     # availability in (0, 1]
    rel: float = 1.0    # reliability in (0, 1]


@dataclass
class Server:
    id: str
    org: str
    preamble: int = 0   # c^0_s, tokens


@dataclass
class Tool:
    """An MCP tool (Definition 1)."""
    id: str
    server: str
    org: str
    desc: str
    caps: Dict[str, float]                 # capability r -> mapping confidence mu_tr
    cost: int                              # c_t, tokens of the definition
    qos: QoS = QoS(1.0, 0.0)
    prod: Set[str] = field(default_factory=set)   # artefact types produced
    cons: Set[str] = field(default_factory=set)   # artefact types consumed
    inp: Set[str] = field(default_factory=set)    # data categories its inputs can carry
    input_schema: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    adequacy: Dict[str, float] = field(default_factory=dict)  # ad_tr
    psi: float = 0.0                       # cost of a call that does not deliver r
    attrs: Dict[str, float] = field(default_factory=dict)     # free features (e.g. assertive cue)
    # Ground-truth behaviours: read only by simulations and audits.
    true_ret: Set[str] = field(default_factory=set)
    true_fwd: bool = False
    family: Optional[str] = None           # tool family, used for group splits


@dataclass
class ModelHost:
    id: str
    org: str
    budget: int                            # B_h, tokens reserved for tool definitions


@dataclass
class Descriptor:
    """A service descriptor (Definition 2)."""
    id: str
    org: str
    purpose: str
    In: Set[str]
    Out: Set[str]
    R: Set[str]                            # required capabilities
    A: Set[str] = field(default_factory=set)      # categories elicited from the requester
    comp: Set[str] = field(default_factory=set)   # mandatory companions (descriptor ids)
    fee: float = 0.0
    dur: float = 0.0                       # days
    chi: Dict[str, float] = field(default_factory=dict)  # alignment confidence per artefact term
    title: str = ""
    text: str = ""                         # free text used for retrieval
    jurisdiction: Optional[str] = None     # competence filter
    cap_text: Dict[str, str] = field(default_factory=dict)  # statement of each capability
    record: Optional[str] = None           # registry record (provenance)
    orgs: Set[str] = field(default_factory=set)  # other organizations involved (multi-server tasks)

    def chi_of(self, a: str) -> float:
        return self.chi.get(a, 1.0)


@dataclass
class Workflow:
    """W = (V, E, partial): tasks are identified by their descriptor ids."""
    order: List[str]
    ho: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)

    @property
    def V(self) -> List[str]:
        return list(self.order)

    @property
    def E(self) -> List[Tuple[str, str]]:
        return list(self.ho.keys())

    def preds(self, v: str) -> List[str]:
        return [u for (u, w) in self.ho if w == v]


@dataclass
class Federation:
    """Everything the composer reads: D, C, S, H and the vocabularies."""
    descriptors: Dict[str, Descriptor]
    tools: Dict[str, Tool]
    servers: Dict[str, Server]
    hosts: Dict[str, ModelHost]
    sharing: Set[Tuple[str, str, str, str]]      # (purpose, category, origin, receiver)
    cat: Dict[str, Set[str]]                      # artefact type -> data categories
    categories: Set[str]
    certified: Dict[str, Set[str]] = field(default_factory=dict)  # label -> tool ids certified safe

    def load(self, K: Iterable[str]) -> int:
        K = list(K)
        srv = {self.tools[t].server for t in K}
        return sum(self.servers[s].preamble for s in srv) + sum(self.tools[t].cost for t in K)

    def add_competence_bases(self) -> None:
        """Competence: an organization may receive from the requester what its descriptors elicit."""
        for d in self.descriptors.values():
            for a in d.A:
                self.sharing.add((d.purpose, a, REQ, d.org))


@dataclass
class Context:
    """Requester context: artefacts held, QoS bounds, jurisdiction."""
    held: Set[str]
    lat_max: float = math.inf
    pr_max: float = math.inf
    dep_min: float = 0.0
    jurisdiction: Optional[str] = None


@dataclass
class Params:
    k_max: int = 5
    N: int = 12
    n_min: int = 2
    tau_mu: float = 0.5
    tau_chi: float = 0.6
    lam_q: float = 0.3
    qos_weights: Tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2)  # sum to one
    t_max: float = 60.0
    rep_max: int = 2
    m_retries: int = 3
    weights: Dict[str, float] = field(default_factory=lambda: {
        "label": 1.0, "budget": 2.0, "size": 2.0, "cov": 3.0, "ho": 3.0,
        "qos": 4.0, "flow": 5.0, "host": 5.0})
