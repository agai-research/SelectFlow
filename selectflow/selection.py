"""Tool-selection model (Assumption 2, Eqs. 1-4, 11-12) and toolkit quality.

The probability that the agent selects tool t for capability r within K is
    Pr_r(t | K) = a_tr / (1 + gamma * L(K) + sum_{t' in K} a_t'r),
with ln a_tr = beta^T phi(t, r, v) and L(K) the context load in kilotokens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .model import Federation, Params, Tool

KTOK = 1000.0  # loads are expressed in kilotokens


# ---------------------------------------------------------------- features
class Featurizer:
    """phi(t, r, v). The default features follow Sect. 4.3 of the paper.

    sim:     similarity between the tool description and the statement of r
    name:    similarity between the tool name and r
    params:  share of required parameters the task can supply
    assert:  assertive cue in the description (0/1)
    own_org: the tool belongs to the organization of the task (0/1)
    The constant feature lets the model learn the level of attraction
    relative to the no-call option.
    """
    names = ("const", "sim", "name", "params", "assert", "own_org")

    def __init__(self, fed: Federation, sim_fn: Optional[Callable[[str, str], float]] = None):
        self.fed = fed
        self.sim_fn = sim_fn or _token_overlap

    def __call__(self, t: str, r: str, v: Optional[str]) -> np.ndarray:
        tool = self.fed.tools[t]
        d = self.fed.descriptors.get(v) if v else None
        stmt = (d.cap_text.get(r, r) if d else r)
        return np.array([
            1.0,
            (tool.attrs["sim"] if ("sim" in tool.attrs and r in tool.caps) else self.sim_fn(tool.desc, stmt)),
            tool.attrs.get("name_sim", self.sim_fn(tool.id.replace(".", " ").replace("_", " "), stmt)),
            tool.attrs.get("params", 1.0),
            tool.attrs.get("assertive", 0.0),
            1.0 if (d is not None and (tool.org == d.org or tool.org in d.orgs)) else 0.0,
        ])


def _token_overlap(a: str, b: str) -> float:
    A = set(a.lower().replace(".", " ").replace("-", " ").split())
    B = set(b.lower().replace(".", " ").replace("-", " ").split())
    if not A or not B:
        return 0.0
    return len(A & B) / math.sqrt(len(A) * len(B))


# ---------------------------------------------------------------- the model
@dataclass
class SelectionModel:
    beta: np.ndarray
    gamma: float
    featurizer: Featurizer

    def attraction(self, t: str, r: str, v: Optional[str]) -> float:
        return float(np.exp(self.featurizer(t, r, v) @ self.beta))

    def probs(self, K: Sequence[str], r: str, v: Optional[str]) -> Tuple[Dict[str, float], float]:
        fed = self.featurizer.fed
        a = {t: self.attraction(t, r, v) for t in K}
        L = fed.load(K) / KTOK
        denom = 1.0 + self.gamma * L + sum(a.values())
        return {t: a[t] / denom for t in K}, (1.0 + self.gamma * L) / denom


# ---------------------------------------------------------------- values and quality
def qos_deficits(fed: Federation, params: Params) -> Dict[str, float]:
    """dq_t: weighted sum of normalized latency, price, unavailability, unreliability."""
    tools = list(fed.tools.values())
    lat = [t.qos.lat for t in tools]; pr = [t.qos.pr for t in tools]
    def norm(x, lo, hi): return 0.0 if hi <= lo else (x - lo) / (hi - lo)
    wl, wp, wa, wr = params.qos_weights
    out = {}
    for t in tools:
        out[t.id] = (wl * norm(t.qos.lat, min(lat), max(lat)) + wp * norm(t.qos.pr, min(pr), max(pr))
                     + wa * (1 - t.qos.av) + wr * (1 - t.qos.rel))
    return out


def net_value(tool: Tool, r: str, dq: float, lam_q: float) -> float:
    """w_tr = ad (1 - lambda dq) - (1 - ad) psi   (Eq. 3)."""
    ad = tool.adequacy.get(r, 0.0) if r in tool.caps else 0.0
    return ad * (1.0 - lam_q * dq) - (1.0 - ad) * tool.psi


class Quality:
    """Q_r(K), Q(K), diluted values, selection loss, expected occasions."""

    def __init__(self, fed: Federation, params: Params, model: SelectionModel):
        self.fed, self.params, self.model = fed, params, model
        self.dq = qos_deficits(fed, params)

    def w(self, t: str, r: str) -> float:
        return net_value(self.fed.tools[t], r, self.dq[t], self.params.lam_q)

    def Q_r(self, K: Sequence[str], r: str, v: Optional[str]) -> float:
        if not K:
            return 0.0
        p, _ = self.model.probs(K, r, v)
        return sum(p[t] * self.w(t, r) for t in K)

    def Q(self, K: Sequence[str], v: str) -> float:
        return sum(self.Q_r(K, r, v) for r in self.fed.descriptors[v].R)

    def diluted(self, t: str, r: str, v: Optional[str], K: Sequence[str] = ()) -> float:
        """tilde w_tr = w_tr / (1 + gamma delta_t / a_tr), delta_t the load t adds to K."""
        tool = self.fed.tools[t]
        srv = {self.fed.tools[u].server for u in K}
        delta = (tool.cost + (self.fed.servers[tool.server].preamble if tool.server not in srv else 0)) / KTOK
        a = self.model.attraction(t, r, v)
        return self.w(t, r) / (1.0 + self.model.gamma * delta / a)

    def selection_loss(self, K: Sequence[str], v: str) -> float:
        tot = 0.0
        for r in self.fed.descriptors[v].R:
            best = max([self.w(t, r) for t in K] + [0.0])
            tot += best - self.Q_r(K, r, v)
        return tot

    def occasions(self, K: Sequence[str], r: str, v: str) -> float:
        p, _ = self.model.probs(K, r, v)
        dl = sum(p[t] * (self.fed.tools[t].adequacy.get(r, 0.0) if r in self.fed.tools[t].caps else 0.0) for t in K)
        m = self.params.m_retries
        return float(m) if dl <= 0 else (1 - (1 - dl) ** m) / dl


# ---------------------------------------------------------------- estimation (Eq. 12)
@dataclass
class Observation:
    K: List[str]
    r: str
    v: Optional[str]
    chosen: Optional[str]     # tool id, or None for no valid call
    group: str = ""           # key used for group splits (e.g. tool family)


def _tensors(obs: Sequence[Observation], feat: Featurizer):
    n = len(obs); kmax = max(len(o.K) for o in obs); d = len(feat.names)
    Phi = np.zeros((n, kmax, d)); mask = np.zeros((n, kmax)); L = np.zeros(n); ch = -np.ones(n, dtype=int)
    for i, o in enumerate(obs):
        for j, t in enumerate(o.K):
            Phi[i, j] = feat(t, o.r, o.v); mask[i, j] = 1.0
            if o.chosen == t:
                ch[i] = j
        L[i] = feat.fed.load(o.K) / KTOK
    return Phi, mask, L, ch


def _negll(beta, gamma, Phi, mask, L, ch):
    a = np.exp(np.clip(Phi @ beta, -30, 30)) * mask
    base = 1.0 + gamma * L
    denom = base + a.sum(1)
    idx = np.arange(len(ch))
    num = np.where(ch >= 0, a[idx, np.maximum(ch, 0)], base)
    nll = -(np.log(num) - np.log(denom)).sum()
    # gradient w.r.t. beta
    p = a / denom[:, None]
    g = (p[:, :, None] * Phi).sum(1).sum(0)
    chosen_phi = np.where((ch >= 0)[:, None], Phi[idx, np.maximum(ch, 0)], 0.0)
    g -= chosen_phi.sum(0)
    return nll, g


def fit(obs: Sequence[Observation], feat: Featurizer, gamma_fixed: Optional[float] = None,
        gamma_max: float = 5.0, signed: bool = True, load_cap: Optional[float] = None) -> Tuple[SelectionModel, float]:
    """Maximum likelihood: concave in beta for fixed gamma; gamma by a 1-D search.

    With signed=True the load effect gamma may be negative (more tools, fewer abstentions), bounded so
    that the no-call weight 1 + gamma*L stays positive up to `load_cap` kilotokens (default: the largest
    load in the data)."""
    Phi, mask, L, ch = _tensors(obs, feat)
    cap = load_cap if load_cap is not None else max(float(L.max()), 1e-6)
    gamma_min = -0.95 / cap if signed else 0.0
    d = Phi.shape[2]
    vals = Phi[mask > 0]
    keep = [0] + [j for j in range(1, d) if np.ptp(vals[:, j]) > 1e-12]   # intercept + varying features
    Pk = Phi[:, :, keep]

    def inner(gamma):
        res = minimize(lambda b: _negll(b, gamma, Pk, mask, L, ch), np.zeros(len(keep)), jac=True, method="L-BFGS-B")
        full = np.zeros(d); full[keep] = res.x
        return res.fun, full

    if gamma_fixed is not None:
        f, b = inner(gamma_fixed)
        return SelectionModel(b, gamma_fixed, feat), f
    r = minimize_scalar(lambda g: inner(g)[0], bounds=(gamma_min, gamma_max), method="bounded",
                        options={"xatol": 1e-3})
    f, b = inner(r.x)
    return SelectionModel(b, float(r.x), feat), f


def predictive_scores(model, obs: Sequence[Observation]) -> Dict[str, float]:
    """Log-loss, Brier score and top-1 accuracy on held-out observations.

    `model` is any object with probs(K, r, v) -> (dict, p_none)."""
    ll, br, top = 0.0, 0.0, 0
    for o in obs:
        p, p0 = model.probs(o.K, o.r, o.v)
        outcomes = dict(p); outcomes[None] = p0
        ll -= math.log(max(outcomes[o.chosen], 1e-12))
        br += sum((outcomes[k] - (1.0 if k == o.chosen else 0.0)) ** 2 for k in outcomes)
        top += int(max(outcomes, key=outcomes.get) == o.chosen)
    n = len(obs)
    return {"log_loss": ll / n, "brier": br / n, "top1": top / n}


class RelevanceModel:
    """E1 baseline: selection proportional to relevance, constant no-call rate.

    Pr(t | K) = (1 - q) * s_t^tau / sum s^tau, Pr(none) = q. No load effect, no learned features.
    """

    def __init__(self, feat: Featurizer, tau: float = 1.0, q: float = 0.1):
        self.feat, self.tau, self.q = feat, tau, q

    def probs(self, K, r, v):
        s = {t: max(self.feat(t, r, v)[1], 1e-3) ** self.tau for t in K}
        z = sum(s.values())
        return {t: (1 - self.q) * s[t] / z for t in K}, self.q

    @classmethod
    def fit(cls, obs, feat):
        q = sum(o.chosen is None for o in obs) / len(obs)
        best = None
        for tau in np.linspace(0.25, 8, 32):
            m = cls(feat, tau, max(q, 1e-3))
            ll = predictive_scores(m, obs)["log_loss"]
            if best is None or ll < best[0]:
                best = (ll, m)
        return best[1]
