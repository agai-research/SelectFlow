"""Toolkit baselines of E3. Each returns the toolkit offered for one task under one host.

All baselines respect the size bound and the context budget. As in the protocol, they are
followed by a filter on *declared* annotations (tools are trusted as declared), whereas
SelectFlow uses certified labels.
"""
from __future__ import annotations

import copy
import re
from typing import Dict, List, Optional, Sequence

from .. import flows
from ..model import Federation, Params, Workflow
from ..selection import Quality


def _text_sim(a: str, b: str) -> float:
    A, B = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(A & B) / (1e-9 + (len(A) * len(B)) ** 0.5)


def declared_view(fed: Federation) -> Federation:
    """A federation in which every tool is trusted as declared safe (the declared-annotation filter)."""
    f2 = copy.copy(fed)
    f2.certified = {k: set(fed.tools) for k in ["fwd"] + [f"ret:{a}" for a in fed.categories]}
    return f2


def _fill(fed, ranked: Sequence[str], h: str, params: Params, banned=()) -> List[str]:
    K: List[str] = []
    for t in ranked:
        if t in banned or t in K:
            continue
        if len(K) >= params.k_max:
            break
        if fed.load(K + [t]) <= fed.hosts[h].budget:
            K.append(t)
    return K


def _banned(fed, W, v, h, cands, gap):
    P = flows.prohibitions(declared_view(fed), W, v, cands, [h], gap)
    return P.forbidden_under(h)


def dense_topk(fed: Federation, W: Workflow, v: str, h: str, params: Params, k: int,
               sim=None, gap=()) -> List[str]:
    """Relevance retrieval (ToolLLM-style). `sim(tool_desc, task_text)` may be a dense retriever;
    the default is lexical overlap, to be replaced by an embedding model in the real campaign."""
    d = fed.descriptors[v]
    text = " ".join([d.title, d.text] + list(d.cap_text.values()) + sorted(d.R))
    sim = sim or _text_sim
    ranked = sorted(fed.tools, key=lambda t: -sim(fed.tools[t].desc, text))
    p = Params(**{**params.__dict__, "k_max": k})
    return _fill(fed, ranked, h, p, _banned(fed, W, v, h, ranked[: 3 * k], gap))


def qos_single(fed: Federation, W: Workflow, v: str, h: str, params: Params, quality: Quality, gap=()) -> List[str]:
    """Classical QoS-aware composition: one service per capability, the one with the lowest deficit."""
    d = fed.descriptors[v]
    out = []
    provs = {r: [t for t, T in fed.tools.items() if T.caps.get(r, 0) >= params.tau_mu] for r in d.R}
    banned = _banned(fed, W, v, h, sorted({t for ps in provs.values() for t in ps}), gap)
    for r in sorted(d.R):
        ps = [t for t in provs[r] if t not in banned]
        if ps:
            out.append(min(ps, key=lambda t: quality.dq[t]))
    return _fill(fed, list(dict.fromkeys(out)), h, params)


def attraction_rank(fed: Federation, W: Workflow, v: str, h: str, params: Params, quality: Quality,
                    gap=()) -> List[str]:
    """Same constraints and certified labels as SelectFlow, but ranks by attraction instead of optimizing Q."""
    d = fed.descriptors[v]
    cands = [t for t, T in fed.tools.items() if any(T.caps.get(r, 0) >= params.tau_mu for r in d.R)]
    P = flows.prohibitions(fed, W, v, cands, [h], gap)
    banned = P.forbidden_under(h)
    m = quality.model
    ranked = sorted(cands, key=lambda t: -max(m.attraction(t, r, v) for r in d.R if r in fed.tools[t].caps))
    return _fill(fed, ranked, h, params, banned)


class SetLevelAdapter:
    """HYSET (Hong et al., 2026) or COLT (Qu et al., CIKM 2024).

    Wrap the authors' released model: `score_set(task_text, candidate_ids) -> ranked ids`.
    Both must be trained on the same catalogue (co-invocations from the TOUCAN extract)."""

    def __init__(self, scorer=None, name="HYSET"):
        self.scorer, self.name = scorer, name

    def __call__(self, fed, W, v, h, params, gap=()):
        if self.scorer is None:
            raise NotImplementedError(f"{self.name}: plug the released model in SetLevelAdapter(scorer=...)")
        d = fed.descriptors[v]
        ranked = self.scorer(" ".join([d.title, d.text] + list(d.cap_text.values())), list(fed.tools))
        return _fill(fed, ranked, h, params, _banned(fed, W, v, h, ranked[: 3 * params.k_max], gap))


def bm25_sim(fed: Federation):
    """BM25 over tool descriptions (rank_bm25). Lexical baseline usable without model weights."""
    from rank_bm25 import BM25Okapi
    ids = list(fed.tools)
    tok = lambda s: re.findall(r"\w+", s.lower())
    bm = BM25Okapi([tok(fed.tools[t].desc + " " + t.replace(".", " ")) for t in ids])
    cache = {}
    def sim(desc, text):
        if text not in cache:
            cache[text] = dict(zip((fed.tools[t].desc for t in ids), bm.get_scores(tok(text))))
        return float(cache[text].get(desc, 0.0))
    return sim


def dense_sim(fed: Federation, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Dense retriever (sentence-transformers). The ToolLLM retriever or COLT's first-stage checkpoint
    can be passed as model_name. Needs the model weights (Hugging Face) on the machine that runs it."""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    m = SentenceTransformer(model_name)
    descs = sorted({t.desc for t in fed.tools.values()})
    E = dict(zip(descs, m.encode(descs, normalize_embeddings=True)))
    cache = {}
    def sim(desc, text):
        if text not in cache:
            cache[text] = m.encode([text], normalize_embeddings=True)[0]
        return float(np.dot(E[desc], cache[text]))
    return sim


def hyset_precomputed(fed: Federation, W: Workflow, v: str, h: str, params: Params, rankings: Dict[str, dict],
                      gap=(), fill: bool = False) -> List[str]:
    """HYSET from rankings computed on Colab (data/hyset_rankings_*.json), restricted to our catalogue.

    The toolkit is HYSET's predicted set (its set-level output); with fill=True it is completed from
    HYSET's ranking up to k_max. Size bound, context budget and the declared-annotation filter apply."""
    r = rankings.get(v)
    if r is None:
        return []
    ranked = list(r["predicted_set"]) + ([t for t in r["ranking"] if t not in r["predicted_set"]] if fill else [])
    ranked = [t for t in ranked if t in fed.tools]
    return _fill(fed, ranked, h, params, _banned(fed, W, v, h, ranked[: 3 * params.k_max], gap))
