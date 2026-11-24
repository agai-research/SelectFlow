"""From registry records to descriptors, federations, requests and reference workflows.

The annotation here is deterministic and traceable: every dependency comes from the registry
itself (Mitos 'Relevant Procedure' links; requirements stated on the Saudi service pages).
An LLM pass may later refine categories and capability statements (scripts/colab_annotate.py),
but the dependency graph and the reference workflows never come from an LLM.
"""
from __future__ import annotations

import math
import random
import re
from typing import Dict, List, Optional, Set, Tuple

from .derivation import check, forward_chain
from .model import REQ, Descriptor, Federation, ModelHost, QoS, Server, Tool

CATS = ["ID", "BUS", "LOC", "FIN", "HLT"]
KW = {"ID": r"identit|passport|id card|iqama|national id|identification|criminal record|solemn declaration",
      "BUS": r"compan|business|enterprise|commercial|establishment|trade|registration|licen[cs]e|vat",
      "LOC": r"premises|location|lease|building|address|site|shop|property|deed|station|port|vessel",
      "FIN": r"fee|payment|bank|tax|financial|solvency|revenue|debt|contribution|zakat",
      "HLT": r"health|medical|sanitary|hygien"}


def slug(s: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:n]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def cats_of(text: str, default=("BUS",)) -> Set[str]:
    out = {c for c, rx in KW.items() if re.search(rx, text or "", re.I)}
    return out or set(default)


def purpose_of(title: str) -> str:
    t = title.lower()
    for k, p in (("change", "modification"), ("renew", "modification"), ("amend", "modification"), ("update", "modification"),
                 ("notification", "notification"), ("registration", "registration"), ("register", "registration"),
                 ("certificate", "certification"), ("attestation", "certification"), ("licen", "authorization"),
                 ("permit", "authorization"), ("visa", "authorization"), ("approval", "authorization"),
                 ("verify", "information"), ("copy", "information")):
        if k in t:
            return p
    return "authorization"


def short_org(body: str) -> str:
    body = re.split(r"\s+(Authority|Service /|Unit|Directorate)\b", body or "")[0].strip()
    return body.title() if body.isupper() else (body or "Unknown body")


EXTERNAL_ORG = {   # issuers of documents that Mitos records require from other procedures (inferred from the titles)
    "criminal record": "Ministry Of Justice", "judicial solvency": "Courts", "solemn declaration": "Gov.gr",
    "tax debts": "Independent Authority For Public Revenue", "social security": "E-Efka", "eefka": "E-Efka",
    "fire protection": "Fire Service", "cng station": "Ministry Of Environment And Energy"}


def _euros(s: str) -> Optional[float]:
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*€", s or "")
    return float(m.group(1).replace(",", ".")) if m else None


def _days(s: str) -> Optional[float]:
    m = re.search(r"(\d+)\s*(?:-\s*(\d+)\s*)?(working day|day|hour|month|year)", s or "", re.I)
    if not m:
        return None
    n = float(m.group(2) or m.group(1)); u = m.group(3).lower()
    return n / 24 if u == "hour" else n * 30 if u == "month" else n * 365 if u == "year" else n


def _caps(did: str, title: str, out: str, fee) -> Dict[str, str]:
    caps = {f"submit:{did}": f"submit the application for {title}", f"retrieve:{did}": f"obtain the {out.replace('_', ' ')}"}
    if fee:
        caps[f"pay:{did}"] = f"pay the fee for {title}"
    return caps


def mitos_descriptors(raw: List[dict], meaningful_names: bool = True) -> Dict[str, Descriptor]:
    """meaningful_names=False reproduces version 1 of the data (opaque names 'out_grXX'), used in E1, E3 and E6."""
    by_title = {norm(r["title"]): i for i, r in enumerate(raw)}
    outs, seen = [], set()
    for i, r in enumerate(raw):
        o = f"out_gr{i:02d}"
        if meaningful_names:
            o = slug(r["title"], 60)
            if o in seen:
                o = f"{o}_{i:02d}"
        seen.add(o); outs.append(o)
    D: Dict[str, Descriptor] = {}
    ext: Dict[str, str] = {}
    for i, r in enumerate(raw):
        did = f"gr{i:02d}"
        F = r["fields"]
        out = outs[i]
        req = set()
        for dep in r.get("relevant_procedures", []):
            k = norm(dep)
            if k in by_title:
                req.add(outs[by_title[k]])
            else:
                ext.setdefault(k, dep)
                req.add("ext_" + slug(dep, 50))
        fee = _euros(F.get("Fees", ""))
        caps = _caps(did, r["title"], out, fee)
        text = r.get("full_text", "")
        D[did] = Descriptor(did, short_org(F.get("Competent body", "")), purpose_of(r["title"]), req, {out}, set(caps),
                            ({"ID"} | cats_of(text[:4000])) & set(CATS), set(), fee or 0.0,
                            _days(F.get("Deadline of completion", "")) or 0.0, {}, r["title"],
                            (F.get("Description") or "")[:600], None, caps, r.get("record"))
    for k, title in ext.items():       # referenced issuing procedures, outside the exported category
        did = "grx_" + slug(title, 30)
        out = "ext_" + slug(title, 50)
        org = next((o for kw, o in EXTERNAL_ORG.items() if kw in k), "Other Public Body")
        caps = _caps(did, title, out, None)
        D[did] = Descriptor(did, org, purpose_of(title), set(), {out}, set(caps), {"ID"}, set(), 0.0, 0.0, {},
                            title, "Referenced by Mitos records as the issuing procedure of a required document.",
                            None, caps, "referenced")
    return D


def saudi_descriptors(raw: List[dict]) -> Dict[str, Descriptor]:
    D = {}
    for r in raw:
        caps = _caps(r["id"], r["title"], r["delivers"][0], r.get("fee_sar"))
        D[r["id"]] = Descriptor(r["id"], r["org"], r["purpose"], set(r["requires"]), set(r["delivers"]), set(caps),
                                ({"ID"} | cats_of(r["title"] + " " + r["conditions"])) & set(CATS), set(r.get("companions", [])),
                                float(r.get("fee_sar") or 0.0), float(r.get("duration_days") or 0.0), {}, r["title"],
                                r["conditions"][:600], None, caps, r["url"])
    return D


def tool_base(did: str, d: Descriptor) -> str:
    """Unique tool-name stem: organization, title, descriptor id (titles alone collide once truncated)."""
    return f"{slug(d.org, 16)}.{slug(d.title, 30)}_{did}"


def _jaccard(a: str, b: str) -> float:
    A, B = set(re.findall(r"[a-z]{4,}", a.lower())), set(re.findall(r"[a-z]{4,}", b.lower()))
    return len(A & B) / len(A | B) if A | B else 0.0


def build_federation(D: Dict[str, Descriptor], seed: int = 0, sharing_density: float = 1.0,
                     p_third_party: float = 0.15, budget: int = 6000) -> Federation:
    """Tools generated from the descriptors, look-alikes taken from neighbouring procedures of the same
    organization, third-party helper tools whose forwarding behaviour is not certified, model hosts,
    sharing bases (competence, registry-sanctioned hand-offs, host contracts) and certified labels."""
    rng = random.Random(seed)
    cat: Dict[str, Set[str]] = {}
    for d in D.values():
        for a in d.In | d.Out:
            cat.setdefault(a, cats_of(a.replace("_", " ") + " " + (d.title if a in d.Out else ""), ("BUS",)))
    servers, tools = {}, {}
    orgs = sorted({d.org for d in D.values()})
    for o in orgs:
        servers[f"s_{slug(o, 30)}"] = Server(f"s_{slug(o, 30)}", o, 120)

    def add(tid, org, srv, desc, caps, prod=(), cons=(), inp=(), ret=(), ad=0.9, psi=0.0, fwd=False, fam=None):
        tools[tid] = Tool(tid, srv, org, desc, dict(caps), 60 + len(desc) // 4,
                          QoS(rng.uniform(0.3, 3.0), 0.0, rng.uniform(0.97, 1.0), rng.uniform(0.97, 1.0)),
                          set(prod), set(cons), set(inp),
                          {"type": "object", "properties": {"applicant_id": {"type": "string"}, "details": {"type": "string"}}},
                          {r: ad for r in caps}, psi, {}, set(ret), fwd, fam)
        return tools[tid]

    for did, d in D.items():
        s = f"s_{slug(d.org, 30)}"; base = tool_base(did, d)
        out = sorted(d.Out)[0]
        in_cats = set().union(*[cat[a] for a in d.In]) if d.In else set()
        add(f"{base}.submit", d.org, s, f"Submit an application for: {d.title}.", {f"submit:{did}": 0.9},
            cons=d.In, inp=d.A | in_cats, psi=0.2, fam=did)
        add(f"{base}.get", d.org, s, f"Retrieve the issued {out.replace('_', ' ')} ({d.title}).", {f"retrieve:{did}": 0.9},
            prod=d.Out, inp={"ID"}, ret=cat[out], fam=did)
        if f"pay:{did}" in d.R:
            add(f"{base}.pay", d.org, s, f"Pay the fee of: {d.title}.", {f"pay:{did}": 0.9}, inp={"ID", "FIN"}, ret={"FIN"}, fam=did)
    # look-alikes: the submit tool of a neighbouring procedure of the same organization also maps to this capability
    ids = sorted(D)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if D[a].org == D[b].org and _jaccard(D[a].title, D[b].title) >= 0.45:
                for x, y in ((a, b), (b, a)):
                    t = f"{tool_base(y, D[y])}.submit"
                    tools[t].caps[f"submit:{x}"] = 0.8
                    tools[t].adequacy[f"submit:{x}"] = 0.0          # files the wrong procedure
    # third-party helper tools (document upload portals), not certified as non-forwarding
    servers["s_third_party"] = Server("s_third_party", "ThirdParty", 100)
    for did, d in D.items():
        if rng.random() < p_third_party:
            add(f"helper.upload_{slug(d.title, 30)}", "ThirdParty", "s_third_party",
                f"Fill in and upload the forms of {d.title} on your behalf.", {f"submit:{did}": 0.6},
                inp=set(CATS), ad=0.7, psi=0.1, fwd=True, fam=did)
    hosts = {"hP": ModelHost("hP", "PortalContractor", budget), "hG": ModelHost("hG", "GovCloud", budget)}
    sharing: Set[Tuple[str, str, str, str]] = set()
    purposes = {d.purpose for d in D.values()}
    for p in purposes:
        for o in [REQ] + orgs + ["ThirdParty"]:
            for a in ("ID", "BUS", "LOC"):
                sharing.add((p, a, o, "PortalContractor"))          # contract of the portal's host
            for a in CATS:
                sharing.add((p, a, o, "GovCloud"))                  # government cloud: all categories
    producers = {}
    for k, d in D.items():
        for a in d.Out:
            producers.setdefault(a, []).append(d.org)
    for d in D.values():
        for a in d.In:
            for c in cat[a]:
                sharing.add((d.purpose, c, REQ, d.org))             # documents the procedure itself requires
                for o in producers.get(a, []):
                    if rng.random() < sharing_density:
                        sharing.add((d.purpose, c, o, d.org))       # registry-sanctioned exchange
    fed = Federation(D, tools, servers, hosts, sharing, cat, set(CATS))
    fed.add_competence_bases()
    cert = {"fwd": {t for t, T in tools.items() if T.org != "ThirdParty"}}
    for a in CATS:
        cert[f"ret:{a}"] = {t for t, T in tools.items() if T.org != "ThirdParty" and a not in T.true_ret}
    fed.certified = cert
    return fed


def requests_and_references(fed: Federation, min_tasks: int = 2, tau_chi: float = 0.6) -> List[dict]:
    """One request per procedure whose output needs other procedures. Held documents are those that no
    procedure of the federation delivers. The reference workflow is the forward-chaining closure over the
    registry dependencies (deterministic, not produced by an LLM)."""
    produced = {a for d in fed.descriptors.values() for a in d.Out}
    out = []
    for did, d in sorted(fed.descriptors.items()):
        G = set(d.Out)
        need = set().union(*[x.In for x in fed.descriptors.values()]) | G
        held = {a for a in need if a not in produced}
        order, U = forward_chain(fed.descriptors, G, held, tau_chi)
        if order is None or len(order) < min_tasks:
            continue
        rep, W = check(order, G, held, fed.descriptors, tau_chi)
        out.append({"id": f"req_{did}", "text": f"I need to obtain: {d.title}.", "targets": sorted(G), "held": sorted(held),
                    "references": [order], "n_tasks": len(order), "organizations": sorted({fed.descriptors[v].org for v in order}),
                    "closed": rep.empty()})
    return out
