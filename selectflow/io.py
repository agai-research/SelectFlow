"""JSON format of a federation (descriptors, tools, servers, hosts, sharing bases).

Real data (Mitos, Saudi service cards, MCP-Bench tools) are converted into this
format; see data/README_data.md for the field-by-field mapping.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict

from .model import Descriptor, Federation, ModelHost, QoS, Server, Tool


def _sets_to_lists(o: Any):
    if isinstance(o, set):
        return sorted(o)
    if isinstance(o, dict):
        return {k: _sets_to_lists(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sets_to_lists(v) for v in o]
    return o


def dump(fed: Federation, path: str) -> None:
    obj = {
        "descriptors": [_sets_to_lists(asdict(d)) for d in fed.descriptors.values()],
        "tools": [_sets_to_lists(asdict(t)) for t in fed.tools.values()],
        "servers": [asdict(s) for s in fed.servers.values()],
        "hosts": [asdict(h) for h in fed.hosts.values()],
        "sharing": sorted(list(s) for s in fed.sharing),
        "cat": _sets_to_lists(fed.cat),
        "categories": sorted(fed.categories),
        "certified": _sets_to_lists(fed.certified),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def load(path: str) -> Federation:
    with open(path, encoding="utf-8") as f:
        o = json.load(f)
    S = lambda x: set(x or [])
    D = {}
    for d in o["descriptors"]:
        D[d["id"]] = Descriptor(d["id"], d["org"], d["purpose"], S(d["In"]), S(d["Out"]), S(d["R"]), S(d.get("A")),
                                S(d.get("comp")), d.get("fee", 0.0), d.get("dur", 0.0), d.get("chi", {}),
                                d.get("title", ""), d.get("text", ""), d.get("jurisdiction"), d.get("cap_text", {}),
                                d.get("record"), S(d.get("orgs")))
    T = {}
    for t in o["tools"]:
        q = t.get("qos") or {}
        T[t["id"]] = Tool(t["id"], t["server"], t["org"], t["desc"], t["caps"], t["cost"],
                          QoS(q.get("lat", 1.0), q.get("pr", 0.0), q.get("av", 1.0), q.get("rel", 1.0)),
                          S(t.get("prod")), S(t.get("cons")), S(t.get("inp")),
                          t.get("input_schema", {"type": "object", "properties": {}}), t.get("adequacy", {}),
                          t.get("psi", 0.0), t.get("attrs", {}), S(t.get("true_ret")), t.get("true_fwd", False),
                          t.get("family"))
    servers = {s["id"]: Server(**s) for s in o["servers"]}
    hosts = {h["id"]: ModelHost(**h) for h in o["hosts"]}
    fed = Federation(D, T, servers, hosts, {tuple(x) for x in o["sharing"]},
                     {k: set(v) for k, v in o["cat"].items()}, set(o["categories"]),
                     {k: set(v) for k, v in o.get("certified", {}).items()})
    return fed
