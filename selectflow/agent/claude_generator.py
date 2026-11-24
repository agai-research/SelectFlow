"""LLM components of the derivation: interpretation of the request and
grammar-constrained generation of workflows (Sect. 4.2).

The grammar is enforced by a forced tool call whose input schema only admits
the identifiers of the retrieved descriptors (an enum); outputs are checked
again against the enum before use.
"""
from __future__ import annotations

import json
from typing import Dict, List, Set, Tuple

from ..derivation import Report
from ..model import Context, Descriptor
from .claude import ClaudeConfig, _client, _to_dict


def _forced(cfg: ClaudeConfig, system: str, user: str, name: str, schema: dict, model=None) -> dict:
    """Structured answer through a tool call. Models that refuse a forced tool_choice get tool_choice auto,
    an explicit instruction, and up to three attempts."""
    client = _client()
    extra = {"thinking": cfg.thinking} if cfg.thinking is not None else {}
    tools = [{"name": name, "description": "Return the answer.", "input_schema": schema}]
    base = dict(model=model or cfg.model, max_tokens=1024, tools=tools, messages=[{"role": "user", "content": user}], **extra)
    try:
        msg = _to_dict(client.messages.create(system=system, tool_choice={"type": "tool", "name": name}, **base))
        attempts = [msg]
    except Exception as e:
        if "tool_choice" not in str(e):
            raise
        attempts = []
        for _ in range(3):
            msg = _to_dict(client.messages.create(system=system + f" Answer only by calling the tool `{name}`.",
                                                  tool_choice={"type": "auto"}, **base))
            attempts.append(msg)
            if any(b["type"] == "tool_use" for b in msg["content"]):
                break
    for b in attempts[-1]["content"]:
        if b["type"] == "tool_use":
            return b["input"]
    return {}


class ClaudeInterpreter:
    def __init__(self, cfg: ClaudeConfig, artefacts: List[str]):
        self.cfg, self.artefacts = cfg, sorted(artefacts)

    def __call__(self, q: str, ctx: Context) -> Tuple[Set[str], Set[str]]:
        schema = {"type": "object", "properties": {
            "targets": {"type": "array", "items": {"type": "string", "enum": self.artefacts}},
            "held": {"type": "array", "items": {"type": "string", "enum": self.artefacts}}},
            "required": ["targets", "held"]}
        out = _forced(self.cfg, "Map the citizen's request to the documents they want (targets) and "
                      "the documents they already hold. Use only the listed artefact types.",
                      q, "interpretation", schema)
        ok = set(self.artefacts)
        return set(out.get("targets", [])) & ok, set(out.get("held", [])) & ok


class ClaudeGenerator:
    """Generates candidate workflows as ordered lists of descriptor identifiers."""

    def __init__(self, cfg: ClaudeConfig, unconstrained: bool = False):
        self.cfg, self.unconstrained = cfg, unconstrained   # unconstrained=True is the E2 direct-generation baseline

    def _schema(self, Dq: Dict[str, Descriptor]) -> dict:
        item = {"type": "string"} if self.unconstrained else {"type": "string", "enum": sorted(Dq)}
        return {"type": "object", "properties": {"tasks": {"type": "array", "items": item}}, "required": ["tasks"]}

    def _catalogue(self, Dq):
        return json.dumps([{"id": k, "title": d.title, "organization": d.org, "requires": sorted(d.In),
                            "delivers": sorted(d.Out)} for k, d in Dq.items()], ensure_ascii=False)

    def generate(self, q, G, held, Dq, k) -> List[List[str]]:
        out = []
        for i in range(k):
            res = _forced(self.cfg, "Compose an administrative workflow: an ordered list of service "
                          "identifiers such that every document a service requires is held or delivered by an "
                          "earlier service, and the targets are delivered.",
                          f"Request: {q}\nTargets: {sorted(G)}\nHeld: {sorted(held)}\nServices: {self._catalogue(Dq)}\n"
                          f"Candidate {i + 1}.", "workflow", self._schema(Dq))
            out.append(list(res.get("tasks", [])))
        return out

    def revise(self, order, rep: Report, Dq) -> List[str]:
        res = _forced(self.cfg, "Revise the workflow so that the reported problems disappear.",
                      f"Workflow: {order}\nMissing artefacts: {sorted(rep.missing_artefacts())}\n"
                      f"Missing companions: {rep.companions}\nSuperfluous: {rep.superfluous}\n"
                      f"Services: {self._catalogue(Dq)}", "workflow", self._schema(Dq))
        return list(res.get("tasks", []))
