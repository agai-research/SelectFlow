"""End-to-end enactment (E6): an LLM agent calls tools of simulated MCP servers.

Three configurations share this loop:
  * SelectFlow: one episode per task, with the composed toolkit (task-scoped context);
  * all-tools baseline: one episode for the whole request, every tool of the federation loaded;
  * strong baseline: one episode with a client-side tool search and a registry search, or the
    native Tool Search of the Claude API (configured in config.yaml, see the API documentation).
Each round of all pending episodes can be sent as one batch (cheaper, slower).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set

from ..model import Federation
from .claude import ClaudeConfig, Ledger, _client, _to_dict, tool_definition, tool_name


def _overlap(a: str, b: str) -> float:
    A, B = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(A & B) / (1 + len(A | B))


class MockEnvironment:
    """Simulated MCP servers: a tool delivers its artefacts when its inputs are held."""

    def __init__(self, fed: Federation, held: Set[str]):
        self.fed, self.held, self.calls = fed, set(held), []

    def execute(self, tid: str, args: dict) -> dict:
        T = self.fed.tools[tid]
        self.calls.append(tid)
        missing = sorted(T.cons - self.held)
        if missing:
            return {"ok": False, "error": f"missing documents: {missing}"}
        self.held |= T.prod
        return {"ok": True, "delivered": sorted(T.prod)}


class RegistryEnvironment(MockEnvironment):
    """Procedure semantics for registry federations (tools grouped by procedure in Tool.family):
    the application ('.submit') needs the required documents; the output ('.get') is delivered only after
    a successful application and, if the procedure has a fee, a payment ('.pay')."""

    def __init__(self, fed: Federation, held: Set[str]):
        super().__init__(fed, held)
        self.submitted, self.paid = set(), set()
        self.has_fee = {T.family for t, T in fed.tools.items() if t.endswith(".pay")}

    def execute(self, tid: str, args: dict) -> dict:
        T = self.fed.tools[tid]
        self.calls.append(tid)
        fam = T.family
        if tid.endswith(".submit") or T.org == "ThirdParty":
            missing = sorted(T.cons - self.held)
            if missing:
                return {"ok": False, "error": f"missing documents: {missing}"}
            target = fam if tid.endswith(".submit") else fam
            self.submitted.add(target)
            return {"ok": True, "status": "application registered"}
        if tid.endswith(".pay"):
            if fam not in self.submitted:
                return {"ok": False, "error": "no application to pay for"}
            self.paid.add(fam)
            return {"ok": True, "status": "fee paid"}
        if tid.endswith(".get"):
            if fam not in self.submitted:
                return {"ok": False, "error": "no application found"}
            if fam in self.has_fee and fam not in self.paid:
                return {"ok": False, "error": "fee not paid"}
            self.held |= T.prod
            return {"ok": True, "delivered": sorted(T.prod)}
        return super().execute(tid, args)


SEARCH_TOOLS = {"name": "search_tools", "description": "Search the catalogue for tools; returns names and descriptions.",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
SEARCH_REGISTRY = {"name": "search_registry", "description": "Search the registry of administrative procedures.",
                   "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}


@dataclass
class Episode:
    system: str
    K: List[str]
    env: MockEnvironment
    messages: List[dict]
    meta: dict = field(default_factory=dict)
    meta_tools: List[dict] = field(default_factory=list)
    turns: int = 0
    done: bool = False


class Enactor:
    def __init__(self, fed: Federation, cfg: ClaudeConfig, max_turns: int = 12, search_k: int = 5,
                 native_tool_search: Optional[dict] = None):
        self.fed, self.cfg, self.max_turns, self.search_k = fed, cfg, max_turns, search_k
        self.native = native_tool_search
        self.names = {tool_name(t): t for t in fed.tools}

    def params(self, ep: Episode) -> dict:
        tools = [tool_definition(self.fed, t) for t in ep.K] + ep.meta_tools
        native = bool(self.native and ep.meta.get("native_search"))
        if native:   # every tool deferred; the search tool itself is never deferred
            tools = [{**tool_definition(self.fed, t), "defer_loading": True} for t in self.fed.tools]
        if self.cfg.cache and tools and not native:          # deferred tools cannot be cached
            tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
        if native:
            tools = tools + [self.native]
        p = {"model": self.cfg.model, "max_tokens": 1024, "system": ep.system, "tools": tools, "messages": ep.messages}
        if self.cfg.thinking is not None:
            p["thinking"] = self.cfg.thinking
        return p

    def apply(self, ep: Episode, msg: dict) -> None:
        ep.turns += 1
        content = [{k: v for k, v in b.items() if v is not None} for b in msg["content"]]   # drop null fields
        if content:
            ep.messages.append({"role": "assistant", "content": content})
        results = []
        for b in msg["content"]:
            if b["type"] != "tool_use":
                continue
            if b["name"] == "search_tools":
                q = b["input"].get("query", "")
                hits = sorted(self.fed.tools, key=lambda t: -_overlap(q, self.fed.tools[t].desc))[: self.search_k]
                ep.K = list(dict.fromkeys(ep.K + hits))               # loaded for the next turns
                out = [{"name": tool_name(t), "description": self.fed.tools[t].desc} for t in hits]
            elif b["name"] == "search_registry":
                q = b["input"].get("query", "")
                ds = sorted(self.fed.descriptors.values(), key=lambda d: -_overlap(q, d.title + " " + d.text))[:5]
                out = [{"procedure": d.title, "organization": d.org, "requires": sorted(d.In),
                        "delivers": sorted(d.Out)} for d in ds]
            elif b["name"] in self.names and self.names[b["name"]] in self.fed.tools:
                out = ep.env.execute(self.names[b["name"]], b["input"])
            else:
                out = {"ok": False, "error": "unknown tool"}
            results.append({"type": "tool_result", "tool_use_id": b["id"], "content": json.dumps(out)})
        if results:
            ep.messages.append({"role": "user", "content": results})
        if not results or ep.turns >= self.max_turns:
            ep.done = True

    def run(self, episodes: List[Episode], backend: Callable[[List[dict]], List[dict]]) -> None:
        while True:
            pending = [e for e in episodes if not e.done]
            if not pending:
                return
            msgs = backend([self.params(e) for e in pending])
            for e, m in zip(pending, msgs):
                self.apply(e, m)


def threaded_backend(ledger: Ledger, workers: int = 8):
    """Synchronous calls in parallel threads (one round of turns at a time), with retries."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    client = _client()
    def call(p):
        for attempt in range(6):
            try:
                m = _to_dict(client.messages.create(**p)); ledger.add(m.get("usage", {}), batch=False); return m
            except Exception:
                time.sleep(2 ** attempt)
        return {"content": []}
    def run(params_list):
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(call, params_list))
    return run


def sync_backend(ledger: Ledger):
    client = _client()
    def run(params_list):
        out = []
        for p in params_list:
            m = _to_dict(client.messages.create(**p)); ledger.add(m.get("usage", {}), batch=False); out.append(m)
        return out
    return run


def batch_backend(ledger: Ledger, poll: float = 30.0):
    """One batch per round of turns: 50% cheaper, each round waits for its batch."""
    import time
    client = _client()
    def run(params_list):
        b = client.messages.batches.create(requests=[{"custom_id": f"e{i}", "params": p} for i, p in enumerate(params_list)])
        while client.messages.batches.retrieve(b.id).processing_status != "ended":
            time.sleep(poll)
        out: List[Optional[dict]] = [None] * len(params_list)
        for r in client.messages.batches.results(b.id):
            i = int(r.custom_id[1:])
            if r.result.type == "succeeded":
                m = _to_dict(r.result.message); ledger.add(m.get("usage", {}), batch=True); out[i] = m
        return [m or {"content": []} for m in out]
    return run
