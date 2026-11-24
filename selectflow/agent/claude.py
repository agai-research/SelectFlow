"""Claude API harness: selection probes, batch submission, prompt caching,
deduplication and cost accounting.

Nothing here runs without an API key (ANTHROPIC_API_KEY). Prices and model
identifiers come from config.yaml; check them against
https://platform.claude.com/docs/en/about-claude/pricing before a campaign.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..model import Federation

SYSTEM_PROBE = (
    "You are an assistant of a public-service portal. You help a citizen complete one step of an "
    "administrative procedure by calling a tool. Call the single tool that performs the step. If no "
    "available tool performs it, answer in one sentence without calling a tool.")


@dataclass
class ClaudeConfig:
    model: str = "claude-sonnet-5-5"          # freeze the exact identifier used in the paper
    debug_model: str = "claude-haiku-5-5"
    max_tokens: int = 256
    temperature: Optional[float] = None       # None = API default (sampling), needed to observe choice variability
    cache: bool = True
    cache_ttl: Optional[str] = "1h"           # 1-hour cache helps batches that share tool definitions
    price_in: float = 2.0                     # USD per million tokens (standard)
    price_out: float = 10.0
    price_cache_read: float = 0.10
    price_cache_write: float = 2.50
    batch_discount: float = 0.5
    store: str = "results/api_cache.jsonl"    # deduplication store
    system_probe: str = SYSTEM_PROBE
    workers: int = 8                          # parallel requests in synchronous mode
    thinking: Optional[dict] = None           # no thinking before the call: {"type": "disabled"} or, on models
                                              # that refuse it, {"type": "between_tools"} (see the API error message)


def tool_name(tid: str) -> str:
    """API-safe tool name; long identifiers keep a readable prefix and a hash, so names stay unique."""
    n = re.sub(r"[^A-Za-z0-9_-]", "_", tid)
    if len(n) <= 64:
        return n
    return n[:55] + "_" + hashlib.sha1(tid.encode()).hexdigest()[:8]


def tool_definition(fed: Federation, tid: str) -> dict:
    t = fed.tools[tid]
    return {"name": tool_name(tid), "description": t.desc, "input_schema": t.input_schema}


@dataclass
class ProbeSpec:
    """One call occasion: the agent needs capability r of task v and is offered toolkit K."""
    K: List[str]
    r: str
    v: str
    seed: int = 0
    text: Optional[str] = None
    meta: Dict = field(default_factory=dict)

    def key(self) -> str:
        return f"{self.v}|{self.r}|{','.join(self.K)}|{self.seed}"


def probe_text(fed: Federation, spec: ProbeSpec) -> str:
    d = fed.descriptors[spec.v]
    step = d.cap_text.get(spec.r, spec.r.replace("-", " "))
    return spec.text or f"Procedure: {d.title or d.id}. Step to perform now: {step}."


def probe_params(fed: Federation, spec: ProbeSpec, cfg: ClaudeConfig, model: Optional[str] = None) -> dict:
    order = list(spec.K)
    random.Random(spec.seed).shuffle(order)          # position is randomized, not confounded with method
    tools = [tool_definition(fed, t) for t in order]
    if cfg.cache and tools:
        cc = {"type": "ephemeral"}
        if cfg.cache_ttl:
            cc["ttl"] = cfg.cache_ttl
        tools[-1] = {**tools[-1], "cache_control": cc}
    p = {"model": model or cfg.model, "max_tokens": cfg.max_tokens, "system": cfg.system_probe,
         "tools": tools, "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
         "messages": [{"role": "user", "content": probe_text(fed, spec)}]}
    if cfg.temperature is not None:
        p["temperature"] = cfg.temperature
    if cfg.thinking is not None:
        p["thinking"] = cfg.thinking
    return p


def parse_selection(message, K: Sequence[str]) -> Optional[str]:
    """The tool the model called, or None for no valid call."""
    names = {tool_name(t): t for t in K}
    content = message["content"] if isinstance(message, dict) else message.content
    for block in content:
        typ = block["type"] if isinstance(block, dict) else block.type
        if typ == "tool_use":
            name = block["name"] if isinstance(block, dict) else block.name
            return names.get(name)          # a name outside K counts as no valid call
    return None


class Ledger:
    """Token and cost accounting, from the usage field of each response."""

    def __init__(self, cfg: ClaudeConfig):
        self.cfg, self.rows = cfg, []

    def add(self, usage, batch: bool):
        g = (lambda k: usage.get(k, 0) or 0) if isinstance(usage, dict) else (lambda k: getattr(usage, k, 0) or 0)
        row = {"in": g("input_tokens"), "out": g("output_tokens"), "cr": g("cache_read_input_tokens"),
               "cw": g("cache_creation_input_tokens"), "batch": batch}
        self.rows.append(row)

    def cost(self) -> float:
        c = self.cfg; tot = 0.0
        for r in self.rows:
            k = c.batch_discount if r["batch"] else 1.0
            tot += k * (r["in"] * c.price_in + r["out"] * c.price_out + r["cr"] * c.price_cache_read
                        + r["cw"] * c.price_cache_write) / 1e6
        return tot


class Store:
    """Deduplication: identical (params, seed) requests are sent once."""

    def __init__(self, path: str):
        self.path, self.data = path, {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    o = json.loads(line); self.data[o["key"]] = o["message"]

    @staticmethod
    def key(params: dict, seed: int) -> str:
        return hashlib.sha1((json.dumps(params, sort_keys=True) + f"#{seed}").encode()).hexdigest()

    def get(self, k):
        return self.data.get(k)

    def put(self, k, message: dict):
        self.data[k] = message
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"key": k, "message": message}) + "\n")


def _client():
    import anthropic
    return anthropic.Anthropic()          # reads ANTHROPIC_API_KEY


def _to_dict(msg) -> dict:
    return msg.model_dump() if hasattr(msg, "model_dump") else msg


def run_probes(fed: Federation, specs: Sequence[ProbeSpec], cfg: ClaudeConfig, batch: bool = True,
               model: Optional[str] = None, poll: float = 30.0, ledger: Optional[Ledger] = None
               ) -> List[Optional[str]]:
    """Send the probes (batch by default) and return the selected tool for each spec."""
    store = Store(cfg.store)
    ledger = ledger or Ledger(cfg)
    params = [probe_params(fed, s, cfg, model) for s in specs]
    keys = [Store.key(p, s.seed) for p, s in zip(params, specs)]
    todo = [i for i, k in enumerate(keys) if store.get(k) is None]
    todo_unique: Dict[str, int] = {}
    for i in todo:
        todo_unique.setdefault(keys[i], i)
    client = _client() if todo_unique else None
    if todo_unique and batch:
        idxs = list(todo_unique.values())
        for start in range(0, len(idxs), 90000):              # stay under the 100,000-request batch limit
            chunk = idxs[start:start + 90000]
            reqs = [{"custom_id": f"r{i}", "params": params[i]} for i in chunk]
            b = client.messages.batches.create(requests=reqs)
            while True:
                b = client.messages.batches.retrieve(b.id)
                if b.processing_status == "ended":
                    break
                time.sleep(poll)
            for res in client.messages.batches.results(b.id):
                i = int(res.custom_id[1:])
                if res.result.type == "succeeded":
                    m = _to_dict(res.result.message)
                    store.put(keys[i], m); ledger.add(m.get("usage", {}), batch=True)
    elif todo_unique:
        from concurrent.futures import ThreadPoolExecutor
        def call(i):
            for attempt in range(5):
                try:
                    return i, _to_dict(client.messages.create(**params[i]))
                except Exception as e:          # rate limits, overload: back off and retry
                    time.sleep(2 ** attempt)
            return i, None
        with ThreadPoolExecutor(cfg.workers) as ex:
            for i, m in ex.map(call, list(todo_unique.values())):
                if m is not None:
                    store.put(keys[i], m); ledger.add(m.get("usage", {}), batch=False)
    out = []
    for k, s in zip(keys, specs):
        m = store.get(k)
        out.append(parse_selection(m, s.K) if m is not None else None)
    return out
