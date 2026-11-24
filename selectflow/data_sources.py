"""Loaders for the real data sources of the protocol.

* Registry records (Mitos, Saudi service cards) -> descriptors, from a CSV that the
  annotators fill in (template: data/registry_template.csv).
* MCP tool definitions (MCP-Bench servers, or any server answering tools/list) -> tools.
* TOUCAN extract -> offered tools and called tool per trajectory (to train the set-level baseline).
"""
from __future__ import annotations

import csv
import json
import re
from typing import Dict, Iterable, List, Optional, Tuple

from .model import Descriptor, QoS, Server, Tool


def _split(x: str) -> set:
    return {s.strip() for s in (x or "").split(";") if s.strip()}


def descriptors_from_csv(path: str) -> Dict[str, Descriptor]:
    """Columns: id, org, title, purpose, requires, delivers, capabilities, elicited, companions,
    fee, duration_days, record, jurisdiction, text. Lists are separated by ';'.
    capabilities: 'cap-id: statement; cap-id: statement'."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            caps = {}
            for item in _split(row.get("capabilities", "")):
                k, _, stmt = item.partition(":")
                caps[k.strip()] = stmt.strip() or k.strip()
            out[row["id"]] = Descriptor(
                row["id"], row["org"], row["purpose"], _split(row.get("requires")), _split(row.get("delivers")),
                set(caps), _split(row.get("elicited")), _split(row.get("companions")),
                float(row.get("fee") or 0), float(row.get("duration_days") or 0), {}, row.get("title", ""),
                row.get("text", ""), row.get("jurisdiction") or None, caps, row.get("record") or None)
    return out


def approx_tokens(obj) -> int:
    """Rough token count (4 characters per token). Use the API token-counting endpoint for the paper."""
    return max(1, len(json.dumps(obj, ensure_ascii=False)) // 4)


def tools_from_mcp_listing(path: str, org_of_server: Dict[str, str]) -> Tuple[Dict[str, Tool], Dict[str, Server]]:
    """Input: JSON list of {server, name, description, input_schema} (e.g. collected with tools/list)."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    tools, servers = {}, {}
    for r in rows:
        s = r["server"]; org = org_of_server.get(s, "ThirdParty")
        servers.setdefault(s, Server(s, org, 0))
        tid = f"{s}.{r['name']}"
        tools[tid] = Tool(tid, s, org, r.get("description", ""), {}, approx_tokens(r),
                          QoS(1.0, 0.0), input_schema=r.get("input_schema", {"type": "object", "properties": {}}),
                          family=s)
    return tools, servers


_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def toucan_rows(rows: Iterable[dict]) -> List[dict]:
    """Offered tools and the first called tool per trajectory.

    Uses the 'available_tools' field and the assistant tool calls in 'messages'.
    The message format must be checked on the first rows of the extract before use."""
    out = []
    for r in rows:
        avail = r.get("available_tools")
        avail = json.loads(avail) if isinstance(avail, str) else (avail or [])
        names = [a.get("name") or a.get("function", {}).get("name") for a in avail]
        msgs = r.get("messages")
        msgs = json.loads(msgs) if isinstance(msgs, str) else (msgs or [])
        first, calls = None, []
        for m in msgs:
            fc = m.get("function_call") if m.get("role") == "assistant" else None
            if isinstance(fc, str):
                try:
                    fc = json.loads(fc)
                except json.JSONDecodeError:
                    fc = None
            if isinstance(fc, dict) and fc.get("name"):
                calls.append(fc["name"])
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            fc = m.get("function_call")                        # TOUCAN format: {"name", "arguments"}
            if isinstance(fc, str):
                try:
                    fc = json.loads(fc)
                except json.JSONDecodeError:
                    fc = None
            if isinstance(fc, dict) and fc.get("name"):
                first = fc["name"]
            for tc in m.get("tool_calls") or []:
                first = first or tc.get("function", {}).get("name") or tc.get("name"); break
            if first is None and isinstance(m.get("content"), str):
                hit = _TOOL_CALL.search(m["content"])
                if hit:
                    try:
                        first = json.loads(hit.group(1)).get("name")
                    except json.JSONDecodeError:
                        pass
            if first:
                break
        meta = r.get("metadata")
        meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
        servers = [s.get("server_name") for s in meta.get("mcp_servers", [])]
        out.append({"uuid": r.get("uuid"), "subset": r.get("subset_name"), "question": r.get("question"),
                    "offered": [n for n in names if n], "called": first, "calls": list(dict.fromkeys(calls)),
                    "target_tools": [t.strip() for t in (r.get("target_tools") or "").split(",") if t.strip()],
                    "servers": servers,
                    "tool_desc": {(a.get("function") or a).get("name"): (a.get("function") or a).get("description", "")
                                  for a in avail}})
    return out


def load_toucan_jsonl(path: str):
    """Reads a local TOUCAN extract (.jsonl or .jsonl.gz), e.g. one produced by load_toucan_extract."""
    import gzip
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return toucan_rows(json.loads(l) for l in f if l.strip())


def load_toucan_extract(n: int = 20000, split: str = "train", seed: int = 0):
    """Streams an extract of Agent-Ark/Toucan-1.5M from Hugging Face (needs the `datasets` package)."""
    from datasets import load_dataset
    ds = load_dataset("Agent-Ark/Toucan-1.5M", split=split, streaming=True).shuffle(seed=seed)
    rows = []
    for i, r in enumerate(ds):
        if i >= n:
            break
        rows.append(r)
    return toucan_rows(rows)
