"""Tool cards (Sect. 4.1): capability mapping repeated n times; mu_tr = share of repetitions returning r.

Run with the cheap model (Haiku) and reuse the same cards for every method."""
from __future__ import annotations

from typing import Dict, List

from ..model import Federation
from .claude import ClaudeConfig, _client, _to_dict


def map_capabilities(fed: Federation, vocabulary: Dict[str, str], cfg: ClaudeConfig, reps: int = 5,
                     model: str = None) -> Dict[str, Dict[str, float]]:
    client = _client()
    caps = sorted(vocabulary)
    schema = {"type": "object", "properties": {"capabilities": {"type": "array", "items": {"type": "string", "enum": caps}}},
              "required": ["capabilities"]}
    listing = "\n".join(f"- {c}: {vocabulary[c]}" for c in caps)
    out: Dict[str, Dict[str, float]] = {}
    for tid, t in fed.tools.items():
        counts: Dict[str, int] = {}
        for i in range(reps):
            m = _to_dict(client.messages.create(
                model=model or cfg.debug_model, max_tokens=256,
                system="Map the tool to the capabilities it provides, using only the listed capabilities.",
                tools=[{"name": "card", "description": "Return the capabilities.", "input_schema": schema}],
                tool_choice={"type": "tool", "name": "card"},
                messages=[{"role": "user", "content": f"Capabilities:\n{listing}\n\nTool: {t.desc}\nSchema: {t.input_schema}"}]))
            for b in m["content"]:
                if b["type"] == "tool_use":
                    for c in set(b["input"].get("capabilities", [])):
                        counts[c] = counts.get(c, 0) + 1
        out[tid] = {c: n / reps for c, n in counts.items()}
    return out
