"""Offline smoke test of the LLM runners with a fake client (no API call)."""
from selectflow import scenario_greece as G
from selectflow.agent import claude_generator as CG
from selectflow.agent.claude import ClaudeConfig
from selectflow.experiments import runners_llm as XL


class FakeMessages:
    def create(self, **kw):
        name = kw.get("tool_choice", {}).get("name")
        if name == "interpretation":
            inp = {"targets": ["receipt"], "held": []}
        elif name == "workflow":
            inp = {"tasks": ["v1", "v3", "v4"]}          # misses the attestation: critics must repair it
        else:
            return {"content": [{"type": "text", "text": "ok"}]}
        return {"content": [{"type": "tool_use", "id": "u", "name": name, "input": inp}]}


class FakeClient:
    messages = FakeMessages()


def fake_backend(params_list):
    out = []
    for p in params_list:
        used = {b["name"] for m in p["messages"] if m["role"] == "assistant" for b in m["content"] if b.get("type") == "tool_use"}
        nxt = [t["name"] for t in p["tools"] if t["name"] not in used and not t["name"].startswith("search")]
        out.append({"content": [{"type": "tool_use", "id": f"u{len(used)}", "name": nxt[0], "input": {}}]} if nxt
                   else {"content": [{"type": "text", "text": "done"}]})
    return out


def test_e2_e6_offline(monkeypatch):
    monkeypatch.setattr(CG, "_client", lambda: FakeClient())
    fed = G.build()
    reqs = [{"id": "r1", "text": G.REQUEST, "targets": ["receipt"], "held": [], "references": [["v1", "v2", "v3", "v4"]]}]
    r2 = XL.e2(fed, reqs, ClaudeConfig(), G.PARAMS, k=1)
    assert r2["selectflow"]["closed"] == 1.0 and r2["planner"]["closed"] == 1.0
    assert r2["direct"]["closed"] == 0.0 and r2["critics_rejection_rate"] == 1.0
    r6 = XL.e6(fed, reqs, ClaudeConfig(cache=False), G.PARAMS, G.models(fed), fake_backend)
    assert r6["selectflow"]["success"] == 1.0
