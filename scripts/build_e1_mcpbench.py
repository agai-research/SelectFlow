"""Federation for the E1 pilot on real tools and real requests: the MCP-Bench tools (one organization
per server) and the MCP-Bench tasks as call occasions, described by their fuzzy description, which
does not name the tools. The assertive-cue feature is computed from the tool description."""
import json
import re
import sys

sys.path.insert(0, ".")
from selectflow import io
from selectflow.model import Descriptor

fed = io.load("data/mcpbench_federation.json")
ASSERT = re.compile(r"\b(best|most|always|must|recommended|powerful|comprehensive|official|ultimate|"
                    r"advanced|accurate|fast|reliable|complete|preferred|essential|important)\b|!", re.I)
for t in fed.tools.values():
    t.attrs["assertive"] = 1.0 if ASSERT.search(t.desc or "") else 0.0
D = {}
for f in ("single", "multi_2server", "multi_3server"):
    o = json.load(open(f"ext/mcp-bench/tasks/mcpbench_tasks_{f}_runner_format.json"))
    for st in o["server_tasks"]:
        servers = [s.strip() for s in st["server_name"].split("+")]
        for t in st["tasks"]:
            text = (t.get("fuzzy_description") or t["task_description"]).strip()
            D[t["task_id"]] = Descriptor(t["task_id"], servers[0], "task", set(), set(), {"step"},
                                         title=st["server_name"], text=text[:1500],
                                         cap_text={"step": text[:1500]}, orgs=set(servers[1:]),
                                         jurisdiction=None)
            D[t["task_id"]].record = ",".join(t.get("distraction_servers", []))
fed.descriptors = D
io.dump(fed, "data/e1_mcpbench.json")
print(len(D), "tasks,", len(fed.tools), "tools,", sum(t.attrs["assertive"] for t in fed.tools.values()), "assertive descriptions")
