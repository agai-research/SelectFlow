"""Builds a federation file from the MCP-Bench tool definitions (data/mcpbench_tools.json).

Each MCP-Bench server becomes one organization with one server. Token costs are approximate
(4 characters per token) until replaced by the API token-counting endpoint. Capabilities are
filled afterwards by selectflow/agent/capmap.py (Haiku, 5 repetitions)."""
import json
import sys

sys.path.insert(0, ".")
from selectflow import io
from selectflow.data_sources import tools_from_mcp_listing
from selectflow.model import Federation, ModelHost

rows = json.load(open("data/mcpbench_tools.json"))
servers = sorted({r["server"] for r in rows})
tools, srv = tools_from_mcp_listing("data/mcpbench_tools.json", {s: s for s in servers})
fed = Federation({}, tools, srv, {"hP": ModelHost("hP", "Contractor", 8000)}, set(), {}, {"ID", "BUS", "LOC", "FIN", "HLT"})
io.dump(fed, "data/mcpbench_federation.json")
print(len(tools), "tools,", len(srv), "servers ->", "data/mcpbench_federation.json")
