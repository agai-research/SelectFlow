"""Static extraction of tool definitions from the MCP-Bench servers (no server started).

Python servers: functions decorated with @<x>.tool(...) (FastMCP style), description from the
decorator or the docstring, input schema from the signature. TypeScript/JavaScript servers:
server.tool("name", "description", ...) / registerTool and {name: "...", description: "..."} tool lists.
The extraction is approximate; `tools/list` on running servers is the reference (MCP-Bench connector).
"""
import ast
import json
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "mcp-bench"
SERVERS = json.load(open(os.path.join(ROOT, "mcp_servers", "commands.json")))
TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "list": "array", "dict": "object"}


def py_tools(path):
    out = []
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            f = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(f, ast.Attribute) and f.attr == "tool":
                name, desc = node.name, ast.get_docstring(node) or ""
                if isinstance(dec, ast.Call):
                    for kw in dec.keywords:
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant): name = kw.value.value
                        if kw.arg == "description" and isinstance(kw.value, ast.Constant): desc = kw.value.value
                    if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
                        name = dec.args[0].value
                props, req = {}, []
                args = node.args.args
                defaults = [None] * (len(args) - len(node.args.defaults)) + list(node.args.defaults)
                for a, d in zip(args, defaults):
                    if a.arg in ("self", "ctx", "context"):
                        continue
                    t = a.annotation.id if isinstance(a.annotation, ast.Name) else "string"
                    props[a.arg] = {"type": TYPES.get(t, "string")}
                    if d is None:
                        req.append(a.arg)
                out.append({"name": name, "description": " ".join(desc.split())[:1000],
                            "input_schema": {"type": "object", "properties": props, "required": req}})
    return out


TS_CALL = re.compile(r"""(?:\w*[Ss]erver|this\.server)\s*\.\s*(?:tool|registerTool)\(\s*['"`]([\w.-]+)['"`]\s*,\s*(?:\{\s*(?:title\s*:\s*['"`][^'"`]*['"`]\s*,\s*)?description\s*:\s*)?['"`]([^'"`]{5,600})['"`]""", re.S)
TS_LIST = re.compile(r"""name\s*:\s*['"`]([\w.-]+)['"`]\s*,\s*description\s*:\s*['"`]([^'"`]{5,600})['"`]""", re.S)


PY_TOOLCLS = re.compile(r"""Tool\(\s*name\s*=\s*(?:[\w.]+\.)?(?:value|['"]([\w.-]+)['"])[^)]*?description\s*=\s*(?:\(\s*)?['"]([^'"]{5,600})['"]""", re.S)
PY_TOOLCLS2 = re.compile(r"""Tool\(\s*name\s*=\s*['"]([\w.-]+)['"]\s*,\s*description\s*=\s*(?:\(\s*)?['"]([^'"]{5,600})['"]""", re.S)


def py_toolcls(path):
    src = open(path, encoding="utf-8", errors="ignore").read()
    if "Tool(" not in src:
        return []
    return [{"name": n, "description": " ".join(d.split()), "input_schema": {"type": "object", "properties": {}}}
            for n, d in PY_TOOLCLS2.findall(src)]


def ts_tools(path):
    src = open(path, encoding="utf-8", errors="ignore").read()
    hits = TS_CALL.findall(src) or (TS_LIST.findall(src) if "inputSchema" in src or "input_schema" in src else [])
    return [{"name": n, "description": " ".join(d.split()), "input_schema": {"type": "object", "properties": {}}}
            for n, d in hits]


rows = []
for server, spec in SERVERS.items():
    cwd = os.path.normpath(os.path.join(ROOT, "mcp_servers", spec.get("cwd", "").replace("../", "")))
    seen = {}
    for dp, dn, fn in os.walk(cwd):
        if any(x in dp for x in ("node_modules", "test", "example", "dist", "build", "website", "weather_agent", "legacy")):
            continue
        for f in fn:
            p = os.path.join(dp, f)
            ts = (py_tools(p) + py_toolcls(p)) if f.endswith(".py") else ts_tools(p) if f.endswith((".ts", ".js")) else []
            for t in ts:
                seen.setdefault(t["name"], t)
    for t in seen.values():
        rows.append({"server": server, **t})
json.dump(rows, open(sys.argv[2] if len(sys.argv) > 2 else "mcpbench_tools.json", "w"), indent=1, ensure_ascii=False)
by = {}
for r in rows:
    by[r["server"]] = by.get(r["server"], 0) + 1
print(len(rows), "tools from", len(by), "of", len(SERVERS), "servers")
print({s: by.get(s, 0) for s in SERVERS})
