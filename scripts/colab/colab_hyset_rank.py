# =============================================================================
# SelectFlow — HYSET rankings restricted to OUR tools + evaluation on held-out servers.
# Works in the same session as the training, or in a new GPU session:
# in a new session, first upload best.pt (from hyset_results.zip) with the Files panel, into /content.
# Paste into ONE cell and run it. At the end the browser downloads hyset_rankings_v2.zip.
# =============================================================================
import json, os, re, ast, subprocess, sys, zipfile
import torch
ROOT = "/content"; WORK = f"{ROOT}/hyset_work"; OUT = f"{ROOT}/hyset_results_v2"
os.makedirs(WORK, exist_ok=True); os.makedirs(OUT, exist_ok=True); os.chdir(WORK)
if not torch.cuda.is_available():
    raise SystemExit("No GPU: Runtime > Change runtime type > T4 GPU, then run again.")
for repo, url in (("HYSET", "https://github.com/stormwther18/HYSET.git"), ("mcp-bench", "https://github.com/Accenture/mcp-bench.git")):
    if not os.path.isdir(repo):
        subprocess.run(f"git clone -q --depth 1 {url}", shell=True, check=True)
sys.path.insert(0, f"{WORK}/HYSET/src")
from hyset_model import HYSETModel, func_name_of
CKPT = next((p for p in (f"{WORK}/checkpoints/best.pt", f"{ROOT}/best.pt") if os.path.exists(p)), None)
if CKPT is None:
    raise SystemExit("best.pt not found: upload it into /content (Files panel on the left), then run again.")
ENC = "ToolBench/ToolBench_IR_bert_based_uncased"
model = HYSETModel.load(CKPT, ENC, device="cuda")
S = model.scorer
print("checkpoint:", CKPT, "| tools in the model:", len(model.func_names))

# ---------------------------------------------------------------- our tools (same extraction as training)

TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "list": "array", "dict": "object"}
TS_CALL = re.compile(r"""(?:\w*[Ss]erver|this\.server)\s*\.\s*(?:tool|registerTool)\(\s*['"`]([\w.-]+)['"`]\s*,\s*(?:\{\s*(?:title\s*:\s*['"`][^'"`]*['"`]\s*,\s*)?description\s*:\s*)?['"`]([^'"`]{5,600})['"`]""", re.S)
TS_LIST = re.compile(r"""name\s*:\s*['"`]([\w.-]+)['"`]\s*,\s*description\s*:\s*['"`]([^'"`]{5,600})['"`]""", re.S)
PY_TOOLCLS = re.compile(r"""Tool\(\s*name\s*=\s*['"]([\w.-]+)['"]\s*,\s*description\s*=\s*(?:\(\s*)?['"]([^'"]{5,600})['"]""", re.S)

def py_tools(path):
    src = open(path, encoding="utf-8", errors="ignore").read(); out = []
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    f = dec.func if isinstance(dec, ast.Call) else dec
                    if isinstance(f, ast.Attribute) and f.attr == "tool":
                        name, desc = node.name, ast.get_docstring(node) or ""
                        if isinstance(dec, ast.Call):
                            for kw in dec.keywords:
                                if kw.arg == "name" and isinstance(kw.value, ast.Constant): name = kw.value.value
                                if kw.arg == "description" and isinstance(kw.value, ast.Constant): desc = kw.value.value
                        out.append((name, " ".join(desc.split())[:1000]))
    except SyntaxError:
        pass
    if "Tool(" in src:
        out += [(n, " ".join(d.split())) for n, d in PY_TOOLCLS.findall(src)]
    return out

def ts_tools(path):
    src = open(path, encoding="utf-8", errors="ignore").read()
    hits = TS_CALL.findall(src) or (TS_LIST.findall(src) if "inputSchema" in src or "input_schema" in src else [])
    return [(n, " ".join(d.split())) for n, d in hits]

servers = json.load(open("mcp-bench/mcp_servers/commands.json"))
ours = []
for server, spec in servers.items():
    cwd = os.path.normpath(os.path.join("mcp-bench/mcp_servers", spec.get("cwd", "").replace("../", "")))
    seen = {}
    for dp, dn, fn in os.walk(cwd):
        if any(x in dp for x in ("node_modules", "test", "example", "dist", "build", "website", "weather_agent", "legacy")):
            continue
        for f in fn:
            p = os.path.join(dp, f)
            for n, d in (py_tools(p) if f.endswith(".py") else ts_tools(p) if f.endswith((".ts", ".js")) else []):
                seen.setdefault(n, d)
    ours += [{"id": f"{server}.{n}", "server": server, "name": n, "description": d} for n, d in seen.items()]
log(len(ours), "MCP-Bench tools in our catalogue")

EXTRA = f"{ROOT}/extra_tools.json"
if os.path.exists(EXTRA):
    ours += json.load(open(EXTRA)); print("administrative tools added")
idx_of = {n: i for i, n in enumerate(model.func_names)}
ours_idx, back = [], {}
for t in ours:
    fn = func_name_of(t["server"].replace(" ", "_"), t["name"])
    if fn in idx_of:
        ours_idx.append(idx_of[fn]); back[idx_of[fn]] = t["id"]
print(len(ours_idx), "of our", len(ours), "tools found in the model")
OURS = torch.tensor(ours_idx, device="cuda")

@torch.no_grad()
def retrieve_ours(text, k1=15, k_pool=20):
    """HYSET's retrieval (shortlist, best set, greedy ranking) restricted to our catalogue."""
    q = model.encode_texts([text])[0]
    single = S.singleton_scores(q)
    mask = torch.full_like(single, -float("inf")); mask[OURS] = single[OURS]
    k1 = min(k1, len(ours_idx) - 1); k_pool = min(k_pool, len(ours_idx))
    s0 = torch.topk(mask, k=k1).indices
    Z = S.Z.weight; M = S.interaction_matrix(S.m_max)
    comp = ((Z @ M) @ Z[s0].T).max(dim=1).values
    cm = torch.full_like(comp, -float("inf")); cm[OURS] = comp[OURS]; cm[s0] = -float("inf")
    shortlist = tuple(s0.tolist() + torch.topk(cm, k=k_pool - k1).indices.tolist())
    best, score = S.predict_set(q, shortlist)
    ranking = S.greedy_ranking(q, shortlist, length=min(S.m_max, len(shortlist)))
    rest = [i for i in torch.argsort(mask, descending=True).tolist() if i in back]
    order = list(dict.fromkeys(list(ranking) + list(shortlist) + rest))
    return {"predicted_set": [back[i] for i in best], "ranking": [back[i] for i in order], "predicted_score": float(score)}

reqs = []
for f in ("single", "multi_2server", "multi_3server"):
    o = json.load(open(f"mcp-bench/tasks/mcpbench_tasks_{f}_runner_format.json"))
    for st in o["server_tasks"]:
        for t in st["tasks"]:
            reqs.append({"id": t["task_id"], "text": (t.get("fuzzy_description") or t["task_description"]).strip(),
                         "servers": [s.strip() for s in st["server_name"].split("+")]})
EXTRA_Q = f"{ROOT}/extra_queries.json"
if os.path.exists(EXTRA_Q):
    reqs += [dict(q, servers=[]) for q in json.load(open(EXTRA_Q))]; print("administrative queries added")
rankings, hit = {}, 0
for q in reqs:
    rankings[q["id"]] = retrieve_ours(q["text"])
    hit += any(t.split(".")[0] in q["servers"] for t in rankings[q["id"]]["ranking"][:5])
json.dump(rankings, open(f"{OUT}/hyset_rankings.json", "w"), indent=1, ensure_ascii=False)
info = {"checkpoint": CKPT, "encoder": ENC, "our_tools_in_model": len(ours_idx), "tasks": len(reqs),
        "top5_contains_task_server": hit / len(reqs)}

# ---------------------------------------------------------------- held-out evaluation (if the training files are still here)
D = f"{WORK}/data_selectflow"
if os.path.exists(f"{D}/instruction/G1_query.json"):
    items = json.load(open(f"{D}/instruction/G1_query.json"))
    test = set(json.load(open(f"{D}/test_query_ids/test.json")))
    items = [it for it in items if it["query_id"] in test]
    r5 = pr = rc = ex = n = 0
    for it in items:
        gold = {func_name_of(a, b) for a, b in it["relevant APIs"]} & set(idx_of)
        if not gold:
            continue
        res = model.retrieve(it["query"], k1=15, k_pool=20, ranking_length=S.m_max).as_dict()
        pred = set(res["predicted_set"]); top5 = set(res["ranking"][:5])
        r5 += len(gold & top5) / len(gold); n += 1
        pr += len(gold & pred) / max(1, len(pred)); rc += len(gold & pred) / len(gold); ex += pred == gold
    info.update({"test_queries": n, "recall@5": r5 / n, "set_precision": pr / n, "set_recall": rc / n, "exact_set": ex / n})
else:
    info["test_evaluation"] = "skipped: training files not in this session"
json.dump(info, open(f"{OUT}/run_info.json", "w"), indent=1)
print(json.dumps(info, indent=1))
zpath = f"{ROOT}/hyset_rankings_v2.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for f in os.listdir(OUT):
        z.write(f"{OUT}/{f}", f)
from google.colab import files
files.download(zpath)
