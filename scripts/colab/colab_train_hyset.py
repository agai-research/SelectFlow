# =============================================================================
# SelectFlow — training of the HYSET baseline (Hong et al., 2026) on Google Colab
#
# 1. Runtime > Change runtime type > T4 GPU (or better), then Save.
# 2. Paste this whole file into ONE cell and click "Run all".
# 3. Optional but recommended: put extra_tools.json and extra_queries.json in /content (Files panel).
# 4. Keep the tab open (about 1 to 3 hours). Accept the Google Drive authorization when asked:
#    data and checkpoint are saved there, so a crash or a disconnection can be resumed by running again.
# At the end the browser downloads hyset_results.zip: upload it in the conversation with Claude.
# =============================================================================
import ast, json, os, random, re, subprocess, sys, time, zipfile

N_TOUCAN = 30000          # TOUCAN trajectories used for training
EPOCHS, PATIENCE, SEED = 10, 3, 42
ENCODERS = ["ToolBench/ToolBench_IR_bert_based_uncased",       # ToolLLM retriever, paper-aligned BERT setting
            "sentence-transformers/all-mpnet-base-v2"]          # fallback, 768 dimensions as well
ROOT = "/content" if os.path.isdir("/content") else ("/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd())
if os.path.isdir("/kaggle/input"):     # Kaggle: copy the uploaded JSON files (Add data > Upload) next to the script
    import shutil
    for dp, dn, fn in os.walk("/kaggle/input"):
        for f in fn:
            if f in ("extra_tools.json", "extra_queries.json"):
                shutil.copy(os.path.join(dp, f), os.path.join(ROOT, f))
WORK = f"{ROOT}/hyset_work"; OUT = f"{ROOT}/hyset_results"
KEEP = WORK                            # prepared data and checkpoint; repositories stay on the local disk
try:                                   # persist the expensive parts on Google Drive (resume after a crash)
    from google.colab import drive
    drive.mount("/content/drive")
    KEEP = "/content/drive/MyDrive/selectflow_hyset"
except Exception:
    pass
os.makedirs(KEEP, exist_ok=True)
os.makedirs(WORK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
def sh(cmd, check=True):
    log("$", cmd); r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if r.stdout: log(r.stdout[-4000:])
    if r.returncode and check:
        log(r.stderr[-4000:]); raise RuntimeError(f"command failed: {cmd}")
    return r

# ----------------------------------------------------------------------------- 0. environment
import torch
log("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Settings > Accelerator > GPU T4 x2.")
sh(f"{sys.executable} -m pip install -q datasets sentencepiece safetensors")
os.chdir(WORK)
if not os.path.isdir("HYSET"):
    sh("git clone -q --depth 1 https://github.com/stormwther18/HYSET.git")
if not os.path.isdir("mcp-bench"):
    sh("git clone -q --depth 1 https://github.com/Accenture/mcp-bench.git")
sys.path.insert(0, f"{WORK}/HYSET/src")
from hyset_model import func_name_of

# ----------------------------------------------------------------------------- 1. our catalogue: MCP-Bench tools
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
EXTRA = f"{ROOT}/extra_tools.json"   # optional: administrative tools later supplied by Claude
if os.path.exists(EXTRA):
    ours += json.load(open(EXTRA)); log("extra tools added:", EXTRA)

# ----------------------------------------------------------------------------- 2. TOUCAN -> HYSET format
from datasets import get_dataset_config_names, load_dataset
def toucan_stream():
    try:
        configs = get_dataset_config_names("Agent-Ark/Toucan-1.5M")
    except Exception:
        configs = [None]
    log("TOUCAN configurations:", configs)
    # small shuffle buffers: four buffers of 20,000 rows (about 28 KB each) exhausted Colab's memory
    its = [iter(load_dataset("Agent-Ark/Toucan-1.5M", c, split="train", streaming=True)
                .shuffle(seed=SEED, buffer_size=1500)) for c in configs]
    while its:
        for it in list(its):
            try:
                yield next(it)
            except StopIteration:
                its.remove(it)

def parse_row(r):
    avail = r.get("available_tools"); avail = json.loads(avail) if isinstance(avail, str) else (avail or [])
    desc = {}
    for a in avail:
        f = a.get("function") or a
        if f.get("name"):
            desc[f["name"]] = f.get("description", "") or ""
    msgs = r.get("messages"); msgs = json.loads(msgs) if isinstance(msgs, str) else (msgs or [])
    calls = []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        fc = m.get("function_call")
        if isinstance(fc, str):
            try: fc = json.loads(fc)
            except ValueError: fc = None
        if isinstance(fc, dict) and fc.get("name"):
            calls.append(fc["name"])
        for tc in m.get("tool_calls") or []:
            n = (tc.get("function") or tc).get("name")
            if n: calls.append(n)
    meta = r.get("metadata"); meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
    srv = [s.get("server_name") for s in meta.get("mcp_servers", [])] or ["?"]
    return desc, list(dict.fromkeys(c for c in calls if c in desc)), srv[0], r.get("question") or ""

def split_name(n):
    i = n.rfind("-")
    return (n[:i], n[i + 1:]) if i > 0 else ("misc", n)

D = f"{KEEP}/data_selectflow"
PREPARED = os.path.exists(f"{D}/hyset_corpus.json") and os.path.exists(f"{D}/state.json")
corpus = {}   # HYSET format: function name -> entry (see HYSET/src/build_hyset_corpus.py)
def add(tool, api, text, category):
    fn = func_name_of(tool, api)
    if fn not in corpus:
        corpus[fn] = {"idx": len(corpus), "category": category, "tool_name": tool, "api_name": api,
                      "text": f"{category} {tool} {api} {text}".strip()[:2000],
                      "full_api_json": {"category_name": category, "tool_name": tool, "api_name": api,
                                        "api_description": text, "required_parameters": [],
                                        "optional_parameters": [], "method": "MCP"}}
if PREPARED:                          # resume: data already prepared in an earlier run
    corpus = json.load(open(f"{D}/hyset_corpus.json"))
    queries = json.load(open(f"{D}/instruction/G1_query.json"))
    st = json.load(open(f"{D}/state.json")); name_map, test_srv = st["name_map"], set(st["test_srv"])
    log("prepared data found: skipping the TOUCAN download")
else:
    queries = []
    for r in toucan_stream():
        desc, calls, server, q = parse_row(r)
        for n, d in desc.items():
            t, a = split_name(n); add(t, a, d, "TOUCAN")
        if calls and q and len(calls) <= 5:
            queries.append({"query_id": str(r.get("uuid") or len(queries)), "query": q,
                            "relevant APIs": [list(split_name(c)) for c in calls], "server": server})
            if len(queries) % 5000 == 0:
                log(len(queries), "TOUCAN queries collected")
        if len(queries) >= N_TOUCAN:
            break
    name_map = {}
    for t in ours:
        tool, api = t["server"].replace(" ", "_"), t["name"]
        add(tool, api, t["description"], "SelectFlow"); name_map[func_name_of(tool, api)] = t["id"]
    rnd = random.Random(SEED)
    srvs = sorted({q["server"] for q in queries}); rnd.shuffle(srvs)
    test_srv = set(srvs[: max(1, int(0.15 * len(srvs)))])
    os.makedirs(f"{D}/instruction", exist_ok=True); os.makedirs(f"{D}/test_query_ids", exist_ok=True)
    json.dump(corpus, open(f"{D}/hyset_corpus.json", "w"), ensure_ascii=False)
    json.dump(queries, open(f"{D}/instruction/G1_query.json", "w"), ensure_ascii=False)
    for g in ("G2_query.json", "G3_query.json"):
        json.dump([], open(f"{D}/instruction/{g}", "w"))
    json.dump([q["query_id"] for q in queries if q["server"] in test_srv], open(f"{D}/test_query_ids/test.json", "w"))
    json.dump({"name_map": name_map, "test_srv": sorted(test_srv)}, open(f"{D}/state.json", "w"))
log(f"corpus {len(corpus)} tools, {len(queries)} queries, {len(test_srv)} test servers")

# ----------------------------------------------------------------------------- 3. training
from transformers import AutoModel
ENC = None
for e in ENCODERS:
    try:
        AutoModel.from_pretrained(e); ENC = e; break
    except Exception as ex:
        log("encoder unavailable:", e, type(ex).__name__)
log("encoder:", ENC)
CK = f"{KEEP}/checkpoints"
t0 = time.time()
if os.path.exists(f"{CK}/best.pt"):    # resume: training already finished in an earlier run
    log("checkpoint found: skipping training")
else:
  sh(f"cd {WORK}/HYSET && {sys.executable} src/train_hyset.py --corpus {D}/hyset_corpus.json "
   f"--instruction_dir {D}/instruction --test_id_dir {D}/test_query_ids --encoder {ENC} --encoder_type bert "
   f"--d_z 768 --eta 0 --epochs {EPOCHS} --patience {PATIENCE} --seed {SEED} --checkpoint_dir {CK} --device cuda")
log(f"training time {time.time() - t0:.0f} s")
if not os.path.exists(f"{CK}/best.pt"):
    raise SystemExit("Training produced no checkpoint: send log.txt or the error shown above to Claude.")

# ----------------------------------------------------------------------------- 4. evaluation on held-out servers
test_items = [q for q in queries if q["server"] in test_srv]
os.makedirs(f"{D}/eval_instruction", exist_ok=True)
json.dump(test_items, open(f"{D}/eval_instruction/G1_query.json", "w"), ensure_ascii=False)
for g in ("G2_query.json", "G3_query.json"):
    json.dump([], open(f"{D}/eval_instruction/{g}", "w"))
sh(f"cd {WORK}/HYSET && {sys.executable} src/evaluate_hyset.py --checkpoint {CK}/best.pt --encoder {ENC} "
   f"--instruction_dir {D}/eval_instruction --test_id_dir {D}/test_query_ids --output_dir {OUT}/evaluation "
   f"--device cuda", check=False)

# ----------------------------------------------------------------------------- 5. rankings for SelectFlow's experiments
from hyset_model import HYSETModel
model = HYSETModel.load(f"{CK}/best.pt", ENC, device="cuda")
func_names = list(model.func_names)
reqs = []
for f in ("single", "multi_2server", "multi_3server"):
    o = json.load(open(f"mcp-bench/tasks/mcpbench_tasks_{f}_runner_format.json"))
    for st in o["server_tasks"]:
        for t in st["tasks"]:
            reqs.append({"id": t["task_id"], "text": (t.get("fuzzy_description") or t["task_description"]).strip()})
EXTRA_Q = f"{ROOT}/extra_queries.json"   # optional: citizen requests later supplied by Claude
if os.path.exists(EXTRA_Q):
    reqs += json.load(open(EXTRA_Q))
rankings = {}
for q in reqs:
    res = model.retrieve(q["text"], k1=15, k_pool=20, ranking_length=model.m_max).as_dict()
    ranked = list(dict.fromkeys(list(res["ranking"]) + list(res["shortlist"])))   # ranking (<= m_max), then shortlist
    rankings[q["id"]] = {"predicted_set": [func_names[i] if isinstance(i, int) else i for i in res["predicted_set"]],
                         "ranking": [func_names[i] if isinstance(i, int) else i for i in ranked],
                         "predicted_score": res.get("predicted_score")}
    for k in ("predicted_set", "ranking"):
        rankings[q["id"]][k] = [name_map.get(n, n) for n in rankings[q["id"]][k]]
json.dump(rankings, open(f"{OUT}/hyset_rankings.json", "w"), indent=1, ensure_ascii=False)
json.dump({"encoder": ENC, "n_queries": len(queries), "n_corpus": len(corpus), "n_ours": len(ours),
           "test_servers": sorted(test_srv), "epochs": EPOCHS, "seed": SEED}, open(f"{OUT}/run_info.json", "w"), indent=1)
open(f"{OUT}/log.txt", "w").write("\n".join(LOG))

# ----------------------------------------------------------------------------- 6. package
zpath = f"{ROOT}/hyset_results.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for dp, dn, fn in os.walk(OUT):
        for f in fn:
            z.write(os.path.join(dp, f), os.path.relpath(os.path.join(dp, f), OUT))
    if os.path.getsize(f"{CK}/best.pt") < 400e6:      # the checkpoint itself, if it is small enough
        z.write(f"{CK}/best.pt", "best.pt")
log("File to upload in the conversation:", zpath)
try:
    from google.colab import files
    files.download(zpath)
except ImportError:
    pass
