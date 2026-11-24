# =============================================================================
# SelectFlow — citizen formulations (Claude) verified by a model of another family (Qwen, open weights)
#
# Before running:
#   1. Runtime > Change runtime type > T4 GPU > Save.
#   2. Left panel, key icon "Secrets": add ANTHROPIC_API_KEY (a NEW key), enable "Notebook access".
#   3. Files panel: upload selectflow_requests_input.json (given by Claude) into /content.
# Then paste this file into ONE cell and click "Run all". About 30 to 60 minutes.
# At the end the browser downloads selectflow_requests_verified.zip: upload it in the conversation.
# =============================================================================
import json, os, re, subprocess, sys, time, zipfile
ROOT = "/content"; OUT = f"{ROOT}/requests_verified"; os.makedirs(OUT, exist_ok=True)
INP = f"{ROOT}/selectflow_requests_input.json"
if not os.path.exists(INP):
    raise SystemExit("Upload selectflow_requests_input.json into /content first (Files panel).")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anthropic", "bitsandbytes", "accelerate"], check=True)
from google.colab import userdata
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
import anthropic, torch
data = json.load(open(INP))          # {"requests": [...], "procedures": {...}}
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"

# ----------------------------------------------------------------------------- 1. formulations (Claude)
SCHEMA = {"type": "object", "properties": {
    "vague_en": {"type": "string"}, "fr": {"type": "string"}, "ar": {"type": "string"}},
    "required": ["vague_en", "fr", "ar"]}
SYS = ("You write requests that citizens or business owners would type into a public-service assistant. "
       "Given an administrative goal, write: (1) vague_en: a short, informal English request that states the goal "
       "in everyday words, without the official procedure name; (2) fr: the same need in natural French; "
       "(3) ar: the same need in natural Modern Standard Arabic. Do not mention documents that are not needed. "
       "Do not add any other goal.")
form_path = f"{OUT}/formulations.json"
forms = json.load(open(form_path)) if os.path.exists(form_path) else {}
for i, rq in enumerate(data["requests"]):
    if rq["id"] in forms:
        continue
    for attempt in range(4):
        try:
            m = client.messages.create(model=MODEL, max_tokens=600, system=SYS,
                tools=[{"name": "requests", "description": "Return the three formulations.", "input_schema": SCHEMA}],
                tool_choice={"type": "tool", "name": "requests"},
                messages=[{"role": "user", "content": f"Administrative goal: {rq['goal']}\nCountry: {rq['country']}"}])
            forms[rq["id"]] = next(b.input for b in m.content if b.type == "tool_use"); break
        except Exception as e:
            print("retry", rq["id"], type(e).__name__); time.sleep(3 * (attempt + 1))
    if i % 10 == 0:
        json.dump(forms, open(form_path, "w"), ensure_ascii=False, indent=1); print(i, "/", len(data["requests"]))
json.dump(forms, open(form_path, "w"), ensure_ascii=False, indent=1)
print(len(forms), "requests formulated")

# ----------------------------------------------------------------------------- 2. verification (Qwen, another family)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
VERIFIER = None
for name in ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct"):
    try:
        kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)} if "7B" in name else {"torch_dtype": torch.float16}
        tok = AutoTokenizer.from_pretrained(name)
        llm = AutoModelForCausalLM.from_pretrained(name, device_map="auto", **kw)
        VERIFIER = name; break
    except Exception as e:
        print("verifier unavailable:", name, type(e).__name__)
print("verifier:", VERIFIER)

def ask(prompt):
    msgs = [{"role": "system", "content": "You are a strict verifier. Answer with JSON only."},
            {"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(llm.device)
    out = llm.generate(ids, max_new_tokens=120, do_sample=False)
    txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else {"raw": txt}
    except ValueError:
        return {"raw": txt}

checks = {}
for rq in data["requests"]:
    f = forms.get(rq["id"])
    if not f:
        continue
    checks[rq["id"]] = {}
    for k in ("vague_en", "fr", "ar"):
        checks[rq["id"]][k] = ask(
            f"Goal: {rq['goal']}\nRequest written by a citizen: {f[k]}\n"
            'Does the request express exactly this goal, with no other goal added? '
            'Return {"faithful": true or false, "reason": "<10 words>"}')
proc_checks = {}
for pid, p in data["procedures"].items():
    proc_checks[pid] = ask(
        f"Administrative procedure: {p['title']}\nDescription: {p['text'][:800]}\n"
        f"Annotated purpose: {p['purpose']}\nAnnotated personal-data categories the applicant provides: {p['categories']}\n"
        "(ID identity, BUS business, LOC premises or location, FIN financial or tax, HLT health)\n"
        'Are the purpose and the categories correct? Return {"purpose_ok": true or false, "categories_ok": true or false, '
        '"missing_categories": [...], "wrong_categories": [...]}')
json.dump({"verifier": VERIFIER, "formulation_checks": checks, "procedure_checks": proc_checks},
          open(f"{OUT}/verification.json", "w"), ensure_ascii=False, indent=1)
ok = [v.get("faithful") for c in checks.values() for v in c.values()]
print("faithful formulations:", sum(1 for x in ok if x is True), "/", len(ok))
zpath = f"{ROOT}/selectflow_requests_verified.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in os.listdir(OUT):
        z.write(f"{OUT}/{fn}", fn)
from google.colab import files
files.download(zpath)
