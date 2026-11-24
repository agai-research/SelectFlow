"""Tool cards from the independent descriptions (version 3): 5 mappings per tool with the cheap model.
Resumable; saves after every tool.   python scripts/capmap_v3.py [n_tools]"""
import json, os, sys, threading, time
sys.path.insert(0, ".")
from concurrent.futures import ThreadPoolExecutor
import anthropic
from selectflow import io
c = anthropic.Anthropic(); OUT = "data/v3/capability_mapping.json"; lock = threading.Lock()
done = json.load(open(OUT)) if os.path.exists(OUT) else {}
jobs = []
for name in ("greece", "saudi"):
    fed = io.load(f"data/v3/{name}.json")
    for tid, T in fed.tools.items():
        if tid not in done:
            vocab = {r: d.cap_text[r] for d in fed.descriptors.values() if T.org in ("ThirdParty", d.org) for r in d.R}
            jobs.append((tid, T.desc, vocab))
jobs = jobs[: int(sys.argv[1]) if len(sys.argv) > 1 else 60]

def call(desc, vocab):
    caps = sorted(vocab)
    schema = {"type": "object", "properties": {"capabilities": {"type": "array", "items": {"type": "string", "enum": caps}}}, "required": ["capabilities"]}
    listing = "\n".join(f"- {k}: {vocab[k]}" for k in caps)
    for a in range(3):
        try:
            m = c.messages.create(model="claude-haiku-5-5", max_tokens=300, thinking={"type": "disabled"},
                system="Map the tool to the capabilities it provides, using only the listed capabilities. A tool usually provides one.",
                tools=[{"name": "card", "description": "Return the capabilities.", "input_schema": schema}], tool_choice={"type": "tool", "name": "card"},
                messages=[{"role": "user", "content": f"Capabilities:\n{listing}\n\nTool description: {desc}"}])
            return next((set(b.input.get("capabilities", [])) for b in m.content if b.type == "tool_use"), set())
        except Exception:
            time.sleep(2 * (a + 1))
    return set()

def one(j):
    tid, desc, vocab = j
    counts = {}
    with ThreadPoolExecutor(5) as ex:
        for s in ex.map(lambda _: call(desc, vocab), range(5)):
            for k in s: counts[k] = counts.get(k, 0) + 1
    with lock:
        done[tid] = {k: v / 5 for k, v in counts.items()}
        json.dump(done, open(OUT, "w"), indent=1)

with ThreadPoolExecutor(6) as ex:
    list(ex.map(one, jobs))
print(len(done), "of 308 tools mapped")
