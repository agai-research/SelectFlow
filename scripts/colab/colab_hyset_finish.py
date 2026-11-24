# =============================================================================
# SelectFlow — finish the HYSET run WITHOUT retraining.
# Paste into a NEW cell of the SAME Colab session (do not restart it) and run only this cell.
# =============================================================================
import json, os, zipfile
for v in ("CK", "ENC", "OUT", "ROOT", "WORK", "name_map", "queries", "corpus", "ours", "test_srv", "LOG"):
    if v not in globals():
        raise SystemExit("The session was reset: rerun the full colab_train_hyset.py (corrected version).")
if not os.path.exists(f"{CK}/best.pt"):
    raise SystemExit("No checkpoint found: rerun the full colab_train_hyset.py (corrected version).")

from hyset_model import HYSETModel
model = HYSETModel.load(f"{CK}/best.pt", ENC, device="cuda")
reqs = []
for f in ("single", "multi_2server", "multi_3server"):
    o = json.load(open(f"{WORK}/mcp-bench/tasks/mcpbench_tasks_{f}_runner_format.json"))
    for st in o["server_tasks"]:
        for t in st["tasks"]:
            reqs.append({"id": t["task_id"], "text": (t.get("fuzzy_description") or t["task_description"]).strip()})
rankings = {}
for q in reqs:
    res = model.retrieve(q["text"], k1=15, k_pool=20, ranking_length=model.m_max).as_dict()
    ranked = list(dict.fromkeys(list(res["ranking"]) + list(res["shortlist"])))   # ranking first, then shortlist
    rankings[q["id"]] = {"predicted_set": [name_map.get(n, n) for n in res["predicted_set"]],
                         "ranking": [name_map.get(n, n) for n in ranked],
                         "predicted_score": res.get("predicted_score")}
json.dump(rankings, open(f"{OUT}/hyset_rankings.json", "w"), indent=1, ensure_ascii=False)
json.dump({"encoder": ENC, "n_queries": len(queries), "n_corpus": len(corpus), "n_ours": len(ours),
           "test_servers": sorted(test_srv), "m_max": model.m_max}, open(f"{OUT}/run_info.json", "w"), indent=1)
LOG.append(f"rankings for {len(rankings)} MCP-Bench tasks")
open(f"{OUT}/log.txt", "w").write("\n".join(LOG))
print("evaluation files:", os.listdir(f"{OUT}/evaluation") if os.path.isdir(f"{OUT}/evaluation") else "none")

zpath = f"{ROOT}/hyset_results.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for dp, dn, fn in os.walk(OUT):
        for f in fn:
            z.write(os.path.join(dp, f), os.path.relpath(os.path.join(dp, f), OUT))
    if os.path.getsize(f"{CK}/best.pt") < 400e6:
        z.write(f"{CK}/best.pt", "best.pt")
print("File to upload in the conversation:", zpath)
from google.colab import files
files.download(zpath)
