"""Training data for HYSET (and COLT) from a TOUCAN extract, plus the corpus entries of our catalogues.

  python scripts/toucan_to_hyset.py --toucan data/toucan_sample.jsonl.gz --fed data/mcpbench_federation.json \
         --fed data/greece.json --out ext/HYSET/data_selectflow

Writes hyset_corpus.json (tools), instruction/G1_query.json (query -> set of called tools),
test_query_ids/test.json (15% of the queries, split by server) and name_map.json (our tool id -> corpus name).
Train with: ./train_hyset.sh --corpus <out>/hyset_corpus.json --instruction_dir <out>/instruction
            --test_id_dir <out>/test_query_ids --encoder $HYSET_ENCODER_PATH --eta 0 ...
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, ".")
from selectflow import io
from selectflow.data_sources import load_toucan_jsonl


def split_name(n):
    i = n.rfind("-")
    return (n[:i], n[i + 1:]) if i > 0 else ("misc", n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toucan", required=True); ap.add_argument("--fed", action="append", default=[])
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rows = load_toucan_jsonl(a.toucan)
    corpus, index, name_map = [], {}, {}

    def add(tool, api, desc, category):
        key = (tool, api)
        if key not in index:
            index[key] = len(corpus)
            corpus.append({"idx": len(corpus), "category": category, "tool_name": tool, "api_name": api,
                           "text": f"{category} {tool} {api} {desc}".strip(),
                           "full_api_json": {"category_name": category, "tool_name": tool, "api_name": api,
                                             "api_description": desc, "required_parameters": [],
                                             "optional_parameters": [], "method": "MCP"}})
        return key

    queries = []
    for r in rows:
        for n in r["offered"]:
            t, api = split_name(n); add(t, api, r["tool_desc"].get(n, "") or "", "TOUCAN")
        rel = [list(split_name(n)) for n in r["calls"] if n in r["offered"]]
        if rel and r["question"]:
            queries.append({"query_id": r["uuid"], "query": r["question"], "relevant APIs": rel,
                            "server": (r["servers"] or ["?"])[0]})
    for path in a.fed:                       # our catalogues: retrievable at inference time
        fed = io.load(path)
        for tid, T in fed.tools.items():
            k = add(T.server.replace(" ", "_"), tid.split(".")[-1], T.desc, "SelectFlow")
            name_map[tid] = list(k)
    rnd = random.Random(a.seed)
    servers = sorted({q["server"] for q in queries}); rnd.shuffle(servers)
    test_servers = set(servers[: max(1, int(0.15 * len(servers)))])
    test_ids = [q["query_id"] for q in queries if q["server"] in test_servers]
    os.makedirs(os.path.join(a.out, "instruction"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "test_query_ids"), exist_ok=True)
    json.dump(corpus, open(os.path.join(a.out, "hyset_corpus.json"), "w"), ensure_ascii=False)
    json.dump(queries, open(os.path.join(a.out, "instruction", "G1_query.json"), "w"), ensure_ascii=False)
    for g in ("G2_query.json", "G3_query.json"):
        json.dump([], open(os.path.join(a.out, "instruction", g), "w"))
    json.dump(test_ids, open(os.path.join(a.out, "test_query_ids", "test.json"), "w"))
    json.dump(name_map, open(os.path.join(a.out, "name_map.json"), "w"), ensure_ascii=False)
    print(len(corpus), "corpus entries,", len(queries), "queries,", len(test_ids), "test queries,",
          len(name_map), "SelectFlow tools mapped")


if __name__ == "__main__":
    main()
