"""Extract from TOUCAN (Hugging Face: Agent-Ark/Toucan-1.5M) to a local .jsonl.gz, same format as the
raw dataset rows. Needs `pip install datasets` and access to huggingface.co.

  python scripts/extract_toucan.py --n 20000 --out data/toucan_20k.jsonl.gz [--config SFT] [--subset multi-turn]
"""
import argparse
import gzip
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000); ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="dataset configuration, if the dataset defines several")
    ap.add_argument("--subset", default=None, help="keep only this subset_name (e.g. single-turn-original)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("Agent-Ark/Toucan-1.5M", a.config, split="train", streaming=True).shuffle(seed=a.seed, buffer_size=50000)
    n = 0
    with gzip.open(a.out, "wt", encoding="utf-8") as f:
        for r in ds:
            if a.subset and r.get("subset_name") != a.subset:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
            if n >= a.n:
                break
    print(n, "rows ->", a.out)


if __name__ == "__main__":
    main()
