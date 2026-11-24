"""Exporters for the two real registries. Run on a machine with normal Internet access
(the sandbox used to write this code cannot reach these sites).

  python scripts/export_registries.py mitos --category "Category:Businessmen" --out data/raw_mitos.json
  python scripts/export_registries.py saudi --urls data/saudi_urls.txt --out data/raw_saudi.json
  python scripts/export_registries.py draft --raw data/raw_mitos.json --out data/mitos_draft.csv

The draft CSV follows data/registry_template.csv; annotators then name the artefacts
(requires/delivers), the capabilities and the elicited data categories.
"""
import argparse
import csv
import json
import re
import time

LABELS_MITOS = ["Competent body", "Fees", "Deadline of completion", "Validity duration",
                "Number of required documents", "Description", "Official title", "Provided to"]
LABELS_SAUDI = ["Steps", "Conditions", "Required Documents", "Service Requirements", "Service Fees", "Fees",
                "Execution Duration", "Service Duration", "Duration", "Service Channels", "Beneficiaries",
                "Target Category", "Service Level Agreement"]


def fields_from_text(text: str, labels):
    """Value of each label = the text between the label and the next label."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out, cur = {}, None
    for l in lines:
        hit = next((lab for lab in labels if l.lower() == lab.lower()), None)
        if hit:
            cur = hit; out.setdefault(cur, [])
        elif cur:
            out[cur].append(l)
    return {k: " ".join(v)[:2000] for k, v in out.items()}


def relevant_procedures(text: str):
    """Mitos documents that are issued by another procedure: 'Relevant Procedure: <title>'."""
    return sorted({m.strip() for m in re.findall(r"Relevant Procedure:\s*\n?\s*([^\n]+)", text)})


def html_text(html: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer"]):
        t.decompose()
    return soup.get_text("\n")


def mitos(category: str, out: str, api="https://en.mitos.gov.gr/api.php", pause=1.0):
    import requests
    s = requests.Session(); titles, cont = [], {}
    while True:
        r = s.get(api, params={"action": "query", "list": "categorymembers", "cmtitle": category,
                               "cmlimit": 500, "format": "json", **cont}).json()
        titles += [m["title"] for m in r["query"]["categorymembers"]]
        if "continue" not in r:
            break
        cont = r["continue"]
    recs = []
    for t in titles:
        if not t.startswith("ΔΔ:"):
            continue
        html = s.get(api, params={"action": "parse", "page": t, "prop": "text", "format": "json"}).json()
        text = html_text(html.get("parse", {}).get("text", {}).get("*", ""))
        rid = re.search(r"\b(\d{6})\b", text)
        recs.append({"title": t.replace("ΔΔ:", ""), "record": rid.group(1) if rid else None,
                     "fields": fields_from_text(text, LABELS_MITOS), "relevant_procedures": relevant_procedures(text)})
        time.sleep(pause)
    json.dump(recs, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(recs), "records ->", out)


def saudi(urls_file: str, out: str, pause=1.0):
    import requests
    recs = []
    for u in [l.strip() for l in open(urls_file) if l.strip()]:
        text = html_text(requests.get(u, timeout=30).text)
        title = next((l.strip() for l in text.splitlines() if len(l.strip()) > 8), u)
        recs.append({"url": u, "title": title, "fields": fields_from_text(text, LABELS_SAUDI)})
        time.sleep(pause)
    json.dump(recs, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(recs), "service cards ->", out)


def days(x: str) -> str:
    m = re.search(r"(\d+)\s*(?:-\s*(\d+)\s*)?(day|working day|hour|month)", x or "", re.I)
    if not m:
        return ""
    n = float(m.group(2) or m.group(1)); unit = m.group(3).lower()
    return str(n / 24 if unit == "hour" else n * 30 if unit == "month" else n)


def draft(raw: str, out: str):
    recs = json.load(open(raw, encoding="utf-8"))
    cols = ["id", "org", "title", "purpose", "requires", "delivers", "capabilities", "elicited", "companions",
            "fee", "duration_days", "record", "jurisdiction", "text"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols); w.writeheader()
        for i, r in enumerate(recs):
            F = r["fields"]
            fee = F.get("Fees") or F.get("Service Fees") or ""
            w.writerow({"id": f"d{i}", "org": F.get("Competent body", ""), "title": r["title"], "purpose": "",
                        "requires": "; ".join(r.get("relevant_procedures", [])),   # to be renamed as artefacts
                        "delivers": "", "capabilities": "", "elicited": "", "companions": "",
                        "fee": "0" if "without cost" in fee.lower() or fee.lower().startswith(("free", "none")) else "",
                        "duration_days": days(F.get("Deadline of completion") or F.get("Execution Duration")
                                              or F.get("Service Duration") or F.get("Duration")),
                        "record": r.get("record") or r.get("url", ""), "jurisdiction": "",
                        "text": (F.get("Description") or F.get("Steps") or "")[:500]})
    print("draft ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["mitos", "saudi", "draft"])
    ap.add_argument("--category", default="Category:Businessmen"); ap.add_argument("--urls")
    ap.add_argument("--raw"); ap.add_argument("--out", required=True)
    ap.add_argument("--api", default="https://en.mitos.gov.gr/api.php", help="MediaWiki API endpoint of Mitos")
    a = ap.parse_args()
    {"mitos": lambda: mitos(a.category, a.out, api=a.api), "saudi": lambda: saudi(a.urls, a.out),
     "draft": lambda: draft(a.raw, a.out)}[a.source]()
