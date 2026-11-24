# =============================================================================
# SelectFlow — export of the Mitos (Greece) and Saudi service records
# Paste into ONE Colab cell and click "Run all". Nothing to upload.
# At the end, the browser downloads selectflow_registries.zip:
# upload this single file in the conversation with Claude.
# =============================================================================
import csv, io, json, os, re, sys, time, zipfile, subprocess
from urllib.parse import quote, unquote

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "requests", "beautifulsoup4"], check=True)
    import requests
    from bs4 import BeautifulSoup

OUT = "/content/selectflow_registries" if os.path.isdir("/content") else "selectflow_registries"
os.makedirs(OUT, exist_ok=True)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

S = requests.Session()
S.headers["User-Agent"] = "SelectFlow-research/1.0 (University of Jendouba; academic use)"

def get(url, **kw):
    for attempt in range(4):
        try:
            r = S.get(url, timeout=40, **kw)
            if r.status_code == 200:
                return r
            log("  HTTP", r.status_code, url[:120])
        except Exception as e:
            log("  error", type(e).__name__, url[:120])
        time.sleep(2 * (attempt + 1))
    return None

def html_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "noscript"]):
        t.decompose()
    return re.sub(r"\n\s*\n+", "\n", soup.get_text("\n"))

def fields_from_text(text, labels):
    out, cur = {}, None
    for l in [x.strip() for x in text.splitlines() if x.strip()]:
        hit = next((lab for lab in labels if l.lower() == lab.lower()), None)
        if l.startswith("Relevant Procedure"):
            cur = None; continue
        if hit:
            cur = hit; out.setdefault(cur, [])
        elif cur:
            out[cur].append(l)
    return {k: " ".join(v)[:2000] for k, v in out.items()}

def days(x):
    m = re.search(r"(\d+)\s*(?:-\s*(\d+)\s*)?(working day|day|hour|month)", x or "", re.I)
    if not m:
        return ""
    n = float(m.group(2) or m.group(1)); u = m.group(3).lower()
    return str(n / 24 if u == "hour" else n * 30 if u == "month" else n)

# ----------------------------------------------------------------------------- Mitos
BASE = "https://en.mitos.gov.gr"
LABELS_MITOS = ["Competent body", "Fees", "Deadline of completion", "Validity duration",
                "Number of required documents", "Description", "Official title", "Provided to",
                "Required documents", "Steps", "Notes"]
CATEGORIES = ["Category:Businessmen"]
KEYWORDS = ["shop of sanitary interest", "restaurant", "food service", "notification of establishment",
            "fire protection", "tax registration", "commencement of operations", "commercial registry",
            "business licence", "health certificate", "employer registration", "building permit"]
MAX_MITOS = 400

def find_api():
    for api in (BASE + "/api.php", BASE + "/w/api.php", BASE + "/index.php/api.php"):
        r = get(api, params={"action": "query", "meta": "siteinfo", "format": "json"})
        if r is not None:
            try:
                if "query" in r.json():
                    log("Mitos API found:", api); return api
            except ValueError:
                pass
    log("Mitos API not found: falling back to HTML pages"); return None

def titles_from_api(api):
    titles = []
    for cat in CATEGORIES:
        cont = {}
        while True:
            r = get(api, params={"action": "query", "list": "categorymembers", "cmtitle": cat,
                                 "cmlimit": 500, "format": "json", **cont})
            if r is None:
                break
            j = r.json(); titles += [m["title"] for m in j.get("query", {}).get("categorymembers", [])]
            if "continue" not in j:
                break
            cont = j["continue"]
    for kw in KEYWORDS:
        r = get(api, params={"action": "query", "list": "search", "srsearch": kw, "srlimit": 50, "format": "json"})
        if r is not None:
            titles += [m["title"] for m in r.json().get("query", {}).get("search", [])]
    return titles

def titles_from_html():
    titles = []
    for cat in CATEGORIES:
        r = get(BASE + "/index.php/" + quote(cat))
        if r is None:
            continue
        for a in BeautifulSoup(r.text, "html.parser").find_all("a", href=True):
            t = unquote(a["href"].split("/index.php/")[-1]).replace("_", " ")
            if t.startswith("ΔΔ:"):
                titles.append(t)
    return titles

def page_text(api, title):
    if api:
        r = get(api, params={"action": "parse", "page": title, "prop": "text", "format": "json"})
        if r is not None:
            try:
                return html_text(r.json()["parse"]["text"]["*"])
            except (KeyError, ValueError):
                pass
    r = get(BASE + "/index.php/" + quote(title.replace(" ", "_")))
    return html_text(r.text) if r is not None else ""

def export_mitos():
    api = find_api()
    titles = titles_from_api(api) if api else titles_from_html()
    titles = [t for t in dict.fromkeys(titles) if t.startswith("ΔΔ:")][:MAX_MITOS]
    log(len(titles), "Mitos procedures to fetch")
    recs = []
    for i, t in enumerate(titles, 1):
        text = page_text(api, t)
        if not text:
            continue
        rid = re.search(r"\b(\d{6})\b", text)
        recs.append({"title": t.replace("ΔΔ:", "").strip(), "record": rid.group(1) if rid else None,
                     "url": BASE + "/index.php/" + quote(t.replace(" ", "_")),
                     "fields": fields_from_text(text, LABELS_MITOS),
                     "relevant_procedures": sorted({m.strip() for m in re.findall(r"Relevant Procedure:\s*\n?\s*([^\n]+)", text)}),
                     "full_text": text[:15000]})
        if i % 25 == 0:
            log("  ", i, "/", len(titles)); json.dump(recs, open(f"{OUT}/raw_mitos.json", "w", encoding="utf-8"), ensure_ascii=False)
        time.sleep(1.0)
    json.dump(recs, open(f"{OUT}/raw_mitos.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(len(recs), "Mitos records saved")
    return recs

# ----------------------------------------------------------------------------- Saudi Arabia
SAUDI_URLS = [
    "https://mc.gov.sa/en/eservices/Pages/ServiceDetails.aspx?sID=38",
    "https://mc.gov.sa/en/eservices/Pages/ServiceDetails.aspx?sID=2",
    "https://mc.gov.sa/en/eservices/Pages/ServiceDetails.aspx?sID=21",
    "https://mc.gov.sa/en/eservices/Pages/ServiceDetails.aspx?sID=52",
    "https://www.balady.gov.sa/en/services/issuing-commercial-license",
    "https://moj.gov.sa/english/eServices/Pages/ServiceDetailsNew.aspx?itemId=359",
] + [f"https://mc.gov.sa/en/eservices/Pages/ServiceDetails.aspx?sID={i}" for i in range(1, 80)
     if i not in (2, 21, 38, 52)]
LABELS_SAUDI = ["Steps", "Conditions", "Required Documents", "Service Requirements", "Service Fees", "Fees",
                "Execution Duration", "Service Duration", "Duration", "Service Channels", "Beneficiaries",
                "Target Category", "Service Level Agreement", "Service Description", "Description"]

def export_saudi():
    recs = []
    for u in SAUDI_URLS:
        r = get(u)
        if r is None:
            continue
        text = html_text(r.text)
        F = fields_from_text(text, LABELS_SAUDI)
        if not F:                         # empty page, error page or content rendered by JavaScript
            continue
        title = next((l.strip() for l in text.splitlines() if len(l.strip()) > 8), u)
        recs.append({"url": u, "title": title, "fields": F, "full_text": text[:15000]})
        time.sleep(1.0)
    json.dump(recs, open(f"{OUT}/raw_saudi.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(len(recs), "Saudi service cards saved (of", len(SAUDI_URLS), "URLs tried)")
    return recs

# ----------------------------------------------------------------------------- drafts
COLS = ["id", "org", "title", "purpose", "requires", "delivers", "capabilities", "elicited", "companions",
        "fee", "duration_days", "record", "jurisdiction", "text"]

def draft(recs, path, prefix):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLS); w.writeheader()
        for i, r in enumerate(recs):
            F = r["fields"]; fee = F.get("Fees") or F.get("Service Fees") or ""
            w.writerow({"id": f"{prefix}{i}", "org": F.get("Competent body", ""), "title": r["title"],
                        "purpose": "", "requires": "; ".join(r.get("relevant_procedures", [])),
                        "delivers": "", "capabilities": "", "elicited": "", "companions": "",
                        "fee": "0" if re.search(r"without cost|free|no fee", fee, re.I) else "",
                        "duration_days": days(F.get("Deadline of completion") or F.get("Execution Duration")
                                              or F.get("Service Duration") or F.get("Duration")),
                        "record": r.get("record") or r.get("url", ""), "jurisdiction": "",
                        "text": (F.get("Description") or F.get("Service Description") or F.get("Steps") or "")[:500]})
    log("draft ->", path)

# ----------------------------------------------------------------------------- run
t0 = time.time()
mitos = export_mitos()
saudi = export_saudi()
draft(mitos, f"{OUT}/mitos_draft.csv", "gr")
draft(saudi, f"{OUT}/saudi_draft.csv", "sa")
log(f"done in {time.time() - t0:.0f} s: {len(mitos)} Mitos records, {len(saudi)} Saudi cards")
open(f"{OUT}/log.txt", "w").write("\n".join(LOG))
zpath = OUT + ".zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in os.listdir(OUT):
        z.write(os.path.join(OUT, fn), fn)
print("\nFile to upload in the conversation:", zpath)
try:
    from google.colab import files
    files.download(zpath)
except ImportError:
    pass
