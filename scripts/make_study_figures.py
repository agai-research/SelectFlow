"""All figures and tables of the experimental section (no API call)."""
import json, os, random, sys
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from selectflow import io
from selectflow.selection import Featurizer, Observation, RelevanceModel, fit

NAVY, TEAL, CHAR, SKY, MINT, SAND, ROSE, PLUM = "#1f3a5f", "#2a7f7f", "#4a4a4a", "#a8c5da", "#c9e4de", "#e8d5b7", "#e6c3c3", "#b9a7c9"
plt.rcParams.update({"boxplot.medianprops.color": "#1f3a5f", "font.family": "serif", "font.size": 8, "axes.edgecolor": CHAR, "axes.labelcolor": CHAR, "xtick.color": CHAR,
                     "ytick.color": CHAR, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "axes.titlesize": 8, "axes.titleweight": "bold"})
D = "results/figures"; T = "results/tables"; os.makedirs(D, exist_ok=True); os.makedirs(T, exist_ok=True)
J = lambda p: json.load(open(p))
L = lambda p: [json.loads(l) for l in open(p)]
rng = np.random.default_rng(0)


def boot(vals, n=2000):
    v = np.asarray(vals, float)
    if len(v) == 0: return (np.nan, np.nan)
    b = rng.choice(v, (n, len(v))).mean(1); return np.percentile(b, 2.5), np.percentile(b, 97.5)


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{D}/{name}.pdf"); fig.savefig(f"{D}/{name}.png", dpi=200); plt.close(fig)


def letter(ax, s): ax.set_title(s, loc="left")


# =============================================================== Fig. S1: tool-selection behaviour (8 panels)
ex, va = L("/dev/null") if False else J("results/e1_full_explicit_observations.json"), J("results/e1_full_vague_observations.json")
ind = J("results/e1_full_indep_observations.json")
conds = ["base", "lookalike", "assertive", "load10", "load30", "load60"]
clab = ["Base", "Look-\nalike", "Asser-\ntive", "+10", "+30", "+60"]
def share(rows, cond, what="star"):
    per = {}
    for r in rows:
        if r["cond"] != cond: continue
        x = (r["chosen"] == r["star"]) if what == "star" else (r["chosen"] is None) if what == "none" else (r["chosen"] == r.get("look"))
        per.setdefault(r["cap"], []).append(float(x))
    return {k: float(np.mean(v)) for k, v in per.items()}
fig, axs = plt.subplots(2, 4, figsize=(11, 5.2))
ax = axs[0, 0]; x = np.arange(len(conds)); w = 0.38
ax.bar(x - w / 2, [np.mean(list(share(ex, c).values())) for c in conds], w, color=NAVY, edgecolor=CHAR, lw=0.3, label="Precise step")
ax.bar(x + w / 2, [np.mean(list(share(va, c).values())) for c in conds], w, color=SKY, edgecolor=CHAR, lw=0.3, label="Citizen phrasing")
ax.set_xticks(x); ax.set_xticklabels(clab, fontsize=6.5); ax.set_ylim(0, 1.25); ax.set_yticks([0, .2, .4, .6, .8, 1]); ax.set_ylabel("Intended tool selected"); ax.legend(fontsize=6.5, loc="upper center", ncol=2); letter(ax, "(a) Selection by condition")
ax = axs[0, 1]; ntools = [3, 13, 33, 63]; lc = ["base", "load10", "load30", "load60"]
for rows, c, lab in ((va, TEAL, "No call, citizen"), (va, NAVY, "Intended, citizen"), (ex, SKY, "Intended, precise")):
    what = "none" if lab.startswith("No") else "star"
    m = [np.mean(list(share(rows, k, what).values())) for k in lc]; ci = [boot(list(share(rows, k, what).values())) for k in lc]
    ax.plot(ntools, m, "o-", color=c, ms=3, label=lab); ax.fill_between(ntools, [a for a, b in ci], [b for a, b in ci], color=c, alpha=0.15)
ax.set_xticks(ntools); ax.set_xlabel("Tools offered"); ax.set_ylim(0, 1.05); ax.legend(fontsize=6.5, loc="center right"); letter(ax, "(b) Effect of the number of tools")
ax = axs[0, 2]
bottom = np.zeros(len(conds))
for kk, (lvl, c) in enumerate(((1.0, NAVY), (2 / 3, TEAL), (1 / 3, SKY), (0.0, ROSE))):
    v = np.array([np.mean([abs(x - lvl) < 1e-6 for x in share(va, cc).values()]) for cc in conds])
    ax.bar(range(len(conds)), v, bottom=bottom, color=c, edgecolor=CHAR, lw=0.3, label=["3 of 3", "2 of 3", "1 of 3", "0 of 3"][kk]); bottom += v
ax.set_xticks(range(len(conds))); ax.set_xticklabels(clab, fontsize=6.5); ax.set_ylabel("Share of capabilities (citizen phrasing)"); ax.set_ylim(0, 1.25); ax.set_yticks([0, .2, .4, .6, .8, 1]); ax.legend(fontsize=6, ncol=4, loc="upper center", title="Repetitions with the intended tool", title_fontsize=6)
letter(ax, "(c) Consistency across repetitions")
# (d) calibration on held-out capabilities (citizen phrasing)
fed = io.load("data/e1_full_vague_federation.json") if os.path.exists("data/e1_full_vague_federation.json") else io.load("data/e1_full_federation.json")
feat = Featurizer(fed)
caps = sorted({r["cap"] for r in va}); random.Random(1).shuffle(caps); tr = set(caps[: int(0.7 * len(caps))])
O = lambda R: [Observation(r["K"], r["r"], r["v"], r["chosen"]) for r in R]
m_s, _ = fit(O([r for r in va if r["cap"] in tr]), feat); rel = RelevanceModel.fit(O([r for r in va if r["cap"] in tr]), feat)
test = [r for r in va if r["cap"] not in tr]
ax = axs[0, 3]
for model, c, lab in ((m_s, NAVY, "Tool-selection model"), (rel, ROSE, "Relevance only")):
    p = np.array([model.probs(r["K"], r["r"], r["v"])[0][r["star"]] for r in test]); y = np.array([r["chosen"] == r["star"] for r in test], float)
    bins = np.quantile(p, np.linspace(0, 1, 7)); idx = np.clip(np.digitize(p, bins[1:-1]), 0, 5)
    ax.plot([p[idx == b].mean() for b in range(6) if (idx == b).any()], [y[idx == b].mean() for b in range(6) if (idx == b).any()], "o-", color=c, ms=3, label=lab)
ax.plot([0.4, 1], [0.4, 1], ":", color=CHAR, lw=0.8); ax.set_xlim(0.4, 1); ax.set_ylim(0.4, 1); ax.set_xlabel("Predicted probability (intended tool)"); ax.set_ylabel("Observed frequency"); ax.legend(fontsize=6.5); letter(ax, "(d) Calibration (held-out)")
ax = axs[1, 0]
vals = [np.mean(list(share(ex, "lookalike", "look").values())), np.mean(list(share(ind, "lookalike", "look").values())), np.mean(list(share(va, "lookalike", "look").values()))]
ax.bar(["Title\ndescr.\nprecise", "Indep.\ndescr.\nprecise", "Title\ndescr.\ncitizen"], vals, color=[NAVY, TEAL, SKY], edgecolor=CHAR, lw=0.3)
for i, v in enumerate(vals): ax.text(i, v + 0.003, f"{v:.1%}", ha="center", fontsize=6.5)
ax.set_ylabel("Calls diverted to the look-alike"); letter(ax, "(e) Look-alike diversion")
ax = axs[1, 1]; kinds = ["submit", "pay", "retrieve"]; x = np.arange(3)
for k, (rows, c, lab) in enumerate(((ex, NAVY, "Precise"), (va, SKY, "Citizen"))):
    vv = [np.mean([r["chosen"] == r["star"] for r in rows if r["cond"] == "base" and r["r"].startswith(t)] or [np.nan]) for t in kinds]
    ax.bar(x + (k - 0.5) * 0.38, vv, 0.38, color=c, edgecolor=CHAR, lw=0.3, label=lab)
ax.set_xticks(x); ax.set_xticklabels(["Apply", "Pay", "Obtain"]); ax.set_ylim(0, 1.05); ax.legend(fontsize=6.5, loc="lower right"); letter(ax, "(f) By kind of step")
ax = axs[1, 2]; x = np.arange(2)
for k, (rows, c, lab) in enumerate(((ex, NAVY, "Precise"), (va, SKY, "Citizen"))):
    vv = [np.mean([r["chosen"] == r["star"] for r in rows if r["cond"] == "base" and r["v"].startswith(p)]) for p in ("gr", "sa")]
    ax.bar(x + (k - 0.5) * 0.38, vv, 0.38, color=c, edgecolor=CHAR, lw=0.3, label=lab)
ax.set_xticks(x); ax.set_xticklabels(["Greece", "Saudi Arabia"]); ax.set_ylim(0, 1.05); ax.legend(fontsize=6.5, loc="lower right"); letter(ax, "(g) By country")
ax = axs[1, 3]; sg = J("results/e1_signed_gamma.json")
names = ["Model,\nsigned\nload", "Model,\nload ≥ 0", "Relevance\nonly"]; vv = [sg["signed"]["log_loss"], sg["nonneg"]["log_loss"], sg["relevance"]["log_loss"]]
ax.bar(names, vv, color=[NAVY, SKY, ROSE], edgecolor=CHAR, lw=0.3)
for i, v in enumerate(vv): ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=6.5)
ax.set_ylabel("Held-out log-loss (lower is better)"); ax.set_ylim(0.4, 0.58); letter(ax, "(h) Prediction of selections")
save(fig, "s1_selection")

# =============================================================== Fig. S2: number of offered tools (4 panels)
if os.path.exists("results/e3_sizes.json"):
    z = J("results/e3_sizes.json"); obs = J("results/e3_sizes_observations.json")
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.6))
    for i, (q, lab) in enumerate((("intended", "Intended tool selected"), ("wrong", "Wrong tool selected"), ("none", "No call"), ("in_tokens", "Input tokens per call"))):
        ax = axs[i]
        for meth, c, name in (("selectflow", NAVY, "SelectFlow ($k_{\\max}$)"), ("bm25", ROSE, "BM25 top-$k$")):
            ks = sorted(int(k) for k in z[meth]); m = [z[meth][str(k)][q] for k in ks]
            ax.plot(ks, m, "o-", color=c, ms=3, label=name)
            if q != "in_tokens":
                ci = [boot([float(r[q]) for r in obs if r["method"] == meth and r["k"] == k]) for k in ks]
                ax.fill_between(ks, [a for a, b in ci], [b for a, b in ci], color=c, alpha=0.15)
        ax.set_xlabel("Maximum number of tools $k$"); ax.set_ylabel(lab); letter(ax, f"({'abcd'[i]}) {lab}")
        if i == 0: ax.legend(fontsize=6.5)
    save(fig, "s2_toolkit_size")

# =============================================================== Fig. S3: E3 comparisons (4 panels)
e3 = J("results/e3_real.json"); cost = J("results/costs.json")["E3_per_probe"]; o3 = J("results/e3_real_observations.json")
meths = ["selectflow", "bm25_topk", "attraction", "qos_single", "tool_search"]; mn = ["SelectFlow", "BM25", "Attraction", "QoS single", "Tool Search"]
cols = [NAVY, ROSE, SAND, PLUM, TEAL]
fig, axs = plt.subplots(1, 4, figsize=(11, 2.8))
ax = axs[0]; it = np.array([e3[m]["intended"] for m in meths]); wr = np.array([e3[m]["wrong"] for m in meths]); no = np.array([e3[m]["none"] for m in meths])
ax.bar(mn, it, color=NAVY, label="Intended"); ax.bar(mn, wr, bottom=it, color=SAND, label="Wrong"); ax.bar(mn, no, bottom=it + wr, color="white", edgecolor=CHAR, lw=0.3, label="No call")
ax.tick_params(axis="x", rotation=30); ax.set_ylim(0, 1.25); ax.set_yticks([0, .2, .4, .6, .8, 1]); ax.legend(fontsize=6.5, loc="upper center", ncol=3); letter(ax, "(a) Outcome of the call")
ax = axs[1]
for m, n, c in zip(meths, mn, cols):
    ax.scatter(cost[m]["usd_per_1000_probes"], e3[m]["intended"], color=c, s=28, edgecolor=CHAR, lw=0.3)
    ax.annotate(n, (cost[m]["usd_per_1000_probes"], e3[m]["intended"]), fontsize=6.5, xytext=(4, {"attraction": -9, "bm25_topk": 4}.get(m, 3)), textcoords="offset points")
ax.set_xlabel("Cost per 1,000 calls (USD)"); ax.set_ylabel("Intended tool selected"); letter(ax, "(b) Accuracy against cost")
ax = axs[2]; kinds = ["submit", "pay", "retrieve"]; x = np.arange(3); w = 0.16
for k, (m, n, c) in enumerate(zip(meths, mn, cols)):
    ax.bar(x + (k - 2) * w, [np.mean([r["intended"] for r in o3 if r["method"] == m and r["r"].startswith(t)]) for t in kinds], w, color=c, edgecolor=CHAR, lw=0.3, label=n)
ax.set_xticks(x); ax.set_xticklabels(["Apply", "Pay", "Obtain"]); ax.set_ylim(0, 1.3); ax.set_yticks([0, .2, .4, .6, .8, 1]); ax.legend(fontsize=5.5, ncol=3, loc="upper center"); letter(ax, "(c) By kind of step")
ax = axs[3]; x = np.arange(2)
for k, (m, n, c) in enumerate(zip(meths, mn, cols)):
    ax.bar(x + (k - 2) * w, [np.mean([r["intended"] for r in o3 if r["method"] == m and r["fed"] == f]) for f in ("greece", "saudi")], w, color=c, edgecolor=CHAR, lw=0.3)
ax.set_xticks(x); ax.set_xticklabels(["Greece", "Saudi Arabia"]); ax.set_ylim(0, 1); letter(ax, "(d) By country")
save(fig, "s3_toolkits")

# =============================================================== Fig. S4: workflow complexity (4 panels)
R6 = L("results/e6_real_records.jsonl"); R2 = L("results/e2_real_records.jsonl")
reqs = {r["id"]: r for n in ("greece", "saudi") for r in J(f"data/requests_{n}.json")}
for r in R6: r["n"] = len(reqs[r["id"]]["references"][0]); r["orgs"] = len(reqs[r["id"]]["organizations"])
ns = sorted({r["n"] for r in R6})
fig, axs = plt.subplots(1, 4, figsize=(11, 2.7))
ax = axs[0]; x = np.arange(len(ns)); w = 0.26
for k, (m, n, c) in enumerate((("selectflow", "SelectFlow", NAVY), ("all_tools", "All tools", SKY), ("tool_search", "Tool Search", MINT))):
    ax.bar(x + (k - 1) * w, [np.mean([r[f"{m}_success"] for r in R6 if r["n"] == L_]) for L_ in ns], w, color=c, edgecolor=CHAR, lw=0.3, label=n)
ax.set_xticks(x); ax.set_xticklabels([f"{L_}\n(n={sum(r['n'] == L_ for r in R6)})" for L_ in ns]); ax.set_xlabel("Tasks in the workflow"); ax.set_ylabel("Document obtained"); ax.set_ylim(0, 1.3); ax.set_yticks([0, .2, .4, .6, .8, 1]); ax.legend(fontsize=6.5, loc="upper center", ncol=3); letter(ax, "(a) Success by complexity")
ax = axs[1]
pos = np.arange(len(ns)); d1 = [[r["selectflow_calls"] for r in R6 if r["n"] == L_] for L_ in ns]; d2 = [[r["all_tools_calls"] for r in R6 if r["n"] == L_] for L_ in ns]
b1 = ax.boxplot(d1, positions=pos - 0.18, widths=0.3, patch_artist=True, flierprops=dict(markersize=2)); b2 = ax.boxplot(d2, positions=pos + 0.18, widths=0.3, patch_artist=True, flierprops=dict(markersize=2))
for b in b1["boxes"]: b.set_facecolor(NAVY)
for b in b2["boxes"]: b.set_facecolor(SKY)
ax.set_xticks(pos); ax.set_xticklabels(ns); ax.set_xlabel("Tasks in the workflow"); ax.set_ylabel("Tool calls per request")
ax.legend([b1["boxes"][0], b2["boxes"][0]], ["SelectFlow", "All tools"], fontsize=6.5); letter(ax, "(b) Tool calls by complexity")
ax = axs[2]; n2 = sorted({r["n_ref"] for r in R2})
ax.bar([str(n) for n in n2], [np.mean([r["llm_first_candidate"]["closed"] for r in R2 if r["n_ref"] == n]) for n in n2], color=ROSE, edgecolor=CHAR, lw=0.3, label="First candidate")
ax.plot([str(n) for n in n2], [np.mean([r["selectflow"]["exact"] for r in R2 if r["n_ref"] == n]) for n in n2], "o-", color=NAVY, ms=3, label="After critics and revision")
ax.set_xlabel("Tasks in the reference workflow"); ax.set_ylabel("Correct workflow"); ax.set_ylim(0, 1.05); ax.legend(fontsize=6.5); letter(ax, "(c) Derivation by complexity")
ax = axs[3]
bp = ax.boxplot([[r["plan_time_s"] for r in R6 if r["n"] == L_] for L_ in ns], patch_artist=True, flierprops=dict(markersize=2))
for b in bp["boxes"]: b.set_facecolor(MINT)
ax.set_xticklabels(ns); ax.set_xlabel("Tasks in the workflow"); ax.set_ylabel("Planning time (s)"); letter(ax, "(d) Planning time by complexity")
save(fig, "s4_complexity")

# =============================================================== Fig. S5: parameters (4 panels)
P = J("results/studies/s5_parameters.json"); Gd = J("results/studies/s5_grid.json")
fig, axs = plt.subplots(1, 4, figsize=(11, 2.7))
for i, (par, lab) in enumerate((("k_max", "Maximum toolkit size $k_{\\max}$"), ("budget", "Context budget $B_h$ (tokens)"), ("tau_mu", "Mapping threshold $\\tau_\\mu$"))):
    ax = axs[i]; xs = sorted(v["value"] for v in P.values() if v.get("param") == par)
    get = lambda q: [next(v[q] for v in P.values() if v.get("param") == par and v["value"] == x) for x in xs]
    ax.plot(xs, get("feasible"), "o-", color=NAVY, ms=3, label="Tasks equipped")
    ax.plot(xs, get("p_intended"), "s-", color=TEAL, ms=3, label="Predicted intended")
    ax.plot(xs, get("p_none"), "^-", color=ROSE, ms=3, label="Predicted no call")
    if par == "budget":
        for ln in ax.get_lines(): ln.set_xdata(range(len(xs)))
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs); ax.set_xlim(-0.3, len(xs) - 0.7)
    ax.set_xlabel(lab); ax.set_ylim(0, 1.0); letter(ax, f"({'abc'[i]}) {lab.split(' $')[0]}")
    if i == 0: ax.legend(fontsize=6.5)
ax = axs[3]; ds = sorted({v["density"] for v in Gd.values()}, reverse=True); bs = sorted({v["budget"] for v in Gd.values()})
M_ = np.array([[Gd[f"{d}|{b}"]["equipped"] for b in bs] for d in ds])
im = ax.imshow(M_, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for a in range(len(ds)):
    for b in range(len(bs)): ax.text(b, a, f"{M_[a, b]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if M_[a, b] > 0.6 else CHAR)
ax.set_xticks(range(len(bs))); ax.set_xticklabels(bs); ax.set_yticks(range(len(ds))); ax.set_yticklabels(ds)
ax.set_xlabel("Context budget (tokens)"); ax.set_ylabel("Density of sharing bases"); letter(ax, "(d) Tasks equipped")
save(fig, "s5_parameters")

# =============================================================== Fig. S6: certification (3 panels)
C = J("results/studies/s6_certification.json")
fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.6))
eps = [x["eps"] for x in C["eps"]]
for m, c, n in (("conformal", NAVY, "Conformal selection"), ("threshold", SAND, "Score threshold 0.5"), ("declared", ROSE, "Declared annotations")):
    axs[0].plot(eps, [x[m]["fdr"] for x in C["eps"]], "o-", color=c, ms=3, label=n); axs[1].plot(eps, [x[m]["power"] for x in C["eps"]], "o-", color=c, ms=3, label=n)
axs[0].plot([0, 0.3], [0, 0.3], ":", color=CHAR, lw=0.8); axs[0].set_xlabel("Target level $\\epsilon$"); axs[0].set_ylabel("Share of unsafe certified tools"); axs[0].legend(fontsize=6.5); letter(axs[0], "(a) False certifications")
axs[1].set_xlabel("Target level $\\epsilon$"); axs[1].set_ylabel("Share of safe tools certified"); letter(axs[1], "(b) Coverage")
af = [x["audit"] for x in C["audit"]]
axs[2].plot(af, [x["conformal"]["fdr"] for x in C["audit"]], "o-", color=NAVY, ms=3, label="False certifications")
axs[2].plot(af, [x["conformal"]["power"] for x in C["audit"]], "s-", color=TEAL, ms=3, label="Coverage")
axs[2].axhline(0.1, ls=":", color=CHAR, lw=0.8); axs[2].set_xlabel("Share of tools audited"); axs[2].legend(fontsize=6.5); letter(axs[2], "(c) Audit effort ($\\epsilon=0.1$)")
save(fig, "s6_certification")

# =============================================================== Fig. S7: scalability (3 panels)
S = J("results/studies/s4_scalability.json"); mi = [r for r in S["milp"] if r["seed"] == 0]
fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.6))
ax = axs[0]; Ns = sorted({r["N"] for r in mi})
for nV, c in zip(sorted({r["tasks"] for r in mi}), [SKY, TEAL, NAVY, PLUM, ROSE]):
    R_ = sorted([r for r in mi if r["tasks"] == nV], key=lambda r: r["N"])
    ax.plot([r["N"] for r in R_], [r["time_s"] for r in R_], "o-", color=c, ms=3, label=f"{nV} task{'s' if nV > 1 else ''}, joint")
    ax.plot([r["N"] for r in R_], [r["time_decomposed_s"] for r in R_], "x--", color=c, ms=3, lw=0.7)
ax.set_yscale("log"); ax.axhline(10, ls=":", color=CHAR, lw=0.8); ax.set_xlabel("Candidate tools per task $N$"); ax.set_ylabel("Solve time (s, log)"); ax.legend(fontsize=5.5); letter(ax, "(a) Joint (solid) vs per task (dashed)")
ax = axs[1]
for nV, c in ((4, TEAL), (6, PLUM), (8, ROSE)):
    R_ = sorted([r for r in mi if r["tasks"] == nV], key=lambda r: r["N"])
    ax.plot([r["N"] for r in R_], [100 * (r["Q_decomposed"] - r["Q_joint"]) / r["Q_decomposed"] for r in R_], "o-", color=c, ms=3, label=f"{nV} tasks")
ax.set_xlabel("Candidate tools per task $N$"); ax.set_ylabel("Quality lost at the time limit (%)"); ax.legend(fontsize=6.5); letter(ax, "(b) Gap of the time-limited joint program")
ax = axs[2]; cs = sorted({r["catalogue"] for r in S["fc"]})
ax.plot(cs, [np.mean([r["time_s"] for r in S["fc"] if r["catalogue"] == c_]) for c_ in cs], "o-", color=NAVY, ms=3)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Service descriptors in the catalogue"); ax.set_ylabel("Forward-chaining time (s)"); letter(ax, "(c) Derivation search")
save(fig, "s7_scalability")

# =============================================================== Fig. S8: end to end, robustness, flows (4 panels)
e6 = J("results/e6_real.json"); ind6 = J("results/e6_independent.json"); mp = J("results/e6_mapped.json"); rc = J("results/real_composition.json")["e4_real"]
fig, axs = plt.subplots(1, 4, figsize=(11, 2.7))
ax = axs[0]; x = np.arange(3); w = 0.26
for k, (m, n, c) in enumerate((("selectflow", "SelectFlow", NAVY), ("all_tools", "All tools", SKY), ("tool_search", "Tool Search", MINT))):
    ax.bar(x + (k - 1) * w, [e6["by_federation"]["greece"][m], e6["by_federation"]["saudi"][m], e6[m]["success"]], w, color=c, edgecolor=CHAR, lw=0.3, label=n)
ax.set_xticks(x); ax.set_xticklabels(["Greece", "Saudi", "All"]); ax.set_ylim(0, 1.3); ax.set_yticks([0, .2, .4, .6, .8, 1]); ax.set_ylabel("Document obtained"); ax.legend(fontsize=6.5, loc="upper center", ncol=3); letter(ax, "(a) 82 requests")
ax = axs[1]; x = np.arange(3)
sfv = [1.0, ind6["selectflow"]["independent_descriptions"], mp["selectflow_mapped_cards"]]
atv = [ind6["all_tools"]["title_based_descriptions_same_requests"], ind6["all_tools"]["independent_descriptions"], ind6["all_tools"]["independent_descriptions"]]
tsv = [ind6["tool_search"]["title_based_descriptions_same_requests"], ind6["tool_search"]["independent_descriptions"], ind6["tool_search"]["independent_descriptions"]]
for k, (v, c) in enumerate(((sfv, NAVY), (atv, SKY), (tsv, MINT))): ax.bar(x + (k - 1) * w, v, w, color=c, edgecolor=CHAR, lw=0.3)
ax.set_xticks(x); ax.set_xticklabels(["Title-based", "Indep.\ndescriptions", "Indep. descr.\n+ mapping"], fontsize=6.5); ax.set_ylim(0, 1.05); letter(ax, "(b) Robustness, 45 requests")
ax = axs[2]
bp = ax.boxplot([[r[f"{m}_calls"] for r in R6] for m in ("selectflow", "all_tools", "tool_search")], patch_artist=True, flierprops=dict(markersize=2))
for b, c in zip(bp["boxes"], (NAVY, SKY, MINT)): b.set_facecolor(c)
ax.set_xticklabels(["SelectFlow", "All tools", "Tool Search"]); ax.set_ylabel("Tool calls per request"); letter(ax, "(c) Tool calls")
ax = axs[3]; x = np.arange(2)
for k, (lab, c, n) in enumerate((("certified", NAVY, "Certified labels"), ("declared", ROSE, "Declared annotations"))):
    v = [rc[f"greece_{lab}"]["rate"], rc[f"saudi_{lab}"]["rate"]]; ax.bar(x + (k - 0.5) * 0.38, v, 0.38, color=c, edgecolor=CHAR, lw=0.3, label=n)
    for xi, vi in zip(x + (k - 0.5) * 0.38, v): ax.text(xi, vi + 0.006, f"{vi:.0%}" if vi else "0", ha="center", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(["Greece", "Saudi Arabia"]); ax.set_ylabel("Tasks opening a denied flow"); ax.set_ylim(0, 0.42); ax.legend(fontsize=6.5, loc="upper left"); letter(ax, "(d) Flow control")
save(fig, "s8_end_to_end")
print(sorted(f for f in os.listdir(D) if f.startswith("s") and f.endswith(".pdf")))
