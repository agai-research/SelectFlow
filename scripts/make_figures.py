"""Figures and LaTeX tables of Section 5, from the result files (no API call)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY, TEAL, CHAR, SKY, MINT, SAND, ROSE = "#1f3a5f", "#2a7f7f", "#4a4a4a", "#a8c5da", "#c9e4de", "#e8d5b7", "#e6c3c3"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": CHAR, "axes.labelcolor": CHAR,
                     "xtick.color": CHAR, "ytick.color": CHAR, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
F, T = "results/figures", "results/tables"
J = lambda p: json.load(open(p))


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{F}/{name}.pdf"); fig.savefig(f"{F}/{name}.png", dpi=220); plt.close(fig)


# ---------------------------------------------------------------- Fig. E1: manipulations, explicit vs citizen step texts
ex, va = J("results/e1_full_explicit.json")["effects"], J("results/e1_full_vague.json")["effects"]
conds = ["base", "lookalike", "assertive", "load10", "load30", "load60"]
labels = ["Base\n(3 tools)", "+ look-alike", "+ assertive\ndescription", "+10 tools", "+30 tools", "+60 tools"]
x = np.arange(len(conds)); w = 0.36
fig, ax = plt.subplots(figsize=(6.6, 2.6))
ax.bar(x - w / 2, [ex[c]["star"] for c in conds], w, color=NAVY, edgecolor=CHAR, linewidth=0.4, label="Step named with the official title")
ax.bar(x + w / 2, [va[c]["star"] for c in conds], w, color=SKY, edgecolor=CHAR, linewidth=0.4, label="Step phrased by a citizen")
ax.plot(x + w / 2, [va[c]["none"] for c in conds], "o", color=TEAL, ms=4, label="No call (citizen phrasing)")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.05); ax.set_ylabel("Share of selections")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=3, fontsize=8)
save(fig, "e1_manipulations")

# ---------------------------------------------------------------- Fig. E3: outcome per method, with cost
e3 = J("results/e3_real.json"); cost = J("results/costs.json")["E3_per_probe"]
meths = ["selectflow", "bm25_topk", "attraction", "qos_single", "tool_search"]
names = ["SelectFlow", "BM25 top-5", "Attraction\nranking", "QoS single\nservice", "Native Tool\nSearch"]
fig, ax = plt.subplots(figsize=(6.6, 2.6))
it = np.array([e3[m]["intended"] for m in meths]); wr = np.array([e3[m]["wrong"] for m in meths]); no = np.array([e3[m]["none"] for m in meths])
ax.bar(names, it, color=NAVY, edgecolor=CHAR, linewidth=0.4, label="Intended tool")
ax.bar(names, wr, bottom=it, color=SAND, edgecolor=CHAR, linewidth=0.4, label="Wrong tool")
ax.bar(names, no, bottom=it + wr, color="white", edgecolor=CHAR, linewidth=0.4, label="No call")
for i, m in enumerate(meths):
    ax.text(i, 1.02, f"{cost[m]['usd_per_1000_probes']:.2f} $ / 1k", ha="center", fontsize=7.5, color=CHAR)
ax.set_ylim(0, 1.1); ax.set_ylabel("Share of probes")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=3, fontsize=8)
save(fig, "e3_toolkits")

# ---------------------------------------------------------------- Fig. E6: end to end, by country
e6 = J("results/e6_real.json"); R6 = [json.loads(l) for l in open("results/e6_real_records.jsonl")]
sys_ = ["selectflow", "all_tools", "tool_search"]; sname = ["SelectFlow", "Agent with all tools", "Agent with Tool Search"]
cols = [NAVY, SKY, MINT]
fig, ax = plt.subplots(figsize=(4.6, 2.5))
groups = ["greece", "saudi", "all"]; gl = ["Greece (67)", "Saudi Arabia (15)", "All (82)"]
x = np.arange(len(groups)); w = 0.26
for k, (m, n, c) in enumerate(zip(sys_, sname, cols)):
    vals = [e6["by_federation"]["greece"][m], e6["by_federation"]["saudi"][m], e6[m]["success"]]
    ax.bar(x + (k - 1) * w, vals, w, color=c, edgecolor=CHAR, linewidth=0.4, label=n)
ax.set_xticks(x); ax.set_xticklabels(gl); ax.set_ylim(0, 1.05); ax.set_ylabel("Requested document obtained")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=2, fontsize=7.5)
save(fig, "e6_end_to_end")

# ---------------------------------------------------------------- Fig. E4: denied flows
e4 = J("results/real_composition.json")["e4_real"]
fig, ax = plt.subplots(figsize=(3.4, 2.4))
x = np.arange(2); w = 0.36
for k, (lab, c) in enumerate((("certified", NAVY), ("declared", ROSE))):
    vals = [e4[f"greece_{lab}"]["rate"], e4[f"saudi_{lab}"]["rate"]]
    ax.bar(x + (k - 0.5) * w, vals, w, color=c, edgecolor=CHAR, linewidth=0.4,
           label="Certified labels" if lab == "certified" else "Declared annotations")
    for xi, v in zip(x + (k - 0.5) * w, vals):
        ax.text(xi, v + 0.008, f"{v:.0%}" if v else "0", ha="center", fontsize=7.5, color=CHAR)
ax.set_xticks(x); ax.set_xticklabels(["Greece", "Saudi Arabia"]); ax.set_ylabel("Tasks opening a denied flow")
ax.set_ylim(0, 0.42); ax.legend(fontsize=7.5, loc="upper left")
save(fig, "e4_flows")

# ---------------------------------------------------------------- LaTeX tables
e2 = J("results/e2_real.json")
open(f"{T}/e2_derivation.tex", "w").write(r"""\begin{table}[t]
\caption{Derivation on the 82 citizen requests (E2). Targets confirmed by the requester}
\label{tab:e2}
\centering\small
\begin{tabular}{@{}lccc@{}}
\toprule
 & Closed & Equal to the reference & Time (s) \\
\midrule
""" + f"""Constrained generation, first candidate & {e2['llm_first_candidate']['closed']:.2f} & {e2['llm_first_candidate']['exact']:.2f} & -- \\\\
SelectFlow (critics, revision, forward chaining) & {e2['selectflow']['closed']:.2f} & {e2['selectflow']['exact']:.2f} & {e2['time_s']['selectflow']:.1f} \\\\
Direct generation, whole catalogue in the prompt & {e2['direct']['closed']:.2f} & {e2['direct']['exact']:.2f} & {e2['time_s']['direct']:.1f} \\\\
""" + r"""\midrule
""" + f"""\\multicolumn{{4}}{{@{{}}l}}{{Interpretation of the request: targets found exactly {e2['interpretation_exact']:.2f}, at least one target {e2['interpretation_found']:.2f}}} \\\\
""" + r"""\bottomrule
\end{tabular}
\end{table}
""")
c = J("results/costs.json")["E3_per_probe"]
rows = "\n".join(f"{n.replace(chr(10), ' ')} & {e3[m]['intended']:.3f} & {e3[m]['wrong']:.3f} & {e3[m]['none']:.3f} & {c[m]['input_tokens']:.0f} & {c[m]['usd_per_1000_probes']:.2f} \\\\"
                 for m, n in zip(meths, names))
diff = lambda m: e3[f"selectflow_vs_{m}"]["diff_ci"]
open(f"{T}/e3_toolkits.tex", "w").write(r"""\begin{table}[t]
\caption{Toolkit composition on the real federations (E3): 292 capabilities, 3 repetitions, steps phrased by citizens}
\label{tab:e3}
\centering\small
\begin{tabular}{@{}lccccc@{}}
\toprule
Method & Intended & Wrong & No call & Input tokens & \$ / 1,000 probes \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")
e5 = J("results/real_composition.json")["e5_real"]
rows5 = "\n".join(f"{d} & {e5[f'density_{d}_budget_6000']['selectflow']['equipped']:.3f} & {e5[f'density_{d}_budget_6000']['selectflow']['left_to_requester']:.3f} \\\\"
                  for d in ("1.0", "0.75", "0.5"))
open(f"{T}/e5_composition.tex", "w").write(r"""\begin{table}[t]
\caption{Joint composition on the real federations (E5), 82 requests. Joint, task-by-task and no-repair variants coincide}
\label{tab:e5}
\centering\small
\begin{tabular}{@{}lcc@{}}
\toprule
Density of sharing bases & Tasks equipped & Tasks left to the requester \\
\midrule
""" + rows5 + r"""
\bottomrule
\end{tabular}
\end{table}
""")
print(sorted(os.listdir(F)), sorted(os.listdir(T)))
