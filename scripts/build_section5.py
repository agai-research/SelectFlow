"""Generates the experimental section (LaTeX) from the result files."""
import json, os
J = lambda p: json.load(open(p))
L = lambda p: [json.loads(l) for l in open(p)]
pc = lambda x: f"{100 * x:.1f}\\%"
ci = lambda c: f"[{100 * c[1]:+.1f} ; {100 * c[2]:+.1f}]"
e1x, e1v, e1i = J("results/e1_full_explicit.json"), J("results/e1_full_vague.json"), J("results/e1_full_indep.json")
sg = J("results/e1_signed_gamma.json"); e2 = J("results/e2_real.json"); e3 = J("results/e3_real.json"); cost = J("results/costs.json")
e6 = J("results/e6_real.json"); ind6 = J("results/e6_independent.json"); mp = J("results/e6_mapped.json")
rc = J("results/real_composition.json"); P = J("results/studies/s5_parameters.json"); Gd = J("results/studies/s5_grid.json")
C = J("results/studies/s6_certification.json"); S = J("results/studies/s4_scalability.json")
z = J("results/e3_sizes.json") if os.path.exists("results/e3_sizes.json") else None
R6 = L("results/e6_real_records.jsonl"); R2 = L("results/e2_real_records.jsonl")
fs = J("results/formulations_summary.json"); mq = J("results/capability_mapping_quality.json")
V = e1v["effects"]; X = e1x["effects"]; I = e1i["effects"]
pv = lambda k: P[k]
eps = {x["eps"]: x for x in C["eps"]}
mi = [r for r in S["milp"] if r["seed"] == 0]
t = lambda nv, n: next(r for r in mi if r["tasks"] == nv and r["N"] == n)
reqs = {r["id"]: r for n in ("greece", "saudi") for r in J(f"data/requests_{n}.json")}
for r in R6: r["n"] = len(reqs[r["id"]]["references"][0])
import numpy as np
succ_n = {m: {n: np.mean([r[f"{m}_success"] for r in R6 if r["n"] == n]) for n in (2, 3, 4, 5)} for m in ("selectflow", "all_tools", "tool_search")}
fc_n = {n: np.mean([r["llm_first_candidate"]["closed"] for r in R2 if r["n_ref"] == n]) for n in (2, 3, 4, 5)}
o3 = J("results/e3_real_observations.json")
kind = lambda m, k: np.mean([r["intended"] for r in o3 if r["method"] == m and r["r"].startswith(k)])

s = r"""\section{Experimental Studies}\label{sec:evaluation}

This section answers six research questions. \textbf{RQ1}: how does an LLM agent select tools when the toolkit, the wording of the step and the tool descriptions vary, and does the tool-selection model of Sect.~\ref{sec:quality} predict it? \textbf{RQ2}: how many tools should a task be offered? \textbf{RQ3}: does composing toolkits under that model select the intended tool more often, or more cheaply, than retrieval-based and QoS-based composition? \textbf{RQ4}: how do derivation, enactment and planning behave as workflows grow? \textbf{RQ5}: how sensitive is \sysname{} to its parameters, and how far does the exact composition scale? \textbf{RQ6}: does the complete approach serve citizen requests better than agents given all the tools or a tool search, and does flow certification prevent denied flows? Table~\ref{tab:studies} maps the eight studies to these questions.

\begin{table*}[t]
\caption{Experimental studies: objective, varied factors, metrics. ``Measured'' results come from calls to the agent model; ``predicted'' results are computed with the tool-selection model fitted on measured calls}
\label{tab:studies}
\centering\fzs
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}p{0.75cm}p{0.6cm}p{3.8cm}p{4.1cm}p{3.5cm}p{2.0cm}@{}}
\toprule
Study & RQ & Objective & Varied factors & Metrics & Nature \\
\midrule
S1 & RQ1 & Selection behaviour of the agent; validity of the tool-selection model & Wording of the step (precise, citizen); look-alike; assertive description; 3 to 63 tools; title-based or independent descriptions & Intended, wrong and no call; diversion to look-alike; held-out log-loss; calibration & Measured, 5{,}532 calls \\
S2 & RQ2 & Number of tools to offer & $k\in\{1,2,3,5,8,12\}$ for BM25, $k_{\max}\in\{1,2,3,5,8\}$ for \sysname{} & Intended, wrong, no call; input tokens & Measured, 2{,}746 calls \\
S3 & RQ3 & Toolkit composition against baselines & Five composition methods; kind of step; country & Intended, wrong, no call; tokens; cost & Measured, 4{,}380 calls \\
S4 & RQ4 & Derivation and enactment as workflows grow & 2 to 5 tasks; 1 to 5 organizations & Correct workflow; first-candidate correctness; success; tool calls; planning time & Measured, 82 requests \\
S5 & RQ5 & Sensitivity to the parameters of the composer & $k_{\max}$; context budget $B_h$; mapping threshold $\tau_\mu$; candidates $N$; density of sharing bases & Tasks equipped; predicted intended selection and abstention & Predicted, 117 procedures; grid of 24 settings on 82 requests \\
S6 & RQ6 & Certification of tool behaviour and flow control & Target level $\epsilon$; audited fraction; labels used to compose & False certifications; coverage; tasks opening a denied flow & Simulated labels; real federations \\
S7 & RQ5 & Scalability of composition and derivation & Candidates $N$; tasks $|V|$; catalogue size & Solve time; quality lost at the time limit & Synthetic federations \\
S8 & RQ6 & End to end and robustness & Three systems; title-based or independent descriptions and capability mappings & Document obtained; tool calls & Measured, 82 + 45 requests \\
\bottomrule
\end{tabular}
\end{table*}

\subsection{Experimental Setup}

\paragraph{Datasets.} Table~\ref{tab:data} summarizes the data. The \emph{Greek federation} is built from the 81 records of the Mitos registry \cite{mitos2026} in the category of business procedures, extracted through the registry's interface, plus the 12 procedures that these records cite as issuers of required documents (criminal record, tax and social-security clearances, fire protection). The \emph{Saudi federation} gathers 24 procedures of seven organizations (Ministry of Commerce, municipal services, Civil Defense, tax, social insurance, human resources, passports), 21 of them from the organizations' service pages. A procedure's required documents and output come from its record; a dependency exists when a document required by one procedure is the output of another, as the registries state. Each procedure has an application tool, a retrieval tool and, when it has a fee, a payment tool (238 and 54 capabilities). The application tool of a neighbouring procedure of the same organization is mapped to the same capability with no adequacy, which yields \emph{look-alikes} taken from the registries themselves; third-party upload tools that forward data are added to 15\% of the procedures. Tool descriptions exist in two versions: \emph{title-based}, generated from the procedure titles, and \emph{independent}, rewritten by another model call as an organization's IT team would publish them, reusing 19\% of the title words instead of 100\%; in the most demanding setting, the capability mapping of the tools is also recomputed from the independent descriptions only (five passes, precision """ + f"{mq['precision']:.2f}, recall {mq['recall']:.2f}" + r""").

A \emph{request} is created for every procedure whose output needs other procedures: 82 requests, 81 of which span several organizations. Their \emph{reference workflows} are the closures of the registry dependencies, from which no language model takes part. Each request is phrased as a citizen would, without the official title (English), and also in French and Arabic. A second model judged all 246 formulations faithful, and the deterministic annotation of the procedures correct for the purpose in """ + pc(fs["purpose_ok"]) + r""" of cases and for the data categories in """ + pc(fs["categories_ok"]) + r""". MCP-Bench \cite{wang2026mcpbench} (278 tools of 25 servers, 104 tasks) and TOUCAN \cite{toucan2025} (30{,}000 trajectories) are used for the set-level baseline and for a pilot of S1.

\begin{table}[t]
\caption{Datasets}
\label{tab:data}
\centering\fzs
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.3cm}cc@{}}
\toprule
 & Greece & Saudi Arabia \\
\midrule
Procedures (registry + referenced) & 81 + 12 & 24 \\
Organizations & 29 & 7 \\
Capabilities & 238 & 54 \\
Tools (third-party) & 252 (14) & 56 (2) \\
Look-alike mappings & 22 & 6 \\
Sharing bases (density 1) & 1{,}447 & 420 \\
Requests & 67 & 15 \\
Tasks per reference (2/3/4/5) & 38/14/10/5 & 8/5/2/0 \\
Organizations per request (1/2/3/4/5) & 0/40/12/13/2 & 1/12/2/0/0 \\
\midrule
\multicolumn{3}{@{}p{6.6cm}}{MCP-Bench: 278 tools, 25 servers, 104 tasks; TOUCAN: 30{,}000 trajectories} \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Agent, harness and defaults.} The agent is Claude Sonnet~5.5 with a fixed identifier, no reasoning before a call and one tool call per occasion; calls are sent through the provider's batch interface. Each call occasion offers the toolkit and the step, either precise (procedure and action, as written by \sysname{}) or phrased by a citizen. Tools enforce procedure semantics in E2E runs: an application needs the required documents, the document is delivered only after a successful application and, if there is a fee, a payment. The prototype implements Sect.~\ref{sec:approach} in Python; programs are solved with HiGHS. Table~\ref{tab:params} lists the default values.

\begin{table}[t]
\caption{Default parameters}
\label{tab:params}
\centering\fzs
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{3.3cm}r@{}}
\toprule
Parameter & Meaning & Value \\
\midrule
$k_{\max}$ & Maximum tools per task & 5 \\
$N$, $n_{\min}$ & Candidates per task; per capability & 12; 2 \\
$\tau_\mu$, $\tau_\chi$ & Mapping and confirmation thresholds & 0.5; 0.6 \\
$B_h$ & Context budget of a model host & 6{,}000 tokens \\
$\hat\gamma$ & Load effect, fitted (S1) & $-0.09$ per 1k tokens \\
$\epsilon$; audit & Certification level; audited share & 0.1; 0.2 \\
$k$, $n_{rev}$ & Candidates; revisions in derivation & 3; 2 \\
$t_{\max}$ & Solver time limit & 30 s \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Baselines.} Table~\ref{tab:baselines} describes the baselines. They cover the three families of the related work: retrieval of tools by relevance, single-service QoS composition, and agents that select among all tools or search them at run time.

\begin{table*}[t]
\caption{Baselines}
\label{tab:baselines}
\centering\fzs
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}p{2.5cm}p{2.3cm}p{6.6cm}p{1.3cm}p{1.6cm}@{}}
\toprule
Baseline & Family & Principle; information used & Studies & Ref. \\
\midrule
Direct generation & LLM composition & The whole catalogue and the confirmed targets in one prompt; the model returns the workflow; no critic & S4 & \cite{pesl2025compositio} \\
BM25 top-$k$ & Relevance retrieval & The $k$ tools whose descriptions best match the step; same budget and flow filter as \sysname{} & S2, S3 & \cite{qin2024toolllm} \\
QoS single service & QoS composition & One tool per capability, the best by QoS among those mapped to it & S3 & \cite{zeng2004qos} \\
Attraction ranking & Internal control & \sysname{}'s candidates ranked by attraction instead of diluted value & S3 & -- \\
Tool Search (native) & Run-time search & All tools deferred; the agent searches them (BM25) before calling & S3, S8 & \cite{anthropic2026toolsearch} \\
Agent with all tools & Agent selection & One episode with every tool of the federation and the citizen request & S8 & \cite{wang2026mcpbench} \\
HYSET & Set-level retrieval & Hypergraph model of tool co-invocations; trained on TOUCAN; MCP-Bench only (no invocation history for the administrative tools) & Setup & \cite{hyset2026} \\
Declared annotations & Flow control & Tools trusted as declared: no data returned, no forwarding & S6 & -- \\
Score threshold & Certification & Tools certified when the classifier score exceeds 0.5, without error control & S6 & -- \\
\bottomrule
\end{tabular}
\end{table*}

HYSET was trained twice on TOUCAN; it reaches a recall@5 of 0.39 and 0.36 on validation servers but of 0.09 on 55 unseen servers, and a tool of the right server is among its first five for 39\% of the MCP-Bench tasks against 38\% for BM25. A set-level retriever thus needs an invocation history that newly published administrative tools do not have, and it is not used further.

\paragraph{Metrics and statistics.} A call occasion ends with the \emph{intended} tool (adequacy at least 0.5 for the capability), a \emph{wrong} tool, or \emph{no call}. A request succeeds when the requested document is obtained. Differences are reported with 95\% paired bootstrap intervals (in points) over capabilities or requests; binary outcomes are compared with McNemar's test, corrected with Holm's method across baselines.

\subsection{S1: Tool-Selection Behaviour (RQ1)}

\emph{Objective and setting.} For 120 capabilities, the same base toolkit (the intended tool and two tools of other organizations) was offered under controlled variants: a look-alike added, an assertive description on a distractor, or 10, 30 or 60 unrelated tools added (3 to 63 tools, up to 5{,}600 tokens). Each condition was run three times with precise steps and with citizen phrasing (3{,}732 calls), and again with independent descriptions for three conditions (786 calls). The tool-selection model was fitted on 70\% of the capabilities and evaluated on the others.

\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{s1_selection.pdf}
\caption{S1, tool-selection behaviour. (a) Intended selection per condition; (b) effect of the number of offered tools, with 95\% bands; (c) consistency of the three repetitions per capability; (d) calibration of the fitted model on held-out capabilities; (e) calls diverted to the look-alike; (f) by kind of step; (g) by country; (h) held-out log-loss}
\label{fig:s1}
\end{figure*}

\emph{Observations.} Fig.~\ref{fig:s1}(a) shows that the wording of the step dominates: precise steps lead to the intended tool in """ + pc(X["base"]["star"]) + r""" of base calls, citizen phrasing in """ + pc(V["base"]["star"]) + r""", the difference being almost entirely abstention (""" + pc(V["base"]["none"]) + r"""). A look-alike lowers the intended selection to """ + pc(V["lookalike"]["star"]) + r""" (""" + ci(V["lookalike"]["star_minus_base_ci"]) + r""" points) and diverts """ + pc(V["lookalike"]["lookalike_chosen"]) + r""" of the calls; an assertive description has no significant effect (""" + ci(V["assertive"]["star_minus_base_ci"]) + r"""). Contrary to the dilution assumption, offering more tools \emph{raises} the intended selection (Fig.~\ref{fig:s1}(b)): with 10, 30 and 60 extra tools it reaches """ + pc(V["load10"]["star"]) + ", " + pc(V["load30"]["star"]) + " and " + pc(V["load60"]["star"]) + r""" (""" + ci(V["load10"]["star_minus_base_ci"]) + r""" points for +10), because abstention falls from """ + pc(V["base"]["none"]) + " to " + pc(V["load60"]["none"]) + r""" and saturates after 10 tools. Repetitions agree for most capabilities (Fig.~\ref{fig:s1}(c)). Precise steps behave alike for every kind of step and country, while citizen phrasing loses most for payment steps and in Greece (Fig.~\ref{fig:s1}(f,g)). With independent descriptions, precise steps keep """ + pc(I["base"]["star"]) + r""" of intended selections, and look-alikes divert """ + pc(I["lookalike"]["lookalike_chosen"]) + r""" of the calls instead of none (Fig.~\ref{fig:s1}(e)).

\emph{Interpretation.} The model's selection errors are mostly abstentions caused by imprecise steps, not diversions; the decomposition of a request into precise steps is therefore the main lever. Look-alikes matter once descriptions are written independently, which is the realistic case. The fitted load effect is negative ($\hat\gamma=""" + f"{sg['gamma_signed']:.2f}" + r"""$ per thousand tokens); the tool-selection model with a load effect of either sign predicts held-out selections better than relevance alone (log-loss """ + f"{sg['signed']['log_loss']:.3f} against {sg['relevance']['log_loss']:.3f}" + r""", difference """ + f"{sg['signed_vs_relevance'][0]:.3f} [{sg['signed_vs_relevance'][1]:.3f} ; {sg['signed_vs_relevance'][2]:.3f}]" + r""") and is better calibrated (Fig.~\ref{fig:s1}(d,h)); its linear form does not capture the saturation of the load effect.
"""

if z:
    sf, bm = z["selectflow"], z["bm25"]
    ks_b = sorted(bm, key=int); ks_s = sorted(sf, key=int)
    best_b = max(ks_b, key=lambda k: bm[k]["intended"])
    s += r"""
\subsection{S2: Number of Offered Tools (RQ2)}

\emph{Objective and setting.} For the 292 capabilities, the agent received the citizen-phrased step with BM25 toolkits of $k\in\{1,2,3,5,8,12\}$ tools and with \sysname{} toolkits composed with $k_{\max}\in\{1,2,3,5,8\}$ (2{,}746 calls).

\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{s2_toolkit_size.pdf}
\caption{S2, number of offered tools: intended selection, wrong tool, no call and input tokens as the maximum number of tools grows (95\% bands)}
\label{fig:s2}
\end{figure*}

\begin{table}[t]
\caption{S2, outcome by maximum number of tools (citizen phrasing)}
\label{tab:s2}
\centering\fzs
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccccc@{}}
\toprule
Method, $k$ & Tools & Intended & Wrong & No call & Tokens \\
\midrule
""" + "\n".join(f"BM25, {k} & {bm[k]['size']:.1f} & {bm[k]['intended']:.3f} & {bm[k]['wrong']:.3f} & {bm[k]['none']:.3f} & {bm[k]['in_tokens']:.0f} \\\\" for k in ks_b) + r"""
\midrule
""" + "\n".join(f"\\sysname{{}}, {k} & {sf[k]['size']:.1f} & {sf[k]['intended']:.3f} & {sf[k]['wrong']:.3f} & {sf[k]['none']:.3f} & {sf[k]['in_tokens']:.0f} \\\\" for k in ks_s) + r"""
\bottomrule
\end{tabular}
\end{table}

\emph{Observations.} Table~\ref{tab:s2} and Fig.~\ref{fig:s2} show that a single retrieved tool is too few: BM25 with $k=1$ selects the intended tool in """ + pc(bm["1"]["intended"]) + r""" of the occasions and the agent abstains in """ + pc(bm["1"]["none"]) + r""". Intended selection peaks at $k=3$ (""" + pc(bm["3"]["intended"]) + r"""), then levels off around 61--63\% while wrong calls rise from """ + pc(bm["3"]["wrong"]) + " to " + pc(bm["12"]["wrong"]) + r""" and input grows from """ + f"{bm['3']['in_tokens']:.0f} to {bm['12']['in_tokens']:,.0f}".replace(",", "{,}") + r""" tokens. \sysname{} cannot equip tasks with $k_{\max}=1$, since a task needs two or three capabilities; it reaches """ + pc(sf["2"]["intended"]) + r""" with $k_{\max}=2$ and """ + pc(sf["5"]["intended"]) + r""" with $k_{\max}=5$, choosing """ + f"{sf['5']['size']:.1f}" + r""" tools on average whatever the bound, so that its input stays at """ + f"{sf['5']['in_tokens']:.0f}" + r""" tokens. Compared on the same occasions, \sysname{} ($k_{\max}=5$) beats BM25 with two tools by 10.3 points [+3.4 ; +17.1] ($p=0.006$), and does not differ significantly from BM25 with three tools ($-2.7$ [$-7.2$ ; +1.4]) or twelve tools (+0.3 [$-4.5$ ; +4.8]), the latter costing 2.7 times its input.

\emph{Interpretation.} The agent abstains when the only tool offered does not obviously match a vague step, and extra tools beyond three mostly add wrong calls and cost. A retriever tuned to the right $k$ reaches the accuracy of \sysname{}, but the right $k$ depends on the federation and is not known in advance; \sysname{} sets the size of each toolkit from the tool-selection model and stays near the best accuracy at a stable, low cost.
"""

s += r"""
\subsection{S3: Toolkit Composition Compared (RQ3)}

\emph{Objective and setting.} For the 292 capabilities, five methods composed a toolkit and the agent received the citizen-phrased step, three times (4{,}380 calls).

\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{s3_toolkits.pdf}
\caption{S3, toolkit composition. (a) Outcome of the call; (b) intended selection against cost; (c) by kind of step; (d) by country}
\label{fig:s3}
\end{figure*}

\begin{table*}[t]
\caption{S3, toolkit composition: 876 call occasions per method; differences with \sysname{} in points (95\% CI), McNemar's test with Holm's correction}
\label{tab:e3}
\centering\fzs
\setlength{\tabcolsep}{6pt}
\begin{tabular}{@{}lcccccl@{}}
\toprule
Method & Intended & Wrong & No call & Tokens & \$/1k & $\Delta$ \sysname{} \\
\midrule
""" + f"\\sysname{{}} & {e3['selectflow']['intended']:.3f} & {e3['selectflow']['wrong']:.3f} & {e3['selectflow']['none']:.3f} & {cost['E3_per_probe']['selectflow']['input_tokens']:.0f} & {cost['E3_per_probe']['selectflow']['usd_per_1000_probes']:.2f} & -- \\\\\n" + "\n".join(
    f"{n} & {e3[m]['intended']:.3f} & {e3[m]['wrong']:.3f} & {e3[m]['none']:.3f} & {cost['E3_per_probe'][m]['input_tokens']:,.0f} & {cost['E3_per_probe'][m]['usd_per_1000_probes']:.2f} & {100 * e3[f'selectflow_vs_{m}']['diff_ci'][0]:+.1f} {ci(e3[f'selectflow_vs_{m}']['diff_ci'])}, $p$={e3['mcnemar_holm'][m]:.2f} \\\\".replace(",", "{,}", 1 if cost['E3_per_probe'][m]['input_tokens'] >= 1000 else 0)
    for m, n in (("bm25_topk", "BM25 top-5"), ("attraction", "Attraction"), ("qos_single", "QoS single"), ("tool_search", "Tool Search"))) + r"""
\bottomrule
\end{tabular}
\end{table*}

\emph{Observations.} \sysname{} selects the intended tool in """ + pc(e3["selectflow"]["intended"]) + r""" of the occasions (Table~\ref{tab:e3}), more than BM25 (+""" + f"{100 * e3['selectflow_vs_bm25_topk']['diff_ci'][0]:.1f}" + r""" points, not significant), attraction ranking and QoS selection (+""" + f"{100 * e3['selectflow_vs_qos_single']['diff_ci'][0]:.1f}" + r""" points, Holm $p=""" + f"{e3['mcnemar_holm']['qos_single']:.2f}" + r"""$), and as often as Tool Search (""" + pc(e3["tool_search"]["intended"]) + r"""). It has the lowest wrong-tool rate after QoS selection; Tool Search almost never abstains but calls a wrong tool in """ + pc(e3["tool_search"]["wrong"]) + r""" of the occasions and needs """ + f"{cost['E3_per_probe']['tool_search']['input_tokens'] / cost['E3_per_probe']['selectflow']['input_tokens']:.1f}" + r""" times more input. Fig.~\ref{fig:s3}(b) places \sysname{} on the cost--accuracy frontier. All methods fail most on retrieval steps phrased by citizens (intended """ + f"{kind('selectflow', 'retrieve'):.2f}" + r""" for \sysname{} against """ + f"{kind('selectflow', 'submit'):.2f}" + r""" for applications): a citizen asking to ``obtain'' a document tends to call the application tool.

\emph{Interpretation.} With a vague step and a single call occasion, toolkit composition yields a modest gain over retrieval and matches tool search at a third of its cost. Its value comes from what it combines with: the precise steps produced by derivation (S1) and the per-task toolkits of the end-to-end runs (S8).

\subsection{S4: Derivation and Workflow Complexity (RQ4)}

\emph{Objective and setting.} For the 82 requests, derivation was run with the requester's confirmation of the targets, and compared with direct generation; end-to-end outcomes, tool calls and planning time were analysed by the number of tasks (2 to 5) of the reference workflow.

\begin{table}[t]
\caption{S4, derivation on the 82 citizen requests}
\label{tab:e2}
\centering\fzs
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.9cm}ccc@{}}
\toprule
Method & Closed & = reference & Time (s) \\
\midrule
""" + f"First candidate, 30 retrieved descriptors & {e2['llm_first_candidate']['closed']:.2f} & {e2['llm_first_candidate']['exact']:.2f} & -- \\\\\n\\sysname{{}} (critics, revision, chaining) & {e2['selectflow']['closed']:.2f} & {e2['selectflow']['exact']:.2f} & {e2['time_s']['selectflow']:.1f} \\\\\nDirect generation, whole catalogue & {e2['direct']['closed']:.2f} & {e2['direct']['exact']:.2f} & {e2['time_s']['direct']:.1f} \\\\" + r"""
\midrule
\multicolumn{4}{@{}l}{Interpretation of the request: exact targets """ + f"{e2['interpretation_exact']:.2f}, at least one {e2['interpretation_found']:.2f}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{s4_complexity.pdf}
\caption{S4, workflow complexity. (a) Document obtained by number of tasks; (b) tool calls per request; (c) correctness of the first derivation candidate and after critics; (d) planning time}
\label{fig:s4}
\end{figure*}

\emph{Observations.} The first candidate generated over the 30 retrieved descriptors is correct for """ + pc(e2["llm_first_candidate"]["closed"]) + r""" of the requests, because it misses the issuer of a required document; the critics reject every incorrect candidate and revision repairs all of them, so that the workflow returned by \sysname{}, which comes from the language model in all 82 cases, equals the reference (Table~\ref{tab:e2}). Direct generation with the whole catalogue is also correct for every request, in """ + f"{e2['time_s']['direct']:.1f}" + r""" s instead of """ + f"{e2['time_s']['selectflow']:.1f}" + r""" s. The interpretation of the request finds the exact targets for """ + pc(e2["interpretation_exact"]) + r""" of the requests. As workflows grow (Fig.~\ref{fig:s4}), \sysname{} keeps obtaining the document, whereas the agent with all tools falls to """ + pc(succ_n["all_tools"][3]) + " and " + pc(succ_n["all_tools"][4]) + r""" for 3 and 4 tasks and Tool Search to """ + pc(succ_n["tool_search"][4]) + r""" for 4 tasks; tool calls grow linearly with the number of tasks for every system, \sysname{} making fewer, and planning takes 10 to 30 s whatever the size.

\emph{Interpretation.} On registries whose catalogue fits in a prompt, a strong model derives correct workflows once the targets are confirmed; the critics act as a guarantee (Proposition~1) that becomes necessary when the catalogue must be retrieved, as the first candidates show. Interpretation is the weak point, which justifies the confirmation step. The gap between \sysname{} and the agents widens with the number of tasks, since one long episode must chain more applications, payments and retrievals in the right order.

\subsection{S5: Parameters of the Composer (RQ5)}

\emph{Objective and setting.} With the capability mappings recomputed from independent descriptions, every procedure was composed under varying $k_{\max}$, context budget $B_h$, mapping threshold $\tau_\mu$ and number of candidates $N$; the intended selection and the abstention are \emph{predicted} by the fitted tool-selection model. The 82 requests were also composed jointly with repair on a grid of densities of sharing bases and budgets.

\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{s5_parameters.pdf}
\caption{S5, parameters. (a)--(c) Tasks equipped and predicted selection as $k_{\max}$, $B_h$ and $\tau_\mu$ vary; (d) tasks equipped for densities of sharing bases and budgets (82 requests, joint composition with repair)}
\label{fig:s5}
\end{figure*}

\emph{Observations.} With $k_{\max}=1$, only """ + pc(pv("k_max=1")["feasible"]) + r""" of the procedures can be equipped and the predicted abstention is """ + pc(pv("k_max=1")["p_none"]) + r"""; from $k_{\max}=3$ on, """ + pc(pv("k_max=3")["feasible"]) + r""" are equipped and nothing changes (Fig.~\ref{fig:s5}(a)), the composer choosing """ + f"{pv('k_max=5')['size']:.1f}" + r""" tools on average. A budget of 150 tokens equips no task and 300 tokens """ + pc(pv("budget=300")["feasible"]) + r"""; 600 tokens suffice (Fig.~\ref{fig:s5}(b)). Raising $\tau_\mu$ from 0.2 to 1.0 lowers the equipped share from """ + pc(pv("tau_mu=0.2")["feasible"]) + " to " + pc(pv("tau_mu=1.0")["feasible"]) + r""", the predicted selection being unchanged; $N$ has no effect from 2 on. On the grid (Fig.~\ref{fig:s5}(d)), the budget sets a hard threshold (no task equipped below 260 tokens, about half at 320) and the density of sharing bases degrades gracefully, from """ + pc(Gd["1.0|6000"]["equipped"]) + " at density 1 to " + pc(Gd["0.5|6000"]["equipped"]) + " at 0.5 and " + pc(Gd["0.4|6000"]["equipped"]) + r""" at 0.4, the remaining tasks being left to the requester.

\emph{Interpretation.} The composer is robust to its parameters within broad ranges: two or three tools per task and a few hundred tokens are enough, and $\tau_\mu$ trades the share of equipped tasks against the confidence of the mappings. The binding factors are the context budget of the host and the sharing bases that the organizations sign, which are governance decisions rather than tuning choices.

\subsection{S6: Certification and Flow Control (RQ6)}

\emph{Objective and setting.} Conformal selection was evaluated on 3{,}000 tools with known behaviour in four strata, varying the target level $\epsilon$ and the audited fraction (20 repetitions), against a score threshold and declared annotations. On the real federations, the 82 requests were composed with certified labels and with declared annotations, and their tasks were checked against the true behaviour of the tools.

\begin{figure*}[t]
\centering\includegraphics[width=0.78\textwidth]{s6_certification.pdf}
\caption{S6, certification. (a) Share of unsafe tools among certified ones against the target level (dotted: the target); (b) share of safe tools certified; (c) effect of the audited fraction at $\epsilon=0.1$}
\label{fig:s6}
\end{figure*}

\emph{Observations.} Conformal selection keeps the share of unsafe certified tools at or below the target level for every $\epsilon$ (""" + f"{eps[0.05]['conformal']['fdr']:.3f} at 0.05, {eps[0.1]['conformal']['fdr']:.3f} at 0.1, {eps[0.2]['conformal']['fdr']:.3f} at 0.2" + r"""), and its coverage grows with $\epsilon$ (""" + f"{eps[0.05]['conformal']['power']:.2f}, {eps[0.1]['conformal']['power']:.2f}, {eps[0.2]['conformal']['power']:.2f}" + r"""; Fig.~\ref{fig:s6}). The score threshold has a fixed error (""" + f"{eps[0.1]['threshold']['fdr']:.3f}" + r""") that cannot be set; declared annotations certify every tool and let """ + pc(eps[0.1]["declared"]["fdr"]) + r""" unsafe tools through. Auditing 5\% to 40\% of the tools keeps the error below $\epsilon$ and changes coverage little. On the real federations, no task composed with certified labels opens a denied flow, whereas """ + f"{rc['e4_real']['greece_declared']['tasks_with_denied_flow']} of {rc['e4_real']['greece_declared']['tasks']}" + r""" Greek tasks (""" + pc(rc["e4_real"]["greece_declared"]["rate"]) + r""") and """ + f"{rc['e4_real']['saudi_declared']['tasks_with_denied_flow']} of {rc['e4_real']['saudi_declared']['tasks']}" + r""" Saudi tasks (""" + pc(rc["e4_real"]["saudi_declared"]["rate"]) + r""") do with declared annotations (Fig.~\ref{fig:s8}(d)).

\emph{Interpretation.} Certification gives organizations a chosen bound on false certifications at a moderate audit cost, which a fixed threshold cannot; trusting declarations lets the data that tools return and the forwarding of third-party tools open denied flows in a quarter of the tasks.

\subsection{S7: Scalability (RQ5)}

\emph{Objective and setting.} The exact joint program was solved on synthetic federations for 1 to 8 tasks and 4 to 15 candidates per task (time limit 10 s), and compared with solving each task separately; forward chaining was run on catalogues of 100 to 6{,}000 descriptors.

\begin{figure*}[t]
\centering\includegraphics[width=0.78\textwidth]{s7_scalability.pdf}
\caption{S7, scalability. (a) Solve time of the joint program (solid) and of the per-task decomposition (dashed); (b) quality lost by the joint program at the time limit; (c) forward-chaining time against catalogue size}
\label{fig:s7}
\end{figure*}

\emph{Observations.} The joint program is solved in """ + f"{t(1, 15)['time_s']:.2f}" + r""" s for one task with 15 candidates and in """ + f"{t(2, 6)['time_s']:.2f}" + r""" s for two tasks with 6 candidates, but reaches the 10 s limit from two tasks with 10 candidates or four tasks with 8 (Fig.~\ref{fig:s7}(a)); at the limit, its best plan loses up to 3.6\% of quality (Fig.~\ref{fig:s7}(b)). Solving each task separately takes under """ + f"{max(r['time_decomposed_s'] for r in mi):.1f}" + r""" s in every setting and is exact when tasks share no hand-off or workflow-level bound. Forward chaining takes """ + f"{np.mean([r['time_s'] for r in S['fc'] if r['catalogue'] == 1000]):.3f}" + r""" s for 1{,}000 descriptors and """ + f"{np.mean([r['time_s'] for r in S['fc'] if r['catalogue'] == 6000]):.2f}" + r""" s for 6{,}000.

\emph{Interpretation.} The exact joint program suits the workflows of the registries (2 to 5 tasks, few candidates per task), where it is solved within the time limit; beyond, the coupling should be handled by decomposition, solving the tasks separately and the coupled ones jointly. Derivation scales to national registries.

\subsection{S8: End to End and Robustness (RQ6)}

\emph{Objective and setting.} The three systems served the 82 citizen requests; the agents received the same confirmation of the goal as \sysname{}, as the citizen's reply, and needed it in """ + pc(e6["all_tools"]["needed_confirmation"]) + " and " + pc(e6["tool_search"]["needed_confirmation"]) + r""" of the requests. To rule out an advantage from shared wording, 45 requests (all Saudi ones and 30 Greek ones drawn at random) were served again with independent descriptions, then with capability mappings recomputed from them, so that \sysname{} uses no information the agents lack.

\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{s8_end_to_end.pdf}
\caption{S8, end to end. (a) Document obtained for the 82 requests; (b) robustness on 45 requests; (c) tool calls per request; (d) tasks opening a denied flow (S6)}
\label{fig:s8}
\end{figure*}

\begin{table*}[t]
\caption{S8, document obtained; differences with \sysname{} in points (95\% CI), McNemar's test}
\label{tab:e6}
\centering\fzs
\setlength{\tabcolsep}{6pt}
\begin{tabular}{@{}lccc@{}}
\toprule
Setting & \sysname{} & All tools & Tool Search \\
\midrule
""" + f"82 requests, title-based & {pc(e6['selectflow']['success'])} & {pc(e6['all_tools']['success'])} & {pc(e6['tool_search']['success'])} \\\\\n" + \
f"\\quad $\\Delta$, $p$ & -- & {100 * e6['selectflow_vs_all_tools']['diff_ci'][0]:+.1f} {ci(e6['selectflow_vs_all_tools']['diff_ci'])}, {e6['selectflow_vs_all_tools']['mcnemar_p']:.2f} & {100 * e6['selectflow_vs_tool_search']['diff_ci'][0]:+.1f} {ci(e6['selectflow_vs_tool_search']['diff_ci'])}, $<$0.001 \\\\\n" + \
f"45 requests, title-based & 100.0\\% & {pc(ind6['all_tools']['title_based_descriptions_same_requests'])} & {pc(ind6['tool_search']['title_based_descriptions_same_requests'])} \\\\\n" + \
f"45, independent descriptions & {pc(ind6['selectflow']['independent_descriptions'])} & {pc(ind6['all_tools']['independent_descriptions'])} & {pc(ind6['tool_search']['independent_descriptions'])} \\\\\n" + \
f"45, independent descr. and mappings & {pc(mp['selectflow_mapped_cards'])} & {pc(ind6['all_tools']['independent_descriptions'])} & {pc(ind6['tool_search']['independent_descriptions'])} \\\\\n" + \
f"\\quad $\\Delta$, $p$ & -- & {100 * mp['mapped_selectflow_vs_all_tools']['diff_ci'][0]:+.1f} {ci(mp['mapped_selectflow_vs_all_tools']['diff_ci'])}, {mp['mapped_selectflow_vs_all_tools']['mcnemar_p']:.2f} & {100 * mp['mapped_selectflow_vs_tool_search']['diff_ci'][0]:+.1f} {ci(mp['mapped_selectflow_vs_tool_search']['diff_ci'])}, $<$0.001 \\\\\n" + r"""\bottomrule
\end{tabular}
\end{table*}

\emph{Observations.} \sysname{} obtains the document for all 82 requests, against """ + pc(e6["all_tools"]["success"]) + r""" for the agent with all tools and """ + pc(e6["tool_search"]["success"]) + r""" for Tool Search (Table~\ref{tab:e6}), with fewer tool calls (""" + f"{e6['selectflow']['tool_calls']:.1f} against {e6['all_tools']['tool_calls']:.1f} and {e6['tool_search']['tool_calls']:.1f}" + r"""; Fig.~\ref{fig:s8}(c)). The agents do as well in Greece as \sysname{} but fail on 40\% of the Saudi requests, whose procedures chain more organizations. With independent descriptions, the agent with all tools falls from """ + pc(ind6["all_tools"]["title_based_descriptions_same_requests"]) + " to " + pc(ind6["all_tools"]["independent_descriptions"]) + r""" and Tool Search from """ + pc(ind6["tool_search"]["title_based_descriptions_same_requests"]) + " to " + pc(ind6["tool_search"]["independent_descriptions"]) + r""", whereas \sysname{} keeps 100\%, and """ + pc(mp["selectflow_mapped_cards"]) + r""" when its capability mappings also come from the independent descriptions; its remaining failures come from wrong mappings (precision """ + f"{mq['precision']:.2f}" + r""").

\emph{Interpretation.} The agents rely on the match between the citizen's words and the tool descriptions; \sysname{} decomposes the request into precise steps and gives each step a small toolkit, which makes it robust to how organizations describe their tools. The quality of the capability mapping becomes the limiting factor.

\subsection{Summary of Findings}

\begin{table*}[t]
\caption{Answers to the research questions}
\label{tab:findings}
\centering\fzs
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}p{0.8cm}p{14.4cm}@{}}
\toprule
RQ & Finding \\
\midrule
RQ1 & The wording of the step dominates selection (98\% against 77\% intended); errors are abstentions more than diversions; look-alikes divert 4.5--6.1\% of calls; up to 60 tools, more tools reduce abstention; the fitted model predicts selections better than relevance and is better calibrated. \\
RQ2 & One tool is too few (63\% abstention); retrieval peaks at three tools and then adds wrong calls and cost; \sysname{} sets the size per task (2.6 tools on average), matches the best-tuned retrieval and beats retrieval with two tools by 10 points. \\
RQ3 & Composition under the model gains 2.5--3.4 points over retrieval and QoS selection with a vague step, and matches tool search at 30\% of its cost. \\
RQ4 & Critics catch every incorrect candidate and revision repairs them; with a small catalogue, direct generation is also correct; the advantage of \sysname{} end to end grows with the number of tasks. \\
RQ5 & Results are stable for $k_{\max}\ge 3$, $B_h\ge 600$ tokens and $N\ge 2$; budget and sharing bases are the binding factors; the exact program suits registry workflows, decomposition is needed beyond. \\
RQ6 & \sysname{} obtains the document for every request, and keeps a 22- and 64-point margin over the agents when descriptions and mappings are independent; certified labels prevent the denied flows that declared annotations open in 26--33\% of the tasks. \\
\bottomrule
\end{tabular}
\end{table*}

Table~\ref{tab:findings} summarizes the answers. \sysname{}'s benefit comes mostly from verified decomposition into precise steps, small per-task toolkits and certified flows; toolkit composition alone brings a smaller gain, and its model corrects an initial assumption, since more definitions reduce abstentions instead of diluting the selection.

\subsection{Threats to Validity}

\emph{Construct.} The servers are simulated and the tools are generated from the registries, not published by the organizations; independent descriptions reduce, but do not remove, the influence of the generation. The intended tool is defined by the registry adequacy. \emph{Internal.} A single agent model was used, end-to-end runs were done once per request, and parameter effects in S5 are predicted by the fitted model. \emph{External.} The Saudi federation is small (15 requests), and the reference workflows assume complete registries. \emph{Annotation.} Citizen formulations were written by a model and checked by a model of the same provider; a human check of a sample is prepared and remains to be reported. The deterministic annotation of data categories is imperfect (""" + pc(fs["categories_ok"]) + r""" judged correct), although S6 changes little when the categories are corrected.

"""
open("paper/s5_eval.tex", "w").write(s)
print("section written:", len(s.split()), "words")
