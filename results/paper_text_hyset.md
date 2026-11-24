# Text for the paper (English)

## Section 5, baselines (one sentence)

HYSET \cite{hyset2026} is compared on MCP-Bench, where its rankings are restricted to the same 278 tools as the
other methods; it is not retrained on the administrative federations, whose tools have no invocation history.

## Section 5, results (short paragraph)

HYSET learns tool representations from co-invocations. Trained twice on TOUCAN (26,623 and 30,000 queries), it reaches a recall@5 of
0.39 and 0.36 on validation queries from servers seen during training, but of 0.09 on 2,834 queries from 55 unseen servers.
On the 104 MCP-Bench tasks, a tool of the right server is among its five first tools for 39\% of the tasks, against
38\% for BM25. A set-level retriever thus offers no advantage for tools without an invocation history, which is the
situation of services that organizations newly publish as MCP tools; SelectFlow, which models the agent's selection
from tool definitions and observed calls on probes, does not require such a history.

## Threats to validity (one sentence)

HYSET is not evaluated on the administrative federations; given its recall on unseen servers, we expect it to perform
close to the lexical retriever there, but this comparison is left to future work.
