# SelectFlow portal — operational demonstrator

The SelectFlow portal: toolkit composition under model-controlled tool selection for cross-organizational workflows. It demonstrates the preparation phase (tool cards, tool-selection models, behaviour certification) and the request phase (confirmed interpretation, verified derivation, joint composition of model hosts and toolkits under C1–C6, repair with a gap ledger). It does not reproduce any experimental result of the paper.

## Launching

Open the published artifact link, or open `index.html` in a browser. Everything is in that one file (HTML, CSS, JavaScript); the only external resources are two Google Fonts, with system fallbacks.

## Architecture

The page is organized as service layers behind plain interfaces, so that each can be replaced by a real integration:

| Layer | Object | Role |
|---|---|---|
| Data store | `Store` | State, persistence, event log (history) |
| Preparation | `Preparation`, `SelectionModel`, `Certification` | Tool cards, multinomial-logit selection model per host, conformal certification (Benjamini–Hochberg) |
| Interpretation | `Interpretation` | Targets and held documents (simulated; live model optional) |
| Derivation | `Derivation` | Retrieval, constrained generation, four critics, revision, forward chaining |
| Governance | `Governance` | Sharing bases, competence, flow checks, host admissibility |
| Composition | `Composition` | Exact enumeration of toolkits per task and host, C1–C6, non-dominated plans |
| Repair | `Repair` | Violations, minimum-weight corrections, gap ledger |
| UI | `RENDER.*`, `ACTIONS.*` | Twelve pages sharing one state, hash routes (`#dashboard`, `#workspace`, …) |

## Pages

Dashboard · Environment configuration · Organizations and MCP servers · Tools catalog · Service descriptors · Preparation center · Sharing bases and governance · Workflow experimentation (six steps) · Workflow results · Execution history and monitoring · Organizations explorer · About.

## Real data, real models, real servers

- Catalogue: 35 organizations, 33 servers, 308 tools, 117 procedures and 1,867 sharing rules of the Greek (Mitos) and Saudi registries; 82 example requests (English, French, Arabic) with their workflows; 1,866 recorded agent calls used to fit the tool-selection model.
- Language models: Claude in claude.ai (Haiku, Sonnet, Opus 5.5); Anthropic API key, OpenAI-compatible endpoint or Ollama (list and download local models) in a copy opened locally.
- MCP: the connectors of your claude.ai account are listed, described and imported; Google Drive `list_recent_files` can be called live.
- Live agent calls: a toolkit is offered to the model as callable tools and the tool it calls is recorded.

## Simulation (remaining)

- MCP servers use `sim://` endpoints and are never contacted.
- Capability mapping: five noisy text-similarity passes stand in for five language-model passes.
- Tool-selection model: multinomial logit with a load term of estimated sign; training fits it by maximum likelihood on synthetic traces drawn from a hidden parameter.
- Certification: audits reveal a simulated ground truth for a sample; other tools are certified by conformal selection at the chosen ε.
- Interpretation and workflow generation are simulated; interpretation can use a live model call when the hosting platform offers one, after an explicit click and with a visible label.

Every simulated value carries a "Simulated" marker; configured, provider-declared, certified, observed and live values have their own markers.

## Demonstration scenario

“I want to open a restaurant in Kallithea.” The workflow chains tax registration, location attestation, fire protection, shop notification and its companion employer registration across five public bodies. The Fire Service publishes no tool to apply for the standard fire certificate, so composition fails on that task; the minimum-weight repair replaces it by the temporary certificate, and the composition is revalidated. All data are illustrative.

## Persistence

Changes are saved in the browser and, when the page runs in claude.ai with a signed-in viewer, in a private per-user document of the artifact's database (`data/users/<id>/portal-state`). The header shows which mode is active. “Reload the demonstration” restores the seed.

## Connecting real services later

Replace `Interpretation.live` / `Derivation.generate` with calls to an LLM API; replace the simulated servers by an MCP client that lists tools (`tools/list`) and feeds `Preparation.generateCards`; replace `SelectionModel.train` by fitting on logged agent calls; replace `Certification.run` by real audits and a trained classifier; move `Store` behind a REST or database backend. The UI only calls these objects.

## Tests performed

Headless-browser tests: all twelve pages render; the end-to-end scenario (submit, confirm, derive, compose, infeasible, repair, recompose, save) completes; tool cards, model training and certification run; organization creation, governance explanations and history work; state survives a reload.
