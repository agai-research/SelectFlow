# SelectFlow

**Toolkit Composition under Model-Controlled Tool Selection for Cross-Organizational Workflows**

Haithem Mezni, Zaki Brahmi, Fedia Ghedass, Afnan Albahli, Hela Elmannai, Shiyam Alalmaei, Aishah Alsehaim

SelectFlow composes, for every task of a cross-organizational workflow, a model host and the *toolkit* of MCP tools its LLM agent is offered. The agent still selects and calls the tool at run time; SelectFlow predicts that choice with a tool-selection model and composes toolkits under coverage, context budget, data-flow admissibility, toolkit size, hand-off compatibility and robust QoS constraints. Workflows are derived from a requester's confirmed request by constrained generation, deterministic critics and forward chaining, and infeasible compositions are repaired with minimum-weight corrections.

This repository contains the implementation, the experimental campaign and its results, the web portal, and the demo video material.

| Folder | Content |
|---|---|
| [`selectflow/`](selectflow) | Python package: model, selection model, derivation, flows, certification, composition, repair, LLM agent, baselines, experiment runners |
| [`scripts/`](scripts) | Campaign scripts (studies S1–S8), figure and table builders, data exporters, Colab/Kaggle scripts for the HYSET baseline (`scripts/colab`) |
| [`tests/`](tests) | Unit tests (propositions, MILP exactness, certification, end-to-end runs with a fake client) |
| [`data/`](data) | Real registries (Greek Mitos, Saudi official service pages), federations, requests, citizen formulations, MCP-Bench tools |
| [`results/`](results) | Raw observations, fitted models, study results, figures and tables |
| [`portal/`](portal) | Interactive web portal (single-file app) and its sources |
| [`video/`](video) | Demo video material: English subtitles, screenshots |

## Implementation

| Module | Role |
|---|---|
| `selectflow/model.py` | Tools, servers, model hosts, service descriptors, workflows, sharing bases |
| `selectflow/selection.py` | Tool-selection model (multinomial logit with a signed load effect), toolkit quality, maximum-likelihood estimation |
| `selectflow/derivation.py` | Deterministic critics (closure, alignment, companions, minimality), forward chaining, Algorithm 1 |
| `selectflow/flows.py` | Data-flow admissibility, admissible hosts |
| `selectflow/certification.py` | Stratified conformal selection with Benjamini–Hochberg |
| `selectflow/compose.py` | Exact joint composition (mixed-integer program), minimum-weight correction |
| `selectflow/repair.py` | Repair, non-dominated plans |
| `selectflow/agent/` | Claude probes (batches, cache, cost), constrained generation, end-to-end enactment, capability mapping |
| `selectflow/baselines/` | BM25 top-k, QoS-greedy, attraction ranking, HYSET adapter |
| `selectflow/experiments/` | Runners of the studies |

### Install and test

```bash
pip install -r requirements.txt pytest
python -m pytest -q            # 14 tests
python scripts/dryrun.py       # full rehearsal with a simulated agent, no API call
```

### Reproduce the studies

Model calls need `export ANTHROPIC_API_KEY=...`; the model and prices are set in `config.json`. Batches are used by default.

| Study | Script |
|---|---|
| S1 tool selection (fitted model, load effect) | `scripts/e1_full.py`, `scripts/e1_batch.py` |
| S2 toolkit size | `scripts/e3_sizes.py` |
| S3 toolkit composition vs. baselines | `scripts/e3_real.py` |
| S4 workflow derivation | `scripts/e2_chunks.py` |
| S5 parameters | `scripts/study_parameters.py`, `scripts/study_grid.py` |
| S6 certification and flows | `selectflow/experiments/runners.py` |
| S7 scalability | `scripts/study_scalability.py` |
| S8 end to end and robustness | `scripts/e6_chunks.py`, `scripts/e6_real.py`, `scripts/real_composition.py` |
| Figures and tables | `scripts/make_figures.py`, `scripts/make_study_figures.py` |

`results/` holds the raw observations of every run, so figures and statistics can be rebuilt without new calls.

### Data

- **Greece**: 81 procedures of the Mitos registry (category *Businessmen*, en.mitos.gov.gr) plus 12 referenced procedures; exporter `scripts/export_registries.py`.
- **Saudi Arabia**: 24 services from official service pages of the Ministry of Commerce, Balady, Civil Defense, ZATCA, GOSI, the Ministry of Human Resources and Jawazat (`data/saudi_urls.txt`).
- **Requests**: 82 requests with reference workflows; citizen formulations in English, French and Arabic.
- **External** (not redistributed; `scripts/fetch_external.sh`, `scripts/extract_toucan.py`): MCP-Bench, TOUCAN, HYSET, COLT.

## Web portal

**Online: https://agai-research.github.io/SelectFlow/portal/** — or open [`portal/index.html`](portal/index.html) in a browser. It runs SelectFlow on the Greek and Saudi registries (32 bodies, 33 servers, 308 tools, 117 procedures, 1,867 sharing rules) and 95 example requests:

- interpretation, verified derivation, joint composition and repair, with activity diagrams of workflows and toolkits;
- preparation (tool cards, tool-selection model fitted on 1,866 recorded agent calls, conformal certification);
- governance (sharing bases, flow-permission matrix), catalogue management, execution history.

Language models: on the online portal or a local copy, an Anthropic API key, an OpenAI-compatible endpoint or local models through Ollama (keys stay in the browser tab); Claude without a key, and your MCP connectors, when the page is opened in claude.ai. Sources and build: [`portal/src`](portal/src) (`python portal/src/build.py`); button test: `portal/src/test_buttons.py`.

## Demo video

A 3-minute guided tour of the portal (model choice, API key, new request, derivation, composition, servers, tools, connectors, governance), with English subtitles: [`video/`](video).

## Citation

The paper is under review. Please cite it by its title and authors until the publication details are available.
