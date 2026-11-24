#!/bin/sh
# External components (run once). COLT and MCP-Bench come from their official repositories.
set -e
mkdir -p ext && cd ext
[ -d COLT ] || git clone --depth 1 https://github.com/quchangle1/COLT.git
[ -d mcp-bench ] || git clone --depth 1 https://github.com/Accenture/mcp-bench.git
cd ..
python scripts/extract_mcpbench_tools.py ext/mcp-bench data/mcpbench_tools.json
python scripts/build_mcpbench_federation.py
# COLT needs a GPU, PyTorch, BEIR and the Contriever weights from Hugging Face (see ext/COLT/README.md).
# TOUCAN extract (Hugging Face): python -c "from selectflow.data_sources import load_toucan_extract as L; import json; json.dump(L(20000), open('data/toucan_extract.json','w'))"
