#!/bin/sh
# First step of the campaign. Requires: export ANTHROPIC_API_KEY=...
set -e
# 1. Plumbing check with the cheap model on the running scenario (a few cents, synchronous)
python scripts/campaign.py e1 --fed data/greece_restaurant.json --n 200 --debug --sync
# 2. Pilot on the annotated registries, with the frozen model, in batch (go/no-go criterion in results/e1.json:
#    diff_vs_relevance must have its 95% interval above zero)
python scripts/campaign.py e1 --fed data/greece.json --n 2000
