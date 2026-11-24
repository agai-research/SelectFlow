"""HYSET (Hong et al., 2026; https://github.com/stormwther18/HYSET) as the set-level baseline of E3.

Needs the HYSET repository, a checkpoint trained on a corpus that contains our tools
(scripts/toucan_to_hyset.py), the frozen encoder, PyTorch and preferably a GPU."""
from __future__ import annotations

import json
import sys
from typing import Dict, List


def hyset_scorer(repo: str, checkpoint: str, encoder: str, name_map_path: str, device: str = "cuda",
                 k1: int = 15, k_pool: int = 20, ranking_length: int = 10):
    sys.path.insert(0, f"{repo}/src")
    from hyset_model import HYSETModel, func_name_of
    model = HYSETModel.load(checkpoint, encoder, device=device)
    name_map: Dict[str, List[str]] = json.load(open(name_map_path))
    back = {func_name_of(t, a): tid for tid, (t, a) in name_map.items()}

    def scorer(query: str, candidate_ids: List[str]) -> List[str]:
        res = model.retrieve(query, k1=k1, k_pool=k_pool, ranking_length=ranking_length).as_dict()
        names = list(res.get("predicted_set", [])) + list(res.get("ranking", []))
        ranked = [back[n] for n in names if n in back]
        ranked = list(dict.fromkeys(ranked))
        return [t for t in ranked if t in candidate_ids] + [t for t in candidate_ids if t not in ranked]
    return scorer
