"""Paired statistics: bootstrap confidence intervals, Wilcoxon, McNemar, Holm."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import stats


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n: int = 5000, seed: int = 0):
    """Mean of a - b with a 95% percentile interval."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, (n, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def wilcoxon(a, b) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(stats.binomtest(n10, n01 + n10, 0.5).pvalue)


def holm(p: Dict[str, float]) -> Dict[str, float]:
    items = sorted(p.items(), key=lambda kv: kv[1]); m = len(items); out = {}; run = 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v)); out[k] = run
    return out
