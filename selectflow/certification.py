"""Certification of tool behaviours by conformal selection (Sect. 4.1, Eq. 10).

Within each stratum, audited tools are drawn at random, so audited and
unaudited tools are exchangeable. For a label l whose safe value is 0, the
conformal p-value of an unaudited tool t for the hypothesis "t is unsafe" is
    p(t) = (1 + #{audited t': unsafe, s(t') >= s(t)}) / (n_audited + 1),
where s is a classifier score that t is safe (ties counted, which is
conservative). Benjamini-Hochberg at level eps then certifies tools; the
expected proportion of unsafe tools among certified ones is <= eps
(Jin and Candes, JMLR 2023).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


def conformal_pvalues(s_test: Sequence[float], s_cal: Sequence[float], unsafe_cal: Sequence[bool]) -> np.ndarray:
    s_cal = np.asarray(s_cal, float); u = np.asarray(unsafe_cal, bool)
    bad = np.sort(s_cal[u])
    n = len(s_cal)
    s_test = np.asarray(s_test, float)
    # number of unsafe calibration scores >= s
    ge = len(bad) - np.searchsorted(bad, s_test, side="left")
    return (1.0 + ge) / (n + 1.0)


def benjamini_hochberg(p: Sequence[float], eps: float) -> np.ndarray:
    p = np.asarray(p, float); n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    order = np.sort(p)
    ok = np.nonzero(order <= eps * np.arange(1, n + 1) / n)[0]
    if len(ok) == 0:
        return np.zeros(n, bool)          # nothing certified
    j = ok[-1] + 1
    return p <= eps * j / n


def certify(strata: Dict[str, Dict[str, list]], eps: float) -> Set[str]:
    """strata[name] = {"test_ids", "test_scores", "cal_scores", "cal_unsafe"} -> certified ids."""
    out: Set[str] = set()
    for st in strata.values():
        if not st["test_ids"]:
            continue
        p = conformal_pvalues(st["test_scores"], st["cal_scores"], st["cal_unsafe"])
        sel = benjamini_hochberg(p, eps)
        out |= {t for t, s in zip(st["test_ids"], sel) if s}
    return out


def stratified_audit(ids_by_stratum: Dict[str, List[str]], frac: float, rng: np.random.Generator) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Random audit within strata: returns (audited, unaudited) ids per stratum."""
    aud, rest = {}, {}
    for k, ids in ids_by_stratum.items():
        ids = list(ids); rng.shuffle(ids)
        n = max(1, int(round(frac * len(ids))))
        aud[k], rest[k] = ids[:n], ids[n:]
    return aud, rest
