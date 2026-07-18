"""Alias-geometric gap samplers used by the production kernels.

Reconstituted verbatim from the copies inlined in evolution_strip_dfe.py
(header there: "Self-contained (alias helpers inlined from evolution_fast4)").
The production kernels (evolution_strip.py, census2.py, ...) import these
two names from `evolution_fast4`; the round-8 kernel bundle ships the
kernels but not evolution_fast4.py itself, so it is rebuilt here byte-for-byte
from the inlined copy. Bit-exact reproduction of deposited runs at equal seed
(seeds 12,900,000-12,900,002 of ridge_profile_runs.parquet) is the acceptance
test that this reconstitution is faithful.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _gap(prob, alias, tail_start, log1m):
    u = np.random.rand() * prob.shape[0]
    i = np.int64(u)
    if u - i >= prob[i]:
        i = alias[i]
    if i == prob.shape[0] - 1:
        return tail_start + 1 + np.int64(np.log(1.0 - np.random.rand())
                                         / log1m)
    return i + 1


def build_geom_alias(p, table_n=2048):
    G = table_n - 1
    g = np.arange(1, G + 1, dtype=np.float64)
    pmf = p * (1.0 - p) ** (g - 1.0)
    tail = (1.0 - p) ** G
    w = np.concatenate([pmf, [tail]])
    w /= w.sum()
    n = table_n
    prob = np.zeros(n, np.float64)
    alias = np.zeros(n, np.int64)
    scaled = w * n
    small = [i for i in range(n) if scaled[i] < 1.0]
    large = [i for i in range(n) if scaled[i] >= 1.0]
    while small and large:
        s = small.pop(); l = large.pop()
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] -= (1.0 - scaled[s])
        (small if scaled[l] < 1.0 else large).append(l)
    for i in large + small:
        prob[i] = 1.0
        alias[i] = i
    return prob, alias
