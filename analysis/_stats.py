"""_stats.py - generic statistics helpers shared by the cohort analysis scripts (06, 09)."""
import numpy as np
from scipy import stats


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values. NaN p-values pass through as NaN and are excluded from ranking."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return q
    order = idx[np.argsort(p[idx])]
    m = idx.size
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        val = p[i] * m / k
        prev = min(prev, val)
        q[i] = prev
    return q


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def residualize(y, X):
    """Return residuals of y on design X (with intercept added)."""
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_pearson(x, y, covar):
    """Pearson(x, y) controlling for covar (one or more columns). Returns (r, p)."""
    if len(x) < 4:
        return np.nan, np.nan
    rx = residualize(np.asarray(x, float), np.asarray(covar, float))
    ry = residualize(np.asarray(y, float), np.asarray(covar, float))
    if rx.std() == 0 or ry.std() == 0:
        return np.nan, np.nan
    r, p = stats.pearsonr(rx, ry)
    return r, p


def rank_biserial_paired(a, b):
    """Matched-pairs rank-biserial effect size for a-b (drops non-finite and zero differences)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    tot = ranks.sum()
    return (ranks[d > 0].sum() - ranks[d < 0].sum()) / tot
