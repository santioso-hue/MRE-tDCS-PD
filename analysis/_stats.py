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
    """Partial Pearson correlation of x and y controlling for covar (one or more columns). Returns (r, p).
    The p-value uses the covariate-adjusted t-test t = r*sqrt((n-2-k)/(1-r^2)) on df = n-2-k, where k is the
    number of covariate columns. (scipy.pearsonr on the residuals would test on df = n-2 and understate p by
    not paying for the df spent residualizing on the k covariates.)"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    cov = np.asarray(covar, float)
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    n, k = len(x), cov.shape[1]
    if n - 2 - k < 1:
        return np.nan, np.nan
    rx, ry = residualize(x, cov), residualize(y, cov)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan, np.nan
    r = float(np.corrcoef(rx, ry)[0, 1])
    if abs(r) >= 1:
        return r, 0.0
    df = n - 2 - k
    t = r * np.sqrt(df / (1 - r * r))
    return r, float(2 * stats.t.sf(abs(t), df))


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
