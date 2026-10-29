"""Small-sample statistics in pure stdlib: chi-square tests,
Kruskal-Wallis, and Benjamini-Hochberg, for the calibration gate.

No scipy/numpy: p-values for chi-square distributions come from the
regularized upper incomplete gamma function Q(a, x), implemented with
the standard series expansion (x < a + 1) and Lentz continued fraction
(otherwise), after Numerical Recipes (gser/gcf). Accuracy is ~1e-12
relative in the ranges used here -- far beyond what a gate at q = 0.05
can distinguish. Kruskal-Wallis uses the chi-square approximation with
df = k - 1 (with the standard tie correction); with thousands of
observations per group that approximation is not the limiting factor.
"""
from __future__ import annotations

import math

_EPS = 3e-14
_FPMIN = 1e-300
_MAXIT = 1000


def _gamma_p_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x), series expansion."""
    if x == 0:
        return 0.0
    ap = a
    term = 1.0 / a
    total = term
    for _ in range(_MAXIT):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q_fraction(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x), continued fraction."""
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _MAXIT + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gamma_q(a: float, x: float) -> float:
    """Q(a, x): regularized upper incomplete gamma."""
    if a <= 0 or x < 0:
        raise ValueError("gamma_q requires a > 0 and x >= 0")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gamma_p_series(a, x)
    return _gamma_q_fraction(a, x)


def chi2_sf(x: float, df: int) -> float:
    """Survival function P(X >= x) of the chi-square distribution."""
    if x <= 0:
        return 1.0
    return gamma_q(df / 2.0, x / 2.0)


def chi_square_test(table: list) -> dict:
    """Chi-square test of independence on an r x c contingency table.

    Returns statistic, df, p-value, and Cramer's V (effect size).
    Zero-total rows/columns are dropped first.
    """
    rows = [r for r in table if sum(r) > 0]
    if not rows:
        raise ValueError("empty contingency table")
    cols = len(rows[0])
    keep = [j for j in range(cols) if sum(r[j] for r in rows) > 0]
    rows = [[r[j] for j in keep] for r in rows]
    n_rows, n_cols = len(rows), len(keep)
    total = sum(sum(r) for r in rows)
    row_tot = [sum(r) for r in rows]
    col_tot = [sum(rows[i][j] for i in range(n_rows)) for j in range(n_cols)]
    stat = 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            expected = row_tot[i] * col_tot[j] / total
            if expected > 0:
                stat += (rows[i][j] - expected) ** 2 / expected
    df = (n_rows - 1) * (n_cols - 1)
    p = chi2_sf(stat, df) if df > 0 else 1.0
    denom = total * min(n_rows - 1, n_cols - 1)
    cramers_v = math.sqrt(stat / denom) if denom > 0 else 0.0
    return {"statistic": stat, "df": df, "p_value": p,
            "effect_size": cramers_v, "effect_name": "cramers_v"}


def kruskal_wallis(groups: list) -> dict:
    """Kruskal-Wallis H test over k >= 2 samples, chi-square approx.

    Returns H (tie-corrected), df, p-value, and eta-squared-H
    ((H - k + 1) / (N - k), floored at 0) as the effect size.
    A degenerate input (all values identical) returns H = 0, p = 1.
    """
    groups = [list(g) for g in groups if g]
    k = len(groups)
    if k < 2:
        raise ValueError("kruskal_wallis needs at least two groups")
    pooled = sorted((v, gi) for gi, g in enumerate(groups) for v in g)
    n_total = len(pooled)
    ranks = [0.0] * n_total
    tie_sum = 0.0
    i = 0
    while i < n_total:
        j = i
        while j + 1 < n_total and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0      # ranks are 1-based
        for t in range(i, j + 1):
            ranks[t] = mean_rank
        run = j - i + 1
        if run > 1:
            tie_sum += run ** 3 - run
        i = j + 1
    rank_sums = [0.0] * k
    counts = [0] * k
    for (_, gi), r in zip(pooled, ranks):
        rank_sums[gi] += r
        counts[gi] += 1
    h = (12.0 / (n_total * (n_total + 1))
         * sum(rank_sums[gi] ** 2 / counts[gi] for gi in range(k))
         - 3 * (n_total + 1))
    if tie_sum:
        denom = 1.0 - tie_sum / (n_total ** 3 - n_total)
        if denom <= 0:   # every value identical: H is 0 by definition
            return {"statistic": 0.0, "df": k - 1, "p_value": 1.0,
                    "effect_size": 0.0, "effect_name": "eta_squared_h"}
        h /= denom
    h = max(h, 0.0)
    df = k - 1
    p = chi2_sf(h, df)
    eta2 = max(0.0, (h - k + 1) / (n_total - k)) if n_total > k else 0.0
    return {"statistic": h, "df": df, "p_value": p,
            "effect_size": eta2, "effect_name": "eta_squared_h"}


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple:
    """Wilson score interval for a binomial proportion k/n.

    The interval the Phase A-3 calibration controller and its held-out
    validation are defined over (experiments/
    model-calibrated-generation/DESIGN.md). z defaults to the 97.5th
    percentile of the standard normal (95% two-sided). n = 0 returns
    the vacuous interval (0, 1). Endpoints are clamped to [0, 1]; for
    k = 0 the lower endpoint is exactly 0 and for k = n the upper is
    exactly 1 (up to float rounding)."""
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError(f"wilson_interval: k={k} outside 0..{n}")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n
                                   + z2 / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def benjamini_hochberg(p_values: list, q: float = 0.05) -> list:
    """BH-adjusted p-values (same order as the input). A test is
    significant at level q iff its adjusted p-value is <= q."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running = min(running, p_values[idx] * m / rank)
        adjusted[idx] = min(running, 1.0)
    return adjusted
