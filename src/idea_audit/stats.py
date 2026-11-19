# -*- coding: utf-8 -*-
"""Small inter-rater reliability helpers (no external stats package needed)."""
from __future__ import annotations

from collections import Counter
from typing import List, Sequence


def krippendorff_alpha_nominal(units: Sequence[Sequence[object]]) -> float:
    """Krippendorff's alpha for nominal data.

    ``units`` is a sequence of rating lists; ``None`` marks a missing rating.
    """
    filtered = [[x for x in unit if x is not None] for unit in units]
    filtered = [u for u in filtered if len(u) >= 2]
    if not filtered:
        return float("nan")
    n = sum(len(u) for u in filtered)
    if n <= 1:
        return float("nan")

    # observed disagreement
    observed = 0.0
    for unit in filtered:
        m = len(unit)
        counts = Counter(unit)
        disagree = m * m - sum(c * c for c in counts.values())
        observed += disagree / (m - 1)
    observed /= n

    # expected disagreement
    all_values = [x for unit in filtered for x in unit]
    counts = Counter(all_values)
    expected = (n * n - sum(c * c for c in counts.values())) / (n * (n - 1))
    if expected == 0:
        return 1.0
    return 1.0 - observed / expected


def icc_2k(matrix: Sequence[Sequence[float]]) -> float:
    """ICC(2,k) for a pairs x raters matrix (two-way random, average raters)."""
    rows = [list(map(float, r)) for r in matrix if len(r) >= 2]
    if len(rows) < 2:
        return float("nan")
    k = len(rows[0])
    n = len(rows)
    grand = sum(sum(r) for r in rows) / (n * k)
    row_means = [sum(r) / k for r in rows]
    col_means = [sum(r[i] for r in rows) / n for i in range(k)]

    ss_total = sum((x - grand) ** 2 for r in rows for x in r)
    ss_rows = k * sum((m - grand) ** 2 for m in row_means)
    ss_cols = n * sum((m - grand) ** 2 for m in col_means)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1) if k > 1 else 0.0
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0.0
    denom = ms_rows + (ms_cols - ms_error) / n
    if denom == 0:
        return float("nan")
    return (ms_rows - ms_error) / denom
