# -*- coding: utf-8 -*-
"""Analysis helpers for LLM-judge audits.

These functions are intentionally small and dependency-light: pandas/numpy
for tabulation, scipy only for p-values when available.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from . import prompts

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None


def distribution(calls: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per-dimension distribution of raw integer scores."""
    rows = []
    for dim in prompts.DIMENSIONS:
        values = pd.to_numeric(calls[dim], errors="coerce").dropna()
        if values.empty:
            continue
        mode = values.mode().iloc[0]
        rows.append({
            "model": model,
            "dimension": dim,
            "n": int(len(values)),
            "mean": round(float(values.mean()), 3),
            "sd": round(float(values.std(ddof=1)), 3),
            "min": int(values.min()),
            "max": int(values.max()),
            "n_unique": int(values.nunique()),
            "mode": int(mode),
            "mode_share": round(float((values == mode).mean()), 3),
        })
    return pd.DataFrame(rows)


def cross_model_spearman(
    aggregates: Mapping[str, pd.DataFrame],
    dims: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Spearman correlations between per-idea model means."""
    dims = list(dims or prompts.DIMENSIONS)
    names = list(aggregates)
    rows = []
    for dim in dims:
        col = f"{dim}_mean"
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                left = aggregates[a].set_index("idea_id")[col]
                right = aggregates[b].set_index("idea_id")[col]
                joined = pd.concat([left, right], axis=1).dropna()
                if len(joined) < 3:
                    continue
                if spearmanr is not None:
                    rho, p = spearmanr(joined.iloc[:, 0], joined.iloc[:, 1])
                else:
                    rho = joined.iloc[:, 0].rank().corr(joined.iloc[:, 1].rank())
                    p = float("nan")
                rows.append({
                    "dimension": dim, "model_a": a, "model_b": b,
                    "spearman": round(float(rho), 3),
                    "p": None if not np.isfinite(p) else round(float(p), 4),
                    "n": len(joined),
                })
    return pd.DataFrame(rows)


def correlate_with_anchor(
    aggregates: Mapping[str, pd.DataFrame],
    anchor: pd.DataFrame,
    anchor_col: str,
) -> pd.DataFrame:
    """Spearman correlation between model means and an external anchor."""
    rows = []
    for model, agg in aggregates.items():
        merged = agg.merge(anchor[["idea_id", anchor_col]], on="idea_id", how="inner")
        for dim in prompts.DIMENSIONS:
            col = f"{dim}_mean"
            joined = merged[[col, anchor_col]].dropna()
            if len(joined) < 3:
                continue
            if spearmanr is not None:
                rho, p = spearmanr(joined[col], joined[anchor_col])
            else:
                rho = joined[col].rank().corr(joined[anchor_col].rank())
                p = float("nan")
            rows.append({
                "model": model, "dimension": dim, "anchor": anchor_col,
                "spearman": round(float(rho), 3),
                "p": None if not np.isfinite(p) else round(float(p), 4),
                "n": len(joined),
            })
    return pd.DataFrame(rows)


def write_report(
    out_dir: str | Path,
    dist: pd.DataFrame,
    cross: pd.DataFrame,
    validity: pd.DataFrame | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Judge audit report", ""]
    lines += ["## Distribution by dimension", "", dist.to_string(index=False), ""]
    lines += ["## Cross-model Spearman", "", cross.to_string(index=False), ""]
    if validity is not None:
        lines += ["## Anchor validity", "", validity.to_string(index=False), ""]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
