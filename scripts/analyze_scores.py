#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare judge outputs: distribution, cross-model Spearman, anchor validity."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea_audit import analysis  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calls", nargs="+", required=True,
                    help="model=calls_csv pairs, e.g. glm=out/calls_glm.csv")
    ap.add_argument("--aggregates", nargs="+", required=True,
                    help="model=aggregate_csv pairs")
    ap.add_argument("--anchor", default=None,
                    help="optional CSV with idea_id and anchor column")
    ap.add_argument("--anchor-col", default="index_standard")
    ap.add_argument("--out", default="out/analysis")
    args = ap.parse_args()

    calls = {}
    for spec in args.calls:
        name, path = spec.split("=", 1)
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = df[df["parse_ok"].astype(str).str.lower() == "true"]
        calls[name] = df

    aggs = {}
    for spec in args.aggregates:
        name, path = spec.split("=", 1)
        aggs[name] = pd.read_csv(path, encoding="utf-8-sig")

    dist = pd.concat([analysis.distribution(df, name) for name, df in calls.items()], ignore_index=True)
    cross = analysis.cross_model_spearman(aggs)
    validity = None
    if args.anchor:
        anchor = pd.read_csv(args.anchor, encoding="utf-8-sig")
        validity = analysis.correlate_with_anchor(aggs, anchor, args.anchor_col)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dist.to_csv(out / "model_distribution.csv", index=False, encoding="utf-8-sig")
    cross.to_csv(out / "cross_model_correlations.csv", index=False, encoding="utf-8-sig")
    if validity is not None:
        validity.to_csv(out / "anchor_validity.csv", index=False, encoding="utf-8-sig")
    analysis.write_report(out, dist, cross, validity)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
