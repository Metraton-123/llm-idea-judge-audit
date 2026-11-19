#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sample calibration / validation / formal idea pairs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea_audit.sampling import check_constraints, load_ideas, sample_pairs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/sample_ideas.csv")
    ap.add_argument("--out", default="out/pairs.csv")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--formal-log", default=None,
                    help="optional existing sampling CSV; its phase=formal rows are kept")
    args = ap.parse_args()

    ideas = load_ideas(args.input)
    formal = None
    if args.formal_log:
        old = pd.read_csv(args.formal_log, encoding="utf-8-sig")
        formal = old[old.get("phase") == "formal"].to_dict("records") if "phase" in old else None

    pairs = sample_pairs(ideas, seed=args.seed, formal_pairs=formal)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out, index=False, encoding="utf-8-sig")

    report = check_constraints(pairs)
    report_path = out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
