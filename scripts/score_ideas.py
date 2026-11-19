#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Score an idea CSV with one or more OpenAI-compatible judges."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea_audit.client import JudgeConfig  # noqa: E402
from idea_audit.scoring import aggregate, build_items, score_items  # noqa: E402


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/sample_ideas.csv")
    ap.add_argument("--config", default="configs/judges.example.json")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--limit", type=int, default=0, help="limit to first N ideas")
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    items = build_items(df)
    config = load_config(args.config)
    names = args.models or list(config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name in names:
        spec = config[name]
        cfg = JudgeConfig(
            name=name,
            provider=spec["provider"],
            model=spec["model"],
            base_url=spec.get("base_url", ""),
            env_var=spec.get("env_var", ""),
            max_tokens=int(spec.get("max_tokens", 150)),
            extra=spec.get("extra") or {},
        )
        summary[name] = score_items(items, cfg, out, execute=args.execute, limit=args.limit)
        calls = out / f"calls_{name}.csv"
        if args.execute and calls.exists():
            aggregate(calls, out / f"scored_{name}_aggregate.csv")
        print(name, summary[name])
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
