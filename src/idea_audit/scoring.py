# -*- coding: utf-8 -*-
"""Scoring pipeline: build prompts, call a judge, aggregate by idea."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

import pandas as pd

from . import prompts
from .client import JudgeConfig, chat

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_scores(raw: str) -> tuple[dict, bool]:
    if not raw:
        return {}, False
    text = raw.strip()
    m = FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m2 = JSON_RE.search(text)
        if not m2:
            return {}, False
        try:
            data = json.loads(m2.group(0))
        except json.JSONDecodeError:
            return {}, False
    if not isinstance(data, dict):
        return {}, False
    scores = {}
    for dim in prompts.DIMENSIONS:
        try:
            value = int(float(data.get(dim)))
        except (TypeError, ValueError):
            return {}, False
        if not 1 <= value <= 10:
            return {}, False
        scores[dim] = value
    return scores, True


def build_items(df: pd.DataFrame) -> List[dict]:
    items = []
    for row in df.to_dict("records"):
        idea_id = f"{row['call_id']}_{row['idea_index']}"
        idea_text = prompts.build_idea_text(row)
        for rnd in prompts.ROUNDS:
            raw_order = row.get(f"{rnd['name']}_order") or "-".join(prompts.DIMENSIONS)
            order = prompts.dims_from_order(str(raw_order))
            items.append({
                "key": f"{idea_id}|{rnd['name']}",
                "idea_id": idea_id,
                "round": rnd["name"],
                "temperature": rnd["temperature"],
                "top_p": prompts.TOP_P,
                "max_tokens": prompts.MAX_TOKENS,
                "order": "-".join(order),
                "prompt": prompts.build_prompt(idea_text, order),
            })
    return items


CALL_FIELDS = [
    "idea_id", "round", "order", "prompt_sha256", "judge", "model",
    "model_version", "temperature", "max_tokens", "finish_reason",
    "raw_output", "parse_ok", *prompts.DIMENSIONS, "error",
]


def _done_keys(path: Path) -> set:
    done = set()
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("parse_ok", "")).lower() == "true":
                    done.add((row["idea_id"], row["round"]))
    return done


def score_items(
    items: Iterable[dict],
    cfg: JudgeConfig,
    out_dir: str | Path,
    execute: bool = False,
    limit: int = 0,
    sleep: float = 0.05,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calls_path = out_dir / f"calls_{cfg.name}.csv"
    log_path = out_dir / f"calls_{cfg.name}.jsonl"
    done = _done_keys(calls_path)

    selected = list(items)
    if limit:
        selected = selected[: limit * len(prompts.ROUNDS)]

    new_file = not calls_path.exists()
    fh = calls_path.open("a", encoding="utf-8-sig" if new_file else "utf-8", newline="")
    writer = csv.DictWriter(fh, fieldnames=CALL_FIELDS)
    if new_file:
        writer.writeheader()

    called = skipped = failed = 0
    try:
        for item in selected:
            if (item["idea_id"], item["round"]) in done:
                skipped += 1
                continue
            if not execute:
                called += 1
                continue
            result = chat(cfg, item["prompt"], item["temperature"], top_p=item["top_p"])
            scores, ok = parse_scores(result.raw)
            called += 1
            if not result.ok:
                failed += 1
            writer.writerow({
                "idea_id": item["idea_id"], "round": item["round"], "order": item["order"],
                "prompt_sha256": hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest(),
                "judge": cfg.name, "model": cfg.model, "model_version": result.model_version,
                "temperature": item["temperature"], "max_tokens": cfg.max_tokens,
                "finish_reason": result.finish_reason, "raw_output": result.raw,
                "parse_ok": ok, **{d: scores.get(d, "") for d in prompts.DIMENSIONS},
                "error": result.error,
            })
            fh.flush()
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(json.dumps({
                    "judge": cfg.name, "idea_id": item["idea_id"], "round": item["round"],
                    "model_version": result.model_version, "usage": result.usage,
                    "parse_ok": ok, "error": result.error,
                }, ensure_ascii=False) + "\n")
            time.sleep(sleep)
    finally:
        fh.close()
    return {"judge": cfg.name, "called": called, "skipped": skipped, "failed": failed}


def aggregate(calls_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    calls_path, output_path = Path(calls_path), Path(output_path)
    rows = []
    with calls_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("parse_ok", "")).lower() != "true":
                continue
            try:
                values = {d: int(float(row[d])) for d in prompts.DIMENSIONS}
            except (TypeError, ValueError):
                continue
            rows.append({"idea_id": row["idea_id"], **values})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    grouped = df.groupby("idea_id")
    out = grouped[prompts.DIMENSIONS].agg(["mean", "var", "max"])
    out.columns = [f"{dim}_{stat}" for dim, stat in out.columns]
    out["maxgap"] = grouped[prompts.DIMENSIONS].apply(lambda g: (g.max() - g.min()).max())
    out.to_csv(output_path, encoding="utf-8-sig")
    return out
