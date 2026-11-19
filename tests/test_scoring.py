# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea_audit.scoring import build_items, parse_scores  # noqa: E402


def test_parse_scores_valid():
    scores, ok = parse_scores('{"Novelty": 7, "Feasibility": 6, "Soundness": 8, "Significance": 5}')
    assert ok
    assert scores == {"Novelty": 7, "Feasibility": 6, "Soundness": 8, "Significance": 5}


def test_parse_scores_fenced():
    raw = '```json\n{"Novelty": 7, "Feasibility": 6, "Soundness": 8, "Significance": 5}\n```'
    scores, ok = parse_scores(raw)
    assert ok and scores["Novelty"] == 7


def test_parse_scores_invalid():
    assert parse_scores("not json")[1] is False
    assert parse_scores('{"Novelty": 0}')[1] is False


def test_build_items_has_three_rounds():
    df = pd.DataFrame([{
        "call_id": 1, "idea_index": 1,
        "idea_title": "T", "research_question": "R",
        "method_family": "M", "contribution_type": "C", "description": "D",
    }])
    items = build_items(df)
    assert len(items) == 3
    assert {i["round"] for i in items} == {"round1", "round2", "round3"}
