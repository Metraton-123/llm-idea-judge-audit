# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea_audit.sampling import check_constraints, sample_pairs  # noqa: E402


def make_ideas(n_per_cell=25):
    settings = ["vanilla", "high_temperature", "generic_diversity",
                "seed_similarity", "generated_history", "dual_memory"]
    rows = []
    for topic in ["G001", "G002", "G003", "G004", "G005"]:
        for setting in settings:
            for i in range(1, n_per_cell + 1):
                rows.append({
                    "call_id": f"{topic}_{setting}", "group_id": topic,
                    "topic": topic, "setting": setting, "idea_index": i,
                    "description": f"synthetic {topic} {setting} {i}",
                })
    return pd.DataFrame(rows)


def test_sampling_constraints():
    ideas = make_ideas()
    pairs = sample_pairs(ideas, seed=43)
    report = check_constraints(pairs)
    assert report["formal"]["pairs"] == 100
    assert report["calibration"]["pairs"] == 15
    assert report["validation"]["pairs"] == 10
    assert report["validation"]["A"] == 5 and report["validation"]["B"] == 5
    assert report["intersections"] == {
        "calibration_validation": 0,
        "calibration_formal": 0,
        "validation_formal": 0,
    }
    assert report["formal_dual_memory_overlap"] == 0
    assert report["calibration"]["duplicate_ideas"] == 0
    assert report["validation"]["duplicate_ideas"] == 0
