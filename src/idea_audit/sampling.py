# -*- coding: utf-8 -*-
"""Stratified pair sampling for human/LLM idea-comparison studies.

The sampler is deterministic given a seed and a fixed pool.  It enforces:

* each pair contains one ``dual_memory`` idea and one comparison setting;
* three sets (calibration / validation / formal) share no ideas;
* formal comparison A and B use disjoint dual-memory ideas;
* each set has a fixed topic x comparison plan.

Example
-------
>>> from idea_audit.sampling import load_ideas, sample_pairs
>>> ideas = load_ideas("data/sample_ideas.csv")
>>> pairs = sample_pairs(ideas, seed=43)
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd

COMPARISONS = {
    "A": ("dual_memory", "vanilla"),
    "B": ("dual_memory", "generated_history"),
}
TOPICS = ["G001", "G002", "G003", "G004", "G005"]

# Formal: 10 pairs per topic per comparison (5 topics x 2 x 10 = 100)
FORMAL_PLAN = {t: {c: 10 for c in COMPARISONS} for t in TOPICS}
# Calibration: 15 pairs, A7 / B8
CALIBRATION_PLAN = {
    "G001": ["A", "A", "B"],
    "G002": ["A", "B", "B"],
    "G003": ["A", "B", "B"],
    "G004": ["A", "A", "B"],
    "G005": ["A", "B", "B"],
}
# Validation: 10 pairs, A5 / B5
VALIDATION_PLAN = {t: ["A", "B"] for t in TOPICS}


def load_ideas(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = ["call_id", "group_id", "setting", "idea_index", "description"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df.copy()
    df["idea_id"] = df["call_id"].astype(str) + "_" + df["idea_index"].astype(str)
    if "topic" not in df.columns:
        df["topic"] = df["group_id"]
    return df


class _Pool:
    def __init__(self, df: pd.DataFrame):
        if "idea_id" not in df.columns:
            df = df.copy()
            df["idea_id"] = df["call_id"].astype(str) + "_" + df["idea_index"].astype(str)
        self.df = df
        self.by_topic_setting = {
            (str(row.group_id), str(row.setting)): row.idea_id
            for row in df.itertuples()
        }
        self.ids_by_cell: Dict[tuple, List[str]] = defaultdict(list)
        for row in df.itertuples():
            self.ids_by_cell[(str(row.group_id), str(row.setting))].append(str(row.idea_id))
        self.text = dict(zip(df.idea_id, df.description.astype(str)))
        self.used: set = set()

    def available(self, topic: str, setting: str) -> List[str]:
        return [x for x in self.ids_by_cell.get((topic, setting), []) if x not in self.used]

    def draw(self, topic: str, comp: str, rng: random.Random) -> tuple:
        dm, other = COMPARISONS[comp]
        for _ in range(500):
            a_pool = self.available(topic, dm)
            b_pool = self.available(topic, other)
            if not a_pool or not b_pool:
                break
            a = rng.choice(a_pool)
            b = rng.choice(b_pool)
            if a == b:
                continue
            if self.text.get(a, "").strip() == self.text.get(b, "").strip():
                continue
            self.used.update([a, b])
            return a, b
        raise RuntimeError(f"cannot draw pair for topic={topic}, comparison={comp}")


def sample_pairs(
    ideas: pd.DataFrame,
    seed: int = 43,
    formal_pairs: Iterable[Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    """Return a pair table with phases calibration / validation / formal.

    ``formal_pairs`` can supply an already-frozen formal set (e.g. from a
    prior sampling log) so that only calibration/validation are redrawn.
    """
    pool = _Pool(ideas)
    rng = random.Random(seed)
    records: List[dict] = []

    # Formal
    if formal_pairs is not None:
        for p in formal_pairs:
            records.append(dict(p, phase="formal"))
            pool.used.update([str(p["idea_a_id"]), str(p["idea_b_id"])])
    else:
        for comp in COMPARISONS:
            for topic in TOPICS:
                for i in range(FORMAL_PLAN[topic][comp]):
                    a, b = pool.draw(topic, comp, rng)
                    records.append({
                        "phase": "formal", "topic": topic, "comparison": comp,
                        "setting_a": COMPARISONS[comp][0], "setting_b": COMPARISONS[comp][1],
                        "idea_a_id": a, "idea_b_id": b,
                    })

    # Calibration / validation
    for phase, plan in (("calibration", CALIBRATION_PLAN), ("validation", VALIDATION_PLAN)):
        for topic in TOPICS:
            for comp in plan[topic]:
                a, b = pool.draw(topic, comp, rng)
                records.append({
                    "phase": phase, "topic": topic, "comparison": comp,
                    "setting_a": COMPARISONS[comp][0], "setting_b": COMPARISONS[comp][1],
                    "idea_a_id": a, "idea_b_id": b,
                })

    out = pd.DataFrame(records)
    out.insert(0, "pair_id", [
        f"{'C' if ph == 'calibration' else 'V' if ph == 'validation' else 'F'}_{i:03d}"
        for i, ph in enumerate(out.phase, 1)
    ])
    out["random_seed"] = seed
    return out


def check_constraints(pairs: pd.DataFrame, formal_pairs: pd.DataFrame | None = None) -> dict:
    """Compute the design checks reported in the README."""
    report = {}
    for phase in ("calibration", "validation", "formal"):
        sub = pairs[pairs.phase == phase]
        ids = list(sub.idea_a_id.astype(str)) + list(sub.idea_b_id.astype(str))
        report[phase] = {
            "pairs": int(len(sub)),
            "A": int((sub.comparison == "A").sum()),
            "B": int((sub.comparison == "B").sum()),
            "unique_ideas": len(set(ids)),
            "duplicate_ideas": len(ids) - len(set(ids)),
        }
    sets = {
        phase: set(pairs[pairs.phase == phase].idea_a_id.astype(str))
        | set(pairs[pairs.phase == phase].idea_b_id.astype(str))
        for phase in ("calibration", "validation", "formal")
    }
    report["intersections"] = {
        "calibration_validation": len(sets["calibration"] & sets["validation"]),
        "calibration_formal": len(sets["calibration"] & sets["formal"]),
        "validation_formal": len(sets["validation"] & sets["formal"]),
    }
    formal = pairs[pairs.phase == "formal"]
    fa = formal[formal.comparison == "A"]
    fb = formal[formal.comparison == "B"]
    dm_a = set(fa[fa.setting_a == "dual_memory"].idea_a_id.astype(str))
    dm_b = set(fb[fb.setting_a == "dual_memory"].idea_a_id.astype(str))
    report["formal_dual_memory_overlap"] = len(dm_a & dm_b)
    return report
