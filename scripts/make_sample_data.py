#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the synthetic 750-row sample used by the quickstart."""
from __future__ import annotations

import csv
import random
from pathlib import Path

TOPICS = [("G001", "computer vision"), ("G002", "NLP"), ("G003", "embodied AI"),
          ("G004", "ML theory"), ("G005", "AI for science")]
SETTINGS = ["vanilla", "high_temperature", "generic_diversity",
            "seed_similarity", "generated_history", "dual_memory"]


def main() -> None:
    rng = random.Random(7)
    out = Path(__file__).resolve().parents[1] / "data" / "sample_ideas.csv"
    rows = []
    for group, topic in TOPICS:
        for setting in SETTINGS:
            for i in range(1, 26):
                rows.append({
                    "call_id": f"{group}-{setting}", "group_id": group, "topic": topic,
                    "setting": setting, "idea_index": i,
                    "idea_title": f"Synthetic idea {group}-{setting}-{i}",
                    "research_question": f"How can method {i} improve {topic}?",
                    "method_family": rng.choice(["Transformer", "Diffusion", "RL", "Hybrid"]),
                    "contribution_type": rng.choice(["New method", "New application", "Benchmark", "Analysis"]),
                    "description": f"Synthetic description for {group}/{setting}/{i}. "
                                   f"This row exists only to exercise the pipeline; it is not real research content.",
                })
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
