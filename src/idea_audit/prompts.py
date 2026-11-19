# -*- coding: utf-8 -*-
"""Frozen scoring prompt for the four-dimensional idea rubric.

The rubric is kept byte-for-byte stable so that scores from different
model runs are comparable.  Do not edit the string literals unless you
also bump ``PROMPT_VERSION`` and re-run all scorers.

Derived from the original study's ``score_new_experiment.py`` prompt.
"""
from __future__ import annotations

import hashlib
from typing import List, Mapping, Sequence

PROMPT_VERSION = "v1.1-2026-11"

RUBRICS = {
    "Novelty": {
        "rubric": """
1-3: Trivial, well-known, or poorly defined. No contribution.
4-6: Incremental improvement with some novelty.
7-8: Novel approach with sound reasoning and clear contribution.
9-10: Paradigm-shifting, opens new research directions.
"""
    },
    "Feasibility": {
        "rubric": """
1-3: Requires unavailable resources or theoretical breakthroughs.
4-6: Feasible but requires significant engineering or large datasets.
7-8: Achievable with moderate effort and existing tools.
9-10: Immediately testable off-the-shelf.
"""
    },
    "Soundness": {
        "rubric": """
1-3: Serious logical gaps or unsupported assumptions.
4-6: Reasonably sound with some weak links.
7-8: Strong logical flow, well-supported.
9-10: Flawless logic, fully justified.
"""
    },
    "Significance": {
        "rubric": """
1-3: Trivial or incremental contribution.
4-6: Modest contribution to a specific subfield.
7-8: Substantial impact, would advance the field.
9-10: Groundbreaking, would open new research directions.
"""
    },
}

DIMENSIONS: List[str] = ["Novelty", "Feasibility", "Soundness", "Significance"]

ROUNDS = [
    {"name": "round1", "temperature": 0.0},
    {"name": "round2", "temperature": 0.1},
    {"name": "round3", "temperature": 0.0},
]
TOP_P = 0.95
MAX_TOKENS = 150

PROMPT_HEADER = (
    "You are a rigorous AI research reviewer. "
    "Evaluate the following research idea on these four dimensions. "
    "Each dimension is scored on a 1-10 scale.\n\n"
)
PROMPT_FOOTER = (
    "Return ONLY a JSON object with all four scores, like this:\n"
    '{"Novelty": 7, "Feasibility": 6, "Soundness": 8, "Significance": 5}\n'
    "\n"
    "Do not include any other text."
)


def build_prompt(idea_text: str, dims: Sequence[str]) -> str:
    """Build the scoring prompt for one idea and one dimension order."""
    prompt = PROMPT_HEADER
    for dim in dims:
        prompt += f"**{dim} (1-10):** {RUBRICS[dim]['rubric']}\n\n"
    prompt += f"Idea: {idea_text}\n\n"
    prompt += PROMPT_FOOTER
    return prompt


def build_idea_text(row: Mapping[str, object]) -> str:
    """Serialize one idea row to the exact text used by the scorer."""
    return (
        f"Title: {row['idea_title']}\n"
        f"Research Question: {row['research_question']}\n"
        f"Method Family: {row['method_family']}\n"
        f"Contribution Type: {row['contribution_type']}\n"
        f"Description: {row['description']}"
    )


def dims_from_order(order: str) -> List[str]:
    """Parse a stored dimension order such as ``Novelty-Soundness-...``."""
    dims = order.split("-")
    if sorted(dims) != sorted(DIMENSIONS):
        raise ValueError(f"not a permutation of the four dimensions: {order!r}")
    return dims


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
