# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea_audit import prompts  # noqa: E402


def test_prompt_contains_all_dimensions():
    text = prompts.build_idea_text({
        "idea_title": "T", "research_question": "R",
        "method_family": "M", "contribution_type": "C", "description": "D",
    })
    prompt = prompts.build_prompt(text, prompts.DIMENSIONS)
    for dim in prompts.DIMENSIONS:
        assert f"**{dim} (1-10):**" in prompt
    assert "Return ONLY a JSON object" in prompt


def test_dims_from_order_roundtrip():
    order = "Soundness-Feasibility-Significance-Novelty"
    assert sorted(prompts.dims_from_order(order)) == sorted(prompts.DIMENSIONS)


def test_prompt_hash_is_stable():
    text = "Title: T\nResearch Question: R\nMethod Family: M\nContribution Type: C\nDescription: D"
    prompt = prompts.build_prompt(text, ["Novelty", "Feasibility", "Soundness", "Significance"])
    assert prompts.prompt_sha256(prompt) == prompts.prompt_sha256(prompt)
    assert len(prompts.prompt_sha256(prompt)) == 64
