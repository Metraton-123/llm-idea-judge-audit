# -*- coding: utf-8 -*-
"""Utilities for auditing LLM judges of research ideas."""
from .prompts import DIMENSIONS, RUBRICS, build_idea_text, build_prompt, prompt_sha256

__all__ = ["DIMENSIONS", "RUBRICS", "build_idea_text", "build_prompt", "prompt_sha256"]
__version__ = "0.1.0"
