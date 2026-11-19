# -*- coding: utf-8 -*-
"""Minimal OpenAI-compatible chat client used by the scoring scripts.

Only the Python standard library is required for API calls.  API keys are
read from environment variables and are never written to logs.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

KEY_ENV = {
    "qwen": "DSH_QWEN_API_KEY",
    "zhipu": "DSH_ZHIPU_API_KEY",
    "deepseek": "DSH_DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
}

DEFAULT_BASE_URLS = {
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
}


@dataclass
class JudgeConfig:
    """Configuration for one judge.

    ``extra`` is merged into the request body (e.g. ``enable_thinking`` or
    ``reasoning_effort``).  Keys are read from ``env_var`` at call time.
    """

    name: str
    provider: str
    model: str
    base_url: str = ""
    env_var: str = ""
    max_tokens: int = 150
    extra: Mapping[str, Any] = field(default_factory=dict)
    timeout: int = 180

    def __post_init__(self) -> None:
        if not self.base_url:
            self.base_url = DEFAULT_BASE_URLS[self.provider]
        if not self.env_var:
            self.env_var = KEY_ENV[self.provider]

    def key(self) -> str:
        value = os.environ.get(self.env_var, "").strip()
        if not value:
            raise RuntimeError(
                f"missing API key: set environment variable {self.env_var}"
            )
        return value


@dataclass
class CallResult:
    ok: bool
    raw: str = ""
    model_version: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    attempts: int = 0


def chat(
    cfg: JudgeConfig,
    prompt: str,
    temperature: float,
    top_p: float = 0.95,
    retry_limit: int = 3,
) -> CallResult:
    """Call one chat completion; retries transient errors."""
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": cfg.max_tokens,
    }
    payload.update(dict(cfg.extra))

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.key()}",
    }

    last_err = None
    for attempt in range(1, retry_limit + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            choice = body["choices"][0]
            message = choice.get("message") or {}
            return CallResult(
                ok=True,
                raw=(message.get("content") or "").strip(),
                model_version=body.get("model"),
                finish_reason=choice.get("finish_reason"),
                usage=body.get("usage") or {},
                attempts=attempt,
            )
        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'ignore')[:200]}"
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
        if attempt < retry_limit:
            time.sleep(min(2 ** attempt, 8))
    return CallResult(ok=False, error=last_err, attempts=retry_limit)
