"""Real, provider-agnostic generation clients. NO MOCKS (global rule).

Deterministic settings matching the Nature paper: temperature=0.0, seed=62 where supported,
search tools left to the provider default unless disabled. Requires the relevant API key in env;
if missing we raise (we never fabricate model outputs).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

SEED = 62


class MissingKey(RuntimeError):
    pass


@dataclass
class Model:
    provider: str  # "openai" | "anthropic"
    name: str      # exact model id, logged into results for version provenance


def generate(model: Model, prompt: str, system: str | None = None, max_tokens: int = 1500) -> str:
    """Single-turn completion. Real API call; raises MissingKey if the key is absent."""
    if model.provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise MissingKey("OPENAI_API_KEY not set")
        from openai import OpenAI

        client = OpenAI()
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        resp = client.chat.completions.create(
            model=model.name, messages=msgs, temperature=0.0, seed=SEED, max_tokens=max_tokens
        )
        return resp.choices[0].message.content or ""
    if model.provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise MissingKey("ANTHROPIC_API_KEY not set")
        from anthropic import Anthropic

        client = Anthropic()
        resp = client.messages.create(
            model=model.name, max_tokens=max_tokens, temperature=0.0,
            system=system or "", messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    raise ValueError(f"unknown provider {model.provider}")
