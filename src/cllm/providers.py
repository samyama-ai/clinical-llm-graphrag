"""Real, provider-agnostic generation clients. NO MOCKS (global rule).

Deterministic settings matching the Nature paper: temperature=0.0, seed=62 where supported,
search tools left to the provider default unless disabled. Requires the relevant API key in env;
if missing we raise (we never fabricate model outputs).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

SEED = 62


class MissingKey(RuntimeError):
    pass


@dataclass
class Model:
    provider: str  # "openai" | "anthropic" | "gemini"
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
        kw = {"model": model.name, "messages": msgs, "temperature": 0.0, "seed": SEED}
        # gpt-5.x / o-series require max_completion_tokens; older models use max_tokens.
        if model.name.startswith(("gpt-5", "o1", "o3", "o4")):
            kw["max_completion_tokens"] = max_tokens
        else:
            kw["max_tokens"] = max_tokens
        resp = client.chat.completions.create(**kw)
        return resp.choices[0].message.content or ""
    if model.provider == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            raise MissingKey("GEMINI_API_KEY not set")
        # Use the OpenAI-compatible endpoint so we keep one client path.
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        resp = client.chat.completions.create(
            model=model.name, messages=msgs, temperature=0.0, max_tokens=max_tokens
        )
        return resp.choices[0].message.content or ""
    if model.provider == "claude_cli":
        # Route through the Claude Code CLI (uses the logged-in subscription, no API key/cost).
        exe = shutil.which("claude")
        if not exe:
            raise MissingKey("claude CLI not found on PATH")
        cmd = [exe, "-p", "--output-format", "text"]
        if model.name:
            cmd += ["--model", model.name]
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {r.stderr[:200]}")
        return r.stdout.strip()
    if model.provider == "gemini_cli":
        exe = shutil.which("antigravity") or shutil.which("gemini")
        if not exe:
            raise MissingKey("gemini/antigravity CLI not found on PATH")
        r = subprocess.run([exe, "-p", prompt], input="", capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"gemini CLI failed: {r.stderr[:200]}")
        return r.stdout.strip()
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
