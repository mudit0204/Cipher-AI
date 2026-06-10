"""Shared Groq LLM helper for classification and profile scoring.

Gemini's free tier is too small for this project's scale, so the text-LLM tasks
run on Groq (generous free tier, already used for Whisper transcription). The
client is created lazily and cached.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv

from config import GROQ_LLM_MODEL

load_dotenv()
logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Configure and cache the Groq client on first use."""
    global _client
    if _client is None:
        from groq import Groq

        _client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
    return _client


def groq_complete(
    prompt: str,
    system: Optional[str] = None,
    json_mode: bool = False,
    temperature: float = 0.1,
) -> Optional[str]:
    """Return the model's text response, or ``None`` on any failure.

    ``None`` (rather than a fallback string) lets callers distinguish a real
    answer from an API/rate-limit failure and retry later.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = {
        "model": GROQ_LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = _get_client().chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    except Exception as exc:
        logger.error("Groq completion failed: %s", exc)
        return None
