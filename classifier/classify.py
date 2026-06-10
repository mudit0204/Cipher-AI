"""Post classification using Google Gemini 2.0 Flash.

Loads the API key from .env, runs each post's content_text through the model,
and writes the structured result back to the database.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional

# Allow running this file directly (python classifier/classify.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from classifier.prompts import SYSTEM_PROMPT, build_user_prompt
from config import CLASSIFY_MAX_CHARS, GROQ_RATE_LIMIT_SLEEP
from llm import groq_complete

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

VALID_CATEGORIES = {"CREDIBILITY", "VIRAL", "LEAD_GEN", "MIXED"}


def _fallback() -> dict:
    return {
        "primary_category": "UNKNOWN",
        "secondary_category": None,
        "hook": None,
        "cta_text": None,
        "sentiment": None,
        "has_cta": False,
        "confidence": 0.0,
    }


def classify_post(content_text: str) -> Optional[dict]:
    """Classify a single post's content_text into the marketing schema.

    Returns the parsed dict on success, a terminal UNKNOWN fallback for empty
    content, or ``None`` on an API/parse failure so the caller can leave the post
    unclassified and retry it on a later run (rather than burning it to UNKNOWN).
    """
    if not content_text or not content_text.strip():
        return _fallback()

    raw = groq_complete(
        build_user_prompt(content_text[:CLASSIFY_MAX_CHARS]),
        system=SYSTEM_PROMPT,
        json_mode=True,
    )
    if raw is None:
        return None  # API/rate-limit failure — caller leaves the post for retry
    try:
        data = json.loads(raw)
        # Normalize: ensure all expected keys exist.
        result = _fallback()
        result.update({k: data.get(k, result[k]) for k in result})
        if result["primary_category"] not in VALID_CATEGORIES:
            result["primary_category"] = "UNKNOWN"
        # Models sometimes emit the literal string "null"/"none" for empty fields.
        for k in ("secondary_category", "cta_text", "hook", "sentiment"):
            if isinstance(result[k], str) and result[k].strip().lower() in ("null", "none", ""):
                result[k] = None
        return result
    except Exception as exc:
        logger.error("Could not parse classification JSON: %s", exc)
        return None


def classify_all_posts(db_session) -> None:
    """Classify every post with content_text but no category yet."""
    from db import Post

    posts = (
        db_session.query(Post)
        .filter(Post.primary_category.is_(None))
        .filter(Post.content_text.isnot(None))
        .all()
    )
    total = len(posts)
    print(f"Classifying {total} posts...")

    counts: dict[str, int] = {}
    consecutive_failures = 0
    for i, post in enumerate(posts, start=1):
        result = classify_post(post.content_text)
        if result is None:
            # API failure — leave this post unclassified (NULL) for a later retry.
            consecutive_failures += 1
            if consecutive_failures >= 5:
                logger.error(
                    "Aborting classify: %d consecutive API failures — likely "
                    "rate-limited. Classified %d/%d so far; re-run to continue.",
                    consecutive_failures, i - 1 - (consecutive_failures - 1), total,
                )
                break
            time.sleep(GROQ_RATE_LIMIT_SLEEP)
            continue
        consecutive_failures = 0

        post.primary_category = result.get("primary_category")
        post.secondary_category = result.get("secondary_category")
        post.hook = result.get("hook")
        post.cta_text = result.get("cta_text")
        post.sentiment = result.get("sentiment")
        post.has_cta = bool(result.get("has_cta"))
        post.classified_at = datetime.utcnow()

        cat = post.primary_category or "UNKNOWN"
        counts[cat] = counts.get(cat, 0) + 1
        db_session.commit()  # persist each success so an abort keeps progress
        conf = result.get("confidence", 0.0)
        print(f"Classified {i}/{total}: @{post.username} - {cat} ({conf} confidence)")

        time.sleep(GROQ_RATE_LIMIT_SLEEP)

    db_session.commit()
    print("\nClassification summary:")
    for cat, n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {cat}: {n}")


def classify_single_test() -> None:
    """Classify a hardcoded sample post and pretty-print the result."""
    sample = (
        "[CAPTION]: Stop doing fasted cardio thinking it burns more fat. "
        "Research shows total daily calorie balance is what matters, not timing. "
        "Want a plan that actually works? DM me PLAN to get started.\n\n"
        "[Engagement: 5400 likes | 210 comments | 0 views]"
    )
    result = classify_post(sample)
    print(json.dumps(result, indent=2) if result else "API failure (None) - check key/quota")


def main() -> None:
    classify_single_test()


if __name__ == "__main__":
    main()
