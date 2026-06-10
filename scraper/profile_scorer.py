"""Scoring and ranking of candidate profiles.

Combines a weighted, deterministic score (engagement, follower tier, posting
frequency, audience quality) with an AI-judged bio relevance from Gemini.
"""
from __future__ import annotations

import logging
import math
import os
import sys
import time
from typing import Optional

# Allow running this file directly (python scraper/profile_scorer.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GROQ_RATE_LIMIT_SLEEP, SCORE_WEIGHTS, TOP_N_PROFILES
from llm import groq_complete

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _tier_benchmark(followers: int) -> float:
    """Expected baseline engagement rate for a follower tier."""
    if followers > 1_000_000:
        return 0.01
    if followers > 100_000:
        return 0.015
    if followers > 10_000:
        return 0.03
    return 0.05


def score_profile(profile: dict, bio_relevance: float) -> float:
    """Compute the weighted relevance score for a profile.

    Args:
        profile: dict with at least followers, engagement_rate, posts_per_week,
            avg_likes, avg_comments.
        bio_relevance: 0.0-1.0 relevance of the bio to fat loss coaching.

    Returns:
        Weighted sum in roughly the 0.0-1.0 range.
    """
    followers = int(profile.get("followers") or 0)
    er = float(profile.get("engagement_rate") or 0.0)
    avg_likes = float(profile.get("avg_likes") or 0.0)
    avg_comments = float(profile.get("avg_comments") or 0.0)
    posts_per_week = float(profile.get("posts_per_week") or 0.0)

    # 1. Engagement relative to tier benchmark, capped at 1.0.
    benchmark = _tier_benchmark(followers)
    engagement_rate_score = min(er / (benchmark * 2), 1.0)

    # 2. Content relevance (passed through from Gemini).
    content_relevance = max(0.0, min(bio_relevance, 1.0))

    # 3. Follower tier on a log scale, capped at 1.0.
    follower_tier = min(math.log10(max(followers, 1000)) / 7, 1.0)

    # 4. Posting frequency, saturating at 3 posts/week.
    posting_frequency = min(posts_per_week / 3, 1.0)

    # 5. Audience quality: healthy comment-to-like ratio scores higher.
    comment_like_ratio = avg_comments / max(avg_likes, 1)
    audience_quality = 1.0 if 0.01 <= comment_like_ratio <= 0.08 else 0.5

    return (
        SCORE_WEIGHTS["engagement_rate"] * engagement_rate_score
        + SCORE_WEIGHTS["content_relevance"] * content_relevance
        + SCORE_WEIGHTS["follower_tier"] * follower_tier
        + SCORE_WEIGHTS["posting_frequency"] * posting_frequency
        + SCORE_WEIGHTS["audience_quality"] * audience_quality
    )


def score_bio_relevance(bio: str) -> float:
    """Ask the LLM (Groq) how relevant a bio is to fat loss coaching (0.0-1.0).

    Returns 0.0 for empty bios, and 0.5 on any error so a single failure doesn't
    sink a profile. Sleeps after the call to respect the rate limit.
    """
    if not bio or not bio.strip():
        return 0.0
    try:
        raw = groq_complete(
            "Rate how relevant this Instagram bio is to fat loss coaching "
            "on a scale of 0.0 to 1.0. Return ONLY a single float number, "
            f"nothing else.\nBio: {bio}"
        )
        if raw is None:
            return 0.5
        # Extract the first float-looking token.
        token = raw.strip().split()[0].strip().rstrip(",")
        return max(0.0, min(float(token), 1.0))
    except Exception as exc:
        logger.warning("Bio relevance scoring failed: %s", exc)
        return 0.5
    finally:
        time.sleep(GROQ_RATE_LIMIT_SLEEP)


def rank_profiles(
    profiles: list[dict],
    api_key: str = "",
    top_n: int = TOP_N_PROFILES,
) -> list[dict]:
    """Score every profile and return the ``top_n`` by relevance_score desc.

    ``api_key`` is accepted for backward compatibility but unused — bio relevance
    now runs on Groq (key read from the environment by the shared helper).
    """
    total = len(profiles)
    scored: list[dict] = []
    for i, profile in enumerate(profiles, start=1):
        bio_relevance = score_bio_relevance(profile.get("bio", ""))
        score = score_profile(profile, bio_relevance)
        profile["relevance_score"] = round(score, 5)
        scored.append(profile)
        print(f"Scoring {i}/{total}: @{profile.get('username')} - score: {score:.4f}")

    scored.sort(key=lambda p: p.get("relevance_score", 0.0), reverse=True)
    return scored[:top_n]


def main() -> None:
    """Test scoring with a dummy profile (no network for the weighted part)."""
    dummy = {
        "username": "demo_coach",
        "followers": 45_000,
        "engagement_rate": 0.04,
        "avg_likes": 1800.0,
        "avg_comments": 90.0,
        "posts_per_week": 4.0,
        "bio": "Fat loss coach | helping busy people lose 20lbs sustainably",
    }
    # Use a fixed bio_relevance so this runs without an API key.
    score = score_profile(dummy, bio_relevance=0.9)
    print(f"Deterministic score for @{dummy['username']}: {score:.4f}")


if __name__ == "__main__":
    main()
