"""Central configuration: constants, hashtag seeds, and scoring weights.

Everything tunable lives here so the rest of the pipeline never hard-codes
magic numbers. Import from this module rather than redefining values.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
# Seed hashtags used to discover candidate fat-loss profiles.
SEED_HASHTAGS: list[str] = [
    "fatloss",
    "fatlosstips",
    "weightlossjourney",
    "weightlosstransformation",
    "caloriedeficit",
    "fatlosscoach",
    "weightlosscoach",
    "fitnesscoach",
    "fatlossjourney",
    "weightlosstips",
    "macros",
    "iifym",
    "bodytransformation",
    "fatlossnutrition",
    "sustainableweightloss",
]

# --------------------------------------------------------------------------- #
# Profile scoring
# --------------------------------------------------------------------------- #
# Relative importance of each scoring component. Must sum to 1.0.
SCORE_WEIGHTS: dict[str, float] = {
    "engagement_rate": 0.30,
    "content_relevance": 0.25,
    "follower_tier": 0.20,
    "posting_frequency": 0.15,
    "audience_quality": 0.10,
}

# How many top profiles to keep after ranking.
TOP_N_PROFILES: int = 30

# How many candidate profiles to fetch metadata for during scoring. Capped well
# below the full discovered set because Instagram's web_profile_info endpoint is
# heavily rate-limited (a fresh burner gets 429'd after a few hundred requests).
# Candidates are ranked by hashtag co-occurrence so the cap keeps the most
# relevant ones.
MAX_PROFILES_TO_SCORE: int = 50

# How many posts to scrape per ranked profile.
POSTS_PER_PROFILE: int = 10

# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
# LLM for classification + bio-relevance scoring. Gemini's free tier is far too
# small (gemini-2.0-flash = 0/day, gemini-2.5-flash = 20/day), so these run on
# Groq instead, which has a much more generous free tier. Swap to
# "llama-3.1-8b-instant" for higher daily volume at some quality cost.
GROQ_LLM_MODEL: str = "llama-3.1-8b-instant"
# Groq free tier allows ~30 requests/minute => ~2s between calls.
GROQ_RATE_LIMIT_SLEEP: float = 2.1
# Cap content_text sent to the classifier. Groq's free tier is token-budgeted
# (llama-3.3-70b: 12k tokens/min, 100k/day), and full reel transcripts blow it.
# The marketing category is evident from the caption + opening, so truncating
# keeps quality while stretching the daily budget. ~4000 chars ~= 1000 tokens.
CLASSIFY_MAX_CHARS: int = 4000

# Gemini model — only still used by the dashboard's optional insight generator.
GEMINI_MODEL: str = "gemini-2.5-flash"

# Legacy Gemini pacing constant, kept for backward compatibility.
GEMINI_RATE_LIMIT_SLEEP: float = 13.0

# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
# Anchor paths to the project root (this file's directory) so they resolve the
# same no matter what working directory a process is launched from — otherwise
# `streamlit run dashboard/app.py` launched from dashboard/ would open a
# different (empty) database than `python main.py` does.
_ROOT: str = os.path.dirname(os.path.abspath(__file__))

DB_PATH: str = os.path.join(_ROOT, "fat_loss_insights.db")

# Directory for scratch artifacts (downloaded audio, exports, etc.).
DATA_DIR: str = os.path.join(_ROOT, "data")

# Where to stash downloaded Reel audio before transcription.
REELS_AUDIO_DIR: str = os.path.join(DATA_DIR, "reels")
