"""Cross-profile analytics over the classified posts.

Every function takes a pandas DataFrame (built by :func:`load_data`) and returns
a DataFrame / Series / dict suitable for the dashboard. Functions are defensive:
empty inputs return empty results rather than raising.
"""
from __future__ import annotations

import logging
import os
import sys

# Allow running this file directly (python analysis/metrics.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LEAD_CATEGORIES = ("LEAD_GEN", "MIXED")


def load_data(db_session) -> pd.DataFrame:
    """Load posts joined with their profile's follower count into a DataFrame.

    Adds an ``engagement_rate`` column = (likes + comments) / max(followers, 1).
    """
    from db import Post, Profile

    rows = (
        db_session.query(
            Post.post_id,
            Post.username,
            Post.caption,
            Post.likes,
            Post.comments,
            Post.views,
            Post.media_type,
            Post.post_url,
            Post.posted_at,
            Post.primary_category,
            Post.secondary_category,
            Post.hook,
            Post.cta_text,
            Post.sentiment,
            Post.has_cta,
            Profile.followers,
        )
        .join(Profile, Post.username == Profile.username)
        .all()
    )
    df = pd.DataFrame(rows, columns=[
        "post_id", "username", "caption", "likes", "comments", "views",
        "media_type", "post_url", "posted_at", "primary_category",
        "secondary_category", "hook", "cta_text", "sentiment", "has_cta",
        "followers",
    ])
    if df.empty:
        return df

    for col in ("likes", "comments", "followers"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["engagement_rate"] = (df["likes"] + df["comments"]) / df["followers"].clip(lower=1)
    return df


def content_mix_by_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Return the per-profile percentage breakdown across categories."""
    if df.empty:
        return pd.DataFrame()
    counts = (
        df.groupby(["username", "primary_category"]).size().unstack(fill_value=0)
    )
    row_sums = counts.sum(axis=1).clip(lower=1)
    return counts.div(row_sums, axis=0) * 100


def engagement_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, and std of engagement_rate grouped by category."""
    if df.empty:
        return pd.DataFrame()
    return df.groupby("primary_category")["engagement_rate"].agg(
        ["mean", "median", "std"]
    )


def top_posts_by_category(df: pd.DataFrame, category: str, n: int = 10) -> pd.DataFrame:
    """Top ``n`` posts by likes within a given category."""
    cols = ["username", "likes", "comments", "hook", "cta_text", "post_url", "posted_at"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    subset = df[df["primary_category"] == category]
    return subset.sort_values("likes", ascending=False).head(n)[cols]


def top_hooks(df: pd.DataFrame, category: str, n: int = 20) -> list[str]:
    """Hook strings from the top ``n`` highest-liked posts in a category."""
    if df.empty:
        return []
    subset = df[df["primary_category"] == category].sort_values("likes", ascending=False)
    hooks = subset["hook"].head(n).tolist()
    return [h for h in hooks if h and str(h).strip() and str(h).lower() != "none"]


def cta_pattern_frequency(df: pd.DataFrame) -> pd.Series:
    """Top 20 most common CTA phrases among LEAD_GEN and MIXED posts."""
    if df.empty:
        return pd.Series(dtype=int)
    subset = df[df["primary_category"].isin(LEAD_CATEGORIES)]
    ctas = subset["cta_text"].dropna()
    ctas = ctas[ctas.astype(str).str.strip().str.lower() != "none"]
    return ctas.value_counts().head(20)


def lead_gen_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """For each lead-gen post, find the category of the previous 1 and 2 posts.

    Posts are ordered chronologically per profile, so prev/prev2 represent the
    content that "warmed up" the audience before the lead-gen push.
    """
    cols = ["username", "likes", "primary_category", "prev_category", "prev2_category"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    ordered = df.sort_values(["username", "posted_at"]).copy()
    ordered["prev_category"] = ordered.groupby("username")["primary_category"].shift(1)
    ordered["prev2_category"] = ordered.groupby("username")["primary_category"].shift(2)
    return ordered[ordered["primary_category"].isin(LEAD_CATEGORIES)][cols]


def format_vs_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """Mean engagement_rate by (media_type, primary_category)."""
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["media_type", "primary_category"])["engagement_rate"]
        .mean()
        .unstack(fill_value=0)
    )


def posting_frequency_by_archetype(df_profiles: pd.DataFrame) -> pd.DataFrame:
    """Mean posts_per_week grouped by archetype."""
    if df_profiles.empty:
        return pd.DataFrame()
    return df_profiles.groupby("archetype")["posts_per_week"].mean().to_frame("avg_posts_per_week")


def generate_insights_summary(df: pd.DataFrame) -> dict:
    """Produce a compact summary dict for the dashboard / strategy generator."""
    if df.empty:
        return {
            "total_profiles": 0,
            "total_posts": 0,
            "category_breakdown": {},
            "best_category_for_engagement": None,
            "most_common_cta": None,
            "top_performing_format": None,
            "avg_engagement_by_category": {},
        }

    category_breakdown = df["primary_category"].value_counts().to_dict()
    avg_eng = df.groupby("primary_category")["engagement_rate"].mean()
    best_category = avg_eng.idxmax() if not avg_eng.empty else None

    ctas = cta_pattern_frequency(df)
    most_common_cta = ctas.index[0] if not ctas.empty else None

    fmt = df.groupby("media_type")["engagement_rate"].mean()
    top_format = fmt.idxmax() if not fmt.empty else None

    return {
        "total_profiles": int(df["username"].nunique()),
        "total_posts": int(len(df)),
        "category_breakdown": {k: int(v) for k, v in category_breakdown.items()},
        "best_category_for_engagement": best_category,
        "most_common_cta": most_common_cta,
        "top_performing_format": top_format,
        "avg_engagement_by_category": {k: float(v) for k, v in avg_eng.items()},
    }


def _mock_df() -> pd.DataFrame:
    """A small synthetic DataFrame for standalone testing."""
    import datetime as dt

    base = dt.datetime(2026, 1, 1)
    rows = []
    cats = ["CREDIBILITY", "VIRAL", "LEAD_GEN", "MIXED"]
    media = ["reel", "carousel", "image"]
    for i in range(24):
        rows.append({
            "post_id": f"p{i}",
            "username": f"coach{i % 3}",
            "caption": f"caption {i}",
            "likes": 500 + i * 37,
            "comments": 20 + i * 3,
            "views": 1000 + i * 50,
            "media_type": media[i % 3],
            "post_url": f"https://instagram.com/p/p{i}",
            "posted_at": base + dt.timedelta(days=i),
            "primary_category": cats[i % 4],
            "secondary_category": None,
            "hook": f"Hook number {i}",
            "cta_text": "DM me PLAN" if cats[i % 4] in LEAD_CATEGORIES else None,
            "sentiment": "educational",
            "has_cta": cats[i % 4] in LEAD_CATEGORIES,
            "followers": 50_000 + (i % 3) * 10_000,
        })
    df = pd.DataFrame(rows)
    df["engagement_rate"] = (df["likes"] + df["comments"]) / df["followers"].clip(lower=1)
    return df


def main() -> None:
    df = _mock_df()
    print("== content_mix_by_profile ==")
    print(content_mix_by_profile(df), "\n")
    print("== engagement_by_category ==")
    print(engagement_by_category(df), "\n")
    print("== top_posts_by_category(VIRAL) ==")
    print(top_posts_by_category(df, "VIRAL", 3), "\n")
    print("== top_hooks(CREDIBILITY) ==")
    print(top_hooks(df, "CREDIBILITY", 5), "\n")
    print("== cta_pattern_frequency ==")
    print(cta_pattern_frequency(df), "\n")
    print("== lead_gen_sequence ==")
    print(lead_gen_sequence(df), "\n")
    print("== format_vs_engagement ==")
    print(format_vs_engagement(df), "\n")
    print("== generate_insights_summary ==")
    import json

    print(json.dumps(generate_insights_summary(df), indent=2, default=str))


if __name__ == "__main__":
    main()
