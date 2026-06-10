"""Pipeline orchestrator for the Fat Loss Insights Engine.

Run individual steps or the whole pipeline:

    python main.py --step discover
    python main.py --step score
    python main.py --step scrape
    python main.py --step enrich
    python main.py --step classify
    python main.py --step analyze
    python main.py --step all

Each step persists to the database before the next begins, so the pipeline is
resumable across days. Steps print their elapsed time and never crash silently.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime

from dotenv import load_dotenv

from config import (
    DATA_DIR,
    MAX_PROFILES_TO_SCORE,
    POSTS_PER_PROFILE,
    SEED_HASHTAGS,
    TOP_N_PROFILES,
)
from db import Profile, create_tables, get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

DISCOVERED_FILE = os.path.join(DATA_DIR, "discovered_profiles.txt")
DISCOVERED_JSON = os.path.join(DATA_DIR, "discovered_profiles.json")
INSIGHTS_FILE = os.path.join(DATA_DIR, "insights_summary.json")


def _ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _instagram_login():
    from scraper.insta_scraper import login

    username = os.getenv("INSTAGRAM_USERNAME", "")
    password = os.getenv("INSTAGRAM_PASSWORD", "")  # optional with cookie auth
    browser = os.getenv("INSTAGRAM_COOKIE_BROWSER", "edge")
    # Default to a cookies.txt dropped next to main.py if not set in .env.
    default_cookie = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "instagram_cookies.txt")
    cookiefile = os.getenv("INSTAGRAM_COOKIE_FILE", "") or default_cookie
    if not username:
        raise RuntimeError("INSTAGRAM_USERNAME missing in .env")
    return login(username, password, browser=browser, cookiefile=cookiefile)


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def step_discover() -> None:
    from scraper.insta_scraper import discover_profiles_from_hashtags

    L = _instagram_login()
    candidates = discover_profiles_from_hashtags(L, SEED_HASHTAGS, posts_per_tag=60)

    # Drop private accounts (their posts can't be scraped) and rank by hashtag
    # co-occurrence — accounts appearing under more seed tags are more relevant.
    public = [rec for rec in candidates.values() if not rec.get("is_private")]
    ranked = sorted(
        public,
        key=lambda r: (r.get("tag_count", 0), r.get("is_verified", False), r["username"]),
        reverse=True,
    )
    dropped = len(candidates) - len(public)
    print(f"Found {len(candidates)} candidates "
          f"({dropped} private dropped, {len(ranked)} scrapeable). "
          f"Scoring will use the top {MAX_PROFILES_TO_SCORE}.")

    _ensure_dirs()
    with open(DISCOVERED_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(r["username"] for r in ranked))
    with open(DISCOVERED_JSON, "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2)
    print(f"Saved ranked candidates to {DISCOVERED_FILE} and {DISCOVERED_JSON}")


def step_score() -> None:
    from scraper.insta_scraper import get_profile_metadata, save_profiles_to_db
    from scraper.profile_scorer import rank_profiles

    if not os.path.exists(DISCOVERED_FILE):
        raise RuntimeError(f"{DISCOVERED_FILE} not found — run --step discover first")
    with open(DISCOVERED_FILE, encoding="utf-8") as f:
        usernames = [u.strip() for u in f if u.strip()]

    # Cap how many candidates we fetch metadata for — web_profile_info is heavily
    # rate-limited, so scoring the full discovered set would get the account 429'd.
    # The discover step writes usernames ranked by relevance, so this keeps the best.
    if len(usernames) > MAX_PROFILES_TO_SCORE:
        print(f"Capping {len(usernames)} candidates to top {MAX_PROFILES_TO_SCORE} for scoring")
        usernames = usernames[:MAX_PROFILES_TO_SCORE]

    api_key = os.getenv("GEMINI_API_KEY", "")
    L = _instagram_login()

    profiles: list[dict] = []
    consecutive_failures = 0
    for i, username in enumerate(usernames, start=1):
        print(f"Fetching metadata {i}/{len(usernames)}: @{username}")
        meta = get_profile_metadata(L, username)
        if meta:
            profiles.append(meta)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            # Many failures in a row means Instagram is throttling/blocking us;
            # stop rather than hammer the account (and risk a ban).
            if consecutive_failures >= 10:
                logger.error(
                    "Aborting score: %d consecutive profile failures — likely "
                    "rate-limited. Collected %d profiles so far.",
                    consecutive_failures, len(profiles),
                )
                break
        # Pace every attempt (success OR failure) so we never burst requests,
        # which is what triggers/extends Instagram's rate-limit penalty.
        if i < len(usernames):
            time.sleep(random.uniform(3, 7))

    ranked = rank_profiles(profiles, api_key, top_n=TOP_N_PROFILES)
    with get_session() as session:
        save_profiles_to_db(ranked, session)
    print(f"Saved top {len(ranked)} profiles to database")


def step_scrape() -> None:
    from scraper.insta_scraper import (
        prune_to_profiles,
        save_posts_to_db,
        scrape_posts_for_profile,
        trim_profile_posts,
    )

    with get_session() as session:
        profiles = (
            session.query(Profile)
            .order_by(Profile.relevance_score.desc())
            .limit(TOP_N_PROFILES)
            .all()
        )
        usernames = [p.username for p in profiles]

    L = _instagram_login()
    total = 0
    for i, username in enumerate(usernames, start=1):
        print(f"Scraping {i}/{len(usernames)}: @{username}")
        posts = scrape_posts_for_profile(L, username, POSTS_PER_PROFILE)
        with get_session() as session:
            save_posts_to_db(posts, session)
            # Enforce the per-profile cap (older runs may have stored more).
            trim_profile_posts(session, username, POSTS_PER_PROFILE)
        total += len(posts)

    # Enforce the profile-set cap: drop any profiles (and their posts) left over
    # from earlier runs so the DB equals the current top-N set.
    with get_session() as session:
        removed = prune_to_profiles(session, usernames)
    print(f"Scraped and saved {total} posts total "
          f"(kept top {len(usernames)} profiles, pruned {removed} stale)")


def step_enrich() -> None:
    from enrichment.ocr import enrich_all_posts

    with get_session() as session:
        enrich_all_posts(session)


def step_classify() -> None:
    from classifier.classify import classify_all_posts

    with get_session() as session:
        classify_all_posts(session)


def step_analyze() -> None:
    from analysis.metrics import generate_insights_summary, load_data

    with get_session() as session:
        df = load_data(session)
    summary = generate_insights_summary(df)
    text = json.dumps(summary, indent=2, default=str)
    print(text)
    _ensure_dirs()
    with open(INSIGHTS_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved insights to {INSIGHTS_FILE}")


def step_probe() -> None:
    """Diagnostic: make exactly ONE web_profile_info request and report status.

    Use to check whether the burner is still rate-limited before running the
    full ``score`` batch. Target a specific account with ``PROBE_USERNAME`` in
    .env, otherwise the first discovered candidate (or a fallback) is used.
    """
    from scraper.insta_scraper import probe_profile

    target = os.getenv("PROBE_USERNAME", "").lstrip("@")
    if not target and os.path.exists(DISCOVERED_FILE):
        with open(DISCOVERED_FILE, encoding="utf-8") as f:
            target = next((line.strip() for line in f if line.strip()), "")
    if not target:
        target = "instagram"  # well-known public account fallback

    L = _instagram_login()
    print(f"Probing exactly one profile: @{target}")
    report = probe_profile(L, target)

    if report["ok"]:
        print(f"[CLEAR] HTTP 200 for @{report.get('username') or target} "
              f"(followers={report.get('followers')}). "
              f"Rate limit looks CLEAR - safe to run --step score.")
    else:
        retry = report.get("retry_after")
        retry_txt = f"Retry-After={retry}s" if retry else "no Retry-After header"
        err = f" - {report['error']}" if report.get("error") else ""
        print(f"[BLOCKED] HTTP {report.get('status')} for @{target} ({retry_txt}){err}")
        print("Still rate-limited / not usable. Wait longer, then probe again before scoring.")


STEPS = {
    "discover": step_discover,
    "score": step_score,
    "scrape": step_scrape,
    "enrich": step_enrich,
    "classify": step_classify,
    "analyze": step_analyze,
    "probe": step_probe,
}

# Pipeline order for `--step all`. 'probe' is a diagnostic and intentionally excluded.
ORDER = ["discover", "score", "scrape", "enrich", "classify", "analyze"]


def run_step(name: str) -> None:
    print(f"\n=== STEP: {name} ===  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    start = time.time()
    try:
        STEPS[name]()
    except Exception as exc:
        logger.error("Step '%s' failed: %s", name, exc)
    finally:
        print(f"--- {name} finished in {time.time() - start:.1f}s ---")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fat Loss Insights Engine pipeline")
    parser.add_argument(
        "--step",
        required=True,
        choices=list(STEPS.keys()) + ["all"],
        help="Which pipeline step to run",
    )
    args = parser.parse_args()

    _ensure_dirs()
    create_tables()

    if args.step == "all":
        for name in ORDER:
            run_step(name)
            time.sleep(5)
    else:
        run_step(args.step)


if __name__ == "__main__":
    main()
