"""Instaloader wrappers for discovering, fetching, and persisting Instagram data.

Every network loop enforces randomized sleeps to avoid tripping Instagram's
rate limits and banning the burner account. No function is allowed to crash the
pipeline: exceptions are caught, logged, and surfaced as ``None``/empty results.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from typing import Optional

# Allow running this file directly (python scraper/insta_scraper.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import instaloader
from instaloader import Hashtag, Profile

from db import Post, Profile as ProfileModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The session cookies are imported from Edge on Windows; present a matching
# Windows/Edge User-Agent so Instagram does not flag a session/UA mismatch
# (which throttles detailed endpoints with HTTP 429).
INSTAGRAM_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0"
)
# Public web app id required by Instagram's web_profile_info endpoint.
WEB_APP_ID = "936619743392459"


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
def _activate_cookie_jar(
    loader: instaloader.Instaloader,
    jar,
    username: str,
    source: str,
) -> str:
    """Load instagram.com cookies from ``jar`` into ``loader`` and verify login.

    Returns the username Instagram actually considers logged in. Raises with an
    actionable message if no usable Instagram session results.
    """
    session = loader.context._session
    imported = 0
    for cookie in jar:
        if "instagram.com" in (cookie.domain or ""):
            session.cookies.set(
                cookie.name, cookie.value,
                domain=cookie.domain, path=cookie.path or "/",
            )
            imported += 1
    if imported == 0:
        raise RuntimeError(
            f"No instagram.com cookies found in {source}. Make sure @{username} "
            f"is logged in there, then re-run."
        )
    logger.info("Imported %d Instagram cookies from %s", imported, source)

    who = loader.test_login()
    if not who:
        raise RuntimeError(
            f"Cookies from {source} were loaded but Instagram does not consider "
            f"them logged in. Re-export a fresh session for @{username} (clear any "
            f"'confirm it's you' checkpoint first), then re-run."
        )
    if who.lower() != username.lower():
        # Refuse to silently operate the wrong account — these cookies belong to
        # a different login than INSTAGRAM_USERNAME requests. (Guards against, e.g.,
        # reusing an old account's cookies for a new burner.)
        raise RuntimeError(
            f"{source} is logged into @{who}, not @{username}. Log @{username} "
            f"into the browser and re-export its cookies, or set INSTAGRAM_USERNAME "
            f"to @{who}."
        )
    loader.context.username = who
    return who


def _import_session_from_file(
    loader: instaloader.Instaloader,
    username: str,
    cookiefile: str,
) -> str:
    """Load cookies from a Netscape-format cookies.txt exported by a browser.

    This is the most robust path on Windows: a browser extension exports the
    already-decrypted cookies, sidestepping Chromium 127+ app-bound encryption
    that defeats direct cookie-DB reads.
    """
    from http.cookiejar import MozillaCookieJar

    jar = MozillaCookieJar(cookiefile)
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse cookie file {cookiefile} ({exc}). Export it in "
            f"Netscape/cookies.txt format."
        ) from exc
    return _activate_cookie_jar(loader, jar, username, f"file {cookiefile}")


def _import_session_from_browser(
    loader: instaloader.Instaloader,
    username: str,
    browser: str,
) -> str:
    """Load Instagram cookies directly from a logged-in browser's cookie store.

    NOTE: On Windows this fails for Chromium 127+ (Chrome/Edge) due to app-bound
    cookie encryption — use the cookies.txt path instead. Works for Firefox.
    """
    from yt_dlp.cookies import extract_cookies_from_browser

    try:
        jar = extract_cookies_from_browser(browser)
    except Exception as exc:  # locked DB, app-bound encryption, no profile, ...
        raise RuntimeError(
            f"Could not read {browser} cookies ({exc}). On recent Edge/Chrome "
            f"this is app-bound encryption — export a cookies.txt instead and set "
            f"INSTAGRAM_COOKIE_FILE."
        ) from exc
    return _activate_cookie_jar(loader, jar, username, browser)


def login(
    username: str,
    password: str = "",
    browser: str = "edge",
    cookiefile: str = "",
) -> instaloader.Instaloader:
    """Return a logged-in Instaloader instance.

    Authentication order, safest and most reliable first:
      1. Reuse a saved session file (no network auth at all).
      2. Load a Netscape cookies.txt exported from the browser (``cookiefile``).
      3. Import cookies directly from a logged-in ``browser`` (Firefox only on
         Windows; Edge/Chrome are blocked by app-bound encryption).
      4. Fall back to password login only if a password is supplied.

    The session is saved after any successful login so later runs take path 1.
    """
    loader = instaloader.Instaloader(
        user_agent=INSTAGRAM_USER_AGENT,  # match the Edge/Windows session the cookies came from
        download_videos=False,
        download_comments=False,
        download_pictures=False,
        save_metadata=False,
        compress_json=False,
        request_timeout=30.0,  # default is 300s — a hung connection would block for 5 min
    )
    try:
        loader.load_session_from_file(username)
        logger.info("Reused saved session for @%s", username)
        return loader
    except FileNotFoundError:
        pass

    try:
        if cookiefile and os.path.exists(cookiefile):
            logger.info("No saved session; loading cookies for @%s from %s",
                        username, cookiefile)
            who = _import_session_from_file(loader, username, cookiefile)
        else:
            if cookiefile:
                logger.warning("Cookie file %s not found; trying %s directly",
                               cookiefile, browser)
            logger.info("No saved session; importing %s cookies for @%s",
                        browser, username)
            who = _import_session_from_browser(loader, username, browser)
        loader.save_session_to_file()
        logger.info("Saved session for @%s for future runs", who)
        return loader
    except Exception as exc:
        if not password:
            raise
        logger.warning("Cookie login failed (%s); trying password login", exc)
        loader.login(username, password)
        loader.save_session_to_file()
        return loader


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _media_dicts(obj) -> list:
    """Recursively collect every ``media`` object in a hashtag sections payload.

    Instagram returns sections with different ``layout_type``s: ``media_grid``
    holds posts under ``layout_content['medias']``, while ``one_by_two_left``
    uses ``fill_items`` / ``one_by_two_item`` (no ``medias`` key at all). Rather
    than enumerate every layout, we walk the whole structure and grab any value
    stored under a ``media`` key — robust to layout changes.
    """
    found: list = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "media" and isinstance(value, dict):
                found.append(value)
            else:
                found.extend(_media_dicts(value))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(_media_dicts(value))
    return found


def _users_from_hashtag(hashtag: Hashtag, limit: int) -> tuple[dict[str, dict], int]:
    """Extract owner user records from a hashtag's already-fetched sections.

    Reads the ``top`` and ``recent`` sections that ``Hashtag.from_name`` already
    downloaded — no per-post network requests. We deliberately do NOT use
    Instaloader's iterators here: ``get_posts()`` raises
    ``KeyError('more_available')`` on current Instagram, and
    ``get_top_posts()``'s SectionIterator raises ``KeyError('medias')`` on any
    non-``media_grid`` layout, aborting an otherwise-usable response.

    Returns ``({username: {is_private, is_verified, full_name}}, media_scanned)``
    deduplicated within this one hashtag.
    """
    node = getattr(hashtag, "_node", {}) or {}
    users: dict[str, dict] = {}
    scanned = 0
    for key in ("top", "recent"):
        if scanned >= limit:
            break
        section = node.get(key) or {}
        for media in _media_dicts(section.get("sections") or []):
            if scanned >= limit:
                break
            scanned += 1
            user = media.get("user") or {}
            uname = user.get("username")
            if uname and uname not in users:
                users[uname] = {
                    "is_private": bool(user.get("is_private")),
                    "is_verified": bool(user.get("is_verified")),
                    "full_name": user.get("full_name") or "",
                }
    return users, scanned


def discover_profiles_from_hashtags(
    L: instaloader.Instaloader,
    hashtags: list[str],
    posts_per_tag: int = 60,
) -> dict[str, dict]:
    """Collect candidate profiles from the top/recent posts of each seed hashtag.

    Makes only one request per hashtag (``Hashtag.from_name``) and parses owner
    records out of the response — no per-post fetches. Each hashtag is isolated
    in a try/except so one failure does not abort discovery.

    Returns ``{username: record}`` where each record carries ``tag_count`` (how
    many seed hashtags the account appeared in — a relevance signal), the
    ``tags`` list, and light signals (``is_private``, ``is_verified``,
    ``full_name``) used to rank and filter candidates before the rate-limited
    scoring step.
    """
    candidates: dict[str, dict] = {}
    for tag in hashtags:
        tag = tag.lstrip("#")
        logger.info("Discovering from #%s ...", tag)
        try:
            hashtag = Hashtag.from_name(L.context, tag)
            found, scanned = _users_from_hashtag(hashtag, posts_per_tag)
            for uname, sig in found.items():
                rec = candidates.get(uname)
                if rec is None:
                    rec = {"username": uname, "tag_count": 0, "tags": [], **sig}
                    candidates[uname] = rec
                rec["tag_count"] += 1
                rec["tags"].append(tag)
            logger.info("#%s -> %d media scanned, %d users from tag, %d unique total",
                        tag, scanned, len(found), len(candidates))
        except Exception as exc:
            logger.error("Hashtag #%s failed: %s", tag, exc)
        # Pause between hashtag scrapes (per project rate-limit policy).
        time.sleep(random.uniform(10, 20))
    return candidates


# --------------------------------------------------------------------------- #
# Profile metadata
# --------------------------------------------------------------------------- #
def _infer_archetype(bio: str, is_business: bool) -> str:
    """Best-effort archetype guess from bio keywords."""
    low = (bio or "").lower()
    if any(k in low for k in ("md", "dr.", "doctor", "rd", "dietitian", "phd")):
        return "doctor"
    if any(k in low for k in ("coach", "coaching", "trainer", "program")):
        return "coach"
    if is_business:
        return "influencer"
    return "creator"


# Mobile app User-Agent for the i.instagram.com private API. The web
# ``web_profile_info`` endpoint is aggressively IP-rate-limited (instant 429),
# but the ``users/{pk}/info`` and ``feed/user/{pk}`` endpoints have separate,
# usable limits.
MOBILE_USER_AGENT = (
    "Instagram 269.0.0.18.75 Android (26/8.0.0; 480dpi; 1080x1920; OnePlus; "
    "ONEPLUS A6013; OnePlus6T; qcom; en_US; 314665256)"
)


def _mobile_get(L: instaloader.Instaloader, path: str, params: Optional[dict] = None) -> dict:
    """GET a path on the i.instagram.com private API, returning parsed JSON.

    Raises RuntimeError (including any ``Retry-After``) on non-200 so callers can
    log it and skip the profile.
    """
    resp = L.context._session.get(
        f"https://i.instagram.com/{path}",
        params=params or {},
        headers={"User-Agent": MOBILE_USER_AGENT, "X-IG-App-ID": WEB_APP_ID},
    )
    if resp.status_code != 200:
        retry = resp.headers.get("Retry-After")
        extra = f" (Retry-After: {retry}s)" if retry else ""
        raise RuntimeError(f"{path} HTTP {resp.status_code}{extra}")
    return resp.json()


def _user_id_for(L: instaloader.Instaloader, username: str) -> str:
    """Resolve a username to its numeric user id.

    Uses Instaloader's search-backed ``Profile.from_username`` (a GraphQL search
    that still works) — needed because the username->id ``web_profile_info``
    endpoint is rate-limited and the private API endpoints below want a ``pk``.
    """
    return str(Profile.from_username(L.context, username).userid)


def probe_profile(L: instaloader.Instaloader, username: str) -> dict:
    """Make a single profile-info request for diagnostics; never raises.

    Returns a status dict: ``ok``, ``status`` (HTTP code), ``retry_after`` (if
    Instagram sent the header), plus ``username``/``followers`` when successful or
    ``error`` on failure. Exercises the same private-API path that scoring uses.
    """
    try:
        uid = _user_id_for(L, username)
        resp = L.context._session.get(
            f"https://i.instagram.com/api/v1/users/{uid}/info/",
            headers={"User-Agent": MOBILE_USER_AGENT, "X-IG-App-ID": WEB_APP_ID},
        )
    except Exception as exc:
        return {"ok": False, "status": None, "retry_after": None, "error": str(exc)}

    report = {
        "ok": resp.status_code == 200,
        "status": resp.status_code,
        "retry_after": resp.headers.get("Retry-After"),
    }
    if resp.status_code == 200:
        try:
            user = resp.json().get("user") or {}
            report["username"] = user.get("username")
            report["followers"] = user.get("follower_count")
        except Exception as exc:
            report["ok"] = False
            report["error"] = f"HTTP 200 but body unparseable: {exc}"
    return report


def get_profile_metadata(L: instaloader.Instaloader, username: str) -> Optional[dict]:
    """Fetch a profile and compute engagement metrics from its last ~12 posts.

    Returns a dict matching the Profile model fields, or ``None`` on failure.
    Resolves the user id, then reads the private-API ``users/{pk}/info`` (profile
    fields) and ``feed/user/{pk}`` (recent posts) endpoints — these work while
    the web ``web_profile_info`` endpoint is IP-rate-limited.
    """
    try:
        uid = _user_id_for(L, username)
        time.sleep(random.uniform(1, 2))
        info = (_mobile_get(L, f"api/v1/users/{uid}/info/").get("user")) or {}
        time.sleep(random.uniform(1, 2))
        feed = _mobile_get(L, f"api/v1/feed/user/{uid}/", {"count": 12})
    except Exception as exc:
        logger.error("Could not load profile @%s: %s", username, exc)
        return None

    likes_list: list[int] = []
    comments_list: list[int] = []
    dates: list[datetime] = []
    for item in (feed.get("items") or [])[:12]:
        likes_list.append(item.get("like_count") or 0)
        comments_list.append(item.get("comment_count") or 0)
        ts = item.get("taken_at")
        if ts:
            dates.append(datetime.utcfromtimestamp(ts))

    avg_likes = sum(likes_list) / len(likes_list) if likes_list else 0.0
    avg_comments = sum(comments_list) / len(comments_list) if comments_list else 0.0
    followers = info.get("follower_count") or 0
    engagement_rate = (
        (avg_likes + avg_comments) / followers if followers > 0 else 0.0
    )

    # posts_per_week from the date spread of the sampled posts.
    posts_per_week = 0.0
    if len(dates) >= 2:
        span_days = (max(dates) - min(dates)).days or 1
        posts_per_week = len(dates) / (span_days / 7.0)

    bio = info.get("biography") or ""
    is_business = bool(info.get("is_business"))
    result = {
        "username": username,
        "followers": followers,
        "following": info.get("following_count") or 0,
        "post_count": info.get("media_count") or 0,
        "bio": bio,
        "archetype": _infer_archetype(bio, is_business),
        "avg_likes": round(avg_likes, 2),
        "avg_comments": round(avg_comments, 2),
        "engagement_rate": round(engagement_rate, 5),
        "posts_per_week": round(posts_per_week, 2),
        "relevance_score": 0.0,  # filled in by profile_scorer
        "is_business": is_business,
        "scraped_at": datetime.utcnow(),
    }
    return result


# --------------------------------------------------------------------------- #
# Post scraping
# --------------------------------------------------------------------------- #
_FEED_MEDIA_TYPE = {1: "image", 2: "reel", 8: "carousel"}


def _best_image_url(node: dict) -> Optional[str]:
    """First (highest-res) image candidate URL from a feed media node."""
    candidates = (node.get("image_versions2") or {}).get("candidates") or []
    return candidates[0].get("url") if candidates else None


def _feed_item_to_post(item: dict, username: str) -> dict:
    """Map an i.instagram.com feed item to our Post model fields."""
    code = item.get("code")
    post_url = f"https://www.instagram.com/p/{code}/"
    media_type = _FEED_MEDIA_TYPE.get(item.get("media_type"), "image")
    caption = (item.get("caption") or {}).get("text") or ""

    carousel_urls: list[str] = []
    video_url: Optional[str] = None
    image_url: Optional[str] = None
    if media_type == "carousel":
        for child in item.get("carousel_media") or []:
            url = _best_image_url(child)
            if url:
                carousel_urls.append(url)
        image_url = carousel_urls[0] if carousel_urls else None
    elif media_type == "reel":
        video_url = post_url  # hand the post URL to yt-dlp later, not the CDN url
        image_url = _best_image_url(item)
    else:
        image_url = _best_image_url(item)

    return {
        "post_id": code,
        "username": username,
        "caption": caption,
        "likes": item.get("like_count") or 0,
        "comments": item.get("comment_count") or 0,
        "views": item.get("play_count") or item.get("view_count") or 0,
        "media_type": media_type,
        "video_url": video_url,
        "image_url": image_url,
        "carousel_urls": json.dumps(carousel_urls),
        "hashtags": json.dumps(re.findall(r"#(\w+)", caption)),
        "post_url": post_url,
        "posted_at": (datetime.utcfromtimestamp(item["taken_at"])
                      if item.get("taken_at") else None),
    }


def scrape_posts_for_profile(
    L: instaloader.Instaloader,
    username: str,
    limit: int = 50,
) -> list[dict]:
    """Scrape up to ``limit`` posts for a profile, mapped to Post model fields.

    Uses the private ``feed/user/{pk}`` endpoint (paginated via ``next_max_id``)
    instead of Instaloader's ``Profile.get_posts()``, which hits a dead GraphQL
    query (HTTP 400) on current Instagram.
    """
    posts: list[dict] = []
    try:
        uid = _user_id_for(L, username)
    except Exception as exc:
        logger.error("Cannot scrape @%s: %s", username, exc)
        return posts

    max_id: Optional[str] = None
    try:
        while len(posts) < limit:
            params: dict = {"count": 33}
            if max_id:
                params["max_id"] = max_id
            data = _mobile_get(L, f"api/v1/feed/user/{uid}/", params)
            items = data.get("items") or []
            for item in items:
                if len(posts) >= limit:
                    break
                try:
                    posts.append(_feed_item_to_post(item, username))
                except Exception as exc:
                    logger.warning("Skipping a post for @%s: %s", username, exc)
            if not items or not data.get("more_available"):
                break
            max_id = data.get("next_max_id")
            time.sleep(random.uniform(2, 5))
    except Exception as exc:
        logger.error("Post iteration failed for @%s: %s", username, exc)

    logger.info("Scraped %d posts for @%s", len(posts), username)
    return posts


# --------------------------------------------------------------------------- #
# Persistence (upsert)
# --------------------------------------------------------------------------- #
def save_profiles_to_db(profiles: list[dict], session) -> None:
    """Upsert profile dicts using SQLAlchemy ``merge``."""
    for data in profiles:
        try:
            session.merge(ProfileModel(**data))
        except Exception as exc:
            logger.error("Failed to save profile %s: %s",
                         data.get("username"), exc)
    session.commit()


def save_posts_to_db(posts: list[dict], session) -> None:
    """Upsert post dicts, skipping any post_id already present."""
    for data in posts:
        try:
            exists = session.get(Post, data["post_id"])
            if exists is not None:
                continue
            session.merge(Post(**data))
        except Exception as exc:
            logger.error("Failed to save post %s: %s",
                         data.get("post_id"), exc)
    session.commit()


def prune_to_profiles(session, keep_usernames: list[str]) -> int:
    """Delete profiles (and their posts) whose username is not in keep_usernames.

    The pipeline is otherwise add-only, so old profiles from earlier runs with
    different caps accumulate. Calling this keeps the DB equal to the current
    top-N set. Returns the number of profiles deleted.
    """
    keep = list(keep_usernames)
    n = session.query(ProfileModel).filter(~ProfileModel.username.in_(keep)).count()
    session.query(Post).filter(~Post.username.in_(keep)).delete(synchronize_session=False)
    session.query(ProfileModel).filter(~ProfileModel.username.in_(keep)).delete(synchronize_session=False)
    session.commit()
    return n


def trim_profile_posts(session, username: str, keep: int) -> int:
    """Keep only the most recent ``keep`` posts for a profile; delete the rest.

    Lowering POSTS_PER_PROFILE doesn't shrink already-stored posts (save is
    dedup-only), so this enforces the cap. Returns the number of posts deleted.
    """
    keep_ids = [
        pid for (pid,) in session.query(Post.post_id)
        .filter(Post.username == username)
        .order_by(Post.posted_at.desc())
        .limit(keep)
    ]
    if not keep_ids:
        return 0
    deleted = (
        session.query(Post)
        .filter(Post.username == username, ~Post.post_id.in_(keep_ids))
        .delete(synchronize_session=False)
    )
    session.commit()
    return deleted


def main() -> None:
    """Smoke test: log in and print the authenticated username."""
    import os

    from dotenv import load_dotenv

    load_dotenv()
    username = os.getenv("INSTAGRAM_USERNAME", "")
    password = os.getenv("INSTAGRAM_PASSWORD", "")
    browser = os.getenv("INSTAGRAM_COOKIE_BROWSER", "edge")
    cookiefile = os.getenv("INSTAGRAM_COOKIE_FILE", "")
    if not username:
        print("Set INSTAGRAM_USERNAME in .env first.")
        return
    L = login(username, password, browser=browser, cookiefile=cookiefile)
    print(f"Logged in as: {L.context.username}")


if __name__ == "__main__":
    main()
