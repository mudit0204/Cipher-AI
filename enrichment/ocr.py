"""OCR for images/carousels plus the unified content_text assembler.

The EasyOCR reader is created lazily on first use rather than at import time —
the first call downloads ~200MB of models, and we don't want a bare ``import``
(e.g. during a smoke test) to trigger that. Behaviour is otherwise "init once".
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
from typing import Optional

# Allow running this file directly (python enrichment/ocr.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_reader = None  # lazily initialized EasyOCR reader (singleton)


def _get_reader():
    """Return a cached EasyOCR reader, building it on first call (CPU only)."""
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def ocr_image_url(image_url: str) -> str:
    """Download an image and return its OCR text, or "" on failure."""
    if not image_url:
        return ""
    try:
        from PIL import Image

        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        import numpy as np

        result = _get_reader().readtext(np.array(img), detail=0, paragraph=True)
        return " ".join(result).strip()
    except Exception as exc:
        logger.warning("OCR failed for %s: %s", image_url, exc)
        return ""


def extract_carousel_text(carousel_urls: list[str]) -> str:
    """OCR each carousel slide, prefixing with a slide marker."""
    slides: list[str] = []
    for i, url in enumerate(carousel_urls, start=1):
        text = ocr_image_url(url)
        if text:
            slides.append(f"[Slide {i}]: {text}")
    return "\n".join(slides)


def assemble_content_text(post: dict) -> str:
    """Build the unified content_text string sent to the AI classifier.

    Combines caption, video transcript, carousel/image OCR, and an engagement
    footer. Each source is optional and added only when it yields content.
    """
    parts: list[str] = []

    caption = (post.get("caption") or "").strip()
    if caption:
        parts.append(f"[CAPTION]: {caption}")

    media_type = post.get("media_type")

    if media_type == "reel" and post.get("video_url"):
        try:
            from enrichment.transcribe import transcribe_post

            transcript = transcribe_post(post)
            if transcript:
                parts.append(f"[VIDEO TRANSCRIPT]: {transcript}")
        except Exception as exc:
            logger.warning("Transcript step failed: %s", exc)

    if media_type == "carousel" and post.get("carousel_urls"):
        try:
            urls = post["carousel_urls"]
            if isinstance(urls, str):
                urls = json.loads(urls)
            text = extract_carousel_text(urls or [])
            if text:
                parts.append(f"[CAROUSEL SLIDES]:\n{text}")
        except Exception as exc:
            logger.warning("Carousel OCR step failed: %s", exc)

    if media_type == "image" and post.get("image_url"):
        text = ocr_image_url(post["image_url"])
        if len(text) > 30:
            parts.append(f"[IMAGE TEXT]: {text}")

    parts.append(
        f"[Engagement: {post.get('likes', 0)} likes | "
        f"{post.get('comments', 0)} comments | {post.get('views', 0)} views]"
    )
    return "\n\n".join(parts)


def enrich_all_posts(db_session) -> None:
    """Assemble content_text for every post that lacks it, committing in batches."""
    from db import Post

    posts = db_session.query(Post).filter(Post.content_text.is_(None)).all()
    total = len(posts)
    print(f"Enriching {total} posts...")

    for i, post in enumerate(posts, start=1):
        try:
            # Read attributes explicitly rather than post.__dict__: db_session
            # is committed below every 10 posts, which expires the ORM objects,
            # leaving post.__dict__ empty for subsequent posts. Attribute access
            # reloads expired values; __dict__ does not.
            post.content_text = assemble_content_text({
                "caption": post.caption,
                "media_type": post.media_type,
                "video_url": post.video_url,
                "carousel_urls": post.carousel_urls,
                "image_url": post.image_url,
                "likes": post.likes,
                "comments": post.comments,
                "views": post.views,
            })
        except Exception as exc:
            logger.error("Enrichment failed for %s: %s", post.post_id, exc)
            post.content_text = ""
        if i % 10 == 0:
            db_session.commit()
            print(f"Enriched {i}/{total}")
    db_session.commit()
    print(f"Enriched {total} posts with content_text")


def main() -> None:
    """Test assemble_content_text on a dummy image post (no OCR triggered)."""
    dummy = {
        "caption": "3 fat loss myths that are keeping you stuck. Save this!",
        "media_type": "image",
        "image_url": None,
        "carousel_urls": "[]",
        "video_url": None,
        "likes": 1200,
        "comments": 45,
        "views": 0,
    }
    print(assemble_content_text(dummy))


if __name__ == "__main__":
    main()
