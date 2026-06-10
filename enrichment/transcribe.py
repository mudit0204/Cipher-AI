"""Transcribe Reel audio using Groq's hosted Whisper.

``transcribe_post`` is the high-level entry point: it downloads the Reel audio
(if needed) and returns the transcript text, or an empty string on any failure.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

# Allow running this file directly (python enrichment/transcribe.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def transcribe_audio(audio_path: Optional[str]) -> str:
    """Transcribe an MP3 file with Groq Whisper, returning plain text.

    Returns "" if the path is missing, the file does not exist, or any error
    occurs during transcription.
    """
    if not audio_path or not os.path.exists(audio_path):
        return ""
    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
                language="en",
            )
        # response_format="text" returns a plain string.
        text = result if isinstance(result, str) else getattr(result, "text", "")
        return (text or "").strip()
    except Exception as exc:
        logger.error("Transcription failed for %s: %s", audio_path, exc)
        return ""


def transcribe_post(post: dict) -> str:
    """Download and transcribe a Reel post's audio.

    Returns "" for non-reel posts, posts without a video URL, or on failure.
    """
    if post.get("media_type") != "reel":
        return ""
    video_url = post.get("video_url")
    if not video_url:
        return ""

    from enrichment.download_audio import download_reel_audio

    audio_path = download_reel_audio(video_url)
    return transcribe_audio(audio_path)


def main() -> None:
    """Test with a sample public Reel URL (replace with a real one)."""
    sample = {
        "media_type": "reel",
        "video_url": "https://www.instagram.com/reel/EXAMPLE/",
    }
    print("Transcript:", transcribe_post(sample)[:500])


if __name__ == "__main__":
    main()
