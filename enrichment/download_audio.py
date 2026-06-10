"""Download Reel audio with yt-dlp for later transcription.

Idempotent: if the target MP3 already exists it is reused. Cross-platform —
the default output directory comes from config rather than a hard-coded /tmp.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from typing import Optional

# Allow running this file directly (python enrichment/download_audio.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import REELS_AUDIO_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _ffmpeg_location() -> Optional[str]:
    """Locate an ffmpeg/ffprobe directory for yt-dlp's audio extraction.

    ffmpeg is a system binary, not a pip dependency. Order: ``FFMPEG_LOCATION``
    env var, then the bundled ``tools/ffmpeg`` directory, then PATH.
    """
    env = os.getenv("FFMPEG_LOCATION")
    if env and os.path.isdir(env):
        return env
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bundled = os.path.join(root, "tools", "ffmpeg")
    if os.path.exists(os.path.join(bundled, "ffmpeg.exe")) or \
            os.path.exists(os.path.join(bundled, "ffmpeg")):
        return bundled
    found = shutil.which("ffmpeg")
    return os.path.dirname(found) if found else None


def download_reel_audio(post_url: str, output_dir: str = REELS_AUDIO_DIR) -> Optional[str]:
    """Download the audio track of a Reel as a 96 kbps MP3.

    Args:
        post_url: Instagram post/reel URL.
        output_dir: directory to write the MP3 into (created if missing).

    Returns:
        Path to the MP3 on success, ``None`` on any failure.
    """
    if not post_url:
        return None

    os.makedirs(output_dir, exist_ok=True)

    try:
        import yt_dlp
    except ImportError as exc:
        logger.error("yt-dlp not installed: %s", exc)
        return None

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "96",
            }
        ],
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }

    ffmpeg_dir = _ffmpeg_location()
    if ffmpeg_dir:
        ydl_opts["ffmpeg_location"] = ffmpeg_dir
    else:
        logger.warning("ffmpeg not found — Reel audio extraction will fail. "
                       "Install ffmpeg or set FFMPEG_LOCATION.")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(post_url, download=False)
            video_id = info.get("id")
            if not video_id:
                logger.error("No video id resolved for %s", post_url)
                return None

            mp3_path = os.path.join(output_dir, f"{video_id}.mp3")
            if os.path.exists(mp3_path):
                logger.info("Audio already downloaded: %s", mp3_path)
                return mp3_path

            ydl.download([post_url])

        if os.path.exists(mp3_path):
            return mp3_path

        logger.error("Expected MP3 not found after download: %s", mp3_path)
        return None
    except Exception as exc:
        logger.error("Audio download failed for %s: %s", post_url, exc)
        return None


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python enrichment/download_audio.py <reel_url>")
        return
    path = download_reel_audio(sys.argv[1])
    print(path)


if __name__ == "__main__":
    main()
