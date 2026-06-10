# Fat Loss Insights Engine — Project Context

## What this project does
Scrapes Instagram fat loss content, classifies posts using AI,
and surfaces marketing patterns through a Streamlit dashboard.
Target: 40 profiles, ~2000 posts for MVP.

## Stack (ALL FREE TIER)
- Instagram data: Instaloader (Python, open source)
- Video audio download: yt-dlp
- Transcription: Groq API (Whisper large-v3-turbo, free tier)
- OCR: EasyOCR (local, no API)
- Classification: Google Gemini 2.0 Flash API (free: 1M tokens/day)
- Storage: SQLite via SQLAlchemy
- Dashboard: Streamlit (local)
- Python 3.11+

## Critical constraints
- Instaloader: ALWAYS add time.sleep(random.uniform(2,6)) between
  profile requests and time.sleep(random.uniform(10,20)) between
  hashtag scrapes. Failure to do so will ban the burner account.
- Gemini free tier: 15 requests/minute max. Always sleep(4.1) between calls.
- Groq free tier: generous limits, but batch wisely.
- Never use the user's personal Instagram account.

## Database
SQLite file: fat_loss_insights.db
Two main tables: profiles, posts
All models in db.py using SQLAlchemy ORM
Session factory: get_session() context manager

## Content categories (classifier output)
CREDIBILITY - educational, expert positioning, science-based
VIRAL        - transformation, recipe, challenge, entertainment
LEAD_GEN     - DM trigger, coaching offer, scarcity CTA
MIXED        - credibility + lead gen in same post

## Key files
config.py          - all constants, hashtag seeds, scoring weights
db.py              - SQLAlchemy models + session factory
scraper/           - Instaloader wrappers
enrichment/        - yt-dlp + Groq Whisper + EasyOCR
classifier/        - Gemini classification + prompts
analysis/          - pandas metrics computation
dashboard/app.py   - Streamlit UI
main.py            - orchestrator with --step flags

## Coding conventions
- All functions typed with type hints
- Each module has a main() for standalone testing
- Errors are caught and logged, never crash the pipeline
- Rate limits enforced with time.sleep in every loop
- Load .env with python-dotenv at top of every module that uses APIs
