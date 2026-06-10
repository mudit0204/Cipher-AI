# Fat Loss Insights Engine

Fat Loss Insights Engine is a local Python pipeline for discovering Instagram fat-loss creators, scoring profiles, scraping posts, enriching media, classifying content with AI, and surfacing patterns in a Streamlit dashboard.

## What it does

- Discovers public Instagram accounts from seed hashtags.
- Scores candidate profiles and keeps the top profiles for scraping.
- Scrapes posts and stores them in SQLite.
- Downloads audio, runs transcription and OCR, and enriches post metadata.
- Classifies posts into content categories for marketing analysis.
- Generates summary metrics and a local Streamlit dashboard.

## Stack

- Python 3.11+
- Instaloader for Instagram scraping
- yt-dlp for media download
- Groq for transcription
- EasyOCR for local OCR
- Google Gemini 2.0 Flash for classification and strategy generation
- SQLAlchemy with SQLite for storage
- Streamlit, Pandas, and Plotly for the dashboard

## Project Structure

- `main.py` - pipeline orchestrator
- `db.py` - SQLAlchemy models and session factory
- `config.py` - constants, seed hashtags, and scoring settings
- `scraper/` - Instagram discovery, scoring, and scraping helpers
- `enrichment/` - audio download, transcription, and OCR
- `classifier/` - classification prompts and batch classification
- `analysis/` - metrics and insight generation
- `dashboard/app.py` - Streamlit dashboard

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the required API and Instagram credentials.

## Environment Variables

The pipeline expects these values in `.env`:

- `INSTAGRAM_USERNAME`
- `INSTAGRAM_PASSWORD` or cookie-based auth via `instagram_cookies.txt`
- `INSTAGRAM_COOKIE_BROWSER`
- `INSTAGRAM_COOKIE_FILE`
- `GEMINI_API_KEY`
- `PROBE_USERNAME` for the optional rate-limit probe step

## Usage

Run the pipeline step by step or end to end:

```bash
python main.py --step discover
python main.py --step score
python main.py --step scrape
python main.py --step enrich
python main.py --step classify
python main.py --step analyze
python main.py --step all
python main.py --step probe
```

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Output

- SQLite database: `fat_loss_insights.db`
- Discovered profiles: `data/discovered_profiles.txt` and `data/discovered_profiles.json`
- Insights summary: `data/insights_summary.json`

## Important Constraints

- Add delays between Instagram requests to avoid rate limits and account bans.
- Keep Gemini calls paced at roughly one request every 4.1 seconds.
- Do not use a personal Instagram account for scraping.

## Notes

The pipeline is resumable. Each stage writes to disk or the database before the next stage runs, so you can stop and continue later without starting over.