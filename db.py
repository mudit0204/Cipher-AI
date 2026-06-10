"""SQLAlchemy models and session management for the Fat Loss Insights Engine.

Two tables: ``profiles`` and ``posts``. Every other module reads from or
writes to this layer. Use :func:`get_session` as a context manager so commits
and rollbacks are handled consistently.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from config import DB_PATH

Base = declarative_base()


class Profile(Base):
    """An Instagram profile and its computed engagement metrics."""

    __tablename__ = "profiles"

    username = Column(String, primary_key=True)
    followers = Column(Integer)
    following = Column(Integer)
    post_count = Column(Integer)
    bio = Column(Text)
    archetype = Column(String)  # coach | doctor | creator | influencer
    avg_likes = Column(Float)
    avg_comments = Column(Float)
    engagement_rate = Column(Float)
    posts_per_week = Column(Float)
    relevance_score = Column(Float)
    is_business = Column(Boolean)
    scraped_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Profile @{self.username} followers={self.followers}>"


class Post(Base):
    """A single Instagram post with raw, enriched, and classified fields."""

    __tablename__ = "posts"

    post_id = Column(String, primary_key=True)
    username = Column(String, ForeignKey("profiles.username"))
    caption = Column(Text)
    likes = Column(Integer)
    comments = Column(Integer)
    views = Column(Integer)
    media_type = Column(String)  # image | carousel | reel
    video_url = Column(Text)
    image_url = Column(Text)
    carousel_urls = Column(Text)  # JSON string
    hashtags = Column(Text)  # JSON string
    post_url = Column(Text)
    posted_at = Column(DateTime)

    # Enrichment
    content_text = Column(Text)  # caption + transcript + OCR assembled

    # Classification
    primary_category = Column(String)
    secondary_category = Column(String)
    hook = Column(Text)
    cta_text = Column(Text)
    sentiment = Column(String)
    has_cta = Column(Boolean)
    classified_at = Column(DateTime)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Post {self.post_id} @{self.username} {self.media_type}>"


_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def get_engine() -> Engine:
    """Return a cached SQLite engine for ``DB_PATH``."""
    global _engine
    if _engine is None:
        _engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    return _engine


def create_tables() -> None:
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(get_engine())


def _get_session_factory() -> sessionmaker:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), future=True)
    return _SessionFactory


@contextmanager
def get_session() -> Iterator[Session]:
    """Context manager yielding a session, committing on success.

    Rolls back and re-raises on exception, and always closes the session.
    """
    session = _get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    create_tables()
    print("Tables created successfully")
