from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from ..core.config import get_settings

settings = get_settings()

Base = declarative_base()

_engine = None
_SessionLocal = None


def get_engine(database_url: str | None = None):
    global _engine
    url = database_url or settings.database_url
    if _engine is None or (database_url is not None and getattr(_engine, "url", None) != url):
        _engine = create_engine(url, future=True)
    return _engine


def _build_session_factory(database_url: str | None = None):
    engine = get_engine(database_url)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


SessionLocal = _build_session_factory()


def get_db():
    from sqlalchemy.orm import Session

    engine = get_engine()
    session = Session(bind=engine)
    try:
        yield session
    finally:
        session.close()
