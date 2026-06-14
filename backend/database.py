"""
backend/database.py
===================
SQLAlchemy 2.x database layer.

Provides:
  - engine        : shared Engine instance connected to SQLite
  - SessionLocal  : session factory for request-scoped sessions
  - get_db()      : FastAPI dependency that yields a Session and
                    guarantees cleanup even on exceptions
  - create_tables(): idempotent DDL function called at startup
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import settings

logger = logging.getLogger(__name__)

# ── ABSOLUTE DATABASE PATH RESOLUTION ─────────────────────────────────────────

db_url = settings.DATABASE_URL

# Force absolute path resolution to the project root workspace directory
if db_url.startswith("sqlite:///"):
    db_filename = db_url.split(":///")[-1].lstrip("./")
    
    # Locate where database.py sits (inside backend/)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the main project root folder
    ROOT_DIR = os.path.dirname(CURRENT_DIR)
    # Target the single valid database file
    DB_PATH = os.path.join(ROOT_DIR, db_filename)
    
    db_url = f"sqlite:///{DB_PATH}"
    logger.info("🔗 SQLite path forced strictly to root: %s", DB_PATH)
else:
    logger.info("🔗 Connecting via configured database URL strategy.")

# ── Engine ────────────────────────────────────────────────────────────────────

engine = create_engine(
    db_url,
    connect_args={"check_same_thread": False} if db_url.startswith("sqlite://") else {},
    echo=settings.DEBUG,
)

# ── Session factory ───────────────────────────────────────────────────────────

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

# ── Declarative base ──────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


# ── FastAPI dependency ────────────────────────────────────────────────────────

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── Table creation ────────────────────────────────────────────────────────────

def create_tables() -> None:
    import backend.models  # type: ignore
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logger.info("Database tables verified / created successfully.")
    except Exception as exc:
        logger.exception("Failed to create database tables: %s", exc)
        raise


def check_connection() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database connectivity check failed: %s", exc)
        return False