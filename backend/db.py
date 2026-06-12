import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from contextlib import contextmanager
from .models import Base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://aeronexus:aeronexus@localhost:5432/aeronexus"
)

# NullPool for async-safe usage; swap to QueuePool in production
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,
    echo=bool(os.getenv("SQL_ECHO", "")),
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db():
    """Create all tables. Call once at startup or in migrations."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """FastAPI dependency — yields a DB session, closes on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_ctx() -> Session:
    """Context-manager version for scripts and ingestion jobs."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()