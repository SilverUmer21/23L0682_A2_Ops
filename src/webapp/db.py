"""Database trail (Task 5): every request leaves a permanent row - what was decided, when, by which model, how confidently.

SQLAlchemy Core works with SQLite (local) and PostgreSQL (Docker Compose) without code changes; only DATABASE_URL differs.
The uploaded image itself is NEVER stored, only its SHA-256 hash.
"""
from __future__ import annotations

import time
from datetime import UTC, datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.exc import OperationalError

metadata = MetaData()
predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),          # when
    Column("filename", String(255), nullable=False),                        # sanitized upload name
    Column("label", String(20), nullable=False),                            # what was decided
    Column("confidence", Float, nullable=False),                            # how confidently
    Column("prob_ai_generated", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("model_name", String(100), nullable=False),                      # by which model
    Column("model_version", String(50), nullable=False),
    Column("image_sha256", String(64), nullable=False),
    Column("source", String(30), nullable=False),                           # "local-model" or "remote-api"
)


def make_engine(url: str, attempts: int = 15, wait_seconds: float = 2.0):
    """Create the engine and the table. Retries because the database container may still be starting."""
    engine = create_engine(url, pool_pre_ping=True)
    for attempt in range(1, attempts + 1):
        try:
            metadata.create_all(engine)
            return engine
        except OperationalError:
            if attempt == attempts:
                raise
            time.sleep(wait_seconds)
    return engine


def log_prediction(engine, **fields) -> None:
    with engine.begin() as connection:
        connection.execute(insert(predictions).values(created_at=datetime.now(UTC), **fields))


def recent(engine, limit: int = 50, offset: int = 0) -> list[dict]:
    query = select(predictions).order_by(predictions.c.id.desc()).limit(limit).offset(offset)
    with engine.connect() as connection:
        return [dict(row._mapping) for row in connection.execute(query)]


def total(engine) -> int:
    with engine.connect() as connection:
        return int(connection.execute(select(func.count()).select_from(predictions)).scalar_one())


def is_reachable(engine) -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False
