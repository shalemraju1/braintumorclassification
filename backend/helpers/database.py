from __future__ import annotations

from contextlib import contextmanager

import psycopg

from .config import DATABASE_URL


def get_db_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not configured")

    return psycopg.connect(
        DATABASE_URL,
        connect_timeout=10,
        sslmode="require",
    )


def ensure_database_schema(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                prediction TEXT NOT NULL,
                confidence FLOAT NOT NULL,
                risk_level TEXT NOT NULL,
                image_path TEXT,
                heatmap_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute("ALTER TABLE reports ADD COLUMN IF NOT EXISTS image_path TEXT")
        cursor.execute("ALTER TABLE reports ADD COLUMN IF NOT EXISTS heatmap_path TEXT")
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reports_user_created_at
            ON reports (user_id, created_at DESC)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_users_email
            ON users (email)
            """
        )

    connection.commit()


def bootstrap_database() -> None:
    with get_db_connection() as connection:
        ensure_database_schema(connection)
