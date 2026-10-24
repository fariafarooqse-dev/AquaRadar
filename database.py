"""SQLite persistence starter for analyses and maintenance alerts."""

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("aquaradar.db")


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection. The database file is local and git-ignored."""
    return sqlite3.connect(str(db_path))


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create initial tables. Expand with migrations as the schema is agreed."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                filename TEXT NOT NULL,
                location_label TEXT,
                status TEXT NOT NULL,
                predicted_class TEXT,
                confidence REAL,
                model_version TEXT,
                message TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                location_label TEXT,
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'New',
                inspection_outcome TEXT,
                notes TEXT,
                FOREIGN KEY (analysis_id) REFERENCES analyses(id)
            )
            """
        )
