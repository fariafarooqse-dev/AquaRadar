"""SQLite persistence for analyses and maintenance alerts (Member 1 owns the schema).

Rules enforced here (see docs/overall_workflow.md):
  * An analysis row stores the model result exactly as returned by the model service and is never
    edited afterwards. Alerts only LINK to it (analysis_id), so the original result is preserved.
  * An alert needs a non-empty location label and a non-empty reason. analysis_id is optional:
    an operator may raise a manual alert. An alert cannot be raised from an analysis whose
    status is "error" (there is no result to act on).
  * Alert status only moves forward: New -> Under Review -> Inspection Required -> Resolved.
    Resolving requires an inspection outcome; a resolved alert is closed and keeps its outcome.
  * "Potential leak" is a reason to investigate; nothing here records a leak as confirmed except
    the human inspection outcome.
Timestamps are UTC (SQLite CURRENT_TIMESTAMP).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

DEFAULT_DB_PATH = Path("aquaradar.db")

STATUSES = ["New", "Under Review", "Inspection Required", "Resolved"]
OUTCOMES = ["Leak found and repaired", "Leak not found", "Inconclusive", "False alarm"]


@contextmanager
def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open a connection that commits on success, rolls back on error, and always closes."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create tables if missing and add columns introduced after the starter schema."""
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
        _ensure_columns(conn, "analyses", {"notes": "TEXT", "warnings": "TEXT"})
        _ensure_columns(conn, "alerts", {"updated_at": "TEXT"})


def save_analysis(result: Any, filename: str, location_label: Optional[str] = None,
                  notes: Optional[str] = None, db_path: str | Path = DEFAULT_DB_PATH) -> int:
    """Store a model-service result (any object with the AnalysisResult fields). Returns its id."""
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO analyses (filename, location_label, status, predicted_class, confidence, "
            "model_version, message, notes, warnings) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (filename, (location_label or "").strip() or None, result.status, result.predicted_class,
             result.confidence, result.model_version, result.message, (notes or "").strip() or None,
             json.dumps(list(result.warnings))),
        )
        return int(cur.lastrowid)


def get_analysis(analysis_id: int, db_path: str | Path = DEFAULT_DB_PATH) -> Optional[dict]:
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    return dict(row) if row else None


def list_analyses(limit: int = 100, db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def create_alert(location_label: str, reason: str, analysis_id: Optional[int] = None,
                 notes: Optional[str] = None, db_path: str | Path = DEFAULT_DB_PATH) -> int:
    """Create an alert with status 'New'. Raises ValueError if the rules above are not met."""
    location_label = (location_label or "").strip()
    reason = (reason or "").strip()
    if not location_label:
        raise ValueError("A location label is required to create an alert.")
    if not reason:
        raise ValueError("A reason is required to create an alert.")
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        if analysis_id is not None:
            row = conn.execute("SELECT status FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
            if row is None:
                raise ValueError(f"Analysis {analysis_id} does not exist.")
            if row["status"] == "error":
                raise ValueError("An alert cannot be created from a failed analysis. "
                                 "Create a manual alert instead.")
        cur = conn.execute(
            "INSERT INTO alerts (analysis_id, location_label, reason, notes, updated_at) "
            "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (analysis_id, location_label, reason, (notes or "").strip() or None),
        )
        return int(cur.lastrowid)


def update_alert_status(alert_id: int, new_status: str, inspection_outcome: Optional[str] = None,
                        note: Optional[str] = None, db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Move an alert forward. Resolving requires an outcome. Notes are appended, never replaced."""
    if new_status not in STATUSES:
        raise ValueError(f"Unknown status: {new_status!r}")
    if new_status == "Resolved":
        if inspection_outcome not in OUTCOMES:
            raise ValueError(f"Resolving an alert requires an inspection outcome: {OUTCOMES}")
    elif inspection_outcome:
        raise ValueError("An inspection outcome can only be recorded when resolving an alert.")
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT status, notes FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        if row is None:
            raise ValueError(f"Alert {alert_id} does not exist.")
        if row["status"] == "Resolved":
            raise ValueError("A resolved alert is closed; its outcome is preserved.")
        if STATUSES.index(new_status) <= STATUSES.index(row["status"]):
            raise ValueError(f"Status can only move forward (currently {row['status']!r}).")
        notes = row["notes"] or ""
        if (note or "").strip():
            stamp = conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0]
            notes = (notes + "\n" if notes else "") + f"[{stamp} UTC] {note.strip()}"
        conn.execute(
            "UPDATE alerts SET status = ?, inspection_outcome = ?, notes = ?, "
            "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (new_status, inspection_outcome, notes or None, alert_id),
        )


def get_alert(alert_id: int, db_path: str | Path = DEFAULT_DB_PATH) -> Optional[dict]:
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
    return dict(row) if row else None


def list_alerts(status: Optional[str] = None, db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    """Alerts joined with the linked analysis result (None fields for manual alerts)."""
    initialize_database(db_path)
    sql = ("SELECT a.*, an.status AS analysis_status, an.predicted_class, an.confidence, "
           "an.model_version, an.filename FROM alerts a LEFT JOIN analyses an ON an.id = a.analysis_id")
    args: tuple = ()
    if status:
        sql += " WHERE a.status = ?"
        args = (status,)
    with get_connection(db_path) as conn:
        rows = conn.execute(sql + " ORDER BY a.id DESC", args).fetchall()
    return [dict(r) for r in rows]


def summary_counts(db_path: str | Path = DEFAULT_DB_PATH) -> dict:
    """Counts computed from saved records only."""
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        by_analysis = {r["status"]: r["n"] for r in
                       conn.execute("SELECT status, COUNT(*) AS n FROM analyses GROUP BY status")}
        by_alert = {r["status"]: r["n"] for r in
                    conn.execute("SELECT status, COUNT(*) AS n FROM alerts GROUP BY status")}
    return {
        "analyses_total": sum(by_analysis.values()),
        "analyses_by_status": by_analysis,
        "alerts_total": sum(by_alert.values()),
        "alerts_by_status": {s: by_alert.get(s, 0) for s in STATUSES},
        "alerts_open": sum(n for s, n in by_alert.items() if s != "Resolved"),
    }
