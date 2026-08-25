"""
SQLite-backed scan history. Every real analysis the live app runs is
logged here -- previously nothing was persisted; a scan's result went to
the browser and was gone. A real security tool keeps an audit log of past
detections; this is that, scoped to what's actually useful (not a general
ORM/ODM layer -- a single table, a handful of functions).

SQLite specifically, not a full DB server: zero configuration, a single
file (data/scan_history.db), safe concurrent writes via WAL mode, nothing
extra to deploy or maintain alongside the app.
"""
import os
import sqlite3
import threading
from datetime import datetime, timezone

DB_PATH = "data/scan_history.db"
_lock = threading.Lock()  # sqlite3 connections aren't thread-safe to share


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    with _lock:
        conn = _connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_type TEXT,
                source_label TEXT,
                pacx_diagnosis TEXT,
                pacx_confidence REAL,
                gnn_diagnosis TEXT,
                gnn_confidence REAL,
                gnn_family TEXT,
                fused_diagnosis TEXT,
                fused_confidence REAL,
                arbitration TEXT,
                completeness REAL,
                agent2_triggered INTEGER,
                reasoning_source TEXT
            )
        """)
        conn.commit()
        conn.close()


def log_scan(
    input_type=None, source_label=None,
    pacx_diagnosis=None, pacx_confidence=None,
    gnn_diagnosis=None, gnn_confidence=None, gnn_family=None,
    fused_diagnosis=None, fused_confidence=None, arbitration=None,
    completeness=None, agent2_triggered=False, reasoning_source=None,
):
    """Fire-and-forget insert -- never raises into the caller's request path;
    a logging failure shouldn't fail a real analysis."""
    try:
        with _lock:
            conn = _connect()
            conn.execute(
                """INSERT INTO scans (
                    timestamp, input_type, source_label,
                    pacx_diagnosis, pacx_confidence,
                    gnn_diagnosis, gnn_confidence, gnn_family,
                    fused_diagnosis, fused_confidence, arbitration,
                    completeness, agent2_triggered, reasoning_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(), input_type, source_label,
                    pacx_diagnosis, pacx_confidence,
                    gnn_diagnosis, gnn_confidence, gnn_family,
                    fused_diagnosis, fused_confidence, arbitration,
                    completeness, int(bool(agent2_triggered)), reasoning_source,
                ),
            )
            conn.commit()
            conn.close()
    except Exception as e:
        print(f"[!] scan_history.log_scan failed (non-fatal): {e}")


def get_recent_scans(limit=50):
    with _lock:
        conn = _connect()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def get_summary_stats():
    """Aggregate counts for a quick dashboard view."""
    with _lock:
        conn = _connect()
        total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        malicious = conn.execute(
            "SELECT COUNT(*) FROM scans WHERE fused_diagnosis = 'Malicious'"
        ).fetchone()[0]
        agent2_count = conn.execute(
            "SELECT COUNT(*) FROM scans WHERE agent2_triggered = 1"
        ).fetchone()[0]
        by_input_type = conn.execute(
            "SELECT input_type, COUNT(*) as n FROM scans GROUP BY input_type"
        ).fetchall()
        conn.close()
    return {
        "total_scans": total,
        "malicious_count": malicious,
        "benign_count": total - malicious,
        "agent2_triggered_count": agent2_count,
        "by_input_type": {row[0] or "unknown": row[1] for row in by_input_type},
    }
