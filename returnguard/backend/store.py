"""SQLite-backed API key and usage store.

Tables:
  api_keys(key, plan_id, email, created_at, active)
  usage(key, period, count)         period = 'YYYY-MM' (monthly bucket)

Thread-safe for a single-process server via a module lock. Good enough for a
demo / low-concurrency deployment.
"""
from __future__ import annotations

import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .plans import DEFAULT_PLAN, get_plan

DB_PATH = Path(__file__).resolve().parent / "returnguard_api.db"
_LOCK = threading.Lock()


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _LOCK, _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                key        TEXT PRIMARY KEY,
                plan_id    TEXT NOT NULL,
                email      TEXT,
                created_at TEXT NOT NULL,
                active     INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS usage (
                key    TEXT NOT NULL,
                period TEXT NOT NULL,
                count  INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (key, period)
            );
            """
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _period() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def issue_key(plan_id: str = DEFAULT_PLAN, email: str | None = None) -> dict:
    """Create a new API key bound to a plan. Returns the key record."""
    plan = get_plan(plan_id)
    key = "rg_" + secrets.token_urlsafe(24)
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO api_keys(key, plan_id, email, created_at, active) "
            "VALUES (?,?,?,?,1)",
            (key, plan.id, email, _now_iso()),
        )
    return {"api_key": key, "plan": plan.id, "monthly_quota": plan.monthly_quota}


def get_key(key: str) -> sqlite3.Row | None:
    with _LOCK, _conn() as c:
        row = c.execute(
            "SELECT * FROM api_keys WHERE key = ? AND active = 1", (key,)
        ).fetchone()
    return row


def usage_this_period(key: str) -> int:
    with _LOCK, _conn() as c:
        row = c.execute(
            "SELECT count FROM usage WHERE key = ? AND period = ?",
            (key, _period()),
        ).fetchone()
    return row["count"] if row else 0


def increment_usage(key: str, by: int = 1) -> int:
    period = _period()
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO usage(key, period, count) VALUES (?,?,?) "
            "ON CONFLICT(key, period) DO UPDATE SET count = count + ?",
            (key, period, by, by),
        )
        row = c.execute(
            "SELECT count FROM usage WHERE key = ? AND period = ?",
            (key, period),
        ).fetchone()
    return row["count"]


def key_stats(key: str) -> dict | None:
    row = get_key(key)
    if row is None:
        return None
    plan = get_plan(row["plan_id"])
    used = usage_this_period(key)
    return {
        "plan": plan.id,
        "plan_name": plan.name,
        "monthly_quota": plan.monthly_quota,
        "used_this_month": used,
        "remaining": max(0, plan.monthly_quota - used),
        "created_at": row["created_at"],
    }


def ensure_demo_key() -> str:
    """Guarantee a well-known free demo key exists (for the UI 'Try it' console)."""
    demo = "rg_demo_free_key"
    with _LOCK, _conn() as c:
        exists = c.execute(
            "SELECT 1 FROM api_keys WHERE key = ?", (demo,)
        ).fetchone()
        if not exists:
            c.execute(
                "INSERT INTO api_keys(key, plan_id, email, created_at, active) "
                "VALUES (?,?,?,?,1)",
                (demo, DEFAULT_PLAN, "demo@returnguard.local", _now_iso()),
            )
    return demo
