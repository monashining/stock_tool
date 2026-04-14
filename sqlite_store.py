"""
SQLite 儲存層（方案 B）：初始化 schema、寫入群組查詢日誌。
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

DEFAULT_SQLITE_PATH = "data/stock_tool.db"


def _as_bool(env_value: str | None) -> bool:
    return str(env_value or "").strip().lower() in ("1", "true", "yes", "on")


def sqlite_log_enabled_from_env(raw_value: str | None) -> bool:
    """是否啟用 SQLite 寫入（預設開啟）。"""
    if raw_value is None:
        return True
    return _as_bool(raw_value)


def sqlite_path_from_env(raw_value: str | None) -> str:
    """讀取 SQLITE_PATH，未設定時回預設。"""
    p = (raw_value or "").strip()
    return p or DEFAULT_SQLITE_PATH


def _ensure_parent_dir(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def connect_sqlite(db_path: str) -> sqlite3.Connection:
    _ensure_parent_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_sqlite_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at_utc TEXT NOT NULL,
            source_type TEXT,
            source_user_id TEXT,
            source_group_id TEXT,
            raw_text TEXT NOT NULL,
            action TEXT NOT NULL,
            tickers TEXT NOT NULL,
            ok INTEGER NOT NULL,
            error_text TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_query_log_created_at
        ON query_log(created_at_utc)
        """
    )
    conn.commit()


def log_query_event(
    *,
    db_path: str,
    source_type: str,
    source_user_id: str,
    source_group_id: str,
    raw_text: str,
    action: str,
    tickers: Sequence[str],
    ok: bool,
    error_text: Optional[str] = None,
) -> None:
    """寫入一次查詢日誌；供 webhook 呼叫，失敗可由外層吞掉。"""
    now_utc = datetime.now(timezone.utc).isoformat()
    tickers_txt = ",".join(tickers)
    conn = connect_sqlite(db_path)
    try:
        init_sqlite_schema(conn)
        conn.execute(
            """
            INSERT INTO query_log (
                created_at_utc,
                source_type,
                source_user_id,
                source_group_id,
                raw_text,
                action,
                tickers,
                ok,
                error_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_utc,
                source_type,
                source_user_id,
                source_group_id,
                raw_text,
                action,
                tickers_txt,
                1 if ok else 0,
                error_text or "",
            ),
        )
        conn.commit()
    finally:
        conn.close()
