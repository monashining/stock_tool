"""
Inspect SQLite query_log: recent rows, hot tickers, hot groups.

Usage (PowerShell):
  python scripts/inspect_sqlite.py
  python scripts/inspect_sqlite.py --limit 20 --days 7
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlite_store import sqlite_path_from_env


def _split_tickers(tickers_csv: str) -> List[str]:
    return [x.strip() for x in str(tickers_csv or "").split(",") if x.strip()]


def _iter_rows(
    conn: sqlite3.Connection,
    *,
    since_utc_iso: str | None,
    limit: int,
) -> Iterable[sqlite3.Row]:
    if since_utc_iso:
        sql = """
            SELECT *
            FROM query_log
            WHERE created_at_utc >= ?
            ORDER BY id DESC
            LIMIT ?
        """
        return conn.execute(sql, (since_utc_iso, limit)).fetchall()

    sql = """
        SELECT *
        FROM query_log
        ORDER BY id DESC
        LIMIT ?
    """
    return conn.execute(sql, (limit,)).fetchall()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SQLite query_log")
    parser.add_argument("--limit", type=int, default=100, help="max rows to load")
    parser.add_argument("--days", type=int, default=0, help="recent N days only (0=all)")
    args = parser.parse_args()

    db_path = sqlite_path_from_env(os.getenv("SQLITE_PATH"))
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        since = None
        if args.days > 0:
            since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
        rows = list(_iter_rows(conn, since_utc_iso=since, limit=max(1, args.limit)))
    finally:
        conn.close()

    print(f"DB: {db_path}")
    print(f"Rows loaded: {len(rows)}")

    if not rows:
        print("\n=== no data ===")
        return

    print("\n=== latest 10 rows ===")
    for r in rows[:10]:
        print(
            f"{r['created_at_utc']} | {r['source_type']} | {r['action']} | "
            f"{r['tickers']} | ok={r['ok']}"
        )

    ticker_counter: Counter[str] = Counter()
    group_counter: Counter[str] = Counter()
    action_counter: Counter[str] = Counter()
    ok_counter: Counter[str] = Counter()

    for r in rows:
        ticker_counter.update(_split_tickers(r["tickers"]))
        g = str(r["source_group_id"] or "").strip()
        if g:
            group_counter[g] += 1
        action_counter[str(r["action"] or "").strip()] += 1
        ok_counter["ok" if int(r["ok"] or 0) == 1 else "fail"] += 1

    print("\n=== top tickers (10) ===")
    for t, n in ticker_counter.most_common(10):
        print(f"{t}: {n}")

    print("\n=== top groups (5) ===")
    if group_counter:
        for g, n in group_counter.most_common(5):
            print(f"{g}: {n}")
    else:
        print("(no group_id)")

    print("\n=== action distribution ===")
    for a, n in action_counter.most_common():
        print(f"{a}: {n}")

    print("\n=== success rate ===")
    ok_n = ok_counter.get("ok", 0)
    fail_n = ok_counter.get("fail", 0)
    total = ok_n + fail_n
    rate = (ok_n / total * 100.0) if total > 0 else 0.0
    print(f"ok={ok_n}, fail={fail_n}, success_rate={rate:.1f}%")


if __name__ == "__main__":
    main()
