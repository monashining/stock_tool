"""
初始化方案 B（SQLite）資料庫。

用法（PowerShell）：
  python scripts/init_sqlite.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlite_store import (
    connect_sqlite,
    init_sqlite_schema,
    sqlite_path_from_env,
)


def main() -> None:
    db_path = sqlite_path_from_env(os.getenv("SQLITE_PATH"))
    conn = connect_sqlite(db_path)
    try:
        init_sqlite_schema(conn)
    finally:
        conn.close()
    print(f"SQLite schema ready: {db_path}")


if __name__ == "__main__":
    main()
