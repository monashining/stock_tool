"""
Local Streamlit dashboard for SQLite query_log inspection.

Run:
  streamlit run monitor.py
"""
from __future__ import annotations

import os
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd
import streamlit as st

from sqlite_store import sqlite_path_from_env


def _split_tickers(tickers_csv: str) -> List[str]:
    return [x.strip() for x in str(tickers_csv or "").split(",") if x.strip()]


def _load_query_log(db_path: str, days: int, limit: int) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        sql = """
            SELECT
                id,
                created_at_utc,
                source_type,
                source_user_id,
                source_group_id,
                raw_text,
                action,
                tickers,
                ok,
                error_text
            FROM query_log
        """
        params: list[object] = []
        where = []
        if days > 0:
            since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            where.append("created_at_utc >= ?")
            params.append(since)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, limit))
        df = pd.read_sql_query(sql, conn, params=params)
        return df
    finally:
        conn.close()


def _ticker_rank(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    c: Counter[str] = Counter()
    for csv in df.get("tickers", pd.Series([], dtype="object")):
        c.update(_split_tickers(str(csv)))
    rows = [{"ticker": k, "count": v} for k, v in c.most_common(top_n)]
    return pd.DataFrame(rows)


def _group_rank(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    s = (
        df.get("source_group_id", pd.Series([], dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    s = s[s != ""]
    vc = s.value_counts().head(top_n)
    if vc.empty:
        return pd.DataFrame(columns=["group_id", "count"])
    return vc.rename_axis("group_id").reset_index(name="count")


def _daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["day", "queries"])
    tmp = df.copy()
    tmp["created_at_utc"] = pd.to_datetime(tmp["created_at_utc"], errors="coerce", utc=True)
    tmp = tmp.dropna(subset=["created_at_utc"])
    if tmp.empty:
        return pd.DataFrame(columns=["day", "queries"])
    tmp["day"] = tmp["created_at_utc"].dt.strftime("%Y-%m-%d")
    vc = tmp["day"].value_counts().sort_index()
    return vc.rename_axis("day").reset_index(name="queries")


def main() -> None:
    st.set_page_config(page_title="Query Monitor", layout="wide")
    st.title("Query Monitor")

    db_path = sqlite_path_from_env(os.getenv("SQLITE_PATH"))

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.caption(f"SQLite path: {db_path}")
    with col2:
        days = st.number_input("Recent days", min_value=0, max_value=3650, value=7, step=1)
    with col3:
        limit = st.number_input("Row limit", min_value=10, max_value=50000, value=1000, step=10)

    try:
        df = _load_query_log(db_path, int(days), int(limit))
    except Exception as e:
        st.error(f"Failed to read SQLite: {type(e).__name__}: {e}")
        st.stop()

    if df.empty:
        st.info("No data yet. Send some queries to your bot first.")
        return

    total = len(df)
    ok_n = int((df["ok"] == 1).sum()) if "ok" in df.columns else 0
    fail_n = total - ok_n
    success_rate = (ok_n / total * 100.0) if total > 0 else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{total}")
    m2.metric("Success", f"{ok_n}")
    m3.metric("Fail", f"{fail_n}")
    m4.metric("Success Rate", f"{success_rate:.1f}%")

    left, right = st.columns(2)

    with left:
        st.subheader("Top Tickers")
        tickers_df = _ticker_rank(df, top_n=10)
        st.dataframe(tickers_df, use_container_width=True, hide_index=True)

        st.subheader("Top Groups")
        groups_df = _group_rank(df, top_n=10)
        st.dataframe(groups_df, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Daily Query Trend")
        trend_df = _daily_trend(df)
        if trend_df.empty:
            st.caption("No parsable timestamps.")
        else:
            st.line_chart(trend_df.set_index("day")["queries"], use_container_width=True)

        st.subheader("Action Distribution")
        action_df = (
            df["action"]
            .fillna("")
            .astype(str)
            .value_counts()
            .rename_axis("action")
            .reset_index(name="count")
        )
        st.dataframe(action_df, use_container_width=True, hide_index=True)

    st.subheader("Recent Queries")
    show_cols = [
        "created_at_utc",
        "source_type",
        "source_group_id",
        "source_user_id",
        "action",
        "tickers",
        "ok",
        "raw_text",
        "error_text",
    ]
    keep_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[keep_cols], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
