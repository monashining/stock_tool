from sqlite_store import (
    connect_sqlite,
    init_sqlite_schema,
    log_query_event,
    sqlite_log_enabled_from_env,
    sqlite_path_from_env,
)


def test_sqlite_path_from_env_default():
    assert sqlite_path_from_env(None).endswith("stock_tool.db")
    assert sqlite_path_from_env("").endswith("stock_tool.db")


def test_sqlite_log_enabled_from_env():
    assert sqlite_log_enabled_from_env(None) is True
    assert sqlite_log_enabled_from_env("1") is True
    assert sqlite_log_enabled_from_env("true") is True
    assert sqlite_log_enabled_from_env("0") is False


def test_log_query_event_writes_row(tmp_path):
    db_path = str(tmp_path / "test.db")
    log_query_event(
        db_path=db_path,
        source_type="group",
        source_user_id="u1",
        source_group_id="g1",
        raw_text="查 3037",
        action="compact",
        tickers=("3037.TW",),
        ok=True,
        error_text="",
    )

    conn = connect_sqlite(db_path)
    try:
        init_sqlite_schema(conn)
        row = conn.execute(
            "SELECT source_type, raw_text, action, tickers, ok FROM query_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert row == ("group", "查 3037", "compact", "3037.TW", 1)
