from line_group_query_bot import (
    is_help_command,
    normalize_ticker,
    parse_group_query_command,
)


def test_normalize_ticker():
    assert normalize_ticker("3037") == "3037.TW"
    assert normalize_ticker("3037.tw") == "3037.TW"
    assert normalize_ticker("foo") is None


def test_parse_commands():
    c = parse_group_query_command("速查 3037 3189")
    assert c is not None and c.action == "multi_ultra"
    assert len(c.tickers_normalized) == 2

    c = parse_group_query_command("查 3037")
    assert c is not None and c.action == "compact" and c.ticker_normalized == "3037.TW"
    assert c.tickers_normalized == ("3037.TW",)
    c = parse_group_query_command("完整 2330")
    assert c is not None and c.action == "full"
    c = parse_group_query_command("撿便宜 2330")
    assert c is not None and c.action == "dip" and not c.has_position_mode
    c = parse_group_query_command("持股 2330")
    assert c is not None and c.action == "position" and c.has_position_mode
    c = parse_group_query_command("2330.TW")
    assert c is not None and c.action == "compact"


def test_parse_multi_tickers_space_and_comma():
    c = parse_group_query_command("查 3037 3189 2367")
    assert c is not None
    assert c.tickers_normalized == ("3037.TW", "3189.TW", "2367.TW")
    assert c.tickers_raw == ("3037", "3189", "2367")

    c2 = parse_group_query_command("查 3037,3189")
    assert c2 is not None
    assert c2.tickers_normalized == ("3037.TW", "3189.TW")


def test_parse_multi_rejects_invalid_token():
    assert parse_group_query_command("查 3037 xx") is None


def test_parse_multi_dedupes_normalized():
    c = parse_group_query_command("查 3037 3037.TW")
    assert c is not None
    assert len(c.tickers_normalized) == 1


def test_help():
    assert is_help_command("help")
    assert is_help_command("/help")
    assert is_help_command("幫助")
    assert is_help_command("@bot help")
    assert is_help_command("help ?")
    assert not is_help_command("查 1")
