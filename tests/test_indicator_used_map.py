from indicator_used_map import (
    build_fail_lines_from_used_map,
    build_fail_lines_short_from_used_map,
    get_primary_risk_category,
    get_primary_risk_rule_and_category,
    ordered_fail_norm_and_category,
    phrase_for_short_display,
    rule_item,
    severity_for_normalized_rule,
)


def test_build_fail_lines_from_used_map():
    um = {
        "Gate": [
            rule_item("a", "Gate A", None, "t", True),
            rule_item("b", "Gate B", None, "t", False, note="n1"),
        ],
        "Trigger": [],
        "Guard": [],
        "Chip Notes": [],
    }
    lines = build_fail_lines_from_used_map(um, max_lines=5)
    assert len(lines) == 1
    assert "Gate B" in lines[0]
    assert "n1" in lines[0]


def test_get_primary_risk_rule_and_category():
    um = {
        "Guard": [rule_item("a", "Guard 先", None, "t", False)],
        "Gate": [rule_item("b", "Gate 後", None, "t", False)],
        "Trigger": [],
        "Chip Notes": [],
    }
    r, cat = get_primary_risk_rule_and_category(um)
    assert r == "Guard 先" and cat == "Guard"
    assert get_primary_risk_category(um) == "Guard"


def test_fail_lines_priority_order():
    """Guard 應排在 Gate 之前（與 CATEGORY_PRIORITY 一致）。"""
    um = {
        "Gate": [
            rule_item("g", "Gate 未過", None, "t", False),
        ],
        "Guard": [
            rule_item("gu", "Guard 未過", None, "t", False),
        ],
        "Trigger": [],
        "Chip Notes": [],
    }
    lines = build_fail_lines_from_used_map(um, max_lines=5)
    assert len(lines) == 2
    assert lines[0].startswith("- Guard")
    assert "Guard 未過" in lines[0]
    assert lines[1].startswith("- Gate")
    short = build_fail_lines_short_from_used_map(um, max_lines=5)
    assert short == ["Guard 未過", "Gate 未過"]


def test_phrase_for_short_display():
    assert phrase_for_short_display("未站上均線") == "結構未轉強"
    assert phrase_for_short_display("跌破防守線") == "防守線失守"
    assert phrase_for_short_display("其他") == "其他"
    assert severity_for_normalized_rule("未站上均線") == "medium"
    assert severity_for_normalized_rule("未知") == "neutral"


def test_severity_sort_orders_medium_before_neutral():
    """語氣 medium 應排在 neutral 前（同批 FAIL 去重後）。"""
    um = {
        "Guard": [rule_item("a", "G1", None, "t", False)],
        "Gate": [
            rule_item("b", "未站上 EMA20", None, "t", False),
            rule_item("c", "未站上均線", None, "t", False),
            rule_item("d", "第三條", None, "t", False),
        ],
        "Trigger": [],
        "Chip Notes": [],
    }
    pairs = ordered_fail_norm_and_category(um)
    norms = [p[0] for p in pairs]
    assert norms[0] == "未站上均線"
    assert norms[1] == "G1"
    assert norms[2] == "第三條"


def test_fail_lines_short_dedup():
    """別名合併 + 去重後條數減少。"""
    um = {
        "Guard": [
            rule_item("a", "未站上 EMA20", None, "t", False),
            rule_item("b", "未站上均線", None, "t", False),
        ],
        "Gate": [],
        "Trigger": [],
        "Chip Notes": [],
    }
    short = build_fail_lines_short_from_used_map(um, max_lines=5)
    assert len(short) == 1
    assert short[0] == "結構未轉強"


def test_compact_uses_only_first_two_after_dedup():
    """別名合併後仍多條時，max_lines=2 只取前兩條（精簡上游與 formatter 一致）。"""
    um = {
        "Guard": [rule_item("a", "G1", None, "t", False)],
        "Gate": [
            rule_item("b", "未站上 EMA20", None, "t", False),
            rule_item("c", "未站上均線", None, "t", False),
            rule_item("d", "第三條", None, "t", False),
        ],
        "Trigger": [],
        "Chip Notes": [],
    }
    short = build_fail_lines_short_from_used_map(um, max_lines=2)
    assert short == ["結構未轉強", "G1"]
    r, _cat = get_primary_risk_rule_and_category(um)
    assert r == "結構未轉強"


def test_fail_lines_short_no_prefix():
    um = {
        "Guard": [rule_item("a", "收盤位置要強", None, "t", False, note="n")],
        "Gate": [],
        "Trigger": [],
        "Chip Notes": [],
    }
    assert build_fail_lines_short_from_used_map(um) == ["收盤位置要強"]


def test_fail_lines_max():
    um = {
        "Gate": [
            rule_item("a", "G1", None, "t", False),
            rule_item("b", "G2", None, "t", False),
            rule_item("c", "G3", None, "t", False),
        ],
        "Trigger": [
            rule_item("d", "T1", None, "t", False),
        ],
        "Guard": [],
        "Chip Notes": [],
    }
    lines = build_fail_lines_from_used_map(um, max_lines=2)
    assert len(lines) == 2
