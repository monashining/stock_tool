"""LINE 推播尾段與截斷。"""

from __future__ import annotations

from types import SimpleNamespace

from line_push_formatter import (
    LINE_PUSH_MAX_CHARS,
    append_line_push_tail,
    build_line_push_payload,
    fuse_one_line_verdict_with_primary_risk,
    truncate_line_push,
    ultra_compact_one_line,
)


def test_ultra_compact_one_line():
    assert ultra_compact_one_line("觀望", ["防守線失守"]) == "防守線失守 → 觀望"
    assert ultra_compact_one_line("觀望", []) == "觀望"
    assert ultra_compact_one_line("", ["僅風險"]) == "僅風險"


def test_build_line_push_payload_ultra_compact_head():
    g = SimpleNamespace(
        break_close=10.0,
        guard_close=9.5,
        buffer_pct=1.5,
        ema_today=10.2,
    )
    text = build_line_push_payload(
        mode="compact",
        ticker="2330.TW",
        close_price=500.0,
        score=55,
        has_position=False,
        decision=None,
        one_line_verdict="觀望",
        bottom_now={"status": "ALLOW", "score": 2},
        top_now={"status": "WATCH", "score": 1},
        guard=g,
        defense_name="EMA5",
        fail_lines=["- G｜x"],
        fail_lines_short=["防守線失守"],
        merge_primary_risk_into_verdict=True,
        ultra_compact_head=True,
    )
    assert "防守線失守 → 觀望" in text


def test_ultra_compact_head_fallback_when_short_risk_all_blank():
    """短風險僅空白字串時不套用 ultra（避免無主風險仍出現異常標題）。"""
    g = SimpleNamespace(
        break_close=10.0,
        guard_close=9.5,
        buffer_pct=1.5,
        ema_today=10.2,
    )
    text = build_line_push_payload(
        mode="compact",
        ticker="2330.TW",
        close_price=500.0,
        score=55,
        has_position=False,
        decision=None,
        one_line_verdict="觀望",
        bottom_now={"status": "ALLOW", "score": 2},
        top_now={"status": "WATCH", "score": 1},
        guard=g,
        defense_name="EMA5",
        fail_lines=[],
        fail_lines_short=["", "  "],
        merge_primary_risk_into_verdict=True,
        ultra_compact_head=True,
    )
    assert "觀望" in text
    assert "→ 觀望" not in text


def test_fuse_one_line_verdict_with_primary_risk():
    assert fuse_one_line_verdict_with_primary_risk("觀望", ["收盤位置要強"]) == "觀望（收盤位置要強）"
    assert fuse_one_line_verdict_with_primary_risk("已有收盤位置要強", ["收盤位置要強"]) == "已有收盤位置要強"
    assert fuse_one_line_verdict_with_primary_risk("", ["僅風險"]) == "僅風險"
    assert (
        fuse_one_line_verdict_with_primary_risk(
            "觀望",
            ["防守線失守"],
            risk_category="Guard",
            show_risk_category=True,
        )
        == "觀望（防守線失守｜防守）"
    )


def test_fuse_no_duplicate_when_already_in_verdict():
    """一句話已含第一條風險語意時不再括號重複。"""
    assert (
        fuse_one_line_verdict_with_primary_risk("短線轉弱先防守跌破防守線", ["跌破防守線"])
        == "短線轉弱先防守跌破防守線"
    )


def test_compact_truncates_fail_lines_short_to_two():
    """精簡模式尾段風險最多 2 條短句（即使多傳）。"""
    g = SimpleNamespace(
        break_close=10.0,
        guard_close=9.5,
        buffer_pct=1.5,
        ema_today=10.2,
    )
    long_short = ["一", "二", "三", "四"]
    text = build_line_push_payload(
        mode="compact",
        ticker="2330.TW",
        close_price=500.0,
        score=55,
        has_position=False,
        decision=None,
        one_line_verdict="x",
        bottom_now={"status": "ALLOW", "score": 2},
        top_now={"status": "WATCH", "score": 1},
        guard=g,
        defense_name="EMA5",
        fail_lines=["- G｜a"] * 4,
        fail_lines_short=long_short,
    )
    assert "風險：一" in text
    assert "風險（續）：二" in text
    assert "三" not in text


def test_build_line_push_payload_merge_primary_risk():
    g = SimpleNamespace(
        break_close=10.0,
        guard_close=9.5,
        buffer_pct=1.5,
        ema_today=10.2,
    )
    text = build_line_push_payload(
        mode="compact",
        ticker="2330.TW",
        close_price=500.0,
        score=55,
        has_position=False,
        decision=None,
        one_line_verdict="短線觀望",
        bottom_now={"status": "ALLOW", "score": 2},
        top_now={"status": "WATCH", "score": 1},
        guard=g,
        defense_name="EMA5",
        fail_lines=["- Guard｜x"],
        fail_lines_short=["跌破防守線"],
        merge_primary_risk_into_verdict=True,
    )
    assert "短線觀望（跌破防守線）" in text


def test_truncate_line_push():
    short = "a" * 100
    assert truncate_line_push(short) == short
    long = "x" * (LINE_PUSH_MAX_CHARS + 500)
    out = truncate_line_push(long)
    assert len(out) <= LINE_PUSH_MAX_CHARS
    assert "截斷" in out


def test_append_tail_compact_uses_fail_lines_short():
    g = SimpleNamespace(
        break_close=10.0,
        guard_close=9.5,
        buffer_pct=1.5,
        ema_today=10.2,
    )
    lines = ["head"]
    append_line_push_tail(
        lines,
        compact=True,
        bottom_txt="A",
        top_txt="B",
        guard=g,
        defense_name="EMA5",
        fail_lines=["- Guard｜長規則｜note"],
        expert_msg="x",
        fail_lines_short=["短句一", "短句二"],
    )
    joined = "\n".join(lines)
    assert "風險：短句一" in joined
    assert "風險（續）：短句二" in joined
    assert "長規則" not in joined


def test_append_tail_compact_omits_expert():
    g = SimpleNamespace(
        break_close=10.0,
        guard_close=9.5,
        buffer_pct=1.5,
        ema_today=10.2,
    )
    lines = ["head"]
    append_line_push_tail(
        lines,
        compact=True,
        bottom_txt="ALLOW 3/4",
        top_txt="WATCH 2/5",
        guard=g,
        defense_name="EMA5",
        fail_lines=["- Gate｜x"],
        expert_msg="很長的專家文",
    )
    assert any("補充：TURN bottom" in x for x in lines)
    assert any("防守：" in x for x in lines)
    assert any("風險：" in x for x in lines)
    assert not any(x == "專家：" for x in lines)
    assert "很長的專家文" not in "\n".join(lines)


def test_append_tail_full_has_expert():
    lines = ["h"]
    append_line_push_tail(
        lines,
        compact=False,
        bottom_txt="A",
        top_txt="B",
        guard=None,
        defense_name="EMA5",
        fail_lines=[],
        expert_msg="結語",
    )
    assert "專家：" in lines
    assert "結語" in lines


def test_build_line_push_payload_turn_from_dict():
    g = SimpleNamespace(
        break_close=10.0,
        guard_close=9.5,
        buffer_pct=1.5,
        ema_today=10.2,
    )
    compact = build_line_push_payload(
        mode="compact",
        ticker="3037.TW",
        close_price=100.0,
        score=50,
        has_position=True,
        decision=None,
        one_line_verdict="觀望",
        bottom_now={"status": "ALLOW", "score": 3},
        top_now={"status": "WATCH", "score": 2},
        guard=g,
        defense_name="EMA5",
        fail_lines=["- G｜a", "- T｜b"],
        expert_msg="不應出現",
    )
    assert "【3037.TW】" in compact
    assert "補充：TURN bottom｜ALLOW 3/4" in compact
    assert "TURN top｜WATCH 2/5" in compact
    assert "不應出現" not in compact

    full = build_line_push_payload(
        mode="full",
        ticker="3037.TW",
        close_price=100.0,
        score=50,
        has_position=True,
        decision=None,
        one_line_verdict="觀望",
        bottom_now={"status": "ALLOW", "score": 3},
        top_now={"status": "WATCH", "score": 2},
        guard=g,
        defense_name="EMA5",
        fail_lines=["- G｜a"],
        expert_msg="專家段落",
    )
    assert "專家：" in full
    assert "專家段落" in full
