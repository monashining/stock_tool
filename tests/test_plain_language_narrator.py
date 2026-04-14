"""白話翻譯層：撿便宜判斷與 build_plain_language_summary。"""

from __future__ import annotations

from final_decision_resolver import (
    FinalAction,
    FinalColor,
    FinalState,
    ReasonCode,
    ResolvedDecision,
)
from plain_language_narrator import (
    PlainLanguageNarratorInput,
    build_plain_language_summary,
    can_buy_the_dip,
)


def test_can_buy_the_dip_blocks_below_ema5():
    v = can_buy_the_dip(
        close=10.0,
        ema5=11.0,
        trigger_type="BREAKOUT",
        guard_ok=True,
        risk_alert=False,
    )
    assert v.allowed is False


def test_can_buy_the_dip_blocks_trigger_none():
    v = can_buy_the_dip(
        close=12.0,
        ema5=11.0,
        trigger_type="NONE",
        guard_ok=True,
        risk_alert=False,
    )
    assert v.allowed is False


def test_can_buy_the_dip_blocks_guard_fail():
    v = can_buy_the_dip(
        close=12.0,
        ema5=11.0,
        trigger_type="PULLBACK",
        guard_ok=False,
        risk_alert=False,
    )
    assert v.allowed is False


def test_can_buy_the_dip_allows_when_rules_pass():
    v = can_buy_the_dip(
        close=12.0,
        ema5=11.0,
        trigger_type="PULLBACK",
        guard_ok=True,
        risk_alert=False,
    )
    assert v.allowed is True


def test_plain_language_3037_style_scenario():
    d = ResolvedDecision(
        action=FinalAction.REDUCE,
        color=FinalColor.RED,
        state=FinalState.DEFENSIVE_TREND_BREAK,
        primary_reason=ReasonCode.PRICE_BELOW_EMA5,
        reason_codes=[
            ReasonCode.PRICE_BELOW_EMA5,
            ReasonCode.PRICE_ABOVE_EMA20,
        ],
    )
    inp = PlainLanguageNarratorInput(
        close=74.0,
        ema5=74.57,
        ema20=72.56,
        gate_ok=False,
        trigger_ok=False,
        guard_ok=False,
        trigger_type="NONE",
        bottom_status="ALLOW",
        top_status="WATCH",
        has_position=False,
        risk_alert=False,
    )
    s = build_plain_language_summary(d, inp)
    assert "中期" in s.current_state and "短線" in s.current_state
    assert "撿便宜" in s.why_not_buy or "止跌" in s.why_not_buy
    assert len(s.term_notes) >= 1
    assert s.one_line_verdict == "趨勢未壞，但短線先防守"


def test_one_line_verdict_default_on_none_decision():
    inp = PlainLanguageNarratorInput(
        close=100.0,
        ema5=101.0,
        ema20=99.0,
        gate_ok=True,
        trigger_ok=True,
        guard_ok=True,
        trigger_type="NONE",
        bottom_status=None,
        top_status=None,
        has_position=False,
    )
    s = build_plain_language_summary(None, inp)
    assert s.one_line_verdict == "資料不足，暫不判讀"
