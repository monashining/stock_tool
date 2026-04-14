"""統一 ResolvedDecision contract — pytest 表（可落地規格）。"""

from __future__ import annotations

from final_decision_resolver import (
    DecisionInput,
    FinalAction,
    FinalColor,
    FinalState,
    ReasonCode,
    ResolvedDecision,
    build_compact_line_diagnosis,
    get_status_bar_label,
    get_status_bar_label_for_score,
    get_status_bar_title,
    group_reason_codes,
    resolve_final_decision,
)
from position_advice import get_position_advice


def test_close_below_ema5_but_above_ema20_should_reduce():
    """Case 1：收盤跌破 EMA5，但仍在 EMA20 上方。"""
    inp = DecisionInput(
        close=74.0,
        ema5=74.57,
        ema20=72.56,
        weighted_ai_score=72,
        bottom_status="ALLOW",
        exec_guard_ok=True,
        gate_pass=False,
        trigger_pass=False,
        guard_pass=False,
    )
    result = resolve_final_decision(inp)

    assert result.action == FinalAction.REDUCE
    assert result.state == FinalState.DEFENSIVE_TREND_BREAK
    assert result.primary_reason == ReasonCode.PRICE_BELOW_EMA5
    assert ReasonCode.PRICE_ABOVE_EMA20 in result.reason_codes


def test_exec_guard_fail_should_not_hold_even_if_ai_score_high():
    """Case 2：EXEC_GUARD fail 即使 AI 高分，也不能 HOLD。"""
    inp = DecisionInput(
        close=100,
        ema5=95,
        ema20=90,
        weighted_ai_score=82,
        bottom_status="ALLOW",
        exec_guard_ok=False,
        gate_pass=True,
        trigger_pass=True,
        guard_pass=False,
    )
    result = resolve_final_decision(inp)

    assert result.action in (FinalAction.REDUCE, FinalAction.EXIT)
    assert result.state == FinalState.EXEC_GUARD_FAIL


def test_low_ai_score_should_reduce_even_if_bottom_allow():
    """Case 3：AI < 70，bottom=ALLOW，仍不應積極續抱（REDUCE）。"""
    inp = DecisionInput(
        close=110,
        ema5=108,
        ema20=100,
        weighted_ai_score=54,
        bottom_status="ALLOW",
        exec_guard_ok=True,
        gate_pass=True,
        trigger_pass=False,
        guard_pass=False,
    )
    result = resolve_final_decision(inp)

    assert result.action == FinalAction.REDUCE
    assert result.state == FinalState.LOW_AI_SCORE


def test_buy_narrative_only_when_gate_trigger_guard_all_pass():
    """Case 4：Gate / Trigger / Guard 全通過，才能 buy narrative = True。"""
    inp = DecisionInput(
        close=120,
        ema5=115,
        ema20=105,
        weighted_ai_score=78,
        bottom_status="ALLOW",
        exec_guard_ok=True,
        gate_pass=True,
        trigger_pass=True,
        guard_pass=True,
    )
    result = resolve_final_decision(inp)

    assert result.can_buy_narrative is True
    assert result.can_push_line_buy_signal is True


def test_gate_pass_but_trigger_fail_should_block_buy_narrative():
    """Case 5：Gate PASS 但 Trigger FAIL，只能觀察，不可寫推薦買入。"""
    inp = DecisionInput(
        close=120,
        ema5=115,
        ema20=105,
        weighted_ai_score=78,
        bottom_status="ALLOW",
        exec_guard_ok=True,
        gate_pass=True,
        trigger_pass=False,
        guard_pass=True,
    )
    result = resolve_final_decision(inp)

    assert result.can_buy_narrative is False
    assert ReasonCode.NARRATIVE_BUY_BLOCKED in result.reason_codes


def test_hard_stop_should_override_other_positive_signals():
    """Case 6：停損優先。"""
    inp = DecisionInput(
        close=120,
        ema5=115,
        ema20=105,
        weighted_ai_score=88,
        bottom_status="ALLOW",
        exec_guard_ok=True,
        gate_pass=True,
        trigger_pass=True,
        guard_pass=True,
        stop_loss_pct=-6.0,
    )
    result = resolve_final_decision(inp)

    assert result.action == FinalAction.EXIT
    assert result.state == FinalState.HARD_STOP
    assert result.primary_reason == ReasonCode.STOP_LOSS_HIT


def test_status_bar_title_and_label_dual_mode():
    d_hold = ResolvedDecision(
        action=FinalAction.HOLD,
        color=FinalColor.GREEN,
        state=FinalState.HEALTHY_TREND,
        primary_reason=ReasonCode.BOTTOM_ALLOW,
    )
    assert get_status_bar_title(True) == "去留診斷"
    assert get_status_bar_title(False) == "標的狀態"
    assert get_status_bar_label(d_hold, True) == "持股續抱"
    assert get_status_bar_label(d_hold, False) == "標的偏多"
    d_exit = ResolvedDecision(
        action=FinalAction.EXIT,
        color=FinalColor.RED,
        state=FinalState.HARD_STOP,
        primary_reason=ReasonCode.STOP_LOSS_HIT,
    )
    assert get_status_bar_label(d_exit, True) == "出場防守"
    assert get_status_bar_label(d_exit, False) == "暫不建倉"


def test_build_compact_line_diagnosis_with_decision_and_one_line():
    d = ResolvedDecision(
        action=FinalAction.REDUCE,
        color=FinalColor.RED,
        state=FinalState.DEFENSIVE_TREND_BREAK,
        primary_reason=ReasonCode.PRICE_BELOW_EMA5,
        summary_text="收盤若未站回 EMA5，應減碼或出場。",
    )
    text = build_compact_line_diagnosis(
        ticker="3037.TW",
        name="景碩",
        close_price=74.0,
        score=55,
        has_position=True,
        decision=d,
        one_line_verdict="趨勢未壞，但短線先防守",
        summary_fallback="備援",
    )
    assert "去留診斷｜減碼觀望" in text
    assert "一句話：趨勢未壞，但短線先防守" in text
    assert "說明：收盤若未站回 EMA5" in text
    assert "備援" not in text


def test_build_compact_line_diagnosis_no_decision_uses_fallback():
    text = build_compact_line_diagnosis(
        ticker="2330.TW",
        close_price=100.0,
        score=50,
        has_position=False,
        decision=None,
        one_line_verdict="",
        summary_fallback="等待回踩成功再觀察。",
    )
    assert "標的狀態｜" in text
    assert "一句話：" in text
    assert "說明：等待回踩成功再觀察。" in text


def test_status_bar_label_for_score_fallback():
    assert get_status_bar_label_for_score(80, True) == "持股續抱"
    assert get_status_bar_label_for_score(80, False) == "標的偏多"
    assert get_status_bar_label_for_score(50, False) == "標的觀望"
    assert get_status_bar_label_for_score(30, False) == "標的轉弱"
    assert get_status_bar_label_for_score(30, True) == "出場防守"


def test_group_reason_codes_buckets():
    g = group_reason_codes(
        [
            ReasonCode.PRICE_BELOW_EMA5,
            ReasonCode.PRICE_ABOVE_EMA20,
            ReasonCode.EXEC_GUARD_FAILED,
            ReasonCode.NARRATIVE_BUY_BLOCKED,
            ReasonCode.GATE_FAILED,
        ]
    )
    assert ReasonCode.PRICE_BELOW_EMA5 in g.price_reasons
    assert ReasonCode.PRICE_ABOVE_EMA20 in g.price_reasons
    assert ReasonCode.EXEC_GUARD_FAILED in g.risk_reasons
    assert ReasonCode.NARRATIVE_BUY_BLOCKED in g.narrative_reasons
    assert ReasonCode.GATE_FAILED in g.narrative_reasons


def test_position_advice_stop_loss_uses_exit_contract():
    adv = get_position_advice(
        current_price=90.0,
        avg_cost=100.0,
        qty=1,
        ema_defense=85.0,
        ema5_short=88.0,
        ema20_trend=80.0,
        ai_score=80.0,
        guard_ok=True,
        bottom_result={"status": "ALLOW", "score": 3},
        top_result={"status": "ALLOW", "score": 0},
    )
    assert adv.resolution is not None
    assert adv.resolution.action == FinalAction.EXIT
    assert adv.resolution.primary_reason == ReasonCode.STOP_LOSS_HIT
    assert adv.level == "error"
