from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import List, Optional, Set

# ---------------------------------------------------------------------------
# 1) 統一 enum（全站唯一契約）
# ---------------------------------------------------------------------------


class FinalAction(str, Enum):
    HOLD = "HOLD"
    WATCH = "WATCH"
    REDUCE = "REDUCE"
    EXIT = "EXIT"


class FinalColor(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class FinalState(str, Enum):
    HEALTHY_TREND = "HEALTHY_TREND"
    DEFENSIVE_TREND_BREAK = "DEFENSIVE_TREND_BREAK"
    EXEC_GUARD_FAIL = "EXEC_GUARD_FAIL"
    LOW_AI_SCORE = "LOW_AI_SCORE"
    WATCHLIST_ONLY = "WATCHLIST_ONLY"
    HARD_STOP = "HARD_STOP"
    NO_CLEAR_EDGE = "NO_CLEAR_EDGE"


class ReasonCode(str, Enum):
    PRICE_BELOW_EMA5 = "PRICE_BELOW_EMA5"
    PRICE_BELOW_DEFENSIVE_LINE = "PRICE_BELOW_DEFENSIVE_LINE"
    PRICE_ABOVE_EMA20 = "PRICE_ABOVE_EMA20"
    EXEC_GUARD_FAILED = "EXEC_GUARD_FAILED"
    AI_SCORE_LT_70 = "AI_SCORE_LT_70"
    BOTTOM_ALLOW = "BOTTOM_ALLOW"
    TOP_WATCH = "TOP_WATCH"
    TOP_BLOCK = "TOP_BLOCK"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    GATE_FAILED = "GATE_FAILED"
    TRIGGER_FAILED = "TRIGGER_FAILED"
    GUARD_FAILED = "GUARD_FAILED"
    NARRATIVE_BUY_BLOCKED = "NARRATIVE_BUY_BLOCKED"
    WATCHLIST_DEFAULT = "WATCHLIST_DEFAULT"


class EntryNarrativeTier(str, Enum):
    BUYABLE = "buyable"
    OBSERVE_NO_BUY = "observe_no_buy"
    NO_BUY = "no_buy"


ACTION_SEVERITY = {
    FinalAction.HOLD: 0,
    FinalAction.WATCH: 1,
    FinalAction.REDUCE: 2,
    FinalAction.EXIT: 3,
}

_COLOR_RANK = {FinalColor.RED: 2, FinalColor.YELLOW: 1, FinalColor.GREEN: 0}

ACTION_UI = {
    FinalAction.HOLD: {"label": "續抱", "emoji": "🟢"},
    FinalAction.WATCH: {"label": "觀望", "emoji": "🟡"},
    FinalAction.REDUCE: {"label": "減碼", "emoji": "🟠"},
    FinalAction.EXIT: {"label": "出場", "emoji": "🔴"},
}


# ---------------------------------------------------------------------------
# 2) Input / Output contract
# ---------------------------------------------------------------------------


@dataclass
class ReasonCodeGroups:
    """將 reason_codes 分組，供 UI 聚合（價格／風控／建倉敘述）。"""

    price_reasons: List[ReasonCode] = field(default_factory=list)
    risk_reasons: List[ReasonCode] = field(default_factory=list)
    narrative_reasons: List[ReasonCode] = field(default_factory=list)


_PRICE_REASON_CODES: Set[ReasonCode] = {
    ReasonCode.PRICE_BELOW_EMA5,
    ReasonCode.PRICE_BELOW_DEFENSIVE_LINE,
    ReasonCode.PRICE_ABOVE_EMA20,
    ReasonCode.BOTTOM_ALLOW,
    ReasonCode.TOP_WATCH,
    ReasonCode.TOP_BLOCK,
}
_RISK_REASON_CODES: Set[ReasonCode] = {
    ReasonCode.STOP_LOSS_HIT,
    ReasonCode.EXEC_GUARD_FAILED,
    ReasonCode.AI_SCORE_LT_70,
}
_NARRATIVE_REASON_CODES: Set[ReasonCode] = {
    ReasonCode.GATE_FAILED,
    ReasonCode.TRIGGER_FAILED,
    ReasonCode.GUARD_FAILED,
    ReasonCode.NARRATIVE_BUY_BLOCKED,
    ReasonCode.WATCHLIST_DEFAULT,
}


def group_reason_codes(codes: List[ReasonCode]) -> ReasonCodeGroups:
    seen: Set[ReasonCode] = set()
    g = ReasonCodeGroups()
    for c in codes:
        if c in seen:
            continue
        seen.add(c)
        if c in _PRICE_REASON_CODES:
            g.price_reasons.append(c)
        elif c in _RISK_REASON_CODES:
            g.risk_reasons.append(c)
        elif c in _NARRATIVE_REASON_CODES:
            g.narrative_reasons.append(c)
        else:
            g.narrative_reasons.append(c)
    return g


@dataclass
class DecisionInput:
    close: float
    ema5: Optional[float] = None
    ema20: Optional[float] = None
    defensive_line: Optional[float] = None

    weighted_ai_score: Optional[float] = None

    bottom_status: Optional[str] = None  # ALLOW / WATCH / BLOCK
    top_status: Optional[str] = None

    exec_guard_ok: Optional[bool] = None

    gate_pass: Optional[bool] = None
    trigger_pass: Optional[bool] = None
    guard_pass: Optional[bool] = None

    pnl_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None

    position_mode: bool = True


@dataclass
class ResolvedDecision:
    action: FinalAction
    color: FinalColor
    state: FinalState

    primary_reason: ReasonCode
    reason_codes: List[ReasonCode] = field(default_factory=list)

    summary_title: str = ""
    summary_text: str = ""
    expert_action_line: str = ""

    trace_lines: List[str] = field(default_factory=list)

    can_buy_narrative: bool = False
    can_push_line_buy_signal: bool = False

    debug_note: Optional[str] = None


def resolve_entry_narrative_tier(
    gate_ok: bool,
    trigger_ok: bool,
    guard_ok: bool,
) -> EntryNarrativeTier:
    if not gate_ok:
        return EntryNarrativeTier.NO_BUY
    if trigger_ok and guard_ok:
        return EntryNarrativeTier.BUYABLE
    return EntryNarrativeTier.OBSERVE_NO_BUY


def _effective_stop_pct(inp: DecisionInput) -> Optional[float]:
    if inp.stop_loss_pct is not None:
        return inp.stop_loss_pct
    return inp.pnl_pct


def _norm_status(s: Optional[str]) -> str:
    t = str(s or "NA").strip().upper()
    return t if t else "NA"


def build_line_message(symbol: str, decision: ResolvedDecision) -> str:
    ui = ACTION_UI[decision.action]
    return (
        f"{ui['emoji']} {symbol}｜{ui['label']}\n"
        f"State: {decision.state.value}\n"
        f"原因: {decision.primary_reason.value}\n"
        f"{decision.summary_text}\n"
        f"{decision.expert_action_line}"
    )


def expert_action_line_markdown(decision: ResolvedDecision) -> str:
    """專家區塊用：套上粗體行動句。"""
    t = (decision.expert_action_line or "").strip()
    if not t:
        return "**行動：觀望**。條件不足。"
    if t.startswith("**"):
        return t
    return f"**{t}**"


def get_status_bar_title(has_position: bool) -> str:
    """儀表條標題：持股語境用『去留』，無持股用『標的』。"""
    return "去留診斷" if has_position else "標的狀態"


def get_status_bar_label(decision: ResolvedDecision, has_position: bool) -> str:
    """依 ResolvedDecision.action 輸出雙模式一句話（不修改裁決，僅 UI 文案）。"""
    if has_position:
        mapping = {
            FinalAction.HOLD: "持股續抱",
            FinalAction.WATCH: "減碼觀望",
            FinalAction.REDUCE: "減碼觀望",
            FinalAction.EXIT: "出場防守",
        }
        return mapping.get(decision.action, "續抱/觀察")
    mapping = {
        FinalAction.HOLD: "標的偏多",
        FinalAction.WATCH: "標的觀望",
        FinalAction.REDUCE: "標的轉弱",
        FinalAction.EXIT: "暫不建倉",
    }
    return mapping.get(decision.action, "標的觀望")


def get_status_bar_label_for_score(score: int, has_position: bool) -> str:
    """無 ResolvedDecision 時，用綜合分數做同款雙模式標籤。"""
    if has_position:
        if score >= 70:
            return "持股續抱"
        if score >= 40:
            return "減碼觀望"
        return "出場防守"
    if score >= 70:
        return "標的偏多"
    if score >= 40:
        return "標的觀望"
    return "標的轉弱"


def build_compact_line_diagnosis(
    *,
    ticker: str,
    name: str = "",
    close_price: float,
    score: int,
    has_position: bool,
    decision: Optional[ResolvedDecision],
    one_line_verdict: str = "",
    summary_fallback: str = "",
) -> str:
    """
    LINE 推播精簡段：與頁面儀表條／白話一句同源。
    格式：標題｜標籤 → 一句話 → 說明（summary_text 優先）→ AI 分數。
    """
    lines: List[str] = []
    lines.append(f"【{ticker}】")
    if name:
        lines.append(f"名稱：{name}")
    lines.append(f"收盤：{close_price:.2f}")
    title = get_status_bar_title(has_position)
    if decision is not None:
        lab = get_status_bar_label(decision, has_position)
        explain = (decision.summary_text or "").strip()
    else:
        lab = get_status_bar_label_for_score(score, has_position)
        explain = ""
    lines.append(f"{title}｜{lab}")
    olv = (one_line_verdict or "").strip() or lab
    lines.append(f"一句話：{olv}")
    if not explain and (summary_fallback or "").strip():
        explain = summary_fallback.strip()
    if explain:
        lines.append(f"說明：{explain}")
    lines.append(f"AI 分數：{score}")
    return "\n".join(lines)


def resolve_final_decision(inp: DecisionInput) -> ResolvedDecision:
    """
    多規則各自產生候選 ResolvedDecision，依 action 嚴重度與燈號 tie-break。
    最後寫入 can_buy_narrative / can_push_line_buy_signal，並可附加 NARRATIVE_BUY_BLOCKED。
    """
    candidates: list[tuple[ResolvedDecision, int]] = []
    seq = 0

    def add(d: ResolvedDecision) -> None:
        nonlocal seq
        seq += 1
        candidates.append((d, seq))

    # 0) Hard stop
    sl = _effective_stop_pct(inp)
    if sl is not None and sl <= -5.0:
        add(
            ResolvedDecision(
                action=FinalAction.EXIT,
                color=FinalColor.RED,
                state=FinalState.HARD_STOP,
                primary_reason=ReasonCode.STOP_LOSS_HIT,
                reason_codes=[ReasonCode.STOP_LOSS_HIT],
                summary_title="硬停損",
                summary_text="已觸發停損條件，優先出場。",
                expert_action_line="行動：出場。已觸發停損，先保護資金。",
                trace_lines=["[RISK] pnl ≤ -5% → EXIT"],
            )
        )

    # 1) Price below EMA5
    if inp.ema5 is not None and inp.close < inp.ema5:
        extra: List[ReasonCode] = []
        if inp.ema20 is not None and inp.close > inp.ema20:
            extra.append(ReasonCode.PRICE_ABOVE_EMA20)
        st = FinalState.DEFENSIVE_TREND_BREAK
        lines = ["[PRICE] close < EMA5 → REDUCE"]
        if ReasonCode.PRICE_ABOVE_EMA20 in extra:
            lines.append("[PRICE] close > EMA20（趨勢未死、短線失控）→ context")
        add(
            ResolvedDecision(
                action=FinalAction.REDUCE,
                color=FinalColor.RED,
                state=st,
                primary_reason=ReasonCode.PRICE_BELOW_EMA5,
                reason_codes=[ReasonCode.PRICE_BELOW_EMA5, *extra],
                summary_title="破線防守",
                summary_text="收盤若未站回 EMA5，應減碼或出場。",
                expert_action_line="行動：減碼／出場。價格已跌破 EMA5，先尊重短線風控。",
                trace_lines=lines,
            )
        )

    # 2) Defensive line breach（與 EMA5 不同時才重複有意義；仍允許同時命中）
    if inp.defensive_line is not None and inp.close < inp.defensive_line:
        add(
            ResolvedDecision(
                action=FinalAction.REDUCE,
                color=FinalColor.RED,
                state=FinalState.DEFENSIVE_TREND_BREAK,
                primary_reason=ReasonCode.PRICE_BELOW_DEFENSIVE_LINE,
                reason_codes=[ReasonCode.PRICE_BELOW_DEFENSIVE_LINE],
                summary_title="跌破保守警戒",
                summary_text="已跌破自選防守／保守警戒，建議先降低風險。",
                expert_action_line="行動：減碼。已跌破保守警戒，先做防守。",
                trace_lines=["[PRICE] close < defensive_line → REDUCE"],
            )
        )

    # 3) Execution guard fail
    if inp.exec_guard_ok is False:
        add(
            ResolvedDecision(
                action=FinalAction.REDUCE,
                color=FinalColor.YELLOW,
                state=FinalState.EXEC_GUARD_FAIL,
                primary_reason=ReasonCode.EXEC_GUARD_FAILED,
                reason_codes=[ReasonCode.EXEC_GUARD_FAILED],
                summary_title="執行保護未通過",
                summary_text="執行保護未通過，不適合積極續抱或加碼。",
                expert_action_line="行動：減碼觀望。執行保護未過，避免主觀硬抱。",
                trace_lines=["[GUARD] exec_guard_ok = False → REDUCE"],
            )
        )

    # 4) AI score（僅持倉模式）
    if (
        inp.position_mode
        and inp.weighted_ai_score is not None
        and inp.weighted_ai_score < 70.0
    ):
        add(
            ResolvedDecision(
                action=FinalAction.REDUCE,
                color=FinalColor.YELLOW,
                state=FinalState.LOW_AI_SCORE,
                primary_reason=ReasonCode.AI_SCORE_LT_70,
                reason_codes=[ReasonCode.AI_SCORE_LT_70],
                summary_title="分數不足",
                summary_text="AI 加權主分低於 70，建議降低積極度。",
                expert_action_line="行動：減碼觀望。分數不足，暫不採積極持有。",
                trace_lines=["[AI] weighted_ai_score < 70 → REDUCE"],
            )
        )

    # 5) Structure okay
    bs = _norm_status(inp.bottom_status)
    if bs == "ALLOW":
        extra_top: List[ReasonCode] = []
        ts = _norm_status(inp.top_status)
        if ts == "WATCH":
            extra_top.append(ReasonCode.TOP_WATCH)
        elif ts == "BLOCK":
            extra_top.append(ReasonCode.TOP_BLOCK)
        add(
            ResolvedDecision(
                action=FinalAction.HOLD,
                color=FinalColor.GREEN,
                state=FinalState.HEALTHY_TREND,
                primary_reason=ReasonCode.BOTTOM_ALLOW,
                reason_codes=[ReasonCode.BOTTOM_ALLOW, *extra_top],
                summary_title="結構偏多",
                summary_text="底部結構仍可觀察，趨勢未完全破壞。",
                expert_action_line="行動：續抱。結構仍在，可依策略防守。",
                trace_lines=[
                    f"[STRUCTURE] bottom = ALLOW, top = {ts} → HOLD",
                ],
            )
        )

    # 6) Fallback
    if not candidates:
        add(
            ResolvedDecision(
                action=FinalAction.WATCH,
                color=FinalColor.YELLOW,
                state=FinalState.NO_CLEAR_EDGE,
                primary_reason=ReasonCode.TRIGGER_FAILED,
                reason_codes=[ReasonCode.TRIGGER_FAILED],
                summary_title="觀望",
                summary_text="目前沒有明確優勢，先觀察。",
                expert_action_line="行動：觀望。等待更明確的條件。",
                trace_lines=["[STRUCTURE] no bottom=ALLOW match → WATCH"],
            )
        )

    def _sort_key(item: tuple[ResolvedDecision, int]) -> tuple:
        d, order = item
        return (
            ACTION_SEVERITY[d.action],
            _COLOR_RANK[d.color],
            -order,
        )

    best = max(candidates, key=_sort_key)
    final_d, _ = best
    same_action = [t for t in candidates if t[0].action == final_d.action]
    final_d, _ = max(same_action, key=_sort_key)

    final = replace(final_d, reason_codes=list(final_d.reason_codes))

    gp = inp.gate_pass is True
    tp = inp.trigger_pass is True
    gup = inp.guard_pass is True
    final.can_buy_narrative = bool(gp and tp and gup)
    final.can_push_line_buy_signal = final.can_buy_narrative

    if not final.can_buy_narrative:
        if ReasonCode.NARRATIVE_BUY_BLOCKED not in final.reason_codes:
            final.reason_codes.append(ReasonCode.NARRATIVE_BUY_BLOCKED)
        if inp.gate_pass is False:
            if ReasonCode.GATE_FAILED not in final.reason_codes:
                final.reason_codes.append(ReasonCode.GATE_FAILED)
        elif inp.trigger_pass is False:
            if ReasonCode.TRIGGER_FAILED not in final.reason_codes:
                final.reason_codes.append(ReasonCode.TRIGGER_FAILED)
        elif inp.guard_pass is False:
            if ReasonCode.GUARD_FAILED not in final.reason_codes:
                final.reason_codes.append(ReasonCode.GUARD_FAILED)

    buy_inputs_explicit = (
        inp.gate_pass is not None
        or inp.trigger_pass is not None
        or inp.guard_pass is not None
    )
    if not final.can_buy_narrative and buy_inputs_explicit:
        final = replace(
            final,
            trace_lines=[
                *list(final.trace_lines),
                "[BUY] Gate/Trigger/Guard not all pass → buy blocked",
            ],
        )

    return final
