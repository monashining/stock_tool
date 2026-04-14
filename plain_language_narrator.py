"""
白話翻譯層：不修改裁決，只把 ResolvedDecision + 現況輸入翻成可讀敘述。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from final_decision_resolver import (
    FinalAction,
    FinalState,
    ReasonCode,
    ResolvedDecision,
)

TERM_EXPLAINER = {
    "EMA5": "最近 5 天的平均成本，常用來看短線有沒有轉弱",
    "EMA20": "最近約 1 個月的平均成本，常用來看中期趨勢有沒有壞掉",
    "Trigger": "真正可以出手的訊號，代表不是只看起來不錯，而是條件真的成立",
    "Guard": "風險保護條件，用來避免買在危險位置",
    "PULLBACK": "拉回後撐住，代表有人在低檔接",
    "BREAKOUT": "突破前面壓力，代表股價重新轉強",
    "CONTINUATION": "原本的上漲趨勢延續，不是反彈而已",
    "bottom ALLOW": "值得繼續盯，但不代表可以直接買",
    "top WATCH": "上方開始有風險，不能太樂觀",
    "top BLOCK": "上方壓力與風險偏高，不宜再用最樂觀假設",
}


@dataclass
class PlainLanguageNarratorInput:
    close: float
    ema5: Optional[float]
    ema20: Optional[float]
    gate_ok: bool
    trigger_ok: bool
    guard_ok: bool
    trigger_type: str
    bottom_status: Optional[str]
    top_status: Optional[str]
    has_position: bool
    """是否視為持倉語境（均價 > 0）。"""
    risk_alert: bool = False
    # 高檔危險量、籌碼背離等加總旗標。
    near_support_red_bar: Optional[bool] = None
    # 可選：接近支撐後收紅（由外層計算後傳入）。


@dataclass
class PlainLanguageSummary:
    current_state: str
    why_not_buy: str
    action_now: str
    what_to_wait_for: str
    # one_line：持股總表、LINE 標題、卡片／手機用一句話
    one_line_verdict: str = ""
    term_notes: List[str] = field(default_factory=list)


@dataclass
class DipBuyVerdict:
    allowed: bool
    explanation: str


def _norm_trigger_type(trigger_type: str) -> str:
    return str(trigger_type or "NONE").strip().upper() or "NONE"


def can_buy_the_dip(
    *,
    close: float,
    ema5: Optional[float],
    trigger_type: str,
    guard_ok: bool,
    risk_alert: bool = False,
    near_support_red_bar: Optional[bool] = None,
) -> DipBuyVerdict:
    """
    只回答「這波下跌比較像可控回踩，還是還在跌勢途中」的撿便宜判斷（白話一句）。

    擋掉條件（使用者規格）：close < EMA5、Trigger = NONE、Guard FAIL；
    另加上 risk_alert 一併擋下。
    允許條件：未擋且（站回／在 EMA5 上、或有效 Trigger 型態、或支撐收紅）。
    """
    tt = _norm_trigger_type(trigger_type)
    cannot = False
    if ema5 is not None and close < ema5:
        cannot = True
    if tt == "NONE":
        cannot = True
    if not guard_ok:
        cannot = True
    if risk_alert:
        cannot = True

    if cannot:
        return DipBuyVerdict(
            False,
            "現在不能算是撿便宜，因為股價還沒有止跌或訊號未齊，風險也還沒被控制住。",
        )

    on_or_above_ema5 = ema5 is not None and close >= ema5
    typed_trigger = tt in ("PULLBACK", "BREAKOUT", "CONTINUATION")
    bounce = near_support_red_bar is True

    if on_or_above_ema5 or typed_trigger or bounce:
        return DipBuyVerdict(
            True,
            "可以開始留意低接，因為價格已回到短線均線之上或已出現有效 Trigger，且 Guard 通過、未亮明顯風險警報。",
        )

    return DipBuyVerdict(
        False,
        "條件尚未齊備：建議等收盤站回 EMA5 或出現回踩／突破類 Trigger，再重新評估。",
    )


def _pick_term_notes(inp: PlainLanguageNarratorInput, codes: List[ReasonCode]) -> List[str]:
    keys: List[str] = []
    if inp.ema5 is not None:
        keys.append("EMA5")
    if inp.ema20 is not None:
        keys.append("EMA20")
    keys.extend(["Trigger", "Guard"])
    if _norm_trigger_type(inp.trigger_type) in TERM_EXPLAINER:
        keys.append(_norm_trigger_type(inp.trigger_type))
    bs = str(inp.bottom_status or "").upper()
    ts = str(inp.top_status or "").upper()
    if bs == "ALLOW":
        keys.append("bottom ALLOW")
    if ts == "WATCH":
        keys.append("top WATCH")
    if ts == "BLOCK":
        keys.append("top BLOCK")

    # reason-based hints
    if ReasonCode.PRICE_BELOW_EMA5 in codes or ReasonCode.PRICE_BELOW_DEFENSIVE_LINE in codes:
        if "EMA5" not in keys and inp.ema5 is not None:
            keys.insert(0, "EMA5")
    out: List[str] = []
    seen = set()
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        if k in TERM_EXPLAINER:
            out.append(f"{k}：{TERM_EXPLAINER[k]}")
    return out[:8]


def build_plain_language_summary(
    decision: Optional[ResolvedDecision],
    inp: PlainLanguageNarratorInput,
) -> PlainLanguageSummary:
    """依裁決 + 現況組出固定四行白話（不覆寫 resolver）。"""
    if decision is None:
        return PlainLanguageSummary(
            current_state="目前資料不足以描述完整狀態。",
            why_not_buy="關鍵價量或均線資料不足，無法判斷是否適合承接。",
            action_now="先補齊資料或縮小解讀範圍，避免硬做決定。",
            what_to_wait_for="等圖表與訊號欄位正常顯示後再評估。",
            one_line_verdict="資料不足，暫不判讀",
            term_notes=[
                f"{k}：{v}" for k, v in list(TERM_EXPLAINER.items())[:4]
            ],
        )

    codes = list(decision.reason_codes)
    act = decision.action
    stt = decision.state
    has = inp.has_position
    below5 = inp.ema5 is not None and inp.close < inp.ema5
    above20 = inp.ema20 is not None and inp.close > inp.ema20
    tt = _norm_trigger_type(inp.trigger_type)

    # --- 預設模板 ---
    current_state = "目前訊號略為分歧，屬於要保守解讀的階段。"
    why_not_buy = "還沒有同時滿足「趨勢、價格位置與風控」的出手條件。"
    action_now = (
        "有持股以防守為主，沒持股寧可多看少做。"
        if has
        else "沒持股先維持觀察，不急著進場。"
    )
    what_to_wait_for = "等短線重新站穩、Trigger 與 Guard 同步好轉，再重新評估。"
    verdict = "訊號分歧，先保守看待"

    # 3037 類：破短守中
    if (
        ReasonCode.PRICE_BELOW_EMA5 in codes
        and ReasonCode.PRICE_ABOVE_EMA20 in codes
        and below5
        and above20
    ):
        current_state = "這檔股票中期趨勢還在，但短線已經轉弱、正在拉回。"
        why_not_buy = (
            "現在不是撿便宜時機，因為短線尚未止跌，貿然承接容易買在還沒跌完的段落。"
        )
        action_now = (
            "有持股先減碼防守；沒有持股先不要急著撿。"
            if has
            else "沒持股先觀望，不要因為中期還在多頭結構就硬接短線跌勢。"
        )
        what_to_wait_for = (
            "等收盤重新站回 EMA5，或出現回踩成功／突破類 Trigger 且 Guard 通過，再考慮進場。"
        )
        verdict = "趨勢未壞，但短線先防守"

    elif act == FinalAction.EXIT or stt == FinalState.HARD_STOP:
        current_state = "目前屬於資金保護優先的階段，訊號偏空或已觸及嚴格出場條件。"
        why_not_buy = "這個位置不適合承接或攤平，應先處理風險。"
        action_now = "依計畫執行減碼或出場，避免擴大虧損。"
        what_to_wait_for = "等結構重新轉強、風險釋放後，再另找新一波機會。"
        verdict = "優先出場，不宜承接"

    elif stt == FinalState.EXEC_GUARD_FAIL or decision.primary_reason == ReasonCode.EXEC_GUARD_FAILED:
        current_state = "價格看起來有機會，但執行保護（Guard）認為這裡還不夠安全。"
        why_not_buy = "Guard 未通過，代表追價或假突破風險仍在，不適合當成舒服買點。"
        action_now = (
            "有持股降低積極度、偏防守；沒持股避免追高。"
            if has
            else "沒持股避免在此追高，寧可等型態站穩。"
        )
        what_to_wait_for = "等收盤強度、量能與 Guard 條件改善，再考慮出手。"
        verdict = "Guard 未過，不宜追高"

    elif stt == FinalState.LOW_AI_SCORE or decision.primary_reason == ReasonCode.AI_SCORE_LT_70:
        current_state = "整體分數偏弱，屬於「不適合積極續抱」的防守區。"
        why_not_buy = "主分偏低時，系統不建議把這裡當加碼或抄底理由。"
        action_now = (
            "有持股縮減積極度或分批降風險；沒持股維持觀察。"
            if has
            else "沒持股維持觀察，不急著進場。"
        )
        what_to_wait_for = "等分數回到較健康區間、價格與結構同步改善。"
        verdict = "分數偏弱，降低積極度"

    elif act == FinalAction.HOLD and stt == FinalState.HEALTHY_TREND:
        current_state = "底部結構仍偏正面，趨勢尚未被完全破壞。"
        why_not_buy = (
            "若 Trigger 尚未點火或 Guard 未通過，仍不會把敘述升級成「可大買」。"
            if not inp.trigger_ok or not inp.guard_ok
            else "仍建議用分批與停損守規則，避免一次押滿。"
        )
        action_now = (
            "有持股可依策略續抱並守好大級防守；沒持股等 Trigger 與 Guard 再對一次。"
            if has
            else "沒持股可持續追蹤，但進場仍應等 Trigger／Guard 條件成立。"
        )
        what_to_wait_for = "留意是否出現有效 Trigger、量能是否健康，以及 top 是否轉為 WATCH／BLOCK。"
        verdict = "趨勢未壞，可續抱"

    elif act == FinalAction.WATCH or stt == FinalState.NO_CLEAR_EDGE:
        current_state = "多空訊號拉鋸，屬於觀察優先、不宜過度解讀的階段。"
        why_not_buy = "尚未出現一致的多頭出手組合（趨勢＋點火＋風控）。"
        action_now = (
            "有持股控制部位與槓桿；沒持股維持觀望。"
            if has
            else "沒持股維持觀望，不勉強進場。"
        )
        what_to_wait_for = "等 Trigger 明朗、Guard 通過，或價格重新站回關鍵均線。"
        verdict = "訊號拉鋸，多看少做"

    elif act == FinalAction.REDUCE and below5 and not above20:
        current_state = "短線與中期結構同步偏空，拉回壓力較大。"
        why_not_buy = "這裡承接勝率與風險報酬都不利，不適合當撿便宜起點。"
        action_now = (
            "有持股優先減碼或嚴守停損；沒持股完全不必急。"
            if has
            else "沒持股不必急著承接，寧可等止跌訊號。"
        )
        what_to_wait_for = "等重新站回 EMA5／EMA20 並搭配 Trigger 轉強，再評估。"
        verdict = "短線轉弱，先防守"

    elif act == FinalAction.REDUCE:
        current_state = "整體偏向降低風險與積極度，市場給的容錯變小。"
        why_not_buy = "訊號不支援在此加碼或攤平，應先處理部位與防守。"
        action_now = (
            "有持股以減碼或調降槓桿為主；沒持股避免逆勢硬接。"
            if has
            else "沒持股避免逆勢硬接，先等風險釋放。"
        )
        what_to_wait_for = "等價格站回短線均線、Guard 好轉，並確認 bottom 結構仍允許。"
        verdict = "風險升溫，先防守"

    # Trigger 白話補述
    if tt == "NONE" and inp.trigger_ok is False:
        why_not_buy = (
            "目前還沒有看到止跌或重新轉強的 Trigger（類型為 NONE），"
            + why_not_buy
        )

    is_3037 = (
        ReasonCode.PRICE_BELOW_EMA5 in codes
        and ReasonCode.PRICE_ABOVE_EMA20 in codes
        and below5
        and above20
    )
    dipv = can_buy_the_dip(
        close=inp.close,
        ema5=inp.ema5,
        trigger_type=inp.trigger_type,
        guard_ok=inp.guard_ok,
        risk_alert=inp.risk_alert,
        near_support_red_bar=inp.near_support_red_bar,
    )
    if not is_3037 and not (act == FinalAction.EXIT or stt == FinalState.HARD_STOP):
        if dipv.allowed and act == FinalAction.HOLD and stt == FinalState.HEALTHY_TREND:
            verdict = "可開始留意低接，但不要追價"
        elif (
            not dipv.allowed
            and below5
            and act in (FinalAction.REDUCE, FinalAction.WATCH)
        ):
            verdict = "還沒止跌，現在不是撿便宜"

    term_notes = _pick_term_notes(inp, codes)
    return PlainLanguageSummary(
        current_state=current_state,
        why_not_buy=why_not_buy,
        action_now=action_now,
        what_to_wait_for=what_to_wait_for,
        one_line_verdict=verdict,
        term_notes=term_notes,
    )
