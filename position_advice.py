from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional
import math

from final_decision_resolver import (
    DecisionInput,
    FinalAction,
    ReasonCode,
    ResolvedDecision,
    resolve_final_decision,
)


AdviceLevel = Literal["info", "success", "warning", "error"]


@dataclass
class PositionAdvice:
    level: AdviceLevel
    headline: str
    bullets: list[str]
    resolution: Optional[ResolvedDecision] = None


def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    try:
        v = int(x)
        return v
    except Exception:
        return None


def profit_pct(current_price: Any, avg_cost: Any) -> Optional[float]:
    c = _to_float(current_price)
    a = _to_float(avg_cost)
    if c is None or a is None or a <= 0:
        return None
    return (c / a - 1.0) * 100.0


def bias_pct(current_price: Any, ma_price: Any) -> Optional[float]:
    c = _to_float(current_price)
    m = _to_float(ma_price)
    if c is None or m is None or m <= 0:
        return None
    return (c / m - 1.0) * 100.0


def get_position_advice(
    *,
    current_price: Any,
    avg_cost: Any,
    qty: Any = None,
    ema_defense: Any = None,
    bottom_result: Optional[dict[str, Any]] = None,
    top_result: Optional[dict[str, Any]] = None,
    ema5_short: Any = None,
    ema20_trend: Any = None,
    ai_score: Any = None,
    guard_ok: Optional[bool] = None,
    gate_pass: Optional[bool] = None,
    trigger_pass: Optional[bool] = None,
    guard_pass: Optional[bool] = None,
    position_mode: bool = True,
) -> PositionAdvice:
    """
    持倉管理（不寫死個人持股）：用「均價/股數 + TURN bottom/top 結果 + 防守均線」
    產出更接近人類交易的具體建議。

    裁決層統一走 resolve_final_decision(DecisionInput)。
    """
    c = _to_float(current_price)
    a = _to_float(avg_cost)
    q = _to_int(qty) if qty is not None else None
    ema = _to_float(ema_defense)

    b_status = str((bottom_result or {}).get("status", "NA"))
    b_score = _to_int((bottom_result or {}).get("score", 0)) or 0
    t_status = str((top_result or {}).get("status", "NA"))
    t_score = _to_int((top_result or {}).get("score", 0)) or 0
    t_conds = (top_result or {}).get("conditions") or {}
    bias_hit = bool(t_conds.get("bias"))  # top 過熱加分

    if c is None:
        return PositionAdvice(
            level="info",
            headline="資料不足：缺少現價",
            bullets=["無法計算持倉損益與防守點。"],
        )

    # 未輸入均價：仍可給一般策略（不涉及個人損益）
    if a is None or a <= 0:
        bullets = []
        if b_status == "ALLOW":
            bullets.append("進場燈為綠燈：偏多方，可續抱；防守點建議放在 EMA5。")
        elif b_status == "WATCH":
            bullets.append("進場燈為黃燈：先觀察，等待站穩與量能延續再加碼。")
        else:
            bullets.append("進場燈為紅燈：偏保守，避免追價。")

        if t_status in ["WATCH", "BLOCK"] or bias_hit:
            bullets.append("漲多轉弱：風險升溫；若續漲可考慮分批落袋。")

        return PositionAdvice(
            level="info",
            headline="可輸入持股均價（Avg Cost）讓建議更精準",
            bullets=bullets,
        )

    p = profit_pct(c, a)
    if p is None:
        return PositionAdvice(
            level="info",
            headline="無法計算損益（均價異常）",
            bullets=["請確認持股均價是否 > 0。"],
        )

    pnl_amt = None
    if q is not None:
        try:
            pnl_amt = (float(c) - float(a)) * float(q)
        except Exception:
            pnl_amt = None

    ema_bias = bias_pct(c, ema) if ema is not None else None
    below_defense = (ema is not None and c < ema) if ema is not None else None

    bullets: list[str] = []
    resolution: Optional[ResolvedDecision] = None
    level: AdviceLevel
    headline: str

    if p <= -5.0:
        resolution = resolve_final_decision(
            DecisionInput(
                close=c,
                pnl_pct=p,
                position_mode=position_mode,
                gate_pass=gate_pass,
                trigger_pass=trigger_pass,
                guard_pass=guard_pass if guard_pass is not None else guard_ok,
            )
        )
        bullets.extend(list(resolution.trace_lines))
        bullets.append("條件：未實現損益 ≤ -5%。")
        bullets.append("建議：若進場燈未回到黃燈／綠燈，請嚴格執行停損或降部位。")
        level = "error"
        headline = resolution.summary_title

    elif _to_float(ema5_short) is not None:
        e5 = _to_float(ema5_short)
        e20 = _to_float(ema20_trend)
        inp = DecisionInput(
            close=c,
            ema5=e5,
            ema20=e20,
            defensive_line=ema,
            weighted_ai_score=_to_float(ai_score),
            bottom_status=b_status if b_status != "NA" else None,
            top_status=t_status if t_status != "NA" else None,
            exec_guard_ok=guard_ok,
            gate_pass=gate_pass,
            trigger_pass=trigger_pass,
            guard_pass=guard_pass if guard_pass is not None else guard_ok,
            position_mode=position_mode,
        )
        resolution = resolve_final_decision(inp)
        act = resolution.action
        if act == FinalAction.EXIT:
            level = "error"
        elif act == FinalAction.REDUCE and resolution.primary_reason == ReasonCode.PRICE_BELOW_EMA5:
            level = "error"
        elif act in (FinalAction.REDUCE, FinalAction.WATCH):
            level = "warning"
        elif act == FinalAction.HOLD:
            level = "success"
        else:
            level = "info"
        headline = resolution.summary_title
        bullets.extend(list(resolution.trace_lines))
        if (
            resolution.primary_reason == ReasonCode.PRICE_BELOW_EMA5
            and ReasonCode.PRICE_ABOVE_EMA20 in resolution.reason_codes
        ):
            bullets.append(
                "情境：趨勢未死（價在 EMA20 上）但短線已失 EMA5—最忌『捨不得砍、慢慢被吃』。"
            )
        if p >= 15.0 and (t_status == "BLOCK" or t_score >= 3) and (
            bias_hit or (ema_bias is not None and ema_bias >= 7.0)
        ):
            bullets.append(
                "補充：已有不小獲利且高檔風險升溫，即使型態尚可也可先分批落袋。"
            )
        if below_defense is True and resolution.primary_reason != ReasonCode.PRICE_BELOW_EMA5:
            if ema is not None:
                bullets.append(
                    f"你選的防守線（{ema:.2f}）已跌破—請一併檢視是否同步減碼。"
                )
            else:
                bullets.append("已跌破自選防守線—請一併檢視是否同步減碼。")
        if t_status in ["WATCH", "BLOCK"] or bias_hit:
            bullets.append("高檔風險升溫：若續漲可先落袋一部分，避免回吐。")

    elif p >= 15.0 and (t_status == "BLOCK" or t_score >= 3) and (bias_hit or (ema_bias is not None and ema_bias >= 7.0)):
        bullets.append("條件：獲利逾 15% 且高檔風險升溫（含乖離過熱）。")
        bullets.append("建議：分批減碼 30–50%（先回收獲利，避免回吐）。")
        if ema is not None:
            bullets.append(f"防守點：跌破防守線（約 {ema:.2f}）可再減碼/出清。")
        level = "error" if t_status == "BLOCK" else "warning"
        headline = "高位階警示：建議分批減碼"

    elif below_defense is True and (t_score >= 2 or t_status in ["WATCH", "BLOCK"]) and p > 0:
        bullets.append("條件：跌破防守線且漲多轉弱（或分數偏高）。")
        bullets.append("建議：先減碼 1/3 或直接落袋（依你的波段/短線習慣）。")
        level = "warning"
        headline = "跌破防守：建議先減碼/落袋"

    elif (-1.0 <= p <= 2.0) and b_score < 2:
        bullets.append("條件：現價接近成本（+2%～-1%）且進場訊號偏弱。")
        bullets.append("建議：以成本價做本金防守；破成本可考慮出場避免小虧變大虧。")
        level = "warning"
        headline = "成本防禦：本金保護優先"

    else:
        if b_status == "ALLOW" and t_status == "ALLOW" and not bias_hit:
            level = "success"
            headline = "趨勢穩定：可續抱"
            bullets.append("進場為綠燈且出場風險燈未亮：偏向續抱。")
        else:
            level = "info"
            headline = "續抱/觀察"
            bullets.append("訊號尚未出現強烈衝突：以防守線控風險、分批操作。")
        if t_status in ["WATCH", "BLOCK"] or bias_hit:
            bullets.append("高檔風險升溫：若續漲可先落袋一部分，避免回吐。")

    bullets.insert(0, f"未實現損益：約 {p:.1f}%")
    if pnl_amt is not None:
        bullets.insert(1, f"未實現損益金額：約 {pnl_amt:,.0f}")
    if ema is not None:
        bullets.append(f"防守線（EMA）參考：約 {ema:.2f}")

    return PositionAdvice(
        level=level, headline=headline, bullets=bullets, resolution=resolution
    )


def humanize_turn_status_label(status: str) -> str:
    """TURN 燈號：推播／白話用（不輸出 ALLOW 等代碼）。"""
    u = str(status or "NA").strip().upper()
    return {
        "ALLOW": "綠燈（偏多）",
        "WATCH": "黃燈（留意）",
        "BLOCK": "紅燈（風險偏高）",
        "NA": "未評估",
    }.get(u, str(status or "NA"))


def build_exit_guide_push_text(
    *,
    close_last: float,
    avg_cost: float,
    exit_style: str,
    ema5: Optional[float],
    ema20: Optional[float],
    bottom_result: Optional[dict[str, Any]] = None,
    top_result: Optional[dict[str, Any]] = None,
    turn_result: Optional[dict[str, Any]] = None,
    advice: PositionAdvice,
    section_heading: str = "📍 下車指南",
) -> str:
    """
    與個股頁「下車指南（白話文決策）」同邏輯，產純文字供 LINE「一般版」使用。
    未輸入均價（LINE 一般使用者）：不提示填均價；一句說明後接與網頁相同的出場情境邏輯（不含損益％子彈）。
    """
    br, tr = bottom_result, top_result
    if isinstance(turn_result, dict) and br is None and tr is None:
        mode_tr = str(turn_result.get("mode", "top"))
        if mode_tr == "bottom":
            br = turn_result
        elif mode_tr == "top":
            tr = turn_result
        else:
            tr = turn_result

    if exit_style == "長線守月線":
        defense_name = "EMA20（月線）"
        defense = ema20
    else:
        defense_name = "EMA5（五日線）"
        defense = ema5

    out: list[str] = []
    out.append(section_heading)
    no_cost = float(avg_cost or 0.0) <= 0
    if no_cost:
        out.append(
            "以下依出場燈號與風險分數說明（一般版不含個人損益％與成本資料）。"
        )
    else:
        out.append(f"持倉診斷摘要：{advice.headline}")
        if advice.bullets:
            for b in advice.bullets[:4]:
                bt = (b or "").strip()
                if not bt or bt.startswith("["):
                    continue
                out.append(f"・{bt}")

    tr_d = tr or turn_result or {}
    score = int(tr_d.get("score", 0) or 0)
    status = str(tr_d.get("status", "NA"))
    mode = str(tr_d.get("mode", "top"))
    bias_hit = bool((tr_d.get("conditions") or {}).get("bias"))

    profit_ratio: Optional[float] = None
    try:
        ac = float(avg_cost)
        if ac > 0:
            profit_ratio = float(close_last) / ac - 1.0
    except Exception:
        profit_ratio = None

    bias_ema5: Optional[float] = None
    if ema5 is not None and float(ema5) > 0:
        try:
            bias_ema5 = float(close_last) / float(ema5) - 1.0
        except Exception:
            bias_ema5 = None

    if exit_style == "積極分批止盈":
        overheat_thr = 0.10
        partial_text = "先賣出 1/2（或至少 1/3）"
        score_trigger = 2
    elif exit_style == "長線守月線":
        overheat_thr = 0.12
        partial_text = "先賣出 1/3"
        score_trigger = 3
    else:
        overheat_thr = 0.10
        partial_text = "先賣出 1/3"
        score_trigger = 3

    overheat = bool(bias_hit) or (
        bias_ema5 is not None and float(bias_ema5) > float(overheat_thr)
    )

    status_zh = humanize_turn_status_label(status)

    if mode == "top":
        if (
            score >= int(score_trigger)
            and defense is not None
            and float(defense) > 0
            and float(close_last) < float(defense)
        ):
            out.append(
                f"🚨【全數或大幅獲利了結】出場風險分數 {score}（滿分 5）且已跌破 {defense_name}"
                f"（{float(defense):.2f}）。這通常是波段轉折點。"
            )
        elif overheat:
            bias_txt = (
                f"{(float(bias_ema5) * 100.0):.1f}%"
                if bias_ema5 is not None
                else "NA"
            )
            out.append(
                f"⚠️【強勢減碼】相對五日線乖離約 {bias_txt}，出場燈為 {status_zh}、風險分數 {score}。"
                f"建議{partial_text}，剩餘持股守 {defense_name}。"
            )
        elif (
            exit_style == "積極分批止盈"
            and status in ["WATCH", "BLOCK"]
            and profit_ratio is not None
            and float(profit_ratio) > 0
            and defense is not None
            and float(defense) > 0
            and float(close_last) >= float(defense)
        ):
            pr_txt = f"{(float(profit_ratio) * 100.0):.1f}%"
            out.append(
                f"🟡【二階段下車：先減碼】出場燈 {status_zh}（風險分數 {score}），"
                f"仍守住 {defense_name}（{float(defense):.2f}）。目前獲利 {pr_txt}；"
                f"建議先賣一半，剩餘跌破 {defense_name} 再全出。"
            )
        elif score >= 2:
            pr_txt = (
                f"{(float(profit_ratio) * 100.0):.1f}%"
                if profit_ratio is not None
                else "NA"
            )
            out.append(
                f"🟡【高位監控】出場燈轉為 {status_zh}（風險分數 {score}）。目前獲利 {pr_txt}。"
                f"建議把「收盤跌破 {defense_name}」當作出場參考點。"
            )
        elif (
            advice.resolution is not None
            and advice.resolution.action
            in (FinalAction.REDUCE, FinalAction.WATCH, FinalAction.EXIT)
        ):
            out.append(
                f"⚠️【與主結論一致】{advice.resolution.summary_title}—此處沿用主結論的風控建議。"
            )
        else:
            out.append("💎【獲利奔跑中】尚未觸發減碼／下車訊號，可續抱。")
    else:
        out.append(
            "此下車句需「漲多高檔」出場分析；請至網頁開啟相關圖表後再推播，"
            "或改用進階版 LINE。"
        )

    return "\n".join(out)
