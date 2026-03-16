from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional
import math


AdviceLevel = Literal["info", "success", "warning", "error"]


@dataclass
class PositionAdvice:
    level: AdviceLevel
    headline: str
    bullets: list[str]


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
) -> PositionAdvice:
    """
    持倉管理（不寫死個人持股）：用「均價/股數 + TURN bottom/top 結果 + 防守均線」
    產出更接近人類交易的具體建議。

    回傳：PositionAdvice(level/headline/bullets)
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
            bullets.append("bottom=ALLOW：偏多方，可續抱；防守點建議放在 EMA5。")
        elif b_status == "WATCH":
            bullets.append("bottom=WATCH：先觀察，等待站穩與量能延續再加碼。")
        else:
            bullets.append("bottom=BLOCK：偏保守，避免追價。")

        if t_status in ["WATCH", "BLOCK"] or bias_hit:
            bullets.append("top 轉弱/過熱：風險升溫；若續漲可考慮分批落袋。")

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

    # --- 規則（可再逐步擴充） ---
    bullets: list[str] = []

    # 1) 位階高 + 已有利潤 + top 風險升溫：強烈減碼
    if p >= 15.0 and (t_status == "BLOCK" or t_score >= 3) and (bias_hit or (ema_bias is not None and ema_bias >= 7.0)):
        bullets.append("條件：Profit > 15% 且 top 風險升溫（含乖離過熱）。")
        bullets.append("建議：分批減碼 30–50%（先回收獲利，避免回吐）。")
        if ema is not None:
            bullets.append(f"防守點：跌破防守線（約 {ema:.2f}）可再減碼/出清。")
        level: AdviceLevel = "error" if t_status == "BLOCK" else "warning"
        headline = "高位階警示：建議分批減碼"

    # 2) top 分數不低 + 跌破 EMA5：建議減碼（獲利保護）
    elif below_defense is True and (t_score >= 2 or t_status in ["WATCH", "BLOCK"]) and p > 0:
        bullets.append("條件：跌破防守線 + top 轉弱（或分數偏高）。")
        bullets.append("建議：先減碼 1/3 或直接落袋（依你的波段/短線習慣）。")
        level = "warning"
        headline = "跌破防守：建議先減碼/落袋"

    # 3) 成本防禦（靠近成本 + bottom 不強）
    elif (-1.0 <= p <= 2.0) and b_score < 2:
        bullets.append("條件：現價接近成本（+2%~-1%）且 bottom 分數不足。")
        bullets.append("建議：以成本價做本金防守；破成本可考慮出場避免小虧變大虧。")
        level = "warning"
        headline = "成本防禦：本金保護優先"

    # 4) 虧損擴大（簡易停損提醒）
    elif p <= -5.0:
        bullets.append("條件：未實現損益 ≤ -5%。")
        bullets.append("建議：若 bottom 未回到 WATCH/ALLOW，請嚴格執行停損或降部位。")
        level = "error"
        headline = "停損提醒：虧損已擴大"

    # 5) 預設：續抱/觀察
    else:
        if b_status == "ALLOW" and t_status == "ALLOW" and not bias_hit:
            level = "success"
            headline = "趨勢穩定：可續抱"
            bullets.append("bottom=ALLOW 且 top 未亮風險：偏向續抱。")
        else:
            level = "info"
            headline = "續抱/觀察"
            bullets.append("訊號尚未出現強烈衝突：以防守線控風險、分批操作。")
        if t_status in ["WATCH", "BLOCK"] or bias_hit:
            bullets.append("top 風險升溫：若續漲可先落袋一部分，避免回吐。")

    # --- 加上數字摘要（可讀性更好） ---
    bullets.insert(0, f"未實現損益：約 {p:.1f}%")
    if pnl_amt is not None:
        bullets.insert(1, f"未實現損益金額：約 {pnl_amt:,.0f}")
    if ema is not None:
        bullets.append(f"防守線（EMA）參考：約 {ema:.2f}")

    return PositionAdvice(level=level, headline=headline, bullets=bullets)

