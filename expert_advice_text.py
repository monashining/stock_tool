"""
與個股頁「咸魚翻身｜AI 專家診斷」同源：供 app / LINE webhook 共用，避免 import app（Streamlit）。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from analysis import check_volume_risk
from final_decision_resolver import (
    DecisionInput,
    ReasonCode,
    expert_action_line_markdown,
    resolve_final_decision,
)
from utils import to_scalar


def _score_display(score, scoring_version: Optional[str]) -> str:
    if scoring_version:
        return f"{score}（模型 {scoring_version}）"
    return str(score)


def generate_expert_advice(
    df,
    ticker_name,
    score,
    risk_metrics,
    is_chip_divergence=False,
    *,
    weighted_ai_score=None,
    bottom_status="NA",
    scoring_version: Optional[str] = None,
):
    if df is None or df.empty or len(df) < 2:
        return "資料不足，無法產生專家講評。"

    latest = df.iloc[-1]

    upper_shadow = latest["High"] - max(latest["Open"], latest["Close"])
    lower_shadow = min(latest["Open"], latest["Close"]) - latest["Low"]
    body = abs(latest["Close"] - latest["Open"])

    analysis = []
    display_name = ticker_name or "此標的"
    sd = _score_display(score, scoring_version)

    is_dangerous_vol = bool(latest.get("Is_Dangerous_Volume"))
    is_pulling_out = is_dangerous_vol and is_chip_divergence

    bias20 = None
    try:
        b = latest.get("Bias20")
        bias20 = float(b) if (b is not None and not pd.isna(b)) else None
    except Exception:
        bias20 = None
    is_overheat = bias20 is not None and float(bias20) > 15.0
    chip_distribution = bool(is_chip_divergence and is_overheat)

    if chip_distribution:
        analysis.append(
            f"**高檔派發中**：{display_name} 評分 {sd}，但出現「乖離過熱（Bias20>15%）+ 籌碼背離」。"
            "這種狀態常見於『外資趁強撤退、投信苦撐』，短線容易急殺洗盤。"
        )
    elif score >= 80:
        analysis.append(f"**強力多頭配置**：{display_name} 評分高達 {sd}，動能強勁。")
    elif score >= 60:
        if is_pulling_out:
            analysis.append(
                f"**高度警戒**：{display_name} 評分 {sd}，但偵測到「拉高出貨」徵兆（爆量黑K + 籌碼背離）。"
            )
        else:
            analysis.append(
                f"**趨勢穩定**：{display_name} 評分 {sd}，屬於標準多方形態。"
            )
    else:
        analysis.append(
            f"**觀望保守**：{display_name} 評分 {sd}，建議等待更明確的點火訊號。"
        )

    if body > 0 and lower_shadow > body * 1.5:
        analysis.append("**下方支撐強勁**：長下影線顯示回測後有買盤承接。")
    if body > 0 and upper_shadow > body * 1.5 and latest.get("BUY_TRIGGER_TYPE") == "BREAKOUT":
        analysis.append("**假突破風險**：創高但上影線過長，須防範套牢壓力。")
    if latest.get("Is_Dangerous_Volume"):
        analysis.append(
            "**注意：高檔放量收黑**。雖然有人接手，但目前空方力道大於多方，建議觀察 3 天。"
        )
    volume_risk = check_volume_risk(df)
    if volume_risk:
        analysis.append(volume_risk)

    if (
        risk_metrics
        and risk_metrics.get("beta", 1) is not None
        and risk_metrics.get("beta", 1) > 1.5
    ):
        analysis.append(
            f"**波動警告**：Beta 偏高 ({risk_metrics['beta']:.2f})，操作需嚴守停損。"
        )

    _close = to_scalar(latest["Close"])
    _ema5 = to_scalar(latest["EMA5"]) if "EMA5" in df.columns else None
    _ema20 = to_scalar(latest["EMA20"]) if "EMA20" in df.columns else None
    _ai = float(weighted_ai_score) if weighted_ai_score is not None else float(score)
    _guard = bool(latest.get("EXEC_GUARD", True))
    _c = float(_close) if _close is not None and not pd.isna(_close) else 0.0
    rd = resolve_final_decision(
        DecisionInput(
            close=_c,
            ema5=float(_ema5) if _ema5 is not None and not pd.isna(_ema5) else None,
            ema20=float(_ema20) if _ema20 is not None and not pd.isna(_ema20) else None,
            weighted_ai_score=_ai,
            bottom_status=str(bottom_status or "NA"),
            exec_guard_ok=_guard,
            gate_pass=bool(latest.get("BUY_GATE", False)),
            trigger_pass=bool(latest.get("BUY_TRIGGER", False)),
            guard_pass=bool(latest.get("EXEC_GUARD", True)),
            position_mode=False,
        )
    )

    if rd.primary_reason == ReasonCode.PRICE_BELOW_EMA5:
        advice = expert_action_line_markdown(rd)
    elif chip_distribution:
        advice = (
            "**行動：減碼觀望**。乖離過熱且籌碼背離，建議先落袋一部分（或至少不再加碼），"
            "以 EMA5/EMA20 作為防守線，收盤守住再談續抱。"
        )
    elif rd.primary_reason == ReasonCode.EXEC_GUARD_FAILED:
        advice = expert_action_line_markdown(rd)
    elif is_pulling_out:
        advice = "**行動：果斷減碼**。偵測到籌碼與量價同步轉惡，建議減碼至 2 成以下或清倉。"
    elif is_dangerous_vol:
        advice = "**行動：減碼觀望**。高檔爆量收黑，守住今日低點再考慮續抱。"
    elif latest.get("BUY_SIGNAL"):
        advice = (
            f"**行動：符合進場標準 ({latest.get('BUY_TRIGGER_TYPE')})**。"
            "建議在平盤附近分批佈局。"
        )
    elif latest.get("Close") < latest.get("SMA20"):
        advice = "**行動：減碼觀望**。跌破 SMA20，在站回前不宜貿然接刀。"
    elif bias20 is not None and abs(float(bias20)) > 10:
        advice = (
            "**行動：持倉可續抱，建倉暫觀望**。結構未破壞但乖離過熱（Bias20≈{:.1f}%），"
            "追高風險大；建倉請等乖離回落。".format(float(bias20))
        )
    else:
        advice = expert_action_line_markdown(rd)

    return "\n\n".join(analysis) + "\n\n---\n" + advice
