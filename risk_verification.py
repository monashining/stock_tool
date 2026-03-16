"""
風險驗證：乖離率檢查 + 動態止損調整

當本益比（P/E）遠高於合理區間，代表情緒面大於基本面，
建議將移動止損從寬鬆改為嚴格，防止短線注意股冷卻後的大幅回撤。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DynamicStopAdvice:
    """動態止損建議"""
    suggested_stop_pct: float      # 建議止損 %
    reason: str
    pe_ttm: Optional[float]
    pe_vs_threshold: str           # 如 "110x vs 合理 30x"
    bias_20_pct: Optional[float]
    is_high_valuation_risk: bool


# 預設：P/E > 此值視為「情緒面主導」，建議收緊止損
PE_OVERHEAT_THRESHOLD = 50.0
PE_EXTREME_THRESHOLD = 80.0
# 乖離率過熱門檻（與 Gate 一致）
BIAS_OVERHEAT_THRESHOLD = 10.0


def get_dynamic_stop_advice(
    pe_ttm: Optional[float],
    bias_20_pct: Optional[float],
    current_stop_pct: float = 10.0,
    *,
    pe_overheat: float = PE_OVERHEAT_THRESHOLD,
    pe_extreme: float = PE_EXTREME_THRESHOLD,
) -> DynamicStopAdvice:
    """
    依本益比與乖離率，建議動態止損幅度。

    邏輯：
    - P/E > 80：極度情緒化，建議 3–4% 止損
    - P/E > 50：情緒面主導，建議 5% 止損
    - P/E 合理 + 乖離過熱：建議 6%
    - 其餘：維持使用者設定
    """
    pe_txt = f"{pe_ttm:.0f}x" if pe_ttm is not None and np.isfinite(pe_ttm) else "N/A"
    bias_txt = f"{bias_20_pct:.1f}%" if bias_20_pct is not None and np.isfinite(bias_20_pct) else "N/A"

    if pe_ttm is not None and np.isfinite(pe_ttm) and pe_ttm >= pe_extreme:
        suggested = 4.0
        reason = (
            f"本益比 {pe_txt} 極高（情緒面主導），建議止損收緊至 {suggested:.0f}%，"
            "防止短線冷卻後大幅回撤。"
        )
        return DynamicStopAdvice(
            suggested_stop_pct=suggested,
            reason=reason,
            pe_ttm=pe_ttm,
            pe_vs_threshold=f"{pe_txt} vs 合理約 20–30x",
            bias_20_pct=bias_20_pct,
            is_high_valuation_risk=True,
        )
    if pe_ttm is not None and np.isfinite(pe_ttm) and pe_ttm >= pe_overheat:
        suggested = 5.0
        reason = (
            f"本益比 {pe_txt} 高於同業，情緒面大於基本面。"
            f"建議止損從 {current_stop_pct:.0f}% 收緊至 {suggested:.0f}%。"
        )
        return DynamicStopAdvice(
            suggested_stop_pct=suggested,
            reason=reason,
            pe_ttm=pe_ttm,
            pe_vs_threshold=f"{pe_txt} vs 合理約 20–30x",
            bias_20_pct=bias_20_pct,
            is_high_valuation_risk=True,
        )
    if bias_20_pct is not None and np.isfinite(bias_20_pct) and abs(bias_20_pct) > BIAS_OVERHEAT_THRESHOLD:
        suggested = min(6.0, current_stop_pct)
        reason = (
            f"乖離率 {bias_txt} 過熱，短線回檔風險升高。"
            f"建議止損收緊至 {suggested:.0f}% 以內。"
        )
        return DynamicStopAdvice(
            suggested_stop_pct=suggested,
            reason=reason,
            pe_ttm=pe_ttm,
            pe_vs_threshold=pe_txt,
            bias_20_pct=bias_20_pct,
            is_high_valuation_risk=False,
        )
    return DynamicStopAdvice(
        suggested_stop_pct=current_stop_pct,
        reason="估值與乖離尚在合理區間，可維持目前止損設定。",
        pe_ttm=pe_ttm,
        pe_vs_threshold=pe_txt,
        bias_20_pct=bias_20_pct,
        is_high_valuation_risk=False,
    )


def risk_verification_from_data(
    df: pd.DataFrame,
    fundamental: Optional[dict],
    symbol: Optional[str] = None,
    current_stop_pct: float = 10.0,
) -> DynamicStopAdvice:
    """
    從 DataFrame 與基本面資料取得風險驗證建議。
    fundamental 來自 fetch_fundamental_snapshot，需含 pe_ttm。
    """
    pe_ttm = None
    if fundamental and isinstance(fundamental, dict):
        pe_ttm = fundamental.get("pe_ttm")
        if pe_ttm is not None:
            try:
                pe_ttm = float(pe_ttm)
            except (TypeError, ValueError):
                pe_ttm = None

    bias_20_pct = None
    if df is not None and not df.empty:
        if "Bias20" in df.columns:
            try:
                bias_20_pct = float(df["Bias20"].iloc[-1])
            except (TypeError, ValueError, IndexError):
                pass
        elif "bias_sma20_pct" in df.columns:
            try:
                bias_20_pct = float(df["bias_sma20_pct"].iloc[-1])
            except (TypeError, ValueError, IndexError):
                pass

    return get_dynamic_stop_advice(
        pe_ttm=pe_ttm,
        bias_20_pct=bias_20_pct,
        current_stop_pct=current_stop_pct,
    )
