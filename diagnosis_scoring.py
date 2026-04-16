"""
單一診斷評分入口：加權分數 + 情境加減分 → 總分（0–100）。
Web（app.py）與 LINE（diagnosis_snapshot）必須只透過此模組計算，避免邏輯分叉。
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from analysis import compute_volume_sum_3d, compute_weighted_score
from data_sources import (
    _is_taiwan_equity_symbol,
    get_weekly_trend_with_meta,
    load_market_index_with_meta,
)
from utils import to_scalar

# 變更加權／情境規則時請 bump，並在 UI／LINE 顯示以利對照
SCORING_VERSION = "1.1"

# 情境分內嵌門檻（單一來源）：改數值時 signature 的 SCORING_CONFIG_HASH 會跟著變，免忘記 bump version。
# 加權分委託 analysis.compute_weighted_score；其原始碼指紋見 WEIGHTED_SCORE_SRC_HASH（inspect 自動計算）。
DIAGNOSIS_CONTEXT_PARAMS: dict[str, int | float | str] = {
    "sell_high_vs_ema20_mult": 1.05,
    "sell_huge_vol_vs_avg5_mult": 2.0,
    "rsi_context_ge": 50,
    "vol_breakout_ratio_min": 1.2,
    "vol_breakout_ratio_max": 1.8,
    "relative_strength_lookback_bars": 21,
    "atr_pct_of_close_min": 0.01,
    "atr_pct_of_close_max": 0.05,
    "higher_low_window": 10,
    "min_bars_higher_low": 20,
    "consecutive_below_ema_days": 3,
    "prior_high_rolling_days": 20,
    "upper_wick_ratio_gt": 0.5,
    "gap_up_open_vs_prior_high_mult": 1.01,
    "bias20_overheated_gt": 10,
    "break_ema20_vol_ratio_gt": 1.3,
    "foreign_3d_net_sum_le": -5000,
    "min_rows_sma60_compare": 6,
    "sma60_prev_iloc_offset": 6,
    "min_bars_vol_shrink": 3,
    "min_bars_two_bar_pattern": 2,
    "min_bars_ema20_slope": 2,
}


def _weighted_score_source_hash() -> str:
    """
    加權函式指紋：優先原始碼，其次 bytecode+常數，最後 qualname（frozen/zipimport 仍可用）。
    """
    fn = compute_weighted_score
    try:
        src = inspect.getsource(fn)
        return hashlib.md5(src.encode("utf-8")).hexdigest()[:8]
    except Exception:
        pass
    try:
        code = fn.__code__
        parts: list[bytes] = [code.co_code]
        for c in code.co_consts:
            if isinstance(c, (str, int, float, bool, type(None))):
                parts.append(bytes(repr(c), "utf-8", errors="replace"))
        blob = b"\x1e".join(parts)
        return hashlib.md5(blob).hexdigest()[:8]
    except Exception:
        pass
    fb = f"fallback:{fn.__module__}.{fn.__qualname__}"
    return hashlib.md5(fb.encode("utf-8")).hexdigest()[:8]


WEIGHTED_SCORE_SRC_HASH = _weighted_score_source_hash()


def _df_index_tz_tag(df: Optional[pd.DataFrame]) -> str:
    if df is None or len(df) < 1:
        return "naive"
    tz = getattr(df.index, "tz", None)
    if tz is None:
        return "naive"
    return str(tz)


def _market_calendar_tag(market_symbol: str) -> str:
    return "TWSE" if _is_taiwan_equity_symbol(str(market_symbol)) else "US"

SCORING_CONFIG_HASH = hashlib.md5(
    (
        json.dumps(DIAGNOSIS_CONTEXT_PARAMS, sort_keys=True, separators=(",", ":"))
        + "|w="
        + WEIGHTED_SCORE_SRC_HASH
    ).encode("utf-8")
).hexdigest()[:16]


@dataclass(frozen=True)
class DiagnosisScoreBundle:
    weighted_score: float
    weighted_reasons: list[str]
    weighted_flags: Any
    context_score: int
    context_reasons: list[str]
    total_score: int
    combined_reasons: list[str]
    # 以下與 app 主控台「指標達成」面板同步（與計分同一套條件）
    weekly_status: str
    ema20_up: Optional[bool]
    trend_mid_ok: bool
    sma60_up: bool
    vol_breakout_ok: bool
    vol_shrink_ok: bool
    vol_ratio_20: Optional[float]
    prior_high: Optional[float]
    higher_low_ok: bool
    rs_ok: bool
    atr_pct_ok: bool
    below_ema20: bool
    upper_wick_bad: bool
    gap_risk: bool
    market_bear: bool
    recent_foreign_sum: Optional[float]


def compute_diagnosis_score_bundle(
    df: pd.DataFrame,
    ema_df: pd.DataFrame,
    *,
    latest: pd.Series,
    ema20: Optional[float],
    ema5: Optional[float],
    close_price: Optional[float],
    current_vol: Optional[float],
    avg_vol_5: Optional[float],
    bias_20_val: Optional[float],
    vol_ma20_val: Optional[float],
    sma20_val: Optional[float],
    sma60_val: Optional[float],
    atr14_val: Optional[float],
    latest_volume: float,
    latest_close: float,
    foreign_net_series: Optional[pd.Series],
    trust_net_series: Optional[pd.Series],
    foreign_net_latest: Optional[float],
    market_symbol: str,
    foreign_divergence_warning: Optional[bool] = None,
    daily_cache_layer: Optional[str] = None,
    period_tag: Optional[str] = None,
) -> DiagnosisScoreBundle:
    """
    market_symbol：用於大盤指數與週線趨勢（請傳與行情 df 一致的解析後代號，例如 2330.TW）。
    foreign_divergence_warning：True＝外資連賣且價漲（扣情境分）；False＝已評估無背離；None＝未評估（不套用背離規則，例如關閉籌碼時）。
    daily_cache_layer：個股日線 OHLCV 命中層（mem/disk/net/batch），供 DEBUG；未傳則視為 unknown。
    period_tag：資料區間與頻率標識（例如 1y_1d、6mo_1d），供 signature；未傳則 unknown。
    """
    P = DIAGNOSIS_CONTEXT_PARAMS
    trigger_type = str(latest.get("BUY_TRIGGER_TYPE", "NONE"))

    is_high_volume_sell = False
    if ema20 is not None and ema5 is not None and not pd.isna(avg_vol_5):
        is_high_price = close_price is not None and close_price > ema20 * float(
            P["sell_high_vs_ema20_mult"]
        )
        is_huge_vol = current_vol is not None and current_vol > avg_vol_5 * float(
            P["sell_huge_vol_vs_avg5_mult"]
        )
        is_selling_pressure = close_price is not None and close_price < to_scalar(
            df["Open"].iloc[-1]
        )
        is_high_volume_sell = bool(
            is_high_price and is_huge_vol and is_selling_pressure
        )

    weighted_score, weighted_reasons, weighted_flags = compute_weighted_score(
        ema20=ema20,
        ema5=ema5,
        close_price=close_price,
        current_vol=current_vol,
        avg_vol_5=avg_vol_5,
        bias_20_val=bias_20_val,
        foreign_net_series=foreign_net_series,
        trust_net_series=trust_net_series,
        vol_sum_3d=compute_volume_sum_3d(df),
        is_dangerous_vol=bool(latest.get("Is_Dangerous_Volume", False)),
    )

    context_score = 0
    context_reasons: list[str] = []
    ema20_up = None
    if ema20 is not None and len(ema_df) >= int(P["min_bars_ema20_slope"]):
        ema20_up = to_scalar(ema_df["EMA20"].iloc[-1]) > to_scalar(
            ema_df["EMA20"].iloc[-2]
        )
    if ema20 is None or ema20_up is None:
        context_reasons.append("EMA20 資料不足，無法判斷趨勢")
    elif close_price is not None and close_price > ema20 and ema20_up:
        context_score += 1
        context_reasons.append("股價在 EMA20 之上且 EMA20 走升，趨勢偏多")
    else:
        context_score -= 1
        context_reasons.append("股價跌破 EMA20 或 EMA20 走平/走弱")

    if ema5 is None or ema20 is None:
        context_reasons.append("EMA5/EMA20 資料不足，無法判斷交叉")
    elif ema5 > ema20:
        context_score += 1
        context_reasons.append("短期動能強於中期 (EMA5 > EMA20)")
    else:
        context_score -= 1
        context_reasons.append("短期動能轉弱 (EMA5 <= EMA20)")

    if not pd.isna(df["RSI14"].iloc[-1]):
        rsi_latest = to_scalar(df["RSI14"].iloc[-1])
        if rsi_latest >= float(P["rsi_context_ge"]):
            context_score += 1
            context_reasons.append("RSI 動能偏多 (>= 50)")
        else:
            context_reasons.append("RSI 動能偏弱 (< 50)")

    if foreign_net_latest is not None:
        if foreign_net_latest > 0:
            context_score += 1
            context_reasons.append("外資買超，籌碼偏多")
        elif foreign_net_latest < 0:
            context_reasons.append("外資賣超，籌碼偏空")

    trend_mid_ok = (
        not pd.isna(sma20_val)
        and not pd.isna(sma60_val)
        and sma20_val > sma60_val
    )
    if trend_mid_ok:
        context_score += 1
        context_reasons.append("SMA20 高於 SMA60，中期趨勢偏多")

    sma60_up = False
    if len(df) >= int(P["min_rows_sma60_compare"]):
        sma60_now = df["SMA60"].iloc[-1]
        sma60_prev = df["SMA60"].iloc[-int(P["sma60_prev_iloc_offset"])]
        if not pd.isna(sma60_now) and not pd.isna(sma60_prev):
            sma60_up = bool(sma60_now > sma60_prev)
            if sma60_up:
                context_score += 1
                context_reasons.append("SMA60 走升，長週期動能偏多")

    vol_ratio_20 = None
    if not pd.isna(vol_ma20_val) and vol_ma20_val > 0:
        vol_ratio_20 = latest_volume / vol_ma20_val

    vol_breakout_ok = (
        trigger_type == "BREAKOUT"
        and vol_ratio_20 is not None
        and float(P["vol_breakout_ratio_min"]) <= vol_ratio_20 <= float(P["vol_breakout_ratio_max"])
    )
    if vol_breakout_ok:
        context_score += 1
        context_reasons.append("突破時放量但不爆量（量能健康）")

    vol_shrink_ok = False
    if len(df) >= int(P["min_bars_vol_shrink"]):
        last2_vol = df["Volume"].tail(2)
        last2_vol_ma = df["VolMA20"].tail(2)
        last2_close = df["Close"].tail(2)
        last2_ema20 = df["EMA20"].tail(2)
        if (
            (last2_vol < last2_vol_ma).all()
            and (last2_close >= last2_ema20).all()
        ):
            vol_shrink_ok = True
            context_score += 1
            context_reasons.append("突破後量縮且價不破，換手健康")

    rs_ok = False
    market_cache_layer = "skipped"
    _rs_n = int(P["relative_strength_lookback_bars"])
    if len(df) >= _rs_n:
        ret_stock_20d = (df["Close"].iloc[-1] / df["Close"].iloc[-_rs_n]) - 1
        mkt, idx_symbol, market_cache_layer = load_market_index_with_meta(market_symbol)
        if len(mkt) >= _rs_n:
            ret_mkt_20d = (mkt["Close"].iloc[-1] / mkt["Close"].iloc[-_rs_n]) - 1
            if ret_stock_20d > ret_mkt_20d:
                rs_ok = True
                context_score += 1
                context_reasons.append(f"相對強勢：近 20 日漲幅優於 {idx_symbol}")

    atr_pct_ok = False
    if atr14_val is not None and not pd.isna(atr14_val) and latest_close > 0:
        atr_pct = atr14_val / latest_close
        if float(P["atr_pct_of_close_min"]) <= atr_pct <= float(P["atr_pct_of_close_max"]):
            atr_pct_ok = True
            context_score += 1
            context_reasons.append("波動度適中（ATR% 在 1%~5%）")

    higher_low_ok = False
    _hlw = int(P["higher_low_window"])
    if len(df) >= int(P["min_bars_higher_low"]):
        low_recent = df["Low"].tail(_hlw).min()
        low_prev = df["Low"].shift(_hlw).tail(_hlw).min()
        if not pd.isna(low_recent) and not pd.isna(low_prev) and low_recent > low_prev:
            higher_low_ok = True
            context_score += 1
            context_reasons.append("低點抬高（Higher Low）")

    weekly_status, weekly_cache_layer = get_weekly_trend_with_meta(market_symbol)
    if weekly_status == "多頭":
        context_score += 1
        context_reasons.append("週線趨勢偏多 (大環境保護小環境)")
    elif weekly_status == "空頭":
        context_score -= 1
        context_reasons.append("週線趨勢偏空 (短線反彈需謹慎)")
    else:
        context_reasons.append("週線趨勢未知")

    below_ema20 = False
    _streak = int(P["consecutive_below_ema_days"])
    if len(ema_df) >= _streak and ema20 is not None:
        last3 = ema_df.tail(_streak)
        below_ema20 = bool((last3["Close"] < last3["EMA20"]).all())
        if below_ema20:
            context_score -= 1
            context_reasons.append("股價連續 3 天站不回 EMA20")

    prior_high = None
    _ph_roll = int(P["prior_high_rolling_days"])
    if "High" in df.columns and len(df) >= int(P["min_bars_two_bar_pattern"]):
        prior_high = to_scalar(df["High"].shift(1).rolling(_ph_roll).max().iloc[-1])
        if not pd.isna(prior_high) and close_price is not None:
            if close_price > prior_high:
                context_score += 1
                context_reasons.append("股價突破前高 (近 20 日高點)")
            else:
                context_reasons.append("股價未突破前高 (近 20 日高點)")

    if ema5 is not None and close_price is not None:
        if close_price < ema5:
            context_score -= 1
            context_reasons.append("股價跌破 EMA5")
        if len(ema_df) >= _streak:
            last3 = ema_df.tail(_streak)
            below_ema5 = (last3["Close"] < last3["EMA5"]).all()
            if below_ema5:
                context_score -= 1
                context_reasons.append("股價連續 3 天站不回 EMA5")

    if is_high_volume_sell:
        context_score -= 3
        context_reasons.append("高檔爆大量收黑，可能有出貨壓力")

    is_dangerous_volume = bool(latest.get("Is_Dangerous_Volume", False))
    if is_dangerous_volume:
        context_score -= 2
        context_reasons.append("高檔放量收黑（疑似危險換手）")

    if bias_20_val is not None and not pd.isna(bias_20_val) and bias_20_val > float(
        P["bias20_overheated_gt"]
    ):
        context_score -= 2
        context_reasons.append("乖離過熱（Bias20 > 10%）")

    upper_wick_bad = False
    if len(df) >= int(P["min_bars_two_bar_pattern"]):
        last2 = df.tail(2)
        upper_wick = last2["High"] - last2[["Open", "Close"]].max(axis=1)
        k_range = last2["High"] - last2["Low"]
        wick_ratio = upper_wick / k_range.replace(0, np.nan)
        if (wick_ratio > float(P["upper_wick_ratio_gt"])).all():
            upper_wick_bad = True
            context_score -= 1
            context_reasons.append("連續 2 天上影線過長")

    if prior_high is not None:
        if latest["High"] > prior_high and close_price is not None and close_price < prior_high:
            context_score -= 2
            context_reasons.append("突破前高後收回（可能假突破）")

    if ema20 is not None and vol_ratio_20 is not None and close_price is not None:
        if close_price < ema20 and vol_ratio_20 > float(P["break_ema20_vol_ratio_gt"]):
            context_score -= 2
            context_reasons.append("跌破 EMA20 且放量（趨勢破壞）")

    gap_risk = False
    if len(df) >= int(P["min_bars_two_bar_pattern"]):
        prev_row = df.iloc[-2]
        if (
            latest["Open"] > prev_row["High"] * float(P["gap_up_open_vs_prior_high_mult"])
            and latest["Low"] < prev_row["Close"]
        ):
            gap_risk = True
            context_score -= 1
            context_reasons.append("跳空上漲後回補缺口")

    market_bear = weekly_status == "空頭"
    if market_bear:
        context_score -= 2
        context_reasons.append("市場趨勢偏空，訊號降級")

    if foreign_divergence_warning is True:
        context_score -= 1
        context_reasons.append("籌碼背離：股價上漲但外資連續賣超")

    recent_foreign_sum = None
    if foreign_net_series is not None and len(foreign_net_series) >= 3:
        recent_foreign_sum = float(foreign_net_series.tail(3).sum())
        if recent_foreign_sum <= float(P["foreign_3d_net_sum_le"]):
            context_score -= 2
            context_reasons.append("外資近 3 天累積賣超超過 5000 張")

    total_score = int(np.clip(weighted_score + context_score, 0, 100))
    combined = list(weighted_reasons) + list(context_reasons)

    if os.environ.get("DEBUG_DIAGNOSIS_SCORE", "").strip() in (
        "1",
        "true",
        "True",
        "yes",
        "YES",
    ):
        _vol = to_scalar(latest.get("Volume", np.nan)) if latest is not None else None
        _vm20 = to_scalar(latest.get("VolMA20", np.nan)) if latest is not None else None
        _daily_layer = daily_cache_layer if daily_cache_layer else "unknown"
        _last_bar_ts = None
        try:
            if df is not None and len(df) >= 1:
                _ix = df.index[-1]
                _last_bar_ts = (
                    _ix.isoformat()
                    if hasattr(_ix, "isoformat")
                    else str(_ix)
                )
        except Exception:
            _last_bar_ts = None
        _len_df = len(df) if df is not None else 0
        _period_tag = period_tag if period_tag else "unknown"
        _idx_tz = _df_index_tz_tag(df)
        _calendar = _market_calendar_tag(market_symbol)
        _sig_raw = "|".join(
            str(x)
            for x in (
                market_symbol,
                _period_tag,
                _len_df,
                _last_bar_ts,
                _idx_tz,
                _calendar,
                SCORING_VERSION,
                SCORING_CONFIG_HASH,
                _daily_layer,
                market_cache_layer,
                weekly_status,
                weekly_cache_layer,
                close_price,
                ema20,
                ema5,
                _vol,
                _vm20,
                rs_ok,
                weighted_score,
                context_score,
                total_score,
                foreign_divergence_warning,
            )
        )
        _signature = hashlib.md5(_sig_raw.encode("utf-8")).hexdigest()
        dbg = {
            "symbol": market_symbol,
            "version": SCORING_VERSION,
            "period_tag": _period_tag,
            "scoring_config_hash": SCORING_CONFIG_HASH,
            "weighted_src_hash": WEIGHTED_SCORE_SRC_HASH,
            "len_df": _len_df,
            "last_bar_ts": _last_bar_ts,
            "tz": _idx_tz,
            "calendar": _calendar,
            "daily_cache_layer": _daily_layer,
            "weekly_cache_layer": weekly_cache_layer,
            "market_cache_layer": market_cache_layer,
            "close": close_price,
            "ema20": ema20,
            "ema5": ema5,
            "bias20": bias_20_val,
            "volume": _vol,
            "vol_ma20": _vm20,
            "weekly_trend": weekly_status,
            "rs_20d_beat_market": rs_ok,
            "chip_divergence_evaluated": foreign_divergence_warning is not None,
            "weighted": float(weighted_score),
            "context": int(context_score),
            "total": int(total_score),
            "signature": _signature,
        }

        def _jdefault(o):
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
                return None
            raise TypeError(repr(o))

        print("[DEBUG_DIAGNOSIS_SCORE] " + json.dumps(dbg, ensure_ascii=False, default=_jdefault))

    return DiagnosisScoreBundle(
        weighted_score=float(weighted_score),
        weighted_reasons=list(weighted_reasons),
        weighted_flags=weighted_flags,
        context_score=int(context_score),
        context_reasons=context_reasons,
        total_score=total_score,
        combined_reasons=combined,
        weekly_status=str(weekly_status),
        ema20_up=ema20_up,
        trend_mid_ok=bool(trend_mid_ok),
        sma60_up=bool(sma60_up),
        vol_breakout_ok=bool(vol_breakout_ok),
        vol_shrink_ok=bool(vol_shrink_ok),
        vol_ratio_20=vol_ratio_20,
        prior_high=prior_high,
        higher_low_ok=bool(higher_low_ok),
        rs_ok=bool(rs_ok),
        atr_pct_ok=bool(atr_pct_ok),
        below_ema20=bool(below_ema20),
        upper_wick_bad=bool(upper_wick_bad),
        gap_risk=bool(gap_risk),
        market_bear=bool(market_bear),
        recent_foreign_sum=recent_foreign_sum,
    )
