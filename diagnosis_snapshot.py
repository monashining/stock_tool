"""
與個股頁同源：載入 OHLCV、指標、加權分數＋情境分、TURN、裁決輸入欄位。
供 LINE 群組查詢等非 Streamlit 入口使用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from analysis import (
    compute_buy_signals,
    compute_indicators,
    compute_risk_metrics,
    compute_volume_sum_3d,
    compute_weighted_score,
)
from data_sources import (
    _load_data_raw,
    fetch_ticker_name,
    fetch_foreign_net_series,
    fetch_trust_net_series,
    get_weekly_trend,
    load_data,
    load_market_index,
)
from expert_advice_text import generate_expert_advice
from final_decision_resolver import DecisionInput, ResolvedDecision, resolve_final_decision
from turn_check_engine import load_turn_config, run_turn_check
from utils import align_by_date, align_net_series_to_price, to_scalar


@dataclass
class LineDiagnosisSnapshot:
    effective_symbol: str
    name: str
    df: pd.DataFrame
    risk: Any
    foreign_net_series: Optional[pd.Series]
    trust_net_series: Optional[pd.Series]
    foreign_divergence_warning: bool
    latest: Any
    ema5: Optional[float]
    ema20: Optional[float]
    close_price: float
    gate_ok: bool
    trigger_ok: bool
    exec_guard_ok: bool
    trigger_type: str
    weighted_score: float
    score: int
    bottom_now: Any
    top_now: Any
    page_resolved_decision: Optional[ResolvedDecision]
    expert_msg: str


def _normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join([str(x) for x in col if x is not None]) for col in df.columns
        ]
    df = df.copy()
    df.columns = df.columns.str.strip()
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_line_diagnosis_snapshot(
    effective_symbol: str,
    *,
    time_range: str = "1y",
    turn_cfg_path: str = "turn_check_config.json",
) -> LineDiagnosisSnapshot:
    name, resolved = fetch_ticker_name(effective_symbol)
    sym = resolved or effective_symbol.strip()

    df = load_data(sym, time_range)
    if df is None or df.empty:
        df = _load_data_raw(sym, time_range)
    if df is None or df.empty:
        raise ValueError(f"無法取得行情：{sym}")

    df = _normalize_ohlcv_df(df)
    df = compute_indicators(df)
    df = compute_buy_signals(df)

    df_risk = load_data(sym, "1y")
    if df_risk is None or df_risk.empty:
        df_risk = df
    mkt_df, _idx_sym = load_market_index(sym)
    risk = compute_risk_metrics(df_risk, mkt_df, rf_annual=0.015)

    foreign_net_series = None
    trust_net_series = None
    foreign_divergence_warning = False
    try:
        foreign_net_series = fetch_foreign_net_series(sym)
        trust_net_series = fetch_trust_net_series(sym)
        foreign_series_aligned = align_net_series_to_price(df, foreign_net_series)
        trust_series_aligned = align_net_series_to_price(df, trust_net_series)
        if foreign_series_aligned is not None:
            foreign_net_series = foreign_series_aligned
        if trust_series_aligned is not None:
            trust_net_series = trust_series_aligned
        if foreign_net_series is not None and not foreign_net_series.empty:
            aligned = align_by_date(df, foreign_net_series)
            if len(aligned) >= 3:
                recent = aligned.tail(3)
                if (recent["net"] < 0).all() and recent["Close"].iloc[-1] > recent["Close"].iloc[0]:
                    foreign_divergence_warning = True
    except Exception:
        foreign_net_series = None
        trust_net_series = None

    latest_row = df.iloc[-1]
    latest_close = to_scalar(latest_row["Close"])
    latest_volume = to_scalar(latest_row["Volume"])
    latest_vol_avg_5 = np.nan
    if sym.endswith(".TW") or sym.endswith(".TWO"):
        vol_avg_5_series = df["Volume"].rolling(5).mean()
        latest_vol_avg_5 = to_scalar(vol_avg_5_series.iloc[-1])

    gate_ok = bool(latest_row.get("BUY_GATE", False))
    trigger_ok = bool(latest_row.get("BUY_TRIGGER", False))
    trigger_type = str(latest_row.get("BUY_TRIGGER_TYPE", "NONE"))
    exec_guard_ok = bool(latest_row.get("EXEC_GUARD", True))

    bias_20_val = to_scalar(latest_row.get("Bias20", np.nan))
    vol_ma20_val = to_scalar(latest_row.get("VolMA20", np.nan))
    sma20_val = to_scalar(latest_row.get("SMA20", np.nan))
    sma60_val = to_scalar(latest_row.get("SMA60", np.nan))
    atr14_val = to_scalar(latest_row.get("ATR14", np.nan))

    df["Vol_Avg_5"] = df["Volume"].rolling(5).mean()
    current_vol = to_scalar(df["Volume"].iloc[-1])
    avg_vol_5 = to_scalar(df["Vol_Avg_5"].iloc[-1])

    ema20 = None
    ema5 = None
    close_price = latest_close
    ema_df = pd.DataFrame()
    if "EMA20" in df.columns:
        ema_df = df.dropna(subset=["EMA20", "EMA5"])

    if ema_df.empty:
        raise ValueError("EMA 資料不足，無法計算診斷。")

    ema_latest = ema_df.iloc[-1]
    close_price = to_scalar(ema_latest["Close"])
    ema20 = to_scalar(ema_latest["EMA20"])
    ema5 = to_scalar(ema_latest["EMA5"])

    is_high_volume_sell = False
    if ema20 is not None and ema5 is not None and not pd.isna(avg_vol_5):
        is_high_price = close_price > ema20 * 1.05
        is_huge_vol = current_vol > avg_vol_5 * 2
        is_selling_pressure = close_price < to_scalar(df["Open"].iloc[-1])
        is_high_volume_sell = is_high_price and is_huge_vol and is_selling_pressure

    weighted_score, _weighted_reasons, _weighted_flags = compute_weighted_score(
        ema20=ema20,
        ema5=ema5,
        close_price=close_price,
        current_vol=current_vol,
        avg_vol_5=avg_vol_5,
        bias_20_val=bias_20_val,
        foreign_net_series=foreign_net_series,
        trust_net_series=trust_net_series,
        vol_sum_3d=compute_volume_sum_3d(df),
        is_dangerous_vol=bool(latest_row.get("Is_Dangerous_Volume", False)),
    )

    context_score = 0
    ema20_up = None
    if ema20 is not None and len(ema_df) >= 2:
        ema20_up = to_scalar(ema_df["EMA20"].iloc[-1]) > to_scalar(
            ema_df["EMA20"].iloc[-2]
        )
    if ema20 is None or ema20_up is None:
        pass
    elif close_price > ema20 and ema20_up:
        context_score += 1
    else:
        context_score -= 1

    if ema5 is None or ema20 is None:
        pass
    elif ema5 > ema20:
        context_score += 1
    else:
        context_score -= 1

    if not pd.isna(df["RSI14"].iloc[-1]):
        rsi_latest = to_scalar(df["RSI14"].iloc[-1])
        if rsi_latest >= 50:
            context_score += 1

    foreign_net_latest = None
    if foreign_net_series is not None and not foreign_net_series.empty:
        foreign_net_latest = foreign_net_series.iloc[-1]
        if foreign_net_latest > 0:
            context_score += 1

    trend_mid_ok = (
        not pd.isna(sma20_val)
        and not pd.isna(sma60_val)
        and sma20_val > sma60_val
    )
    if trend_mid_ok:
        context_score += 1

    sma60_up = False
    if len(df) >= 6:
        sma60_now = df["SMA60"].iloc[-1]
        sma60_prev = df["SMA60"].iloc[-6]
        if not pd.isna(sma60_now) and not pd.isna(sma60_prev):
            sma60_up = sma60_now > sma60_prev
            if sma60_up:
                context_score += 1

    vol_ratio_20 = None
    if not pd.isna(vol_ma20_val) and vol_ma20_val > 0:
        vol_ratio_20 = latest_volume / vol_ma20_val

    if (
        trigger_type == "BREAKOUT"
        and vol_ratio_20 is not None
        and 1.2 <= vol_ratio_20 <= 1.8
    ):
        context_score += 1

    if len(df) >= 3:
        last2_vol = df["Volume"].tail(2)
        last2_vol_ma = df["VolMA20"].tail(2)
        last2_close = df["Close"].tail(2)
        last2_ema20 = df["EMA20"].tail(2)
        if (
            (last2_vol < last2_vol_ma).all()
            and (last2_close >= last2_ema20).all()
        ):
            context_score += 1

    if len(df) >= 21:
        ret_stock_20d = (df["Close"].iloc[-1] / df["Close"].iloc[-21]) - 1
        if len(mkt_df) >= 21:
            ret_mkt_20d = (mkt_df["Close"].iloc[-1] / mkt_df["Close"].iloc[-21]) - 1
            if ret_stock_20d > ret_mkt_20d:
                context_score += 1

    if atr14_val is not None and not pd.isna(atr14_val) and latest_close > 0:
        atr_pct = atr14_val / latest_close
        if 0.01 <= atr_pct <= 0.05:
            context_score += 1

    if len(df) >= 20:
        low_recent = df["Low"].tail(10).min()
        low_prev = df["Low"].shift(10).tail(10).min()
        if not pd.isna(low_recent) and not pd.isna(low_prev) and low_recent > low_prev:
            context_score += 1

    weekly_status = get_weekly_trend(sym)
    if weekly_status == "多頭":
        context_score += 1
    elif weekly_status == "空頭":
        context_score -= 1

    if len(ema_df) >= 3 and ema20 is not None:
        last3 = ema_df.tail(3)
        below_ema20 = (last3["Close"] < last3["EMA20"]).all()
        if below_ema20:
            context_score -= 1

    prior_high = None
    if "High" in df.columns and len(df) >= 2:
        prior_high = to_scalar(df["High"].shift(1).rolling(20).max().iloc[-1])
        if not pd.isna(prior_high):
            if close_price > prior_high:
                context_score += 1

    if ema5 is not None:
        if close_price < ema5:
            context_score -= 1
        if len(ema_df) >= 3:
            last3 = ema_df.tail(3)
            below_ema5 = (last3["Close"] < last3["EMA5"]).all()
            if below_ema5:
                context_score -= 1

    if is_high_volume_sell:
        context_score -= 3

    is_dangerous_volume = bool(latest_row.get("Is_Dangerous_Volume", False))
    if is_dangerous_volume:
        context_score -= 2

    if bias_20_val is not None and not pd.isna(bias_20_val) and bias_20_val > 10:
        context_score -= 2

    if len(df) >= 2:
        last2 = df.tail(2)
        upper_wick = last2["High"] - last2[["Open", "Close"]].max(axis=1)
        k_range = last2["High"] - last2["Low"]
        wick_ratio = upper_wick / k_range.replace(0, np.nan)
        if (wick_ratio > 0.5).all():
            context_score -= 1

    if prior_high is not None:
        if latest_row["High"] > prior_high and close_price < prior_high:
            context_score -= 2

    if ema20 is not None and vol_ratio_20 is not None:
        if close_price < ema20 and vol_ratio_20 > 1.3:
            context_score -= 2

    if len(df) >= 2:
        prev_row = df.iloc[-2]
        if (
            latest_row["Open"] > prev_row["High"] * 1.01
            and latest_row["Low"] < prev_row["Close"]
        ):
            context_score -= 1

    if weekly_status == "空頭":
        context_score -= 2

    if foreign_divergence_warning:
        context_score -= 1

    if foreign_net_series is not None and len(foreign_net_series) >= 3:
        recent_foreign_sum = foreign_net_series.tail(3).sum()
        if recent_foreign_sum <= -5000:
            context_score -= 2

    score = int(np.clip(weighted_score + context_score, 0, 100))

    turn_cfg = load_turn_config(turn_cfg_path)
    bottom_now = None
    top_now = None
    try:
        chip_foreign_3d_page = (
            float(foreign_net_series.dropna().tail(3).sum())
            if foreign_net_series is not None
            and foreign_net_series.dropna().shape[0] >= 3
            else None
        )
        chip_trust_3d_page = (
            float(trust_net_series.dropna().tail(3).sum())
            if trust_net_series is not None and trust_net_series.dropna().shape[0] >= 3
            else None
        )
        df_page_turn = df.copy()
        if "RSI" not in df_page_turn.columns and "RSI14" in df_page_turn.columns:
            df_page_turn["RSI"] = df_page_turn["RSI14"]
        if foreign_net_series is not None and not foreign_net_series.empty:
            df_page_turn["Foreign_Net"] = foreign_net_series.reindex(df_page_turn.index)
        if trust_net_series is not None and not trust_net_series.empty:
            df_page_turn["Trust_Net"] = trust_net_series.reindex(df_page_turn.index)
        bottom_now = run_turn_check(
            df_page_turn,
            mode="bottom",
            cfg=turn_cfg,
            foreign_3d_net=chip_foreign_3d_page,
            trust_3d_net=chip_trust_3d_page,
        )
        top_now = run_turn_check(
            df_page_turn,
            mode="top",
            cfg=turn_cfg,
            foreign_3d_net=chip_foreign_3d_page,
            trust_3d_net=chip_trust_3d_page,
        )
    except Exception:
        bottom_now = None
        top_now = None

    page_resolved_decision: Optional[ResolvedDecision] = None
    if ema5 is not None:
        page_resolved_decision = resolve_final_decision(
            DecisionInput(
                close=float(close_price),
                ema5=float(ema5),
                ema20=float(ema20)
                if ema20 is not None and not pd.isna(ema20)
                else None,
                defensive_line=float(ema5),
                weighted_ai_score=float(weighted_score),
                bottom_status=str(bottom_now.get("status", "NA"))
                if isinstance(bottom_now, dict)
                else None,
                top_status=str(top_now.get("status", "NA"))
                if isinstance(top_now, dict)
                else None,
                exec_guard_ok=exec_guard_ok,
                gate_pass=gate_ok,
                trigger_pass=trigger_ok,
                guard_pass=exec_guard_ok,
                position_mode=False,
            )
        )

    _expert_bottom = "NA"
    if isinstance(bottom_now, dict):
        _expert_bottom = str(bottom_now.get("status", "NA"))

    expert_msg = generate_expert_advice(
        df,
        name or sym,
        score,
        risk,
        is_chip_divergence=foreign_divergence_warning,
        weighted_ai_score=weighted_score,
        bottom_status=_expert_bottom,
    )

    return LineDiagnosisSnapshot(
        effective_symbol=sym,
        name=name or "",
        df=df,
        risk=risk,
        foreign_net_series=foreign_net_series,
        trust_net_series=trust_net_series,
        foreign_divergence_warning=foreign_divergence_warning,
        latest=latest_row,
        ema5=ema5,
        ema20=ema20,
        close_price=float(close_price),
        gate_ok=gate_ok,
        trigger_ok=trigger_ok,
        exec_guard_ok=exec_guard_ok,
        trigger_type=trigger_type,
        weighted_score=float(weighted_score),
        score=score,
        bottom_now=bottom_now,
        top_now=top_now,
        page_resolved_decision=page_resolved_decision,
        expert_msg=expert_msg,
    )


def resolve_decision_for_position_mode(
    snap: LineDiagnosisSnapshot,
    *,
    position_mode: bool,
) -> Optional[ResolvedDecision]:
    """與個股頁 DecisionInput.position_mode 對應（無 session 持倉時仍可比對去留語境）。"""
    b = snap.bottom_now
    t = snap.top_now
    if snap.ema5 is None:
        return snap.page_resolved_decision
    return resolve_final_decision(
        DecisionInput(
            close=float(snap.close_price),
            ema5=float(snap.ema5),
            ema20=float(snap.ema20)
            if snap.ema20 is not None and not pd.isna(snap.ema20)
            else None,
            defensive_line=float(snap.ema5),
            weighted_ai_score=float(snap.weighted_score),
            bottom_status=str(b.get("status", "NA")) if isinstance(b, dict) else None,
            top_status=str(t.get("status", "NA")) if isinstance(t, dict) else None,
            exec_guard_ok=snap.exec_guard_ok,
            gate_pass=snap.gate_ok,
            trigger_pass=snap.trigger_ok,
            guard_pass=snap.exec_guard_ok,
            position_mode=position_mode,
        )
    )
