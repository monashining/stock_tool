"""
回測引擎：BUY_SIGNAL（Gate/Trigger/Guard）與 TURN 策略回測

提供統一介面與指標，方便比較不同策略、調參、追蹤精準度。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from analysis import compute_buy_signals, compute_indicators
from turn_check_engine import backtest_turn_signals, load_turn_config


@dataclass
class BacktestSummary:
    """回測摘要（供 UI 顯示與比較）"""
    strategy: str
    symbol: str
    n_signals: int
    win_rate: float
    avg_return_pct: float
    avg_holding_days: float
    max_drawdown_pct: float
    sharpe_approx: float
    expectancy: float
    profit_factor: float
    trades_df: pd.DataFrame
    # 即時診斷用（回測參數連動）
    hold_days: int = 10
    trailing_stop_pct: float = 0.0
    exit_ema_window: int = 0
    current_status: Optional[Dict[str, Any]] = None


def _ensure_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """確保 df 有 BUY_SIGNAL 所需欄位（需先 compute_indicators）"""
    if df is None or df.empty:
        return df
    if "BUY_SIGNAL" not in df.columns:
        df = compute_indicators(df)
        df = compute_buy_signals(df)
    return df


def backtest_buy_signal(
    df: pd.DataFrame,
    *,
    hold_days: int = 10,
    trailing_stop_pct: float = 0.0,
    exit_ema_window: int = 0,
    min_win_pct: float = 0.0,
) -> pd.DataFrame:
    """
    回測 BUY_SIGNAL（Gate + Trigger + Guard）策略

    - 進場：BUY_SIGNAL 從 False 變 True 的當根收盤
    - 出場（誰先到）：
      1) exit_ema_window >= 2：收盤跌破 EMA(exit_ema_window) 出場（不再依賴 BUY_SIGNAL 降級）
      2) 移動止損：從進場後最高點回檔 trailing_stop_pct%
      3) 持有期滿：hold_days 天

    回傳欄位與 backtest_turn_signals 相容
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "date", "entry_close", "exit_close", "exit_date", "exit_reason",
                "exit_stop", "holding_days", "max_favorable_pct", "max_adverse_pct",
                "return_pct", "final_pct", "giveback_pct", "giveback_ratio",
                "trigger_type",
            ]
        )
    df = _ensure_indicators(df)
    if "BUY_SIGNAL" not in df.columns:
        return pd.DataFrame()

    signal = df["BUY_SIGNAL"].astype(bool)
    signal_mask = signal & (~signal.shift(1).fillna(False))
    if len(signal_mask) > 0:
        signal_mask.iloc[0] = False
    signal_pos = np.where(signal_mask.fillna(False).values)[0]

    close_s = pd.to_numeric(df["Close"], errors="coerce")
    high_s = pd.to_numeric(df["High"], errors="coerce")
    low_s = pd.to_numeric(df["Low"], errors="coerce")
    open_s = pd.to_numeric(df["Open"], errors="coerce") if "Open" in df.columns else close_s

    hold_days = max(1, int(hold_days))
    trailing_ratio = max(0.0, float(trailing_stop_pct or 0.0) / 100.0)
    exit_ema_window = max(0, int(exit_ema_window or 0))

    exit_ema_s = None
    if exit_ema_window >= 2 and "EMA5" in df.columns:
        try:
            ema_col = f"EMA{exit_ema_window}" if f"EMA{exit_ema_window}" in df.columns else "EMA5"
            exit_ema_s = pd.to_numeric(df[ema_col], errors="coerce")
        except Exception:
            exit_ema_s = None

    rows = []
    for i in signal_pos:
        if i + 1 >= len(df):
            continue
        entry_close = float(close_s.iloc[i])
        if not np.isfinite(entry_close) or entry_close <= 0:
            continue
        trigger_type = str(df["BUY_TRIGGER_TYPE"].iloc[i]) if "BUY_TRIGGER_TYPE" in df.columns else "NONE"

        max_exit_pos = min(len(df) - 1, i + hold_days)

        # 出場：純粹依賴「跌破 EMA」與「持有期滿」（不再依賴 BUY_SIGNAL 降級）
        status_exit_pos = int(max_exit_pos)
        status_exit_reason = "HOLDING" if status_exit_pos == (len(df) - 1) else "TIMEOUT"

        if exit_ema_s is not None and max_exit_pos >= (i + 1):
            close_after = close_s.iloc[i + 1 : max_exit_pos + 1]
            ma_after = exit_ema_s.iloc[i + 1 : max_exit_pos + 1]
            ema_break = (
                close_after.notna()
                & ma_after.notna()
                & (close_after < ma_after)
            )
            if bool(ema_break.any()):
                rel = int(np.argmax(ema_break.values))
                status_exit_pos = int(i + 1 + rel)
                status_exit_reason = f"EMA{exit_ema_window}_BREAK"
        # 若 exit_ema_window == 0：維持持有期滿出場（status_exit_pos = max_exit_pos）

        status_exit_price = float(close_s.iloc[status_exit_pos])
        if not np.isfinite(status_exit_price) or status_exit_price <= 0:
            continue

        # 移動止損
        ts_exit_pos = None
        ts_exit_price = None
        ts_reason = None
        if trailing_ratio > 0:
            peak = float(entry_close)
            for j in range(i + 1, max_exit_pos + 1):
                stop = peak * (1.0 - trailing_ratio)
                o, h, l_ = float(open_s.iloc[j]), float(high_s.iloc[j]), float(low_s.iloc[j])
                if np.isfinite(o) and o <= stop:
                    ts_exit_pos = j
                    ts_exit_price = float(o)
                    ts_reason = f"TS_GAP({trailing_stop_pct:.1f}%)"
                    break
                if np.isfinite(l_) and l_ <= stop:
                    ts_exit_pos = j
                    ts_exit_price = float(stop)
                    ts_reason = f"TS({trailing_stop_pct:.1f}%)"
                    break
                if np.isfinite(h):
                    peak = max(peak, h)

        exit_pos = status_exit_pos
        exit_price = status_exit_price
        exit_reason = status_exit_reason
        if (
            ts_exit_pos is not None
            and ts_exit_price is not None
            and ts_exit_pos < status_exit_pos
        ):
            exit_pos = int(ts_exit_pos)
            exit_price = float(ts_exit_price)
            exit_reason = ts_reason or "TS"

        holding_days = int(exit_pos - i)
        if holding_days <= 0:
            continue

        fut_high = high_s.iloc[i + 1 : exit_pos + 1].max()
        fut_low = low_s.iloc[i + 1 : exit_pos + 1].min()
        if pd.isna(fut_high) or pd.isna(fut_low) or fut_low <= 0:
            continue

        max_fav = (float(fut_high) / entry_close - 1.0) * 100.0
        max_adv = (float(fut_low) / entry_close - 1.0) * 100.0
        return_pct = (float(exit_price) / entry_close - 1.0) * 100.0
        giveback_ratio = np.nan
        if np.isfinite(max_fav) and max_fav > 0:
            giveback_ratio = (max_fav - return_pct) / max_fav

        rows.append({
            "date": df.index[i],
            "entry_close": entry_close,
            "exit_close": exit_price,
            "exit_date": df.index[exit_pos],
            "exit_reason": exit_reason,
            "exit_stop": np.nan,
            "holding_days": holding_days,
            "max_favorable_pct": max_fav,
            "max_adverse_pct": max_adv,
            "return_pct": return_pct,
            "final_pct": return_pct,
            "giveback_pct": max_fav - return_pct if np.isfinite(max_fav) else np.nan,
            "giveback_ratio": giveback_ratio,
            "trigger_type": trigger_type,
        })

    return pd.DataFrame(rows)


def compute_backtest_metrics(
    trades_df: pd.DataFrame,
    *,
    win_threshold_pct: float = 0.0,
) -> Dict[str, Any]:
    """從交易明細計算回測指標"""
    if trades_df is None or trades_df.empty:
        return {
            "n_signals": 0,
            "win_rate": np.nan,
            "avg_return_pct": np.nan,
            "avg_holding_days": np.nan,
            "max_drawdown_pct": np.nan,
            "sharpe_approx": np.nan,
            "expectancy": np.nan,
            "profit_factor": np.nan,
        }
    ret = pd.to_numeric(trades_df["return_pct"], errors="coerce")
    final = trades_df.get("final_pct", ret)
    if final is None:
        final = ret
    final = pd.to_numeric(final, errors="coerce").dropna()
    hold = pd.to_numeric(trades_df.get("holding_days", 0), errors="coerce")

    n = len(trades_df)
    win_rate = (final >= win_threshold_pct).mean() if len(final) > 0 else np.nan
    avg_return = float(final.mean()) if len(final) > 0 else np.nan
    avg_hold = float(hold.mean()) if len(hold) > 0 and hold.notna().any() else np.nan

    # 簡易權益曲線 → 最大回撤
    cum = (1 + final / 100.0).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak * 100.0
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    # 簡易 Sharpe（假設每筆獨立，年化）
    ret_std = float(final.std()) if len(final) > 1 else 0.0
    sharpe = (avg_return / ret_std * np.sqrt(252 / max(avg_hold, 1))) if ret_std > 0 else np.nan

    # Profit factor
    gains = final[final > 0].sum()
    losses = abs(final[final < 0].sum())
    pf = float(gains / losses) if losses > 0 and np.isfinite(gains) else (np.inf if gains > 0 else np.nan)

    return {
        "n_signals": n,
        "win_rate": win_rate,
        "avg_return_pct": avg_return,
        "avg_holding_days": avg_hold,
        "max_drawdown_pct": max_dd,
        "sharpe_approx": sharpe,
        "expectancy": avg_return,
        "profit_factor": pf,
    }


def run_backtest(
    df: pd.DataFrame,
    *,
    strategy: str = "buy_signal",
    symbol: str = "",
    hold_days: int = 10,
    trailing_stop_pct: float = 0.0,
    exit_ema_window: int = 0,
    win_threshold_pct: float = 0.0,
    foreign_3d_net: Optional[float] = None,
    trust_3d_net: Optional[float] = None,
    turn_cfg: Optional[Dict] = None,
) -> Optional[BacktestSummary]:
    """
    統一回測入口：依 strategy 選擇 BUY_SIGNAL 或 TURN
    """
    if df is None or df.empty:
        return None
    strategy = (strategy or "buy_signal").lower().strip()

    if strategy == "buy_signal":
        trades = backtest_buy_signal(
            df,
            hold_days=hold_days,
            trailing_stop_pct=trailing_stop_pct,
            exit_ema_window=exit_ema_window,
        )
    elif strategy in ("turn_bottom", "turn_top"):
        mode = "bottom" if strategy == "turn_bottom" else "top"
        trades = backtest_turn_signals(
            df,
            mode=mode,
            cfg=turn_cfg or load_turn_config(),
            foreign_3d_net=foreign_3d_net,
            trust_3d_net=trust_3d_net,
            hold_days=hold_days,
            trailing_stop_pct=trailing_stop_pct,
            exit_ma_window=exit_ema_window,
        )
    else:
        return None

    if trades is None or trades.empty:
        return None
    metrics = compute_backtest_metrics(trades, win_threshold_pct=win_threshold_pct)

    # 即時診斷：從最新 K 線與最後一筆出場推斷現況
    current_status: Optional[Dict[str, Any]] = None
    if df is not None and not df.empty:
        try:
            last_close = float(df["Close"].iloc[-1])
            bias20 = float(df["Bias20"].iloc[-1]) if "Bias20" in df.columns else 0.0
            ema5 = float(df["EMA5"].iloc[-1]) if "EMA5" in df.columns else last_close
            last_high = float(df["High"].iloc[-1]) if "High" in df.columns else last_close
            last_exit = (
                str(trades["exit_reason"].iloc[-1])
                if "exit_reason" in trades.columns and len(trades) > 0
                else "N/A"
            )
            current_status = {
                "is_hot": bias20 > 15,
                "last_exit_reason": last_exit,
                "suggested_action": "HOLD" if last_close > ema5 else "EXIT",
                "last_close": last_close,
                "last_high": last_high,
                "ema5": ema5,
                "bias20": bias20,
            }
        except Exception:
            current_status = None

    return BacktestSummary(
        strategy=strategy,
        symbol=symbol,
        n_signals=int(metrics["n_signals"]),
        win_rate=float(metrics["win_rate"]) if np.isfinite(metrics["win_rate"]) else 0.0,
        avg_return_pct=float(metrics["avg_return_pct"]) if np.isfinite(metrics["avg_return_pct"]) else 0.0,
        avg_holding_days=float(metrics["avg_holding_days"]) if np.isfinite(metrics["avg_holding_days"]) else 0.0,
        max_drawdown_pct=float(metrics["max_drawdown_pct"]) if np.isfinite(metrics["max_drawdown_pct"]) else 0.0,
        sharpe_approx=float(metrics["sharpe_approx"]) if np.isfinite(metrics["sharpe_approx"]) else 0.0,
        expectancy=float(metrics["expectancy"]) if np.isfinite(metrics["expectancy"]) else 0.0,
        profit_factor=float(metrics["profit_factor"]) if np.isfinite(metrics["profit_factor"]) else 0.0,
        trades_df=trades,
        hold_days=hold_days,
        trailing_stop_pct=float(trailing_stop_pct or 0),
        exit_ema_window=int(exit_ema_window or 0),
        current_status=current_status,
    )
