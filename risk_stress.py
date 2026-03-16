"""
風險壓力測試：蒙地卡羅 + 摩擦成本 + 倖存者偏差檢查

系統驗證邏輯：
1. 波動率壓力：若 σ 增加 X%，移動止損策略破功（虧損）的機率？
2. 摩擦成本：證交稅 0.3% + 手續費，實際獲利需扣除
3. 訊號時序：檢查訊號是否集中在特定行情（倖存者偏差）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from price_prediction import compute_gbm_params, run_gbm_monte_carlo


# 台灣交易摩擦成本（單邊）
TAIWAN_TAX_PCT = 0.3          # 證交稅（賣出）
TAIWAN_FEE_PCT = 0.1425      # 手續費（買+賣各約 0.1425%，折扣券商）
ROUND_TRIP_FRICTION_PCT = TAIWAN_TAX_PCT + TAIWAN_FEE_PCT * 2  # 約 0.585%


@dataclass
class TrailingStopStressResult:
    """移動止損蒙地卡羅壓力測試結果"""
    pct_loss: float           # 虧損路徑比例（破功機率）
    pct_win: float
    avg_return_pct: float     # 模擬路徑平均報酬
    median_return_pct: float
    sigma_base: float         # 基準波動率
    sigma_stressed: float     # 壓力情境波動率
    vol_shock_pct: float      # 波動率增幅（%）
    n_sim: int
    hold_days: int
    trailing_stop_pct: float


def _apply_trailing_stop_to_path(
    path: np.ndarray,
    trailing_stop_ratio: float,
) -> tuple[float, int]:
    """
    對單一路徑套用移動止損，回傳 (exit_return_pct, exit_day)
    path: 股價序列（含起點），長度 hold_days+1
    """
    if len(path) < 2 or path[0] <= 0:
        return 0.0, 0
    peak = float(path[0])
    stop = peak * (1.0 - trailing_stop_ratio)
    for d in range(1, len(path)):
        p = float(path[d])
        if not np.isfinite(p) or p <= 0:
            continue
        if p <= stop:
            ret = (p / path[0] - 1.0) * 100.0
            return ret, d
        peak = max(peak, p)
        stop = peak * (1.0 - trailing_stop_ratio)
    # 持有到期
    ret = (path[-1] / path[0] - 1.0) * 100.0
    return ret, len(path) - 1


def trailing_stop_monte_carlo_stress(
    df: pd.DataFrame,
    *,
    trailing_stop_pct: float = 10.0,
    hold_days: int = 10,
    vol_shock_pct: float = 20.0,
    n_sim: int = 10000,
    seed: Optional[int] = 42,
) -> Optional[TrailingStopStressResult]:
    """
    蒙地卡羅壓力測試：若波動率增加 vol_shock_pct%，
    這套 trailing_stop_pct 移動止損策略「破功（虧損）」的機率？

    邏輯：
    - 從歷史資料估計 μ, σ
    - 壓力情境：σ_stressed = σ * (1 + vol_shock_pct/100)
    - 模擬 n_sim 條路徑，每條套用移動止損
    - 計算虧損路徑比例
    """
    if df is None or df.empty:
        return None
    mu, sigma_base = compute_gbm_params(df)
    sigma_stressed = sigma_base * (1.0 + float(vol_shock_pct) / 100.0)
    trailing_ratio = float(trailing_stop_pct) / 100.0
    hold_days = max(1, int(hold_days))

    paths = run_gbm_monte_carlo(
        s0=1.0,  # 歸一化，報酬率不變
        mu=mu,
        sigma=sigma_stressed,
        n_days=hold_days,
        n_sim=n_sim,
        seed=seed,
    )
    # paths shape (n_sim, n_days)，需加上起點
    paths_full = np.column_stack([np.ones((n_sim, 1)), paths])

    returns = []
    for i in range(n_sim):
        ret, _ = _apply_trailing_stop_to_path(paths_full[i], trailing_ratio)
        returns.append(ret)
    returns = np.array(returns)

    n_loss = (returns < 0).sum()
    n_win = (returns >= 0).sum()
    return TrailingStopStressResult(
        pct_loss=float(n_loss) / n_sim * 100.0,
        pct_win=float(n_win) / n_sim * 100.0,
        avg_return_pct=float(np.mean(returns)),
        median_return_pct=float(np.median(returns)),
        sigma_base=sigma_base,
        sigma_stressed=sigma_stressed,
        vol_shock_pct=vol_shock_pct,
        n_sim=n_sim,
        hold_days=hold_days,
        trailing_stop_pct=trailing_stop_pct,
    )


def compute_friction_adjusted_return(
    avg_return_pct: float,
    round_trip_pct: float = ROUND_TRIP_FRICTION_PCT,
) -> float:
    """扣除摩擦成本後的實際報酬"""
    return float(avg_return_pct) - float(round_trip_pct)


def get_signal_date_range(trades_df: pd.DataFrame) -> Optional[tuple[str, str]]:
    """取得訊號日期區間（供倖存者偏差檢查）"""
    if trades_df is None or trades_df.empty:
        return None
    if "date" in trades_df.columns:
        dates = pd.to_datetime(trades_df["date"], errors="coerce").dropna()
    else:
        dates = pd.to_datetime(trades_df.index, errors="coerce")
    if dates.empty:
        return None
    return (str(dates.min().date()), str(dates.max().date()))
