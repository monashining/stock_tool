"""
機率預測模組：GBM 蒙地卡羅、斐波那契回撤/擴展、隱含波動率區間

數學模型：
1. GBM (Geometric Brownian Motion)：S_t = S_0 * exp((μ - σ²/2)t + σW_t)
2. Fibonacci：回撤位 High-(High-Low)×{0.236,0.382,0.5,0.618}，擴展位 Low+(High-Low)×{1.382,1.618,2.618}
3. IV 區間：Stock ± (Stock × σ × sqrt(DTE/365))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# -------------------------
# 1. GBM 蒙地卡羅模擬
# -------------------------
@dataclass
class GBMResult:
    """GBM 蒙地卡羅模擬結果"""
    paths: np.ndarray          # shape (n_sim, n_days)
    dates: pd.DatetimeIndex
    mu: float                  # 年化預期報酬率
    sigma: float               # 年化波動率
    s0: float
    percentile_5: np.ndarray
    percentile_50: np.ndarray
    percentile_95: np.ndarray


def compute_gbm_params(df: pd.DataFrame) -> tuple[float, float]:
    """
    從歷史日線計算 μ（年化報酬率）與 σ（年化波動率）
    假設 252 個交易日/年
    """
    if df is None or df.empty or "Close" not in df.columns:
        return 0.0, 0.2
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 2:
        return 0.0, 0.2
    returns = np.log(close / close.shift(1)).dropna()
    mu_daily = float(returns.mean())
    sigma_daily = float(returns.std())
    # 年化
    mu_annual = mu_daily * 252
    sigma_annual = sigma_daily * np.sqrt(252)
    return mu_annual, sigma_annual


def run_gbm_monte_carlo(
    s0: float,
    mu: float,
    sigma: float,
    n_days: int = 21,
    n_sim: int = 10000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    GBM 蒙地卡羅模擬：S_t = S_0 * exp((μ - σ²/2)t + σW_t)
    回傳 shape (n_sim, n_days) 的股價路徑
    """
    if seed is not None:
        np.random.seed(seed)
    dt = 1.0 / 252.0  # 日
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    # W_t ~ N(0, t)，增量 dW ~ N(0, dt)
    dW = np.random.standard_normal((n_sim, n_days))
    log_returns = drift + diffusion * dW
    log_path = np.log(s0) + np.cumsum(log_returns, axis=1)
    return np.exp(log_path)


def gbm_monte_carlo_full(
    df: pd.DataFrame,
    s0: Optional[float] = None,
    n_days: int = 21,
    n_sim: int = 10000,
    seed: Optional[int] = 42,
) -> Optional[GBMResult]:
    """
    完整 GBM 蒙地卡羅：計算參數、跑模擬、回傳路徑與百分位數
    """
    if df is None or df.empty:
        return None
    mu, sigma = compute_gbm_params(df)
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty:
        return None
    s0_val = float(s0) if s0 is not None and s0 > 0 else float(close.iloc[-1])
    paths = run_gbm_monte_carlo(s0_val, mu, sigma, n_days, n_sim, seed)
    dates = pd.date_range(
        start=df.index[-1],
        periods=n_days + 1,
        freq="B",
    )[1:]
    p5 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    return GBMResult(
        paths=paths,
        dates=dates,
        mu=mu,
        sigma=sigma,
        s0=s0_val,
        percentile_5=p5,
        percentile_50=p50,
        percentile_95=p95,
    )


# -------------------------
# 2. 斐波那契回撤與擴展
# -------------------------
FIB_RETRACEMENT = [0.236, 0.382, 0.5, 0.618]
FIB_EXTENSION = [1.382, 1.618, 2.618]


@dataclass
class FibonacciLevels:
    """斐波那契支撐/壓力位"""
    high: float
    low: float
    range_val: float
    retracement: dict[float, float]   # ratio -> price
    extension: dict[float, float]     # ratio -> price


def compute_fibonacci(
    high: float,
    low: float,
    retracement_ratios: Optional[list[float]] = None,
    extension_ratios: Optional[list[float]] = None,
) -> FibonacciLevels:
    """
    斐波那契回撤（找支撐）與擴展（找目標）
    回撤：High - (High - Low) × ratio
    擴展：Low + (High - Low) × ratio
    """
    if high < low:
        high, low = low, high
    rng = high - low
    retracement_ratios = retracement_ratios or FIB_RETRACEMENT
    extension_ratios = extension_ratios or FIB_EXTENSION
    retracement = {r: high - rng * r for r in retracement_ratios}
    extension = {r: low + rng * r for r in extension_ratios}
    return FibonacciLevels(
        high=high,
        low=low,
        range_val=rng,
        retracement=retracement,
        extension=extension,
    )


def fibonacci_from_df(
    df: pd.DataFrame,
    lookback: int = 60,
) -> Optional[FibonacciLevels]:
    """從 DataFrame 取最近 lookback 日的 High/Low 計算斐波那契"""
    if df is None or df.empty or len(df) < 2:
        return None
    sub = df.tail(lookback)
    high = float(pd.to_numeric(sub["High"], errors="coerce").max())
    low = float(pd.to_numeric(sub["Low"], errors="coerce").min())
    if np.isnan(high) or np.isnan(low) or high <= low:
        return None
    return compute_fibonacci(high, low)


# -------------------------
# 3. 隱含波動率 / 歷史波動率區間
# -------------------------
@dataclass
class VolatilityRange:
    """波動率區間（一倍標準差約 68% 機率）"""
    lower: float
    upper: float
    sigma: float
    dte: int
    is_implied: bool  # True=隱含波動率，False=歷史波動率


def compute_volatility_range(
    stock_price: float,
    sigma_annual: float,
    dte: int = 21,
) -> VolatilityRange:
    """
    一倍標準差區間：Stock ± (Stock × σ × sqrt(DTE/365))
    約 68% 機率落在區間內
    """
    if stock_price <= 0 or sigma_annual <= 0:
        return VolatilityRange(
            lower=stock_price,
            upper=stock_price,
            sigma=0.0,
            dte=dte,
            is_implied=False,
        )
    factor = sigma_annual * np.sqrt(dte / 365.0)
    half_range = stock_price * factor
    return VolatilityRange(
        lower=stock_price - half_range,
        upper=stock_price + half_range,
        sigma=sigma_annual,
        dte=dte,
        is_implied=False,
    )


def volatility_range_from_df(
    df: pd.DataFrame,
    stock_price: Optional[float] = None,
    dte: int = 21,
) -> Optional[VolatilityRange]:
    """從歷史資料計算波動率區間（yfinance 無 IV 時用歷史波動率）"""
    if df is None or df.empty:
        return None
    _, sigma = compute_gbm_params(df)
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    s0 = float(stock_price) if stock_price and stock_price > 0 else float(close.iloc[-1])
    return compute_volatility_range(s0, sigma, dte)
