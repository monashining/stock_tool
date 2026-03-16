from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import math

import pandas as pd


def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def ema_alpha(window: int) -> float:
    """
    EMA alpha = 2 / (N + 1)
    """
    try:
        w = int(window)
    except Exception:
        w = 5
    w = max(1, w)
    return 2.0 / (float(w) + 1.0)


def project_next_ema(*, ema_today: Any, close_next: Any, window: int) -> Optional[float]:
    """
    根據 EMA 遞推公式，推算「下一根」EMA：
    EMA(t+1) = alpha * Close(t+1) + (1-alpha) * EMA(t)
    """
    e = _to_float(ema_today)
    c = _to_float(close_next)
    if e is None or c is None:
        return None
    a = ema_alpha(window)
    return float(a * float(c) + (1.0 - a) * float(e))


def _pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ema_today_from_df(df: pd.DataFrame, *, window: int) -> Optional[float]:
    """
    取得「今日（最後一根）」EMA 值。
    - 若 df 已有 EMA{N} 欄位，優先使用
    - 否則使用 Close 即時計算
    """
    if df is None or df.empty or "Close" not in df.columns:
        return None

    col = _pick_first_existing_col(df, [f"EMA{int(window)}", f"ema{int(window)}"])
    if col:
        v = _to_float(pd.to_numeric(df[col].iloc[-1], errors="coerce"))
        if v is not None:
            return float(v)

    close_n = pd.to_numeric(df["Close"], errors="coerce")
    if close_n.dropna().empty:
        return None
    ema_s = close_n.ewm(span=max(1, int(window)), adjust=False).mean()
    v = _to_float(ema_s.iloc[-1])
    return float(v) if v is not None else None


@dataclass(frozen=True)
class TomorrowGuard:
    """
    明日保命價（以「收盤不破 EMA{N}」為核心的關鍵價位）

    重點結論（數學上是精準的）：
    - 若你用「收盤價 vs 當日 EMA{N}」作為破線判定
    - 那麼「明日是否收盤破線」的臨界價，其實就是「今日 EMA{N}」
      （因為 EMA(t+1) 介於 Close(t+1) 與 EMA(t) 之間）
    """

    as_of: Any
    window: int
    ema_today: float
    break_close: float
    buffer_pct: float
    guard_close: float
    alpha: float


def calc_tomorrow_guard(
    *,
    ema_today: Any,
    window: int = 5,
    buffer_pct: float = 1.5,
    as_of: Any = None,
) -> Optional[TomorrowGuard]:
    """
    用「今日 EMA」計算「明日保命價」：
    - break_close：明日收盤 < break_close ⇒ 明日收盤落在 EMA 下方（破線）
    - guard_close：在 break_close 基礎上加 buffer（更保守的警示價）
    """
    e = _to_float(ema_today)
    if e is None:
        return None
    try:
        w = int(window)
    except Exception:
        w = 5
    w = max(1, w)
    try:
        b = float(buffer_pct or 0.0)
    except Exception:
        b = 0.0
    b = max(0.0, float(b))

    break_close = float(e)
    guard_close = float(e) * (1.0 + float(b) / 100.0)
    return TomorrowGuard(
        as_of=as_of,
        window=w,
        ema_today=float(e),
        break_close=break_close,
        buffer_pct=float(b),
        guard_close=guard_close,
        alpha=float(ema_alpha(w)),
    )


def calc_tomorrow_guard_from_df(
    df: pd.DataFrame,
    *,
    window: int = 5,
    buffer_pct: float = 1.5,
) -> Optional[TomorrowGuard]:
    if df is None or df.empty:
        return None
    ema_today = ema_today_from_df(df, window=window)
    as_of = None
    try:
        as_of = df.index[-1]
    except Exception:
        as_of = None
    return calc_tomorrow_guard(
        ema_today=ema_today,
        window=window,
        buffer_pct=buffer_pct,
        as_of=as_of,
    )

