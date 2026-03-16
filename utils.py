import math

import numpy as np
import pandas as pd


def to_scalar(value):
    if isinstance(value, pd.Series):
        if value.size == 1:
            return value.iloc[0]
        return value.iloc[-1]
    if isinstance(value, pd.DataFrame):
        if value.size == 1:
            return value.iloc[0, 0]
        return value.iloc[-1, -1]
    try:
        return float(value)
    except Exception:
        return value


def safe_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, (int, float, np.floating)):
            v = float(value)
            if math.isfinite(v):
                return v
        return None
    except Exception:
        return None


def parse_portfolio_lines(raw_text: str):
    """
    解析持股清單輸入：
    每行格式：symbol, avg_cost, shares
    - avg_cost / shares 可省略
    - 分隔符支援逗號或空白
    回傳 list[dict]: {"input":..., "symbol":..., "avg_cost":..., "shares":...}
    """
    items = []
    if not raw_text:
        return items

    for line in raw_text.splitlines():
        s = line.strip()
        if not s:
            continue

        s = s.replace("，", ",")
        parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
        if not parts:
            continue

        symbol = parts[0]
        avg_cost = None
        shares = None

        if len(parts) >= 2:
            try:
                avg_cost = float(parts[1])
            except Exception:
                avg_cost = None
        if len(parts) >= 3:
            try:
                shares = int(float(parts[2]))
            except Exception:
                shares = None

        items.append(
            {
                "input": line,
                "symbol": symbol,
                "avg_cost": avg_cost,
                "shares": shares,
            }
        )

    return items


def build_net_series(df_inst, names):
    if df_inst is None or df_inst.empty:
        return None
    if "name" not in df_inst.columns:
        return None
    sub = df_inst[df_inst["name"].isin(names)]
    if sub.empty:
        return None
    if "date" not in sub.columns or "buy" not in sub.columns or "sell" not in sub.columns:
        return None
    sub = sub.copy()
    sub["net"] = sub["buy"] - sub["sell"]
    return sub.groupby("date")["net"].sum()


def align_by_date(price_df: pd.DataFrame, net_s: pd.Series) -> pd.DataFrame:
    if price_df is None or price_df.empty or net_s is None or net_s.empty:
        return pd.DataFrame()
    p = price_df.copy()
    p["d"] = pd.to_datetime(p.index, errors="coerce").date
    p = p.dropna(subset=["d"]).set_index("d")
    n = net_s.copy()
    n.index = pd.to_datetime(n.index, errors="coerce").date
    n = n.dropna()
    out = p[["Close"]].join(n.rename("net"), how="inner")
    return out.dropna()


def align_net_series_to_price(price_df: pd.DataFrame, net_s: pd.Series) -> pd.Series:
    aligned = align_by_date(price_df, net_s)
    if aligned is None or aligned.empty:
        return None
    return aligned["net"]


def normalize_net_series_to_lot(net_s: pd.Series) -> pd.Series:
    if net_s is None or net_s.empty:
        return net_s
    s = pd.to_numeric(net_s, errors="coerce").dropna()
    if s.empty:
        return net_s
    median_abs = s.tail(10).abs().median()
    # 若量級偏大（股），轉成張（1 張 = 1000 股）
    if median_abs >= 100000:
        return net_s / 1000.0
    return net_s


def detect_net_unit_tag(net_s: pd.Series) -> str:
    if net_s is None or net_s.empty:
        return "NA"
    s = pd.to_numeric(net_s, errors="coerce").dropna()
    if s.empty:
        return "NA"
    median_abs = s.tail(10).abs().median()
    return "股(推測)" if median_abs >= 100000 else "張(推測)"
