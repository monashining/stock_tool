import numpy as np
import pandas as pd
import pandas_ta_classic as ta

from data_sources import fetch_fundamental_snapshot, load_market_index
from utils import safe_float


def check_volume_risk(df):
    if df is None or df.empty or len(df) < 2:
        return ""
    latest = df.iloc[-1]
    if "VolMA20" not in df.columns or pd.isna(df["VolMA20"].iloc[-1]):
        return ""

    is_high_vol_black = (latest["Volume"] > df["VolMA20"].iloc[-1] * 2) and (
        latest["Close"] < latest["Open"]
    )

    support_level = latest["Low"]
    if is_high_vol_black:
        return (
            f"**警訊：高檔爆量收黑**。雖然下方有接手買盤，但籌碼已趨向分散。"
            f"短期關鍵在於能否守住今日低點 {support_level:.2f}。"
            "若跌破，代表今日接手的買盤全數套牢，將轉為強大壓力。"
        )
    return ""


def compute_volume_sum_3d(df):
    if df is None or df.empty or "Volume" not in df.columns:
        return None
    vol = pd.to_numeric(df["Volume"].tail(3), errors="coerce").dropna()
    if vol.empty:
        return None
    # TW 成交量為股數，換算成張數（1 張 = 1000 股）以對齊法人張數單位
    return float(vol.sum() / 1000.0)


def compute_lock_thresholds(vol_sum_3d):
    if vol_sum_3d is None or not np.isfinite(vol_sum_3d) or vol_sum_3d <= 0:
        return 5000, 1000
    foreign_th = max(300, min(5000, vol_sum_3d * 0.05))
    trust_th = max(80, min(1000, vol_sum_3d * 0.03))
    return foreign_th, trust_th


def analyze_chip_flow(df, foreign_series, trust_series):
    if df is None or df.empty or (foreign_series is None and trust_series is None):
        return "籌碼面：資料不足，無法評估法人動向。"

    latest = df.iloc[-1]
    if "VolMA20" not in df.columns or pd.isna(df["VolMA20"].iloc[-1]):
        return "籌碼面：資料不足，無法評估法人動向。"

    recent_foreign = foreign_series.tail(3).sum() if foreign_series is not None else 0
    recent_trust = trust_series.tail(3).sum() if trust_series is not None else 0
    vol_sum_3d = compute_volume_sum_3d(df)
    foreign_lock_th, trust_lock_th = compute_lock_thresholds(vol_sum_3d)
    is_black_k = latest["Close"] < latest["Open"]
    is_high_vol = latest["Volume"] > df["VolMA20"].iloc[-1] * 1.5

    if is_high_vol and is_black_k:
        if recent_foreign > 0 or recent_trust > 0:
            return (
                "**大戶承接**：爆量收黑但法人逆勢買超，顯示大戶在支撐位接走恐慌盤。"
            )
        if recent_foreign < 0 and recent_trust < 0:
            return (
                "**散戶接刀**：爆量收黑且法人同步賣出，籌碼由大戶流向散戶，後市壓力沉重。"
            )
        return "**換手整理**：爆量收黑但法人動向不明顯，建議觀察量價與支撐表現。"

    if recent_foreign >= foreign_lock_th or recent_trust >= trust_lock_th:
        if recent_foreign >= foreign_lock_th and recent_trust >= trust_lock_th:
            return "**土洋超級鎖碼**：外資與投信同步強勢佈局，具備波段推升動能。"
        if recent_foreign >= foreign_lock_th:
            return "**外資大舉入駐**：近 3 日強勢佈局，具備波段推升動能。"
        return "**投信作帳發動**：近 3 日積極布局，有機會展開波段走勢。"

    return "**換手整理**：籌碼目前在震盪換手，無明顯單一力道。"


def compute_weighted_score(
    ema20,
    ema5,
    close_price,
    current_vol,
    avg_vol_5,
    bias_20_val,
    foreign_net_series,
    trust_net_series=None,
    vol_sum_3d=None,
    is_dangerous_vol=False,
):
    score = 0
    reasons = []

    trend_ok = (
        ema20 is not None
        and ema5 is not None
        and close_price is not None
        and not pd.isna(ema20)
        and not pd.isna(ema5)
        and not pd.isna(close_price)
        and close_price > ema20
        and ema5 > ema20
    )
    if trend_ok:
        score += 40
        reasons.append("趨勢 40%：Price > EMA20 且 EMA5 > EMA20 ✅")
    else:
        reasons.append("趨勢 40%：未達（Price > EMA20 且 EMA5 > EMA20）")

    volume_ok = False
    if avg_vol_5 is not None and not pd.isna(avg_vol_5) and avg_vol_5 > 0:
        if current_vol is not None and not pd.isna(current_vol):
            volume_ok = current_vol > avg_vol_5 and current_vol < avg_vol_5 * 2
            if volume_ok:
                score += 30
                reasons.append("量能 30%：當日量 > 5 日均量且不爆量 ✅")
            else:
                reasons.append("量能 30%：未達（需 > 5 日均量且 < 2x）")
        else:
            reasons.append("量能 30%：資料不足（當日量缺失）")
    else:
        reasons.append("量能 30%：資料不足（5 日均量缺失）")

    recent_foreign_sum = (
        foreign_net_series.tail(3).sum()
        if foreign_net_series is not None and len(foreign_net_series) >= 3
        else None
    )
    recent_trust_sum = (
        trust_net_series.tail(3).sum()
        if trust_net_series is not None and len(trust_net_series) >= 3
        else None
    )
    chip_ok = False
    if recent_foreign_sum is None and recent_trust_sum is None:
        reasons.append("籌碼 20%：資料不足")
    else:
        chip_ok = (recent_foreign_sum or 0) > 0 or (recent_trust_sum or 0) > 0
        if chip_ok:
            score += 20
            reasons.append("籌碼 20%：外資或投信近 3 日累計買超為正 ✅")
        else:
            reasons.append("籌碼 20%：外資與投信近 3 日累計買超未達")

        foreign_lock_th, trust_lock_th = compute_lock_thresholds(vol_sum_3d)
        foreign_lock = (recent_foreign_sum or 0) >= foreign_lock_th
        trust_lock = (recent_trust_sum or 0) >= trust_lock_th
        if foreign_lock and trust_lock:
            score = min(100, score + 25)
            reasons.append("籌碼加分：土洋超級鎖碼（+25）")
        elif foreign_lock:
            score = min(100, score + 10)
            reasons.append("籌碼加分：外資大舉入駐（+10）")
        elif trust_lock:
            score = min(100, score + 10)
            reasons.append("籌碼加分：投信作帳發動（+10）")

    bias_ok = bias_20_val is not None and not pd.isna(bias_20_val) and 0 <= bias_20_val <= 5
    if bias_ok:
        score += 10
        reasons.append("乖離 10%：Bias20 位於 0%~5% 安全區 ✅")
    else:
        reasons.append("乖離 10%：未達（0%~5% 安全區）")

    # -------- 進階風險濾網：高檔派發（避免「籌碼背離 + 強力多頭」互相打架） --------
    # 若 Bias20 明顯過熱，且外資連 3 日賣超，代表上漲「純度」不高，容易被洗或閃崩
    distribution_warning = False
    try:
        bias_overheat = (
            bias_20_val is not None
            and not pd.isna(bias_20_val)
            and float(bias_20_val) > 15.0
        )
    except Exception:
        bias_overheat = False

    foreign_sell_3d = False
    try:
        if foreign_net_series is not None and len(foreign_net_series) >= 3:
            f3 = pd.to_numeric(foreign_net_series.tail(3), errors="coerce").dropna()
            if len(f3) >= 3:
                foreign_sell_3d = bool((f3 < 0).all())
    except Exception:
        foreign_sell_3d = False

    if bias_overheat and foreign_sell_3d:
        distribution_warning = True
        score = max(0, int(score) - 40)
        reasons.append("派發扣分：Bias20 > 15% 且外資連賣 3 日（-40）")

    if is_dangerous_vol:
        score = max(0, score - 20)
        reasons.append("警訊扣分：高檔爆量黑K (-20)")

    flags = {
        "trend_ok": trend_ok,
        "volume_ok": volume_ok,
        "chip_ok": chip_ok,
        "bias_ok": bias_ok,
        "distribution_warning": distribution_warning,
    }
    return score, reasons, flags


def compute_indicators(df, include_vwap=True):
    df = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA60"] = df["Close"].rolling(60).mean()
    df["VolMA20"] = df["Volume"].rolling(20).mean()
    df["EMA5"] = ta.ema(df["Close"], length=5)
    df["EMA20"] = ta.ema(df["Close"], length=20)
    df["RSI14"] = ta.rsi(df["Close"], length=14)
    df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["Bias20"] = (df["Close"] - df["SMA20"]) / df["SMA20"] * 100
    df["BiasEMA20"] = (df["Close"] - df["EMA20"]) / df["EMA20"] * 100
    if include_vwap and {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        vwap = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()
        df["VWAP"] = vwap

        hhv20 = df["High"].rolling(20, min_periods=20).max()
        breakout = df["Close"] > hhv20.shift(1)
        df["AVWAP"] = np.nan
        if breakout.any():
            anchor_idx = breakout[breakout].index[-1]
            anchored = df.loc[anchor_idx:].copy()
            a_tp = (anchored["High"] + anchored["Low"] + anchored["Close"]) / 3
            a_vwap = (a_tp * anchored["Volume"]).cumsum() / anchored["Volume"].cumsum()
            df.loc[anchor_idx:, "AVWAP"] = a_vwap
    return df


def compute_buy_signals(df):
    df = df.copy()
    if len(df) < 60:
        df["BUY_GATE"] = False
        df["BUY_TRIGGER"] = False
        df["BUY_SIGNAL"] = False
        df["BUY_TRIGGER_TYPE"] = "NONE"
        df["BUY_NOTE"] = "資料不足（需至少 60 天）"
        df["EXEC_GUARD"] = False
        df["EXEC_BLOCK_REASON"] = "資料不足"
        df["Is_Dangerous_Volume"] = False
        return df

    # ---------- Gate（維持你原本邏輯） ----------
    trend_ok = (df["Close"] > df["SMA20"]) & (df["SMA20"] > df["SMA20"].shift(5))
    risk_ok = (df["Volume"] <= 1.5 * df["VolMA20"]) & (df["Bias20"].abs() <= 10)
    gate_ok = trend_ok & risk_ok

    # ---------- Trigger（新增 CONTINUATION） ----------
    trigger_pullback = (df["Bias20"].between(-1, 3)) & (
        df["Close"] > df["Close"].shift(1)
    )
    hhv20 = df["High"].rolling(20, min_periods=20).max()
    trigger_breakout = (df["Close"] > hhv20.shift(1)) & (
        df["Volume"] > 1.2 * df["VolMA20"]
    )
    hhv5 = df["High"].rolling(5, min_periods=5).max()
    trigger_continuation = (df["Bias20"].between(3, 10)) & (
        df["Close"] > hhv5.shift(1)
    )
    trigger_ok = trigger_pullback | trigger_breakout | trigger_continuation

    trigger_type = np.where(
        trigger_breakout,
        "BREAKOUT",
        np.where(
            trigger_pullback,
            "PULLBACK",
            np.where(trigger_continuation, "CONTINUATION", "NONE"),
        ),
    )

    # ---------- Execution Guard（防假突破/被洗掉） ----------
    k_range = (df["High"] - df["Low"]).replace(0, np.nan)
    close_pos = (df["Close"] - df["Low"]) / k_range
    breakout_close_strong = close_pos >= 0.6
    vol_ratio = df["Volume"] / df["VolMA20"].replace(0, np.nan)
    not_crazy_volume = vol_ratio <= 2.0
    not_too_hot = df["Bias20"] <= 9.5
    avwap_support = df["AVWAP"].isna() | (df["Close"] >= df["AVWAP"])
    guard_ok = np.where(
        (trigger_type == "BREAKOUT") | (trigger_type == "CONTINUATION"),
        (breakout_close_strong & not_crazy_volume & not_too_hot & avwap_support),
        avwap_support,
    )

    # ---------- 輸出欄位 ----------
    df["BUY_GATE"] = gate_ok
    df["BUY_TRIGGER"] = trigger_ok
    df["BUY_TRIGGER_TYPE"] = trigger_type
    df["EXEC_GUARD"] = guard_ok
    df["BUY_SIGNAL"] = gate_ok & trigger_ok & guard_ok

    # ---------- Guard 擋下原因（可累積，不覆蓋） ----------
    df["EXEC_BLOCK_REASON"] = ""

    strict_mask = df["BUY_TRIGGER_TYPE"].isin(["BREAKOUT", "CONTINUATION"])

    reason_close = np.where(
        ~breakout_close_strong, "收盤不夠強（上影偏長/假突破風險）", ""
    )
    reason_vol = np.where(~not_crazy_volume, "量能過熱（>2.0x 20日均量）", "")
    reason_hot = np.where(~not_too_hot, "乖離接近過熱上限（>9.5%）", "")

    reason_avwap = np.where(
        ~avwap_support, "跌回 Anchored VWAP 下方（成本線失守）", ""
    )

    tmp = pd.DataFrame(
        {
            "a": pd.Series(reason_close, index=df.index).where(strict_mask, ""),
            "b": pd.Series(reason_vol, index=df.index).where(strict_mask, ""),
            "c": pd.Series(reason_hot, index=df.index).where(strict_mask, ""),
            "d": pd.Series(reason_avwap, index=df.index),
        }
    )

    df.loc[(~df["EXEC_GUARD"]), "EXEC_BLOCK_REASON"] = tmp.apply(
        lambda r: "Guard："
        + "；".join([x for x in [r["a"], r["b"], r["c"], r["d"]] if isinstance(x, str) and x]),
        axis=1,
    )

    df["BUY_NOTE"] = ""
    df["Is_Dangerous_Volume"] = (
        (df["Volume"] > df["VolMA20"] * 2)
        & (df["Close"] < df["Open"])
        & (df["Bias20"] > 10)
    )
    return df


def estimate_target_range(df, symbol):
    if df is None or df.empty:
        return None
    latest = df.iloc[-1]
    close = safe_float(latest.get("Close"))
    atr14 = safe_float(latest.get("ATR14"))
    if close is None:
        return None

    def rolling_high(n):
        if len(df) >= n:
            return safe_float(df["High"].tail(n).max())
        return None

    def rolling_low(n):
        if len(df) >= n:
            return safe_float(df["Low"].tail(n).min())
        return None

    r20 = rolling_high(20)
    r60 = rolling_high(60)
    r252 = rolling_high(252) if len(df) >= 252 else None
    l20 = rolling_low(20)

    tp_atr_2 = close + 2 * atr14 if atr14 is not None else None
    tp_atr_3 = close + 3 * atr14 if atr14 is not None else None
    tp_mm = None
    if r20 is not None and l20 is not None and close >= 0.98 * r20:
        tp_mm = r20 + (r20 - l20)

    tech_candidates_high = [x for x in [r20, r60, r252, tp_atr_3, tp_mm] if x is not None]
    tech_candidates_mid = [x for x in [r20, r60, tp_atr_2] if x is not None]
    tp_tech_high = max(tech_candidates_high) if tech_candidates_high else None
    tp_tech_mid = np.median(tech_candidates_mid) if tech_candidates_mid else None
    tp_tech_low = min([x for x in [close, r20, r60] if x is not None])

    f = fetch_fundamental_snapshot(symbol)
    eps = f.get("eps_ttm") or f.get("eps_fwd")
    tp_fund_low = tp_fund_mid = tp_fund_high = None
    fund_enabled = False
    if eps is not None and eps > 0:
        pe_now = f.get("pe_ttm") or f.get("pe_fwd")
        if pe_now is None:
            pe_now = close / eps if eps != 0 else None
        if pe_now is not None and pe_now > 0:
            pe_low = max(5, 0.8 * pe_now)
            pe_mid = max(5, 1.0 * pe_now)
            pe_high = min(60, 1.2 * pe_now)
            tp_fund_low = eps * pe_low
            tp_fund_mid = eps * pe_mid
            tp_fund_high = eps * pe_high
            fund_enabled = True

    tech_enabled = tp_tech_mid is not None and tp_tech_high is not None
    if fund_enabled and tech_enabled:
        tp_low = min(tp_fund_low, tp_tech_low)
        tp_mid = float(np.median([tp_fund_mid, tp_tech_mid]))
        tp_high = max(tp_fund_high, tp_tech_high)
        confidence = "HIGH"
    elif tech_enabled:
        tp_low, tp_mid, tp_high = tp_tech_low, float(tp_tech_mid), tp_tech_high
        confidence = "MED"
    elif fund_enabled:
        tp_low, tp_mid, tp_high = tp_fund_low, tp_fund_mid, tp_fund_high
        confidence = "MED"
    else:
        tp_low = tp_mid = tp_high = None
        confidence = "LOW"

    return {
        "tp_low": tp_low,
        "tp_mid": tp_mid,
        "tp_high": tp_high,
        "confidence": confidence,
        "model_flags": {
            "fund_enabled": fund_enabled,
            "tech_enabled": tech_enabled,
            "mm_enabled": tp_mm is not None,
            "atr_enabled": atr14 is not None,
        },
        "debug": {
            "close": close,
            "atr14": atr14,
            "R20": r20,
            "R60": r60,
            "R252": r252,
            "tp_atr_2": tp_atr_2,
            "tp_atr_3": tp_atr_3,
            "tp_mm": tp_mm,
            "eps_used": eps,
            "pe_now": f.get("pe_ttm") or f.get("pe_fwd") or (close / eps if eps else None),
        },
    }


def compute_risk_metrics(
    df_stock: pd.DataFrame, df_mkt: pd.DataFrame, rf_annual: float = 0.015
):
    """
    計算投資人風險指標（Beta / Annualized Volatility / Sharpe）
    - df_stock / df_mkt：至少包含 Close 或 Adj Close
    - rf_annual：年化無風險利率（預設 1.5%，你可改成台灣短期利率）
    """
    if df_stock is None or df_stock.empty:
        return None
    if df_mkt is None or df_mkt.empty:
        return None

    s = df_stock.copy()
    m = df_mkt.copy()

    # 以 Adj Close 優先（若無則用 Close）
    s_price_col = "Adj Close" if "Adj Close" in s.columns else "Close"
    m_price_col = "Adj Close" if "Adj Close" in m.columns else "Close"

    s = s[[s_price_col]].rename(columns={s_price_col: "P"})
    m = m[[m_price_col]].rename(columns={m_price_col: "M"})

    # 對齊日期
    aligned = s.join(m, how="inner")
    aligned = aligned.dropna()
    if len(aligned) < 60:
        return {"note": "資料不足（需至少 60 個交易日）"}

    # 日報酬
    aligned["rP"] = aligned["P"].pct_change()
    aligned["rM"] = aligned["M"].pct_change()
    aligned = aligned.dropna()
    if len(aligned) < 60:
        return {"note": "資料不足（報酬序列不足 60 日）"}

    rp = aligned["rP"]
    rm = aligned["rM"]

    # Beta = cov(rP, rM) / var(rM)
    cov = np.cov(rp, rm)[0, 1]
    var = np.var(rm)
    beta = cov / var if var != 0 else None

    # 年化波動率（以 252 個交易日）
    vol_annual = rp.std() * np.sqrt(252)

    # 年化報酬
    mu_annual = rp.mean() * 252

    # Sharpe = (mu - rf) / vol
    sharpe = (mu_annual - rf_annual) / vol_annual if vol_annual != 0 else None

    # Max Drawdown
    cum = (1 + rp).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_drawdown = float(dd.min())

    return {
        "beta": beta,
        "vol_annual": vol_annual,
        "mu_annual": mu_annual,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "window_days": int(len(rp)),
        "rf_annual": rf_annual,
    }


def check_global_buy_strategy(symbol, df, foreign_series):
    """
    全股票通用買進策略
    """
    mkt, _ = load_market_index(symbol) if symbol.endswith(".TW") else (pd.DataFrame(), None)

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    buy_note = str(latest.get("BUY_NOTE", "")).strip()
    if buy_note:
        return False, f"{buy_note}（暫停評估）", "NONE", {
            "k_ratio": 0,
            "vol_ratio": 0,
            "mkt_ret": None,
        }

    k_range = latest["High"] - latest["Low"]
    k_body = abs(latest["Close"] - latest["Open"])
    k_ratio = k_body / k_range if k_range != 0 else 0

    vol_ma20 = df["Volume"].rolling(20).mean().iloc[-1]
    vol_ratio = latest["Volume"] / vol_ma20 if vol_ma20 != 0 else 0

    allow_buy = False
    reason = "不符合買進條件 (觀望)"
    status_type = "NONE"
    mkt_ret = None
    mkt_2d_ret = None
    if len(mkt) >= 3:
        mkt_ret = (mkt["Close"].iloc[-1] / mkt["Close"].iloc[-2]) - 1
        mkt_2d_ret = (mkt["Close"].iloc[-1] / mkt["Close"].iloc[-3]) - 1

    gate_ok = bool(latest.get("BUY_GATE", False))
    trigger_ok = bool(latest.get("BUY_TRIGGER", False))
    trigger_type = str(latest.get("BUY_TRIGGER_TYPE", "NONE"))

    if gate_ok and trigger_ok:
        allow_buy = True
        status_type = "TREND"
        reason = f"情境 1：Gate/Trigger 通過（{trigger_type}）。"
    else:
        is_market_panic = (
            mkt_ret is not None
            and mkt_2d_ret is not None
            and (mkt_ret <= -0.02 or mkt_2d_ret <= -0.035)
        )

        stock_ret = (latest["Close"] / prev["Close"]) - 1
        is_stronger_than_mkt = mkt_ret is not None and stock_ret > mkt_ret

        foreign_ok = True
        if symbol.endswith(".TW") and foreign_series is not None and len(df) > 0:
            fidx = pd.to_datetime(foreign_series.index, errors="coerce")
            fidx = fidx[~pd.isna(fidx)]
            f = foreign_series.copy()
            f.index = fidx
            idx = df.index.intersection(f.index)
            if len(idx) >= 3:
                foreign_ok = f.loc[idx].tail(3).sum() >= 0
        hold_support = latest["Close"] >= latest["SMA20"]

        if is_market_panic and is_stronger_than_mkt and foreign_ok and hold_support:
            allow_buy = True
            status_type = "PANIC"
            reason = "情境 2：恐慌承接。大盤重挫但個股展現支撐，籌碼未散。"

    metrics = {
        "k_ratio": k_ratio,
        "vol_ratio": vol_ratio,
        "mkt_ret": mkt_ret,
    }
    return allow_buy, reason, status_type, metrics
