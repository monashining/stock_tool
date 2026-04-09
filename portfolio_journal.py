"""
持倉交易日誌（MVP）
----------------
只追蹤「有買進」的部位：一筆 trade 主檔 + 持有期間每日 journal（不可覆寫歷史列）。

目錄：data/portfolio/
  - trades.csv
  - trade_daily_journal.csv

trade_daily_journal 的欄位 `date`：該列快照所對應的**日線最後一根 bar 之日期**（持有中日日記；非交易日通常不新增列）。

之後可改為 parquet；本版以 CSV 為預設（免依賴 pyarrow）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Optional
import numpy as np
import pandas as pd


@dataclass
class JournalUpdateReport:
    """update_open_trades_daily 的結果；主控台／持股清單用於顯示明細。"""

    updated: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)

# ---------------------------------------------------------------------------
# 路徑
# ---------------------------------------------------------------------------


def default_data_dir() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "data", "portfolio")


def trades_path(data_dir: str | None = None) -> str:
    return os.path.join(data_dir or default_data_dir(), "trades.csv")


def journal_path(data_dir: str | None = None) -> str:
    return os.path.join(data_dir or default_data_dir(), "trade_daily_journal.csv")


def ensure_dir(data_dir: str | None = None) -> str:
    d = data_dir or default_data_dir()
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Schema（與欄位順序，利於 append / 讀取）
# ---------------------------------------------------------------------------

TRADE_COLUMNS = [
    "trade_id",
    "symbol",
    "entry_date",
    "entry_price",
    "shares",
    "entry_amount",
    "entry_reason",
    "entry_regime",
    "entry_confidence",
    "locked_tp_low",
    "locked_tp_high",
    "dynamic_tp_low_at_entry",
    "dynamic_tp_high_at_entry",
    "valuation_low_at_entry",
    "valuation_high_at_entry",
    "stop_loss_price",
    "status",
    "exit_date",
    "exit_price",
    "exit_reason",
    "realized_pnl",
    "realized_return_pct",
    "notes",
]

JOURNAL_COLUMNS = [
    "trade_id",
    "date",
    "symbol",
    "close",
    "high",
    "low",
    "volume",
    "sma20",
    "sma60",
    "ema5",
    "ema20",
    "atr14",
    "rsi14",
    "bias20",
    "trend_ok",
    "risk_ok",
    "buy_gate",
    "buy_trigger",
    "buy_signal",
    "buy_trigger_type",
    "regime_type",
    "dynamic_tp_low",
    "dynamic_tp_high",
    "valuation_low",
    "valuation_high",
    "locked_tp_low",
    "locked_tp_high",
    "unrealized_pnl",
    "return_since_entry_pct",
    "distance_to_locked_high_pct",
    "distance_to_dynamic_high_pct",
    "drawdown_from_peak_pct",
    "hit_locked_target_today",
    "hit_dynamic_target_today",
    "hit_stop_today",
    "reason_summary",
]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _symbol_key(symbol: str) -> str:
    return (symbol or "").strip().replace(".", "")


def _trade_date_str(d: Any) -> str:
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    return str(d).strip()[:10]


def _normalize_symbol(s: str) -> str:
    return (s or "").strip()


def _last_bar_date_str(df: pd.DataFrame) -> str:
    idx = df.index[-1]
    if hasattr(idx, "date"):
        try:
            return idx.date().isoformat()
        except Exception:
            pass
    return pd.Timestamp(idx).date().isoformat()


def last_bar_date_from_ohlcv_df(df: pd.DataFrame | None) -> Optional[str]:
    """OHLCV DataFrame 最後一根 bar 日期 yyyy-mm-dd；無資料則 None。"""
    if df is None or df.empty:
        return None
    return _last_bar_date_str(df)


def _calendar_days_held(entry_date_s: str, asof_date_s: str | None) -> float:
    """進場日至行情最後 bar 日的日曆天數（>=0）；缺 asof 則 NaN。"""
    if not entry_date_s or not asof_date_s:
        return float("nan")
    try:
        e = pd.Timestamp(str(entry_date_s).strip()[:10]).normalize()
        a = pd.Timestamp(str(asof_date_s).strip()[:10]).normalize()
        return float(max(0, (a - e).days))
    except Exception:
        return float("nan")


def journal_stale_flag_value(mkt_bar_date_s: str, last_journal_date_s: str) -> Any:
    """
    數值層：journal 是否落後「行情最後 bar 日」。

    - True：行情 bar 日晚於最後一筆 journal（宜補寫日誌）。
    - False：兩日相同，或 journal 較新（異常但視為未落後）。
    - NaN：任一日期缺失或無法解析。
    """
    m = (mkt_bar_date_s or "").strip()[:10]
    j = (last_journal_date_s or "").strip()[:10]
    if not m or not j:
        return np.nan
    try:
        tm = pd.Timestamp(m).normalize()
        tj = pd.Timestamp(j).normalize()
    except Exception:
        return np.nan
    if tm > tj:
        return True
    return False


def _journal_regime_ui(val: Any) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none"):
        return "—"
    return s


def _journal_keys_typed(journal_df: pd.DataFrame, trade_ids: set[str]) -> set[tuple[str, str]]:
    """(trade_id, date yyyy-mm-dd) 已存在的 journal 鍵。"""
    if journal_df.empty or not trade_ids:
        return set()
    j = journal_df[journal_df["trade_id"].astype(str).isin(trade_ids)].copy()
    if j.empty:
        return set()
    ds = j["date"].astype(str).str.strip().str[:10]
    return set(zip(j["trade_id"].astype(str), ds))


def _latest_journal_row_per_trade(journal_df: pd.DataFrame) -> dict[str, pd.Series]:
    """每個 trade_id 取日期最新的一列 journal。"""
    if journal_df is None or journal_df.empty:
        return {}
    out: dict[str, pd.Series] = {}
    for tid, grp in journal_df.groupby(journal_df["trade_id"].astype(str)):
        g = grp.copy()
        g["_dsort"] = pd.to_datetime(g["date"].astype(str).str.strip().str[:10], errors="coerce")
        g = g.sort_values("_dsort")
        out[str(tid)] = g.iloc[-1]
    return out


def _journal_float(val: Any) -> float:
    if val is None or (isinstance(val, str) and val.strip() in ("", "nan", "NaN", "None")):
        return float("nan")
    try:
        if isinstance(val, float) and np.isnan(val):
            return float("nan")
    except Exception:
        pass
    try:
        return float(val)
    except Exception:
        return float("nan")


def _journal_bool_ui(val: Any) -> str:
    """總表顯示：是 / 否 / —"""
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return "—"
    if isinstance(val, str):
        lo = val.strip().lower()
        if lo in ("true", "1", "yes", "y", "t"):
            return "是"
        if lo in ("false", "0", "no", "n", "f"):
            return "否"
    try:
        if isinstance(val, float) and np.isnan(val):
            return "—"
    except Exception:
        pass
    try:
        return "是" if bool(val) else "否"
    except Exception:
        return "—"


def _distance_to_high_pct(high: float, close: float) -> float:
    """(high - close) / close * 100，與 journal 欄位定義一致。"""
    if not np.isfinite(high) or not np.isfinite(close) or close == 0:
        return float("nan")
    return ((float(high) - float(close)) / float(close)) * 100.0


# 彙總表「接近鎖定目標」距離上限（%）；僅在尚未已達時與 target_reached_flag 併用。
NEAR_LOCKED_TARGET_THRESHOLD_PCT = 3.0

OPEN_TRADES_SUMMARY_COLS: list[str] = [
    "symbol",
    "trade_id",
    "entry_date",
    "days_held",
    "market_last_bar_date",
    "entry_price",
    "shares",
    "latest_close",
    "return_pct",
    "unrealized_pnl",
    "locked_tp_high",
    "dynamic_tp_high",
    "distance_to_locked_high_pct",
    "target_reached_flag",
    "near_locked_target_flag",
    "distance_to_dynamic_high_pct",
    "drawdown_from_peak_pct",
    "trend_ok",
    "risk_ok",
    "regime_type",
    "last_journal_bar_date",
    "journal_stale_flag",
]


def summarize_open_trades_for_ui(
    latest_close_by_symbol: dict[str, float | None],
    *,
    market_last_bar_date_by_symbol: dict[str, str | None] | None = None,
    data_dir: str | None = None,
) -> pd.DataFrame:
    """
    組 Open Trades 總表：績效 + 目標距離（decision 用）。

    - locked_tp_high：trade master；dynamic / 回撤 / 趨勢 / risk / regime：最新 journal。
    - 距離% 用 latest_close 重算；**持有天數**＝進場日→`market_last_bar_date`（宜傳行情最後 bar）。
    - `market_last_bar_date_by_symbol` 若缺該 symbol，持有天數與行情日欄為空。
    - trend_ok / risk_ok：是、否、—；regime_type：journal 原文或 —。
    - journal_stale_flag：True=journal 落後行情 bar、False=已對齊、NaN=無法比對。
    - target_reached_flag / near_locked_target_flag：供 UI 合併為「目標狀態」；
      已達＝latest_close ≥ locked_tp_high；接近＝未已達且 0 < 距鎖定% ≤ NEAR_LOCKED_TARGET_THRESHOLD_PCT。
    """
    empty = pd.DataFrame(columns=OPEN_TRADES_SUMMARY_COLS)
    ot = list_open_trades(data_dir)
    if ot.empty:
        return empty

    j = load_journal(data_dir)
    last_row_by_tid = _latest_journal_row_per_trade(j)

    last_bar_date: dict[str, str] = {}
    for tid, row in last_row_by_tid.items():
        last_bar_date[tid] = str(row.get("date", "")).strip()[:10]

    mbd_map = market_last_bar_date_by_symbol or {}

    rows: list[dict[str, Any]] = []
    for _, r in ot.iterrows():
        sym = _normalize_symbol(str(r["symbol"]))
        tid = str(r["trade_id"])
        entry_ds = str(r["entry_date"]).strip()[:10]
        mkt_bar = mbd_map.get(sym)
        mkt_bar_s = str(mkt_bar).strip()[:10] if mkt_bar else ""
        days_h = _calendar_days_held(entry_ds, mkt_bar_s if mkt_bar_s else None)

        try:
            ep = float(r["entry_price"])
        except Exception:
            ep = np.nan
        try:
            sh = int(float(r["shares"] or 0))
        except Exception:
            sh = 0
        raw_lc = latest_close_by_symbol.get(sym)
        if raw_lc is None:
            lc = np.nan
        else:
            try:
                lc = float(raw_lc)
            except Exception:
                lc = np.nan
        ur = np.nan
        rp = np.nan
        if sh and np.isfinite(ep) and np.isfinite(lc):
            ur = (lc - ep) * sh
            rp = ((lc / ep) - 1.0) * 100.0 if ep else np.nan

        locked_hi = _journal_float(r.get("locked_tp_high"))

        jrow = last_row_by_tid.get(tid)
        dyn_hi = float("nan")
        dd_peak = float("nan")
        trend_s = "—"
        risk_s = "—"
        regime_s = "—"
        if jrow is not None:
            dyn_hi = _journal_float(jrow.get("dynamic_tp_high"))
            dd_peak = _journal_float(jrow.get("drawdown_from_peak_pct"))
            trend_s = _journal_bool_ui(jrow.get("trend_ok"))
            risk_s = _journal_bool_ui(jrow.get("risk_ok"))
            regime_s = _journal_regime_ui(jrow.get("regime_type"))

        d_lock = _distance_to_high_pct(locked_hi, lc) if np.isfinite(locked_hi) else float("nan")
        d_dyn = _distance_to_high_pct(dyn_hi, lc) if np.isfinite(dyn_hi) else float("nan")

        if np.isfinite(locked_hi) and np.isfinite(lc):
            target_reached = bool(float(lc) >= float(locked_hi))
        else:
            target_reached = np.nan

        if np.isfinite(d_lock):
            _d = float(d_lock)
            if 0.0 < _d <= NEAR_LOCKED_TARGET_THRESHOLD_PCT:
                near_locked = True
            else:
                near_locked = False
        else:
            near_locked = np.nan

        jlast = last_bar_date.get(tid, "")
        stale_flag = journal_stale_flag_value(mkt_bar_s, jlast)

        rows.append(
            {
                "symbol": sym,
                "trade_id": tid,
                "entry_date": entry_ds,
                "days_held": days_h,
                "market_last_bar_date": mkt_bar_s,
                "entry_price": ep,
                "shares": sh,
                "latest_close": lc,
                "return_pct": rp,
                "unrealized_pnl": ur,
                "locked_tp_high": locked_hi,
                "dynamic_tp_high": dyn_hi,
                "distance_to_locked_high_pct": d_lock,
                "target_reached_flag": target_reached,
                "near_locked_target_flag": near_locked,
                "distance_to_dynamic_high_pct": d_dyn,
                "drawdown_from_peak_pct": dd_peak,
                "trend_ok": trend_s,
                "risk_ok": risk_s,
                "regime_type": regime_s,
                "last_journal_bar_date": jlast,
                "journal_stale_flag": stale_flag,
            }
        )
    out = pd.DataFrame(rows)
    return out[OPEN_TRADES_SUMMARY_COLS]


def _next_trade_seq(trades_df: pd.DataFrame, symbol: str, entry_date: str) -> int:
    if trades_df is None or trades_df.empty:
        return 1
    mask = (trades_df["symbol"] == symbol) & (trades_df["entry_date"] == entry_date)
    n = int(mask.sum())
    return n + 1


def load_trades(data_dir: str | None = None) -> pd.DataFrame:
    p = trades_path(data_dir)
    if not os.path.isfile(p):
        return pd.DataFrame(columns=TRADE_COLUMNS)
    df = pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""])
    for c in TRADE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[TRADE_COLUMNS]


def load_journal(data_dir: str | None = None) -> pd.DataFrame:
    p = journal_path(data_dir)
    if not os.path.isfile(p):
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    df = pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""])
    for c in JOURNAL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[JOURNAL_COLUMNS]


def _save_trades(df: pd.DataFrame, data_dir: str | None = None) -> None:
    ensure_dir(data_dir)
    df[TRADE_COLUMNS].to_csv(trades_path(data_dir), index=False, encoding="utf-8-sig")


def _append_journal_rows(rows: list[dict[str, Any]], data_dir: str | None = None) -> None:
    ensure_dir(data_dir)
    p = journal_path(data_dir)
    new_df = pd.DataFrame(rows)
    for c in JOURNAL_COLUMNS:
        if c not in new_df.columns:
            new_df[c] = np.nan
    new_df = new_df[JOURNAL_COLUMNS]
    if os.path.isfile(p):
        old = pd.read_csv(p, encoding="utf-8-sig")
        out = pd.concat([old, new_df], ignore_index=True)
        # 避免同日重複 append：可依 trade_id+date 去重（保留最後一筆）
        out = out.drop_duplicates(subset=["trade_id", "date"], keep="last")
    else:
        out = new_df
    out.to_csv(p, index=False, encoding="utf-8-sig")


def list_open_trades(data_dir: str | None = None) -> pd.DataFrame:
    t = load_trades(data_dir)
    if t.empty:
        return t
    return t[t["status"].astype(str).str.upper() == "OPEN"].copy()


def _infer_regime(latest: pd.Series, df: pd.DataFrame) -> str:
    try:
        c = float(latest.get("Close"))
        s20 = latest.get("SMA20")
        s20 = float(s20) if s20 is not None and pd.notna(s20) else None
        if s20 and c > s20 * 1.02:
            return "TREND"
        if s20 and c < s20 * 0.98:
            return "WEAK"
    except Exception:
        pass
    return "NEUTRAL"


def _reason_summary_row(latest: pd.Series) -> str:
    parts: list[str] = []
    try:
        if bool(latest.get("BUY_GATE")):
            parts.append("Gate通過")
        else:
            parts.append("Gate未通過")
        tt = str(latest.get("BUY_TRIGGER_TYPE", "") or "")
        if tt and tt != "NONE":
            parts.append(f"Trigger={tt}")
        if bool(latest.get("BUY_SIGNAL")):
            parts.append("BUY_SIGNAL=True")
        br = str(latest.get("EXEC_BLOCK_REASON", "") or "").strip()
        if br:
            parts.append(br[:80])
    except Exception:
        pass
    return "｜".join(parts) if parts else ""


def _df_through_asof(df: pd.DataFrame, asof_date: str | None) -> pd.DataFrame:
    """若指定 asof_date，只保留 index <= 該日之資料列（用於進場日非最新 bar 時）。"""
    if df is None or df.empty or not asof_date:
        return df
    try:
        ad = pd.Timestamp(str(asof_date)[:10]).normalize()
    except Exception:
        return df
    idx = pd.to_datetime(df.index, errors="coerce")
    if idx.isna().all():
        return df
    inorm = idx.normalize()
    sub = df.loc[inorm <= ad]
    if sub.empty:
        try:
            pos = int(inorm.searchsorted(ad, side="right")) - 1
            if pos >= 0:
                sub = df.iloc[: pos + 1]
        except Exception:
            pass
    return sub if not sub.empty else df


def build_journal_snapshot(
    trade_row: pd.Series,
    df: pd.DataFrame,
    symbol: str,
    *,
    asof_date: str | None = None,
) -> dict[str, Any]:
    """
    由已含技術指標 + BUY_* 的 df 之最後一列，組出 journal 一筆。
    trade_row：主檔該 trade 的 Series（需有 entry_price, shares, locked_tp_*）
    若傳入 asof_date，會先裁切 df 至該日（含）再以最後一列為快照。
    """
    if df is None or df.empty:
        raise ValueError("df 不可為空")

    from analysis import estimate_target_range

    df = _df_through_asof(df.copy(), asof_date)
    latest = df.iloc[-1]
    idx = df.index[-1]
    d = asof_date or (idx.isoformat()[:10] if hasattr(idx, "isoformat") else str(idx)[:10])

    entry_price = float(trade_row.get("entry_price") or 0)
    shares = int(float(trade_row.get("shares") or 0))
    locked_lo = trade_row.get("locked_tp_low")
    locked_hi = trade_row.get("locked_tp_high")
    try:
        locked_lo_f = float(locked_lo) if locked_lo not in (None, "", "nan") else None
    except Exception:
        locked_lo_f = None
    try:
        locked_hi_f = float(locked_hi) if locked_hi not in (None, "", "nan") else None
    except Exception:
        locked_hi_f = None

    tp = estimate_target_range(df, symbol)

    close = float(latest["Close"])
    high = float(latest["High"])
    low = float(latest["Low"])
    vol = float(latest.get("Volume", np.nan))

    def _f(col: str) -> Optional[float]:
        v = latest.get(col)
        try:
            return float(v) if v is not None and pd.notna(v) else None
        except Exception:
            return None

    dyn_lo = tp.get("tp_low") if tp else None
    dyn_hi = tp.get("tp_high") if tp else None
    val_lo = tp.get("tp_low") if tp else None  # 與估值 blend 同 dict；進階可拆欄
    val_hi = tp.get("tp_high") if tp else None

    # 邏輯上 valuation 若要以基本面為主，estimate_target_range 已內含；debug 可另存
    if tp and tp.get("model_flags", {}).get("fund_enabled"):
        val_lo = tp.get("tp_low")
        val_hi = tp.get("tp_high")

    try:
        if len(df) >= 6 and "SMA20" in df.columns:
            c = float(latest["Close"])
            s20 = float(latest["SMA20"])
            s20_prev = float(df["SMA20"].iloc[-6])
            trend_ok = bool(c > s20 and s20 > s20_prev)
        else:
            trend_ok = bool(latest.get("BUY_GATE"))
    except Exception:
        trend_ok = bool(latest.get("BUY_GATE"))
    try:
        risk_ok = bool(latest.get("BUY_GATE"))
    except Exception:
        risk_ok = False
    try:
        raw_gate = latest.get("BUY_GATE")
        buy_gate = bool(raw_gate) if pd.notna(raw_gate) else False
    except Exception:
        buy_gate = False
    buy_trigger = bool(latest.get("BUY_TRIGGER")) if pd.notna(latest.get("BUY_TRIGGER")) else False
    buy_signal = bool(latest.get("BUY_SIGNAL")) if pd.notna(latest.get("BUY_SIGNAL")) else False
    btt = str(latest.get("BUY_TRIGGER_TYPE", "") or "NONE")

    unrealized = (close - entry_price) * shares if shares and entry_price else None
    ret_pct = ((close / entry_price) - 1.0) * 100.0 if entry_price else None
    dist_locked_hi = (
        ((locked_hi_f / close) - 1.0) * 100.0 if locked_hi_f and close else None
    )
    dist_dyn_hi = (
        ((float(dyn_hi) / close) - 1.0) * 100.0 if dyn_hi is not None and close else None
    )

    hh = float(df["High"].loc[: idx].max()) if len(df) > 0 else high
    dd_peak = ((close / hh) - 1.0) * 100.0 if hh else None

    stop_px = trade_row.get("stop_loss_price")
    try:
        stop_f = float(stop_px) if stop_px not in (None, "", "nan") else None
    except Exception:
        stop_f = None
    hit_stop = stop_f is not None and low <= stop_f
    hit_locked = locked_hi_f is not None and high >= locked_hi_f
    hit_dyn = dyn_hi is not None and high >= float(dyn_hi)

    return {
        "trade_id": str(trade_row["trade_id"]),
        "date": d,
        "symbol": symbol.strip(),
        "close": close,
        "high": high,
        "low": low,
        "volume": vol,
        "sma20": _f("SMA20"),
        "sma60": _f("SMA60"),
        "ema5": _f("EMA5"),
        "ema20": _f("EMA20"),
        "atr14": _f("ATR14"),
        "rsi14": _f("RSI14"),
        "bias20": _f("Bias20"),
        "trend_ok": trend_ok,
        "risk_ok": risk_ok,
        "buy_gate": buy_gate,
        "buy_trigger": buy_trigger,
        "buy_signal": buy_signal,
        "buy_trigger_type": btt,
        "regime_type": _infer_regime(latest, df),
        "dynamic_tp_low": dyn_lo,
        "dynamic_tp_high": dyn_hi,
        "valuation_low": val_lo,
        "valuation_high": val_hi,
        "locked_tp_low": locked_lo_f,
        "locked_tp_high": locked_hi_f,
        "unrealized_pnl": unrealized,
        "return_since_entry_pct": ret_pct,
        "distance_to_locked_high_pct": dist_locked_hi,
        "distance_to_dynamic_high_pct": dist_dyn_hi,
        "drawdown_from_peak_pct": dd_peak,
        "hit_locked_target_today": hit_locked,
        "hit_dynamic_target_today": hit_dyn,
        "hit_stop_today": hit_stop,
        "reason_summary": _reason_summary_row(latest),
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def create_trade(
    symbol: str,
    entry_date: str | date | datetime,
    entry_price: float,
    shares: int = 0,
    entry_reason: str = "MANUAL",
    *,
    df_at_entry: pd.DataFrame | None = None,
    locked_tp_low: float | None = None,
    locked_tp_high: float | None = None,
    stop_loss_price: float | None = None,
    entry_regime: str = "",
    entry_confidence: str = "",
    notes: str = "",
    data_dir: str | None = None,
) -> str:
    """
    建立一筆 OPEN trade，並寫入第一筆 journal（若有 df_at_entry）。
    df_at_entry 需已含 OHLCV；若未含指標，請先在外部呼叫 compute_indicators + compute_buy_signals。
    """
    ensure_dir(data_dir)
    sym = symbol.strip()
    ed = _trade_date_str(entry_date)
    trades_df = load_trades(data_dir)
    seq = _next_trade_seq(trades_df, sym, ed)
    trade_id = f"{_symbol_key(sym)}_{ed}_{seq:03d}"

    amount = float(entry_price) * int(shares) if shares else np.nan

    tp_bundle = None
    if df_at_entry is not None and not df_at_entry.empty:
        from analysis import estimate_target_range

        tp_bundle = estimate_target_range(df_at_entry, sym)

    dyn_lo = dyn_hi = val_lo = val_hi = np.nan
    if tp_bundle:
        dyn_lo = tp_bundle.get("tp_low")
        dyn_hi = tp_bundle.get("tp_high")
        val_lo = tp_bundle.get("tp_low")
        val_hi = tp_bundle.get("tp_high")

    l_lo = locked_tp_low if locked_tp_low is not None else dyn_lo
    l_hi = locked_tp_high if locked_tp_high is not None else dyn_hi

    conf = entry_confidence or (
        (tp_bundle or {}).get("confidence", "MED") if tp_bundle else "MED"
    )

    row = {k: np.nan for k in TRADE_COLUMNS}
    row.update(
        {
            "trade_id": trade_id,
            "symbol": sym,
            "entry_date": ed,
            "entry_price": float(entry_price),
            "shares": int(shares),
            "entry_amount": amount,
            "entry_reason": entry_reason,
            "entry_regime": entry_regime or "UNKNOWN",
            "entry_confidence": conf,
            "locked_tp_low": l_lo,
            "locked_tp_high": l_hi,
            "dynamic_tp_low_at_entry": dyn_lo,
            "dynamic_tp_high_at_entry": dyn_hi,
            "valuation_low_at_entry": val_lo,
            "valuation_high_at_entry": val_hi,
            "stop_loss_price": stop_loss_price if stop_loss_price is not None else np.nan,
            "status": "OPEN",
            "exit_date": np.nan,
            "exit_price": np.nan,
            "exit_reason": np.nan,
            "realized_pnl": np.nan,
            "realized_return_pct": np.nan,
            "notes": notes,
        }
    )

    new_t = pd.DataFrame([{k: row.get(k, np.nan) for k in TRADE_COLUMNS}])
    out_t = pd.concat([trades_df, new_t], ignore_index=True)
    _save_trades(out_t, data_dir)

    if df_at_entry is not None and not df_at_entry.empty:
        trow = pd.Series(row)
        snap = build_journal_snapshot(trow, df_at_entry, sym, asof_date=ed)
        _append_journal_rows([snap], data_dir)

    return trade_id


def close_trade(
    trade_id: str,
    exit_date: str | date | datetime,
    exit_price: float,
    exit_reason: str = "",
    data_dir: str | None = None,
) -> bool:
    trades_df = load_trades(data_dir)
    if trades_df.empty:
        return False
    m = trades_df["trade_id"].astype(str) == trade_id
    if not m.any():
        return False
    i = trades_df.index[m][0]
    if str(trades_df.at[i, "status"]).upper() != "OPEN":
        return False

    ep = float(trades_df.at[i, "entry_price"])
    sh = int(float(trades_df.at[i, "shares"] or 0))
    xp = float(exit_price)
    pnl = (xp - ep) * sh if sh else np.nan
    ret_pct = ((xp / ep) - 1.0) * 100.0 if ep else np.nan

    trades_df.at[i, "status"] = "CLOSED"
    trades_df.at[i, "exit_date"] = _trade_date_str(exit_date)
    trades_df.at[i, "exit_price"] = xp
    trades_df.at[i, "exit_reason"] = exit_reason
    trades_df.at[i, "realized_pnl"] = pnl
    trades_df.at[i, "realized_return_pct"] = ret_pct
    _save_trades(trades_df, data_dir)
    return True


def update_open_trades_daily(
    data_by_symbol: dict[str, pd.DataFrame] | None = None,
    *,
    fetch_df: Callable[[str], pd.DataFrame] | None = None,
    symbols: list[str] | None = None,
    only_missing_today: bool = False,
    asof_date: str | None = None,
    data_dir: str | None = None,
) -> JournalUpdateReport:
    """
    對 OPEN trades 依序更新 journal。

    - symbols=None：更新所有未結案交易。
    - symbols=[...]：只更新代號在清單內之交易（代號需與主檔 symbol 一致）。
    - symbols=[]：不執行任何更新（全數記為 skipped）。
    - only_missing_today：若該 trade 在「預期寫入之 bar 日」已有 journal 列則跳過。
      預期日 = asof_date（有指定時）或資料最後一根 K 線之日（週末則為上一交易日 bar）。

    同一 trade_id + date 若已存在，append 後仍會由 _append_journal_rows 去重保留最後一筆。
    """
    report = JournalUpdateReport()
    open_trades = list_open_trades(data_dir)
    if open_trades.empty:
        return report

    if symbols is not None and len(symbols) == 0:
        for _, trow in open_trades.iterrows():
            report.skipped.append(
                {
                    "trade_id": str(trow["trade_id"]),
                    "symbol": _normalize_symbol(str(trow["symbol"])),
                    "reason": "symbols=[]，未選取任何代號",
                }
            )
        return report

    sym_allow: set[str] | None = None
    if symbols is not None:
        sym_allow = {_normalize_symbol(s) for s in symbols}

    journal_df = load_journal(data_dir)
    open_ids = set(open_trades["trade_id"].astype(str))
    existing_keys = _journal_keys_typed(journal_df, open_ids)

    rows: list[dict[str, Any]] = []

    for _, trow in open_trades.iterrows():
        sym = _normalize_symbol(str(trow["symbol"]))
        tid = str(trow["trade_id"])

        if sym_allow is not None and sym not in sym_allow:
            report.skipped.append(
                {
                    "trade_id": tid,
                    "symbol": sym,
                    "reason": "未在選取之 symbol 清單",
                }
            )
            continue

        df = None
        if data_by_symbol and sym in data_by_symbol:
            df = data_by_symbol[sym]
        elif fetch_df is not None:
            df = fetch_df(sym)
        if df is None or df.empty:
            report.failed.append(
                {
                    "trade_id": tid,
                    "symbol": sym,
                    "reason": "資料為空或抓價失敗",
                }
            )
            continue

        bar_d = _last_bar_date_str(df)
        expected_d = _trade_date_str(asof_date) if asof_date else bar_d

        if only_missing_today and (tid, expected_d) in existing_keys:
            report.skipped.append(
                {
                    "trade_id": tid,
                    "symbol": sym,
                    "reason": f"journal 已含 bar 日 {expected_d}",
                }
            )
            continue

        try:
            snap = build_journal_snapshot(
                trow, df, sym, asof_date=asof_date if asof_date else None
            )
            written_d = str(snap.get("date", expected_d)).strip()[:10]
            rows.append(snap)
            report.updated.append(tid)
            existing_keys.add((tid, written_d))
        except Exception as exc:
            report.failed.append(
                {
                    "trade_id": tid,
                    "symbol": sym,
                    "reason": f"snapshot 失敗：{exc}",
                }
            )

    if rows:
        _append_journal_rows(rows, data_dir)
    return report


@dataclass
class PortfolioJournalConfig:
    """供 App 註冊 fetch 函式時使用"""
    data_dir: str = field(default_factory=default_data_dir)


def prepare_df_for_journal(df: pd.DataFrame) -> pd.DataFrame:
    """補齊技術指標與 BUY_* 欄位（供 journal 使用）"""
    from analysis import compute_buy_signals, compute_indicators

    out = compute_indicators(df)
    out = compute_buy_signals(out)
    return out
