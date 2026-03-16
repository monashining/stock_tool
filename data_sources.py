import numpy as np
import pandas as pd
import pandas_ta as ta
import streamlit as st
import yfinance as yf

from utils import normalize_net_series_to_lot, safe_float

try:
    from FinMind.data import DataLoader

    FINMIND_AVAILABLE = True
except Exception:
    FINMIND_AVAILABLE = False


if FINMIND_AVAILABLE:

    @st.cache_data(ttl=300)
    def load_institutional_data(stock_id, start_date):
        dl = DataLoader()
        return dl.taiwan_stock_institutional_investors(
            stock_id=stock_id,
            start_date=start_date,
        )

    @st.cache_data(ttl=3600)
    def load_tw_stock_names():
        dl = DataLoader()
        info = dl.taiwan_stock_info()
        if info.empty:
            return {}
        return dict(zip(info["stock_id"].astype(str), info["stock_name"]))


@st.cache_data(ttl=3600)
def load_tw_stock_names_from_web():
    name_map = {}
    sources = [
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4",
    ]
    for url in sources:
        try:
            tables = pd.read_html(url, encoding="big5")
        except Exception:
            continue
        if not tables:
            continue
        df = tables[0]
        if df.empty or df.shape[1] < 1:
            continue
        if "有價證券代號及名稱" in df.columns:
            col = "有價證券代號及名稱"
            pairs = df[col].astype(str).str.replace("\u3000", " ")
            parts = pairs.str.split(" ", n=1, expand=True)
            df = pd.DataFrame({"stock_id": parts[0], "stock_name": parts[1]})
        elif "代號" in df.columns and "名稱" in df.columns:
            df = df.rename(columns={"代號": "stock_id", "名稱": "stock_name"})
        else:
            possible = [c for c in df.columns if "代號" in str(c)]
            if possible:
                pairs = df[possible[0]].astype(str).str.replace("\u3000", " ")
                parts = pairs.str.split(" ", n=1, expand=True)
                df = pd.DataFrame({"stock_id": parts[0], "stock_name": parts[1]})
            else:
                continue
        df = df.dropna(subset=["stock_id", "stock_name"])
        for _, row in df.iterrows():
            stock_id = str(row["stock_id"]).strip()
            stock_name = str(row["stock_name"]).strip()
            if stock_id.isdigit() and stock_name:
                name_map[stock_id] = stock_name
    return name_map


@st.cache_data(ttl=300)
def fetch_chip_net_series(symbol: str):
    """
    一次抓外資/投信淨買賣超（共用同一份 FinMind raw），避免重複處理。
    回傳：(foreign_series, trust_series)，單位自動正規化為「張(推測)」。
    """
    if not FINMIND_AVAILABLE or not symbol or not (symbol.endswith(".TW") or symbol.endswith(".TWO")):
        return None, None
    try:
        stock_id = symbol.split(".")[0]
        df_inst = load_institutional_data(stock_id, "2024-01-01")
        if df_inst is None or df_inst.empty:
            return None, None
        if "name" not in df_inst.columns:
            return None, None

        def _build(names: list[str]):
            sub = df_inst[df_inst["name"].isin(names)]
            if sub.empty:
                return None
            if "date" not in sub.columns or "buy" not in sub.columns or "sell" not in sub.columns:
                return None
            tmp = sub.copy()
            tmp["net"] = pd.to_numeric(tmp["buy"], errors="coerce") - pd.to_numeric(
                tmp["sell"], errors="coerce"
            )
            s = tmp.groupby("date")["net"].sum()
            # 轉成 datetime index（方便對齊）
            try:
                s.index = pd.to_datetime(s.index, errors="coerce")
                s = s.dropna()
            except Exception:
                pass
            return normalize_net_series_to_lot(s.sort_index())

        foreign_s = _build(["Foreign_Investor"])
        trust_s = _build(["Investment_Trust", "Investment_Trusts"])
        return foreign_s, trust_s
    except Exception:
        return None, None


@st.cache_data(ttl=300)
def fetch_foreign_net_series(symbol):
    if not FINMIND_AVAILABLE or not (symbol.endswith(".TW") or symbol.endswith(".TWO")):
        return None
    try:
        f, _ = fetch_chip_net_series(symbol)
        return f
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_trust_net_series(symbol):
    if not FINMIND_AVAILABLE or not (symbol.endswith(".TW") or symbol.endswith(".TWO")):
        return None
    try:
        _, t = fetch_chip_net_series(symbol)
        return t
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_institutional_raw(symbol):
    if not FINMIND_AVAILABLE or not (symbol.endswith(".TW") or symbol.endswith(".TWO")):
        return None
    try:
        stock_id = symbol.split(".")[0]
        return load_institutional_data(stock_id, "2024-01-01")
    except Exception:
        return None


def _fetch_last_price_uncached(symbol):
    try:
        info = yf.Ticker(symbol).fast_info
        last_price = info.get("last_price") or info.get("lastPrice")
        if last_price is not None and np.isfinite(last_price):
            return float(last_price)
    except Exception:
        pass
    try:
        df = yf.download(symbol, period="5d", progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        if "Close" not in df.columns:
            return None
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if close.empty:
            return None
        return float(close.iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=60)
def fetch_last_price(symbol):
    return _fetch_last_price_uncached(symbol)


@st.cache_data(ttl=60)
def fetch_last_price_batch(symbols: list[str]):
    """
    批次抓取即時價（有快取）。
    - 使用 ThreadPoolExecutor 平行抓取 fast_info，避免持股清單 N 檔就 N 次卡很久。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    syms = [s.strip() for s in (symbols or []) if isinstance(s, str) and s.strip()]
    if not syms:
        return {}
    # 去重、保序
    seen = set()
    uniq = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    max_workers = min(8, max(1, len(uniq)))
    out: dict[str, float | None] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_last_price_uncached, s): s for s in uniq}
        for fu in as_completed(futs):
            s = futs[fu]
            try:
                out[s] = fu.result()
            except Exception:
                out[s] = None
    return out


@st.cache_data(ttl=3600)
def fetch_fundamental_snapshot(symbol):
    try:
        tk = yf.Ticker(symbol)
        info = tk.info or {}
    except Exception:
        info = {}
    eps_ttm = safe_float(info.get("trailingEps"))
    eps_fwd = safe_float(info.get("forwardEps"))
    pe_ttm = safe_float(info.get("trailingPE"))
    pe_fwd = safe_float(info.get("forwardPE"))
    fcf = safe_float(info.get("freeCashflow"))
    shares = safe_float(info.get("sharesOutstanding"))
    return {
        "eps_ttm": eps_ttm,
        "eps_fwd": eps_fwd,
        "pe_ttm": pe_ttm,
        "pe_fwd": pe_fwd,
        "free_cashflow": fcf,
        "shares_outstanding": shares,
    }


@st.cache_data(ttl=3600)
def fetch_ticker_name(symbol: str):
    if not symbol:
        return None, None
    symbol = symbol.strip()
    candidates = [symbol]
    if "." not in symbol and symbol.isdigit():
        candidates = [f"{symbol}.TW", f"{symbol}.TWO", symbol]

    base = symbol.replace(".TW", "").replace(".TWO", "")
    if base.isdigit():
        if FINMIND_AVAILABLE:
            try:
                name_map = load_tw_stock_names()
                name = name_map.get(base)
                if name:
                    return name, f"{base}.TW"
            except Exception:
                pass
        try:
            name_map = load_tw_stock_names_from_web()
            name = name_map.get(base)
            if name:
                return name, f"{base}.TW"
        except Exception:
            pass

    for cand in candidates:
        try:
            tk = yf.Ticker(cand)
            fi = getattr(tk, "fast_info", None) or {}
            lp = fi.get("last_price") or fi.get("lastPrice")
            if lp is not None and np.isfinite(lp):
                try:
                    info = tk.info or {}
                    name = info.get("longName") or info.get("shortName")
                    if name:
                        return name, cand
                except Exception:
                    pass
                return None, cand
        except Exception:
            pass
        try:
            info = yf.Ticker(cand).info or {}
            name = info.get("longName") or info.get("shortName")
            if name:
                return name, cand
        except Exception:
            continue
    return None, candidates[0] if candidates else None


@st.cache_data(ttl=300)
def load_market_index(symbol):
    if symbol.endswith(".TW") or symbol.endswith(".TWO") or symbol.isdigit():
        index_symbol = "^TWII"
    else:
        index_symbol = "^GSPC"
    mkt = yf.download(index_symbol, period="6mo", progress=False)
    if not mkt.empty and isinstance(mkt.columns, pd.MultiIndex):
        mkt.columns = [c[0] for c in mkt.columns]
    return mkt, index_symbol


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Volume" not in df.columns:
        df["Volume"] = np.nan
    return df.dropna(subset=["Close"]) if "Close" in df.columns else df


def _try_download(symbol, period, threads=True):
    try:
        return yf.download(
            symbol,
            period=period,
            progress=False,
            group_by="column",
            auto_adjust=False,
            threads=threads,
        )
    except Exception:
        return pd.DataFrame()


def _try_history(symbol, period):
    try:
        return yf.Ticker(symbol).history(period=period, auto_adjust=False)
    except Exception:
        return pd.DataFrame()


def _load_data_raw(symbol, period):
    fetch_period = "5y" if period == "5y" else "1y"
    df = _try_download(symbol, fetch_period)
    if df.empty and symbol.endswith(".TW"):
        df = _try_download(symbol.replace(".TW", ".TWO"), fetch_period)
    if df.empty and "." not in symbol:
        df = _try_download(f"{symbol}.TW", fetch_period)
        if df.empty:
            df = _try_download(f"{symbol}.TWO", fetch_period)
    if df.empty:
        df = _try_history(symbol, fetch_period)
    if df.empty and symbol.endswith(".TW"):
        df = _try_history(symbol.replace(".TW", ".TWO"), fetch_period)
    if df.empty and "." not in symbol:
        df = _try_history(f"{symbol}.TW", fetch_period)
        if df.empty:
            df = _try_history(f"{symbol}.TWO", fetch_period)
    if df.empty:
        df = _try_download(symbol, fetch_period, threads=False)
    df = normalize_ohlcv(df)
    if df is None or df.empty:
        return df
    period_map = {
        "1mo": 22,
        "3mo": 63,
        "6mo": 126,
        "1y": 252,
    }
    if period in period_map:
        return df.tail(period_map[period])
    return df


@st.cache_data(ttl=300)  # 盤中 5 分鐘更新
def load_data(symbol, period):
    return _load_data_raw(symbol, period)


@st.cache_data(ttl=300)
def load_data_batch(symbols, period="1y"):
    """
    一次抓多檔資料（比逐檔 yf.download 快很多）
    回傳：dict[str, DataFrame]，key = symbol（和輸入一致）
    """
    symbols = [s.strip() for s in symbols if s and s.strip()]
    if not symbols:
        return {}

    df_all = yf.download(
        tickers=" ".join(symbols),
        period=period,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    out = {}
    if df_all is None or df_all.empty:
        return out

    if not isinstance(df_all.columns, pd.MultiIndex):
        tmp = normalize_ohlcv(df_all.copy())
        out[symbols[0]] = tmp
        return out

    level0 = set(df_all.columns.get_level_values(0)) if isinstance(df_all.columns, pd.MultiIndex) else set()
    for sym in symbols:
        try:
            if sym not in level0:
                out[sym] = pd.DataFrame()
                continue
            sub = normalize_ohlcv(df_all[sym].copy())
            out[sym] = sub
        except Exception:
            out[sym] = pd.DataFrame()

    return out


@st.cache_data(ttl=3600)
def get_weekly_trend(symbol):
    w_df = yf.download(symbol, period="1y", interval="1wk", progress=False)
    if w_df.empty or len(w_df) < 5:
        return "未知"
    if isinstance(w_df.columns, pd.MultiIndex):
        w_df.columns = [col[0] for col in w_df.columns]
    w_ema10 = ta.ema(w_df["Close"], length=10)
    if w_ema10.empty:
        return "未知"
    w_close = w_df["Close"].iloc[-1]
    w_ema10_last = w_ema10.iloc[-1]
    return "多頭" if w_close > w_ema10_last else "空頭"
