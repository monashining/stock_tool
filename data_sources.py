import os
import pickle
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import streamlit as st
import yfinance as yf

from utils import normalize_net_series_to_lot, safe_float

# 程序內記憶體快取（跨 Streamlit cache_data 與直接呼叫 _load_data_raw 共用），
# 降低 yfinance 頻率與 rate limit；**不同 process**（Web 與 LINE worker）仍各自一份。
_OHLCV_MEM_LOCK = threading.Lock()
_OHLCV_MEM_CACHE: dict[Tuple[str, ...], tuple[float, pd.DataFrame]] = {}
# 週線趨勢字串（多頭/空頭/未知）專用
_WEEKLY_STR_MEM_CACHE: dict[Tuple[str, ...], tuple[float, str]] = {}

_YF_DISK_LOCK = threading.Lock()


def _read_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        return v if np.isfinite(v) and v >= 0 else default
    except ValueError:
        return default


def yf_mem_cache_ttl_sec() -> float:
    """環境變數 YF_MEM_CACHE_TTL_SEC（秒），預設 60。"""
    return _read_env_float("YF_MEM_CACHE_TTL_SEC", 60.0)


def yf_disk_cache_ttl_sec() -> float:
    """環境變數 YF_DISK_CACHE_TTL_SEC（秒），預設 60。"""
    return _read_env_float("YF_DISK_CACHE_TTL_SEC", 60.0)


def _yf_disk_cache_dir() -> Path:
    raw = os.environ.get("YF_DISK_CACHE_DIR", "").strip()
    if raw:
        d = Path(raw).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        return d.resolve()
    d = Path(__file__).resolve().parent / "data" / ".yf_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _yf_disk_cache_path(key_parts: Tuple[str, ...]) -> Path:
    safe = "__".join(
        "".join(c if c.isalnum() or c in ".-_" else "_" for c in str(p))[:120]
        for p in key_parts
    )
    return _yf_disk_cache_dir() / f"{safe}.pkl"


def _yf_disk_blob_load(key_parts: Tuple[str, ...]) -> Optional[dict]:
    path = _yf_disk_cache_path(key_parts)
    if not path.is_file():
        return None
    try:
        with _YF_DISK_LOCK, open(path, "rb") as f:
            blob = pickle.load(f)
        ts = float(blob.get("wall_ts", 0))
        if time.time() - ts >= yf_disk_cache_ttl_sec():
            return None
        return blob if isinstance(blob, dict) else None
    except Exception:
        return None


def _yf_disk_blob_write_atomic(key_parts: Tuple[str, ...], payload: dict) -> None:
    path = _yf_disk_cache_path(key_parts)
    tmp = path.with_suffix(".pkl.tmp")
    blob = dict(payload)
    blob["wall_ts"] = time.time()
    try:
        with _YF_DISK_LOCK, open(tmp, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        try:
            if tmp.is_file():
                tmp.unlink()
        except Exception:
            pass


def _yf_disk_cache_read(key_parts: Tuple[str, ...]) -> Optional[pd.DataFrame]:
    blob = _yf_disk_blob_load(key_parts)
    if not blob:
        return None
    df = blob.get("df")
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    return df


def _yf_disk_cache_write(key_parts: Tuple[str, ...], df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    _yf_disk_blob_write_atomic(key_parts, {"df": df.copy()})


def _yf_disk_weekly_trend_read(key_parts: Tuple[str, ...]) -> Optional[str]:
    blob = _yf_disk_blob_load(key_parts)
    if not blob or "weekly_trend" not in blob:
        return None
    return str(blob["weekly_trend"])


def _yf_disk_weekly_trend_write(key_parts: Tuple[str, ...], trend: str) -> None:
    _yf_disk_blob_write_atomic(key_parts, {"weekly_trend": str(trend)})


def _weekly_str_mem_get(key: Tuple[str, ...]) -> Optional[str]:
    now = time.monotonic()
    with _OHLCV_MEM_LOCK:
        ent = _WEEKLY_STR_MEM_CACHE.get(key)
        if not ent:
            return None
        ts, trend = ent
        if now - ts > yf_mem_cache_ttl_sec():
            try:
                del _WEEKLY_STR_MEM_CACHE[key]
            except KeyError:
                pass
            return None
        return str(trend)


def _weekly_str_mem_set(key: Tuple[str, ...], trend: str) -> None:
    with _OHLCV_MEM_LOCK:
        _WEEKLY_STR_MEM_CACHE[key] = (time.monotonic(), str(trend))


def _ohlcv_mem_cache_get(key: Tuple[str, ...]) -> Optional[pd.DataFrame]:
    now = time.monotonic()
    with _OHLCV_MEM_LOCK:
        ent = _OHLCV_MEM_CACHE.get(key)
        if not ent:
            return None
        ts, df = ent
        if now - ts > yf_mem_cache_ttl_sec():
            try:
                del _OHLCV_MEM_CACHE[key]
            except KeyError:
                pass
            return None
        return df


def _ohlcv_mem_cache_set(key: Tuple[str, ...], df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    with _OHLCV_MEM_LOCK:
        _OHLCV_MEM_CACHE[key] = (time.monotonic(), df.copy())

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


# 大盤指數：台股一律 ^TWII（不依 input 字串變體），非台股 ^GSPC
MARKET_INDEX_TWII = "^TWII"
MARKET_INDEX_GSPC = "^GSPC"


def _is_taiwan_equity_symbol(symbol: str) -> bool:
    s = (symbol or "").strip()
    return bool(s.endswith(".TW") or s.endswith(".TWO") or s.isdigit())


def _load_market_index_ohlcv_with_meta(
    index_symbol: str, *, period: str = "6mo"
) -> tuple[pd.DataFrame, str]:
    cache_key = ("mkt", index_symbol, period)
    hit = _ohlcv_mem_cache_get(cache_key)
    if hit is not None:
        return hit.copy(), "mem"
    disk = _yf_disk_cache_read(cache_key)
    if disk is not None:
        _ohlcv_mem_cache_set(cache_key, disk)
        return disk.copy(), "disk"
    mkt = yf.download(index_symbol, period=period, progress=False)
    if not mkt.empty and isinstance(mkt.columns, pd.MultiIndex):
        mkt.columns = [c[0] for c in mkt.columns]
    mkt = normalize_ohlcv(mkt)
    if mkt is not None and not mkt.empty:
        _ohlcv_mem_cache_set(cache_key, mkt)
        _yf_disk_cache_write(cache_key, mkt)
        return mkt, "net"
    return mkt if mkt is not None else pd.DataFrame(), "net"


def _load_market_index_ohlcv(index_symbol: str, *, period: str = "6mo") -> pd.DataFrame:
    df, _layer = _load_market_index_ohlcv_with_meta(index_symbol, period=period)
    return df


def load_market_index_with_meta(
    symbol: str = "",
) -> tuple[pd.DataFrame, str, str]:
    """
    回傳 (大盤日線 DataFrame, 指數代號, cache_layer)。
    cache_layer：mem / disk / net（與個股 OHLCV 同一套 mem+disk）。
    """
    idx = MARKET_INDEX_TWII if _is_taiwan_equity_symbol(str(symbol)) else MARKET_INDEX_GSPC
    df, layer = _load_market_index_ohlcv_with_meta(idx)
    return df, idx, layer


def load_market_index(symbol: str = "") -> tuple[pd.DataFrame, str]:
    """
    回傳 (大盤日線 DataFrame, 指數代號)。
    台股相關標的固定用 ^TWII；其餘用 ^GSPC。與個股 ticker 寫法無關，避免 TWII / ^TWII 混用。
    """
    df, idx, _layer = load_market_index_with_meta(symbol)
    return df, idx


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


def _normalize_symbol_for_cache(symbol: str) -> str:
    """正規化股票代號，確保切換股票時 cache key 一致，避免吃到錯檔資料"""
    if not symbol or not isinstance(symbol, str):
        return str(symbol or "")
    s = symbol.strip()
    if not s:
        return s
    # 台股：純數字補上 .TW，避免 "2367" 與 "2367.TW" 產生不同 cache
    if s.isdigit():
        return f"{s}.TW"
    return s


def _load_data_raw_with_meta(symbol, period) -> tuple[pd.DataFrame, str]:
    """回傳 (df, cache_layer)；layer 為 mem / disk / net。"""
    # 與 load_data 的 symbol 正規化一致，避免 2367 / 2367.TW 打到不同快取槽
    mem_key = (_normalize_symbol_for_cache(str(symbol or "")), str(period or "1y"))
    cached = _ohlcv_mem_cache_get(mem_key)
    if cached is not None:
        return cached.copy(), "mem"
    disk_df = _yf_disk_cache_read(mem_key)
    if disk_df is not None:
        _ohlcv_mem_cache_set(mem_key, disk_df)
        return disk_df.copy(), "disk"

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
        return df, "net"
    period_map = {
        "1mo": 22,
        "3mo": 63,
        "6mo": 126,
        "1y": 252,
    }
    if period in period_map:
        df = df.tail(period_map[period])
    _ohlcv_mem_cache_set(mem_key, df)
    _yf_disk_cache_write(mem_key, df)
    return df.copy(), "net"


def _load_data_raw(symbol, period):
    df, _layer = _load_data_raw_with_meta(symbol, period)
    return df


@st.cache_data(ttl=300)  # 盤中 5 分鐘更新；用正規化 symbol 當 key，切換股票會正確取新資料
def _load_data_cached(normalized_symbol: str, period: str):
    """內部：cache 以 normalized_symbol 為 key；一併快取本次命中的 cache_layer。"""
    df, layer = _load_data_raw_with_meta(normalized_symbol, period)
    if df is None or df.empty:
        return df, layer
    # 回傳 copy，避免 caller 對 df 的修改污染 cache
    return df.copy(), layer


def load_data_with_meta(symbol, period) -> tuple[pd.DataFrame, str]:
    """與 load_data 相同資料，另回傳本次載入的 cache_layer（mem/disk/net）。"""
    norm = _normalize_symbol_for_cache(str(symbol or ""))
    return _load_data_cached(norm, period)


def load_data(symbol, period):
    norm = _normalize_symbol_for_cache(str(symbol or ""))
    df, _layer = _load_data_cached(norm, period)
    return df


@st.cache_data(ttl=300)
def load_data_batch(symbols, period="1y"):
    """
    一次抓多檔資料（比逐檔 yf.download 快很多）
    回傳：dict[str, DataFrame]，key = symbol（和輸入一致）
    """
    symbols = [s.strip() for s in (symbols or []) if s and str(s).strip()]
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


def _fetch_weekly_ohlcv_for_trend(symbol: str) -> pd.DataFrame:
    """週線 OHLCV：與日線類似做代號後備，失敗回傳空 DataFrame。"""
    sym = str(symbol or "").strip()
    norm = _normalize_symbol_for_cache(sym)

    def _one(s: str) -> pd.DataFrame:
        w_df = yf.download(s, period="1y", interval="1wk", progress=False)
        if w_df is None or w_df.empty:
            return pd.DataFrame()
        return w_df

    w_df = _one(norm)
    if not w_df.empty:
        return w_df
    if norm.endswith(".TW"):
        w_df = _one(norm.replace(".TW", ".TWO"))
        if not w_df.empty:
            return w_df
    if sym and sym != norm:
        w_df = _one(sym)
        if not w_df.empty:
            return w_df
    return pd.DataFrame()


def _compute_weekly_label_from_wdf(w_df: pd.DataFrame) -> str:
    if w_df is None or w_df.empty or len(w_df) < 5:
        return "未知"
    w_df = w_df.copy()
    if isinstance(w_df.columns, pd.MultiIndex):
        w_df.columns = [col[0] for col in w_df.columns]
    w_ema10 = ta.ema(w_df["Close"], length=10)
    if w_ema10.empty:
        return "未知"
    w_close = w_df["Close"].iloc[-1]
    w_ema10_last = w_ema10.iloc[-1]
    if pd.isna(w_close) or pd.isna(w_ema10_last):
        return "未知"
    return "多頭" if w_close > w_ema10_last else "空頭"


def get_weekly_trend_with_meta(symbol: str) -> Tuple[str, str]:
    """
    週線趨勢字串 + 快取層級（mem / disk / net）。
    與日線 OHLCV 相同：程序內記憶體 → 磁碟 pickle → yfinance。
    """
    norm = _normalize_symbol_for_cache(str(symbol or ""))
    key: Tuple[str, ...] = ("wk", norm, "1y_1wk")
    mem_hit = _weekly_str_mem_get(key)
    if mem_hit is not None:
        return mem_hit, "mem"
    disk_hit = _yf_disk_weekly_trend_read(key)
    if disk_hit is not None:
        _weekly_str_mem_set(key, disk_hit)
        return disk_hit, "disk"
    w_df = _fetch_weekly_ohlcv_for_trend(symbol)
    trend = _compute_weekly_label_from_wdf(w_df)
    _weekly_str_mem_set(key, trend)
    _yf_disk_weekly_trend_write(key, trend)
    return trend, "net"


def get_weekly_trend(symbol: str) -> str:
    """相容舊介面：僅回傳趨勢字串。"""
    return get_weekly_trend_with_meta(symbol)[0]
