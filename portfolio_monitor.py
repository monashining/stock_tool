from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from datetime import timedelta
from zoneinfo import ZoneInfo
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from analysis import compute_indicators, compute_volume_sum_3d
from precision_diagnosis import diagnose_precision
from utils import align_net_series_to_price, normalize_net_series_to_lot


def _stdout_utf8():
    # Windows console 可能不是 UTF-8，避免列印中文名稱時爆炸
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def _fmt(x: Any, digits: int = 2) -> str:
    v = _to_float(x)
    if v is None:
        return "NA"
    return f"{v:.{digits}f}"


def _fmt_pct(x: Any, digits: int = 1) -> str:
    v = _to_float(x)
    if v is None:
        return "NA"
    return f"{v:.{digits}f}%"


def _fmt_int(x: Any) -> str:
    v = _to_float(x)
    if v is None:
        return "NA"
    return f"{v:,.0f}"


def _is_tw_symbol(sym: str) -> bool:
    s = (sym or "").upper()
    return s.endswith(".TW") or s.endswith(".TWO") or s.isdigit()


def load_tw_name_map_from_web() -> dict[str, str]:
    """
    從 TWSE ISIN 網頁抓「代號→名稱」對照表（上市+上櫃）
    """
    name_map: dict[str, str] = {}
    urls = [
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4",
    ]
    for url in urls:
        try:
            tables = pd.read_html(url, encoding="big5")
        except Exception:
            continue
        if not tables:
            continue
        df = tables[0]
        if df is None or df.empty:
            continue
        col = None
        for c in df.columns:
            if "有價證券代號及名稱" in str(c):
                col = c
                break
        if col is not None:
            pairs = df[col].astype(str).str.replace("\u3000", " ")
            parts = pairs.str.split(" ", n=1, expand=True)
            tmp = pd.DataFrame({"stock_id": parts[0], "stock_name": parts[1]})
        elif "代號" in df.columns and "名稱" in df.columns:
            tmp = df.rename(columns={"代號": "stock_id", "名稱": "stock_name"})[
                ["stock_id", "stock_name"]
            ]
        else:
            continue
        tmp = tmp.dropna(subset=["stock_id", "stock_name"])
        for _, row in tmp.iterrows():
            sid = str(row["stock_id"]).strip()
            nm = str(row["stock_name"]).strip()
            if sid.isdigit() and nm:
                name_map[sid] = nm
    return name_map


def resolve_symbol(stock_id: str) -> str:
    s = (stock_id or "").strip()
    if not s:
        return s
    if "." in s:
        return s
    if s.isdigit():
        # 先走上市，再走上櫃
        return f"{s}.TW"
    return s


def guess_name(stock_id: str, resolved_symbol: str, name_map: dict[str, str]) -> str:
    base = (stock_id or "").strip().replace(".TW", "").replace(".TWO", "")
    if base.isdigit() and base in name_map:
        return name_map[base]
    # fallback: yfinance info
    try:
        info = yf.Ticker(resolved_symbol).info or {}
        nm = info.get("longName") or info.get("shortName") or info.get("symbol")
        if nm:
            return str(nm)
    except Exception:
        pass
    return base or resolved_symbol


def download_price_df(resolved_symbol: str, period: str = "1y") -> tuple[pd.DataFrame, str]:
    df = yf.download(resolved_symbol, period=period, progress=False, auto_adjust=False)
    if df is None or df.empty:
        # 嘗試 OTC fallback
        if resolved_symbol.endswith(".TW"):
            alt = resolved_symbol.replace(".TW", ".TWO")
            df = yf.download(alt, period=period, progress=False, auto_adjust=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                return df, alt
        return pd.DataFrame(), resolved_symbol
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df, resolved_symbol


def get_live_price(resolved_symbol: str) -> Optional[float]:
    try:
        fi = getattr(yf.Ticker(resolved_symbol), "fast_info", None) or {}
        lp = fi.get("last_price") or fi.get("lastPrice")
        if lp is not None and np.isfinite(lp):
            return float(lp)
    except Exception:
        pass
    return None


def _finmind_available() -> bool:
    try:
        from FinMind.data import DataLoader  # noqa: F401

        return True
    except Exception:
        return False


def fetch_inst_net_series_finmind(
    *,
    stock_id: str,
    investor_names: list[str],
    start_date: str = "2024-01-01",
) -> Optional[pd.Series]:
    """
    回傳指定法人的日淨買賣超序列（張/股不確定，會做量級正規化）
    """
    if not _finmind_available():
        return None
    try:
        from FinMind.data import DataLoader

        dl = DataLoader()
        df_inst = dl.taiwan_stock_institutional_investors(
            stock_id=stock_id,
            start_date=start_date,
        )
        if df_inst is None or df_inst.empty:
            return None
        sub = df_inst[df_inst["name"].isin(investor_names)]
        if sub.empty:
            return None
        net = sub.set_index("date")["buy"] - sub.set_index("date")["sell"]
        s = net.groupby(level=0).sum()
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s.dropna()
        if s.empty:
            return None
        return normalize_net_series_to_lot(s.sort_index())
    except Exception:
        return None


def compute_beta_vs_index(close: pd.Series, index_close: pd.Series) -> Optional[float]:
    """
    Beta = Cov(r_stock, r_mkt) / Var(r_mkt)
    """
    try:
        c = pd.to_numeric(close, errors="coerce").dropna()
        m = pd.to_numeric(index_close, errors="coerce").dropna()
        if c.empty or m.empty:
            return None
        df = pd.DataFrame({"c": c, "m": m}).dropna()
        if len(df) < 40:
            return None
        r = df.pct_change().dropna()
        if r.empty:
            return None
        var_m = float(r["m"].var())
        if not np.isfinite(var_m) or var_m == 0:
            return None
        cov = float(r["c"].cov(r["m"]))
        if not np.isfinite(cov):
            return None
        return float(cov / var_m)
    except Exception:
        return None


def read_portfolio_csv(path: str) -> list[dict]:
    df = pd.read_csv(path)
    if df is None or df.empty:
        return []
    df.columns = [str(c).strip() for c in df.columns]

    # 欄位別名
    col_stock = None
    for c in ["stock_id", "symbol", "ticker", "code"]:
        if c in df.columns:
            col_stock = c
            break
    if col_stock is None:
        raise ValueError("CSV 需包含欄位：stock_id（或 symbol/ticker/code）")

    col_buy = None
    for c in ["buy_price", "avg_cost", "cost", "buy", "price"]:
        if c in df.columns:
            col_buy = c
            break
    if col_buy is None:
        raise ValueError("CSV 需包含欄位：buy_price（或 avg_cost/cost）")

    col_shares = None
    for c in ["shares", "qty", "amount"]:
        if c in df.columns:
            col_shares = c
            break

    rows: list[dict] = []
    for _, r in df.iterrows():
        sid = str(r.get(col_stock, "")).strip()
        if not sid:
            continue
        buy = _to_float(r.get(col_buy))
        if buy is None or buy <= 0:
            continue
        shares = _to_int(r.get(col_shares)) if col_shares else None
        rows.append({"stock_id": sid, "buy_price": float(buy), "shares": shares})
    return rows


def send_line_notification(message: str) -> tuple[bool, str]:
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
    if not token:
        return False, "缺少 LINE_CHANNEL_ACCESS_TOKEN 環境變數"
    try:
        from linebot.v3.messaging import (
            ApiClient,
            BroadcastRequest,
            Configuration,
            MessagingApi,
            TextMessage,
        )

        configuration = Configuration(access_token=token)
        with ApiClient(configuration) as api_client:
            api_instance = MessagingApi(api_client)
            api_instance.broadcast(BroadcastRequest(messages=[TextMessage(text=message)]))
        return True, "已群發推播"
    except Exception as exc:
        return False, f"推播失敗：{exc}"


def main():
    _stdout_utf8()

    ap = argparse.ArgumentParser(description="持股健康檢查（含籌碼量化對比）")
    ap.add_argument("--csv", default="my_portfolio.csv", help="持股清單 CSV 路徑")
    ap.add_argument("--period", default="1y", help="yfinance period，例如 6mo/1y/5y")
    ap.add_argument("--use-live-price", action="store_true", help="使用即時價（盤中）")
    ap.add_argument("--send-line", action="store_true", help="遇到 ⚠️/🚨 就推播 LINE")
    ap.add_argument(
        "--start-date",
        default="",
        help="FinMind 起始日期（留空＝自動用最近 180 天；格式 YYYY-MM-DD）",
    )
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"找不到 CSV：{args.csv}")

    portfolio = read_portfolio_csv(args.csv)
    if not portfolio:
        print("持股清單為空（或欄位不完整）。")
        return

    tw_now = datetime.now(ZoneInfo("Asia/Taipei"))
    start_date = str(args.start_date or "").strip()
    if not start_date:
        try:
            start_date = (tw_now.date() - timedelta(days=180)).isoformat()
        except Exception:
            start_date = "2025-01-01"
    is_market_time = (tw_now.weekday() < 5) and (datetime.strptime("09:00", "%H:%M").time() <= tw_now.time() <= datetime.strptime("13:35", "%H:%M").time())
    print(f"監控系統已就緒，開始掃描持股...（台北時間：{tw_now:%Y-%m-%d %H:%M}）")
    if is_market_time and args.use_live_price:
        print("模式：盤中即時價（若取不到則回退收盤價）")

    # 下載市場指數（beta 用）
    idx_tw = None
    idx_us = None
    try:
        idx_tw = yf.download("^TWII", period="6mo", progress=False)
        if isinstance(idx_tw.columns, pd.MultiIndex):
            idx_tw.columns = [c[0] for c in idx_tw.columns]
    except Exception:
        idx_tw = None
    try:
        idx_us = yf.download("^GSPC", period="6mo", progress=False)
        if isinstance(idx_us.columns, pd.MultiIndex):
            idx_us.columns = [c[0] for c in idx_us.columns]
    except Exception:
        idx_us = None

    # 股票名稱 map（台股）
    try:
        tw_name_map = load_tw_name_map_from_web()
    except Exception:
        tw_name_map = {}

    for it in portfolio:
        stock_id = it["stock_id"]
        buy_price = float(it["buy_price"])
        shares = it.get("shares")

        resolved0 = resolve_symbol(stock_id)
        df, resolved = download_price_df(resolved0, period=str(args.period))
        if df is None or df.empty:
            print(f"【{stock_id}】資料不足：yfinance 無法下載 {resolved0}")
            continue

        base = resolved.replace(".TW", "").replace(".TWO", "")
        name = guess_name(stock_id, resolved, tw_name_map)

        df = compute_indicators(df, include_vwap=False)
        df = df.dropna(subset=["Close"]).sort_index()
        last_close = _to_float(df["Close"].iloc[-1])
        if last_close is None:
            print(f"【{stock_id}】資料不足：缺少 Close")
            continue

        current_price = last_close
        if args.use_live_price and is_market_time:
            lp = get_live_price(resolved)
            if lp is not None:
                current_price = float(lp)

        # 籌碼（FinMind）
        foreign_s = fetch_inst_net_series_finmind(
            stock_id=base,
            investor_names=["Foreign_Investor"],
            start_date=str(start_date),
        )
        trust_s = fetch_inst_net_series_finmind(
            stock_id=base,
            investor_names=["Investment_Trust", "Investment_Trusts"],
            start_date=str(start_date),
        )
        # 對齊到股價交易日
        foreign_s = align_net_series_to_price(df, foreign_s) if foreign_s is not None else None
        trust_s = align_net_series_to_price(df, trust_s) if trust_s is not None else None

        f3d = float(foreign_s.tail(3).sum()) if foreign_s is not None and len(foreign_s) >= 3 else None
        t3d = float(trust_s.tail(3).sum()) if trust_s is not None and len(trust_s) >= 3 else None
        foreign_sell_3d = None
        try:
            if foreign_s is not None and len(foreign_s) >= 3:
                last3 = pd.to_numeric(foreign_s.tail(3), errors="coerce").dropna()
                if len(last3) >= 3:
                    foreign_sell_3d = bool((last3 < 0).all())
        except Exception:
            foreign_sell_3d = None

        # 成交量（近3日張數）
        vol3 = compute_volume_sum_3d(df)

        bias20 = _to_float(df["Bias20"].iloc[-1]) if "Bias20" in df.columns else None

        # Beta（對應 TWII / GSPC）
        beta = None
        try:
            if _is_tw_symbol(resolved) and idx_tw is not None and not idx_tw.empty and "Close" in idx_tw.columns:
                beta = compute_beta_vs_index(df["Close"], idx_tw["Close"])
            elif idx_us is not None and not idx_us.empty and "Close" in idx_us.columns:
                beta = compute_beta_vs_index(df["Close"], idx_us["Close"])
        except Exception:
            beta = None

        diag = diagnose_precision(
            foreign_3d_net=f3d,
            trust_3d_net=t3d,
            vol_sum_3d_lot=vol3,
            bias20=bias20,
            beta=beta,
            foreign_sell_3d=foreign_sell_3d,
        )

        pl_pct = ((float(current_price) / float(buy_price)) - 1.0) * 100.0 if buy_price > 0 else None
        pl_amt = None
        if shares is not None and pl_pct is not None:
            try:
                pl_amt = (float(current_price) - float(buy_price)) * float(shares)
            except Exception:
                pl_amt = None

        # 報告（console + LINE）
        lines = []
        lines.append(f"【持股監控：{name}】({resolved})")
        lines.append(f"現價：{_fmt(current_price)}｜成本：{_fmt(buy_price)}｜損益：{_fmt_pct(pl_pct)}")
        if pl_amt is not None:
            lines.append(f"未實現損益金額：約 {pl_amt:,.0f}")
        lines.append(diag.as_one_liner())
        for bline in diag.bullets[:6]:
            lines.append(f"- {bline}")
        if f3d is not None or t3d is not None:
            lines.append(f"外資3日：{_fmt_int(f3d)}｜投信3日：{_fmt_int(t3d)}｜成交量3日：{_fmt_int(vol3)} 張")

        report = "\n".join(lines)
        print("\n" + report + "\n")

        if args.send_line and ("⚠️" in report or "🚨" in report):
            ok, msg = send_line_notification(report)
            print(f"[LINE] {msg}" if ok else f"[LINE] {msg}")


if __name__ == "__main__":
    main()

