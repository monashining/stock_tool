"""
價格警報監控：盤中監控指定股票，觸及關鍵價位時發送 LINE 通知

使用方式：
  python price_alert_monitor.py 3037.TW 534.6 570.24
  python price_alert_monitor.py 3037.TW --stop 534.6 --entry 570.24 --send-line

環境變數：LINE_CHANNEL_ACCESS_TOKEN（與主程式共用）
"""
from __future__ import annotations

import argparse
import sys


def _stdout_utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf


def send_line_alert(message: str) -> tuple[bool, str]:
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
    if not token:
        return False, "缺少 LINE_CHANNEL_ACCESS_TOKEN"
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
        return True, "已推播"
    except Exception as exc:
        return False, str(exc)


def fetch_current_price(symbol: str) -> float | None:
    """抓取即時價（盤中）或最近收盤價"""
    try:
        tk = yf.Ticker(symbol)
        info = getattr(tk, "fast_info", None) or {}
        price = info.get("last_price") or info.get("lastPrice")
        if price is not None:
            return float(price)
        hist = tk.history(period="5d", interval="1d")
        if hist is not None and not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def main():
    _stdout_utf8()
    ap = argparse.ArgumentParser(description="價格警報監控（欣興 3037 等）")
    ap.add_argument("symbol", nargs="?", default="3037.TW", help="股票代號，如 3037.TW")
    ap.add_argument("--stop", type=float, default=534.6, help="保命價／止損價（跌破此價發警報）")
    ap.add_argument("--entry", type=float, default=570.24, help="進場參考價（站回此價發警報）")
    ap.add_argument("--send-line", action="store_true", help="觸發時發送 LINE 通知")
    ap.add_argument("--dry-run", action="store_true", help="只列印不發送")
    args = ap.parse_args()

    symbol = (args.symbol or "3037.TW").strip()
    if "." not in symbol and symbol.isdigit():
        symbol = f"{symbol}.TW"

    price = fetch_current_price(symbol)
    tw_now = datetime.now(ZoneInfo("Asia/Taipei"))

    if price is None:
        print(f"[{tw_now:%H:%M}] 無法取得 {symbol} 價格")
        return 1

    name_hint = "欣興" if "3037" in symbol else symbol
    header = f"🚨 鹹魚翻身即時預警\n標的：{name_hint} ({symbol})\n時間：{tw_now:%Y-%m-%d %H:%M}\n當前價：{price:.2f}"

    triggered = False
    msg = ""

    if price <= args.stop:
        triggered = True
        msg = f"{header}\n\n⚠️ 已觸及 10% 保命價！\n參考價位：{args.stop:.2f}\n趨勢轉弱，請保持空手。"
        print(f"[警報] 跌破保命價 {args.stop:.2f}，當前 {price:.2f}")
    elif price >= args.entry:
        triggered = True
        msg = f"{header}\n\n✅ 已重新站回策略區！\n參考價位：{args.entry:.2f}\n轉強訊號出現，準備進場。"
        print(f"[警報] 站回進場價 {args.entry:.2f}，當前 {price:.2f}")
    else:
        print(f"[{tw_now:%H:%M}] {symbol} 當前 {price:.2f}（保命 {args.stop:.2f}｜進場 {args.entry:.2f}）")

    if triggered and args.send_line and not args.dry_run:
        ok, err = send_line_alert(msg)
        print(f"[LINE] {'成功' if ok else err}")
    elif triggered and args.dry_run:
        print("--- 預覽（dry-run）---")
        print(msg)

    return 0


if __name__ == "__main__":
    exit(main())
