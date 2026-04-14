"""
LINE Messaging API webhook：群組查股（與 stock_query_service 串接）。

環境變數：
  LINE_CHANNEL_SECRET
  LINE_CHANNEL_ACCESS_TOKEN
  LINE_MULTI_ULTRA_SHOW_CATEGORY（可選：1/true/yes/on 時，速查列附主風險分類｜防守／結構…）
  SQLITE_PATH（可選，預設 data/stock_tool.db）
  SQLITE_LOG_ENABLED（可選，預設啟用；0/false/no/off 可停用）

啟動（開發）：
  python line_webhook_app.py
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os

import requests
from dotenv import load_dotenv
from flask import Flask, abort, request

load_dotenv()

from line_group_query_bot import (
    help_message_text,
    is_help_command,
    parse_group_query_command,
)
from line_quick_reply import build_quick_reply_items
from sqlite_store import (
    log_query_event,
    sqlite_log_enabled_from_env,
    sqlite_path_from_env,
)
from stock_query_service import run_group_stock_query

app = Flask(__name__)

LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
SQLITE_PATH = sqlite_path_from_env(os.getenv("SQLITE_PATH"))
SQLITE_LOG_ENABLED = sqlite_log_enabled_from_env(os.getenv("SQLITE_LOG_ENABLED"))


def _multi_ultra_show_category_from_env() -> bool:
    v = os.getenv("LINE_MULTI_ULTRA_SHOW_CATEGORY", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def verify_line_signature(body: bytes, signature: str) -> bool:
    if not LINE_CHANNEL_SECRET or not signature:
        return False
    digest = hmac.new(
        LINE_CHANNEL_SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)


def reply_line_message(
    reply_token: str,
    text: str,
    *,
    quick_reply_ticker_base: str | None = None,
) -> None:
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    msg: dict = {
        "type": "text",
        "text": text[:5000],
    }
    if quick_reply_ticker_base:
        msg["quickReply"] = {"items": build_quick_reply_items(quick_reply_ticker_base)}
    payload = {"replyToken": reply_token, "messages": [msg]}
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()


@app.route("/line/webhook", methods=["POST"])
def line_webhook():
    body = request.get_data()
    signature = request.headers.get("X-Line-Signature", "")

    if not verify_line_signature(body, signature):
        abort(400, description="Invalid signature")

    payload = request.get_json(silent=True) or {}
    events = payload.get("events", [])

    for event in events:
        if event.get("type") != "message":
            continue
        if event.get("message", {}).get("type") != "text":
            continue

        reply_token = event.get("replyToken")
        text = event.get("message", {}).get("text", "")

        if is_help_command(text):
            if reply_token:
                reply_line_message(
                    reply_token, help_message_text(), quick_reply_ticker_base="2330"
                )
            continue

        cmd = parse_group_query_command(text)
        if not cmd:
            continue

        try:
            result_text = run_group_stock_query(
                cmd,
                multi_ultra_show_category=_multi_ultra_show_category_from_env(),
            )
            query_ok = True
            query_err = ""
        except Exception as e:
            result_text = f"查詢失敗：{type(e).__name__}：{e}"
            query_ok = False
            query_err = str(e)

        if SQLITE_LOG_ENABLED:
            src = event.get("source", {}) or {}
            try:
                log_query_event(
                    db_path=SQLITE_PATH,
                    source_type=str(src.get("type", "")),
                    source_user_id=str(src.get("userId", "")),
                    source_group_id=str(src.get("groupId", "")),
                    raw_text=str(text or ""),
                    action=str(cmd.action),
                    tickers=cmd.tickers_normalized,
                    ok=query_ok,
                    error_text=query_err,
                )
            except Exception:
                # 記錄失敗不應影響 webhook 主流程
                pass

        if reply_token:
            qr_base = cmd.ticker_raw
            reply_line_message(
                reply_token, result_text, quick_reply_ticker_base=qr_base
            )

    return "OK", 200


@app.route("/health", methods=["GET"])
def health():
    return {"ok": True}, 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
