"""
LINE Quick Reply：依代碼產生常用指令按鈕（不動核心查詢邏輯，僅附加 payload）。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


def display_code_for_quick_reply(ticker_base: str) -> str:
    """抽出 4 碼；無法辨識時用範例代碼。"""
    s = (ticker_base or "").strip().upper().replace(".TW", "")
    if re.fullmatch(r"\d{4}", s):
        return s
    m = re.search(r"(\d{4})", s)
    if m:
        return m.group(1)
    return "2330"


def build_quick_reply_items(ticker_base: str) -> List[Dict[str, Any]]:
    """回傳 LINE quickReply.items（最多 13 則；此處固定 6 則）。"""
    code = display_code_for_quick_reply(ticker_base)

    def item(label: str, text: str) -> Dict[str, Any]:
        return {
            "type": "action",
            "action": {
                "type": "message",
                "label": label[:20],
                "text": text[:300],
            },
        }

    return [
        item(f"查 {code}", f"查 {code}"),
        item(f"速查 {code}", f"速查 {code}"),
        item(f"完整 {code}", f"完整 {code}"),
        item(f"撿便宜 {code}", f"撿便宜 {code}"),
        item(f"持股 {code}", f"持股 {code}"),
        item("help", "help"),
    ]
