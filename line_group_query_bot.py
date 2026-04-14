"""
LINE 群組文字指令：解析「查／完整／撿便宜／持股」與純代碼，輸出結構化命令。
支援多檔：查 3037 3189、查 3037,3189（空白或逗號分隔）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

TW_STOCK_RE = re.compile(r"^(?:\d{4}|\d{4}\.TW)$", re.IGNORECASE)


@dataclass(frozen=True)
class GroupQueryCommand:
    action: str  # "compact" | "full" | "dip" | "position" | "multi_ultra"
    tickers_normalized: Tuple[str, ...]
    tickers_raw: Tuple[str, ...]
    has_position_mode: bool = False

    @property
    def ticker_normalized(self) -> str:
        return self.tickers_normalized[0]

    @property
    def ticker_raw(self) -> str:
        return self.tickers_raw[0]


def normalize_ticker(raw: str) -> Optional[str]:
    s = raw.strip().upper()
    if re.fullmatch(r"\d{4}", s):
        return f"{s}.TW"
    if re.fullmatch(r"\d{4}\.TW", s):
        return s
    return None


def _parse_ticker_blob(blob: str) -> Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """空白或逗號分隔；無效代碼整段失敗；依 normalized 去重保留順序。"""
    blob = (blob or "").replace(",", " ").strip()
    tokens = [t for t in blob.split() if t]
    if not tokens:
        return None
    seen_norm: set[str] = set()
    norms: list[str] = []
    raws: list[str] = []
    for t in tokens:
        nt = normalize_ticker(t)
        if nt is None:
            return None
        if nt in seen_norm:
            continue
        seen_norm.add(nt)
        norms.append(nt)
        raws.append(t.strip())
    if not norms:
        return None
    return (tuple(raws), tuple(norms))


def parse_group_query_command(text: str) -> Optional[GroupQueryCommand]:
    if not text:
        return None

    s = text.strip()

    if TW_STOCK_RE.fullmatch(s.upper()):
        nt = normalize_ticker(s)
        if nt:
            raw = s.strip()
            return GroupQueryCommand(
                action="compact",
                tickers_normalized=(nt,),
                tickers_raw=(raw,),
                has_position_mode=False,
            )

    patterns = [
        (r"^速查\s+(.+)$", "multi_ultra", False),
        (r"^查\s+(.+)$", "compact", False),
        (r"^完整\s+(.+)$", "full", False),
        (r"^撿便宜\s+(.+)$", "dip", False),
        (r"^持股\s+(.+)$", "position", True),
    ]

    for pattern, action, has_position_mode in patterns:
        m = re.match(pattern, s, flags=re.IGNORECASE)
        if m:
            parsed = _parse_ticker_blob(m.group(1))
            if not parsed:
                return None
            raws, norms = parsed
            return GroupQueryCommand(
                action=action,
                tickers_normalized=norms,
                tickers_raw=raws,
                has_position_mode=has_position_mode,
            )

    return None


def is_help_command(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if t in ("help", "/help", "幫助", "說明", "?", "？"):
        return True
    # 群組常見：@bot help、help ?、請幫助
    toks = [x for x in re.split(r"[\s,，。！？!?/]+", t) if x]
    return any(x in ("help", "幫助", "說明") for x in toks)


def help_message_text() -> str:
    return (
        "可用指令：\n"
        "查 3037\n"
        "查 3037 3189  或  查 3037,3189（多檔精簡列示，最多 5 檔）\n"
        "速查 3037 3189（多檔極簡：代碼 → 結論標籤）\n"
        "完整 3037\n"
        "撿便宜 3037\n"
        "持股 3037\n"
        "\n"
        "亦可直接傳 3037 或 3037.TW（預設精簡診斷）。"
    )
