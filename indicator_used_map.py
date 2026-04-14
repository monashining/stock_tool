"""
Gate / Trigger / Guard / Chip Notes 規則表（與個股頁同源）＋ LINE 風險列 fail_lines。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from utils import to_scalar

# 語意重疊時合併為同一短句（精簡版用）；可再擴充。
FAIL_SHORT_RULE_ALIASES: Dict[str, str] = {
    "未站上 EMA20": "未站上均線",
}

# 短版／一句話用：正規化 key →（口語顯示, 語氣層級）；未命中則顯示原 rule、層級 neutral
# 層級：high / medium / low / positive / neutral — 數字越小越先列（同層保留原本 FAIL 順序）
RISK_PHRASE_ENTRIES: Dict[str, Tuple[str, str]] = {
    "跌破防守線": ("防守線失守", "high"),
    "未站上均線": ("結構未轉強", "medium"),
}

RISK_PHRASE_SEVERITY_RANK = {
    "high": 0,
    "medium": 1,
    "low": 2,
    "positive": 3,
    "neutral": 4,
}

# 風險 FAIL 列印順序：Guard（執行風控）→ Gate → Trigger → Chip（與精簡 LINE 可讀性一致）
CATEGORY_PRIORITY = {
    "Guard": 0,
    "Gate": 1,
    "Trigger": 2,
    "Chip Notes": 3,
}


def _normalize_short_rule(rule: str) -> str:
    s = (rule or "").strip()
    return FAIL_SHORT_RULE_ALIASES.get(s, s)


def phrase_for_short_display(normalized_rule: str) -> str:
    """別名／去重後的 rule 字串 → LINE 短句顯示用。"""
    s = (normalized_rule or "").strip()
    ent = RISK_PHRASE_ENTRIES.get(s)
    return ent[0] if ent else s


def severity_for_normalized_rule(normalized_rule: str) -> str:
    """對應 RISK_PHRASE_ENTRIES 的語氣層級；未設定則 neutral。"""
    s = (normalized_rule or "").strip()
    ent = RISK_PHRASE_ENTRIES.get(s)
    return ent[1] if ent else "neutral"


def _collect_fail_norm_and_category_deduped(
    used_map: dict,
) -> List[Tuple[str, str]]:
    """依 iter_fail_items_ordered 順序，(正規化 rule, 分類)；同 norm 只保留第一次。"""
    seen: Set[str] = set()
    out: List[Tuple[str, str]] = []
    for cat, it in iter_fail_items_ordered(used_map):
        raw = str(it.get("rule", "") or "").strip()
        if not raw:
            continue
        norm = _normalize_short_rule(raw)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append((norm, cat))
    return out


def _sort_norm_cat_pairs_by_phrase_severity(
    pairs: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    decorated = [
        (
            RISK_PHRASE_SEVERITY_RANK.get(severity_for_normalized_rule(n), 99),
            i,
            n,
            c,
        )
        for i, (n, c) in enumerate(pairs)
    ]
    decorated.sort(key=lambda x: (x[0], x[1]))
    return [(x[2], x[3]) for x in decorated]


def ordered_fail_norm_and_category(used_map: dict) -> List[Tuple[str, str]]:
    """去重後依語氣層級排序的 (norm, category)，供短句與 primary risk 共用。"""
    return _sort_norm_cat_pairs_by_phrase_severity(
        _collect_fail_norm_and_category_deduped(used_map)
    )


def iter_fail_items_ordered(used_map: dict):
    """依嚴重度排序後，逐一產出 (category, rule_item dict)（僅 pass is False）。"""
    cats = sorted(
        ("Gate", "Trigger", "Guard", "Chip Notes"),
        key=lambda c: CATEGORY_PRIORITY.get(c, 99),
    )
    for cat in cats:
        for it in (used_map or {}).get(cat, []) or []:
            if it.get("pass", True) is False:
                yield cat, it


def get_primary_risk_rule_and_category(used_map: dict) -> Tuple[str, Optional[str]]:
    """
    第一條短風險：別名合併 → 去重 → 依語氣排序 → 口語化後顯示字串，及分類鍵。
    與 build_fail_lines_short_from_used_map 第一列一致。
    """
    pairs = ordered_fail_norm_and_category(used_map)
    if not pairs:
        return "", None
    n, c = pairs[0]
    return phrase_for_short_display(n), c


def get_primary_risk_category(used_map: dict) -> Optional[str]:
    _rule, cat = get_primary_risk_rule_and_category(used_map)
    return cat


def rule_item(
    key: str,
    rule: str,
    value: Any,
    threshold: str,
    passed: bool,
    note: str = "",
) -> dict:
    return {
        "key": key,
        "rule": rule,
        "value": value,
        "threshold": threshold,
        "pass": passed,
        "note": note,
    }


def build_used_map_for_signals(
    df: pd.DataFrame,
    latest: pd.Series,
    *,
    close_price: float,
    prev_close: Optional[float],
    bias_20_val: Optional[float],
    sma20_val: Optional[float],
    latest_volume: Optional[float],
    vol_ma20_val: Optional[float],
    trigger_type: str,
    foreign_divergence_warning: bool,
    foreign_net_latest: Optional[float],
    foreign_3d_net: Optional[float],
    trust_3d_net: Optional[float],
) -> Dict[str, List[dict]]:
    trigger_ok = bool(latest.get("BUY_TRIGGER", False))

    used_map: Dict[str, List[dict]] = {
        "Gate": [],
        "Trigger": [],
        "Guard": [],
        "Chip Notes": [],
    }

    used_map["Gate"].append(
        rule_item(
            key="BUY_GATE",
            rule="Gate 通過（趨勢 + 風險門檻）",
            value=bool(latest.get("BUY_GATE", False)),
            threshold="BUY_GATE == True",
            passed=bool(latest.get("BUY_GATE", False)),
            note="trend_ok: Close>SMA20 & SMA20上升；risk_ok: Vol<=1.5*VolMA20 & |Bias20|<=10",
        )
    )
    used_map["Trigger"].append(
        rule_item(
            key="BUY_TRIGGER",
            rule="Trigger 通過（PULLBACK/BREAKOUT/CONTINUATION）",
            value=str(latest.get("BUY_TRIGGER_TYPE", "NONE")),
            threshold="BUY_TRIGGER == True",
            passed=bool(latest.get("BUY_TRIGGER", False)),
            note="TriggerType = BUY_TRIGGER_TYPE",
        )
    )
    used_map["Guard"].append(
        rule_item(
            key="EXEC_GUARD",
            rule="Execution Guard 通過（防假突破/過熱/失守 AVWAP）",
            value=bool(latest.get("EXEC_GUARD", True)),
            threshold="EXEC_GUARD == True",
            passed=bool(latest.get("EXEC_GUARD", True)),
            note=str(latest.get("EXEC_BLOCK_REASON", "")).strip(),
        )
    )
    used_map["Chip Notes"].append(
        rule_item(
            key="chip_divergence",
            rule="籌碼背離（價漲但外資連賣）",
            value=bool(foreign_divergence_warning),
            threshold="False（理想狀態）",
            passed=not bool(foreign_divergence_warning),
            note="若為 True：建議 Gate/Trigger 降級或提高停損嚴格度",
        )
    )
    used_map["Gate"].append(
        rule_item(
            key="SMA20",
            rule="收盤 > SMA20",
            value=close_price,
            threshold="Close > SMA20",
            passed=bool(close_price > sma20_val) if sma20_val is not None else False,
        )
    )
    used_map["Gate"].append(
        rule_item(
            key="SMA20_up",
            rule="SMA20 走升（SMA20 > SMA20.shift(5)）",
            value=float(sma20_val) if sma20_val is not None and not pd.isna(sma20_val) else None,
            threshold="SMA20 > SMA20(5日前)",
            passed=bool(df["SMA20"].iloc[-1] > df["SMA20"].shift(5).iloc[-1])
            if len(df) >= 6 and not pd.isna(df["SMA20"].shift(5).iloc[-1])
            else False,
            note="趨勢門檻",
        )
    )
    used_map["Gate"].append(
        rule_item(
            key="Bias20_gate",
            rule="|SMA20乖離| <= 10（避免追高/過熱）",
            value=float(bias_20_val) if bias_20_val is not None and not pd.isna(bias_20_val) else None,
            threshold="abs(Bias20) <= 10",
            passed=bool(abs(bias_20_val) <= 10)
            if bias_20_val is not None and not pd.isna(bias_20_val)
            else False,
            note="風險門檻",
        )
    )
    used_map["Gate"].append(
        rule_item(
            key="Volume_gate",
            rule="Volume <= 1.5×VolMA20（避免爆量失控）",
            value=float(latest_volume) if latest_volume is not None and not pd.isna(latest_volume) else None,
            threshold="Volume <= 1.5×VolMA20",
            passed=bool(latest_volume <= 1.5 * vol_ma20_val)
            if (latest_volume is not None and vol_ma20_val is not None and not pd.isna(vol_ma20_val))
            else False,
            note="風險門檻",
        )
    )

    trigger_pullback_pass = (
        bool(trigger_type == "PULLBACK" and (-1 <= bias_20_val <= 3) and (close_price > prev_close))
        if (
            prev_close is not None
            and bias_20_val is not None
            and not pd.isna(bias_20_val)
        )
        else False
    )
    hhv20_prev = (
        to_scalar(df["High"].rolling(20, min_periods=20).max().shift(1).iloc[-1])
        if len(df) >= 21
        else np.nan
    )
    trigger_breakout_pass = bool(
        trigger_type == "BREAKOUT"
        and (close_price is not None and not pd.isna(hhv20_prev) and close_price > hhv20_prev)
        and (
            latest_volume is not None
            and vol_ma20_val is not None
            and not pd.isna(vol_ma20_val)
            and latest_volume > 1.2 * vol_ma20_val
        )
    )
    hhv5_prev = (
        to_scalar(df["High"].rolling(5, min_periods=5).max().shift(1).iloc[-1])
        if len(df) >= 6
        else np.nan
    )
    trigger_cont_pass = bool(
        trigger_type == "CONTINUATION"
        and (bias_20_val is not None and not pd.isna(bias_20_val) and 3 <= bias_20_val <= 10)
        and (close_price is not None and not pd.isna(hhv5_prev) and close_price > hhv5_prev)
    )

    used_map["Trigger"].append(
        rule_item(
            key="Trigger_type",
            rule="Trigger 類型",
            value=trigger_type,
            threshold="PULLBACK/BREAKOUT/CONTINUATION",
            passed=bool(trigger_ok),
            note="trigger_ok = True 才算點火",
        )
    )
    used_map["Trigger"].append(
        rule_item(
            key="Trigger_pullback",
            rule="回踩買：Bias20 -1~+3 且 Close > 昨收",
            value=f"Bias20={bias_20_val:.2f}, Close={close_price:.2f}, Prev={prev_close:.2f}"
            if (
                bias_20_val is not None
                and prev_close is not None
                and close_price is not None
                and not pd.isna(bias_20_val)
            )
            else None,
            threshold="(-1<=Bias20<=3) & (Close>PrevClose)",
            passed=trigger_pullback_pass,
        )
    )
    used_map["Trigger"].append(
        rule_item(
            key="Trigger_breakout",
            rule="突破買：Close > 前 20 日高 & Volume > 1.2×VolMA20",
            value=f"Close={close_price:.2f}, HHV20_prev={hhv20_prev:.2f}, Vol={int(latest_volume):,}, VolMA20={int(vol_ma20_val):,}"
            if (
                close_price is not None
                and not pd.isna(hhv20_prev)
                and latest_volume is not None
                and vol_ma20_val is not None
                and not pd.isna(vol_ma20_val)
            )
            else None,
            threshold="Close>HHV20.shift(1) & Vol>1.2×VolMA20",
            passed=trigger_breakout_pass,
        )
    )
    used_map["Trigger"].append(
        rule_item(
            key="Trigger_continuation",
            rule="延續買：Bias20 3~10 且 Close > 前 5 日高",
            value=f"Bias20={bias_20_val:.2f}, Close={close_price:.2f}, HHV5_prev={hhv5_prev:.2f}"
            if (
                bias_20_val is not None
                and close_price is not None
                and not pd.isna(hhv5_prev)
                and not pd.isna(bias_20_val)
            )
            else None,
            threshold="(3<=Bias20<=10) & (Close>HHV5.shift(1))",
            passed=trigger_cont_pass,
        )
    )

    k_range_latest = (
        float((latest["High"] - latest["Low"]))
        if ("High" in latest and "Low" in latest and (latest["High"] - latest["Low"]) != 0)
        else np.nan
    )
    close_pos_latest = (
        float((latest["Close"] - latest["Low"]) / k_range_latest)
        if (not pd.isna(k_range_latest))
        else np.nan
    )
    vol_ratio_20_latest = (
        float(latest_volume / vol_ma20_val)
        if (
            latest_volume is not None
            and vol_ma20_val is not None
            and not pd.isna(vol_ma20_val)
            and vol_ma20_val != 0
        )
        else np.nan
    )
    avwap_val = to_scalar(latest.get("AVWAP", np.nan))

    breakout_close_strong_pass = bool(close_pos_latest >= 0.6) if not pd.isna(close_pos_latest) else False
    not_crazy_volume_pass = bool(vol_ratio_20_latest <= 2.0) if not pd.isna(vol_ratio_20_latest) else False
    not_too_hot_pass = (
        bool(bias_20_val <= 9.5)
        if (bias_20_val is not None and not pd.isna(bias_20_val))
        else False
    )
    avwap_support_pass = bool(pd.isna(avwap_val) or close_price >= avwap_val) if (close_price is not None) else False

    is_strict_guard = trigger_type in ["BREAKOUT", "CONTINUATION"]

    used_map["Guard"].append(
        rule_item(
            key="Guard_strict_mode",
            rule="Guard 嚴格模式（BREAKOUT/CONTINUATION 才啟用）",
            value=trigger_type,
            threshold="Trigger in {BREAKOUT, CONTINUATION}",
            passed=is_strict_guard,
            note="Pullback 不走嚴格檢查",
        )
    )
    used_map["Guard"].append(
        rule_item(
            key="Guard_close_strong",
            rule="收盤位置要強（Close_pos >= 0.6）",
            value=float(close_pos_latest) if not pd.isna(close_pos_latest) else None,
            threshold=">= 0.6",
            passed=(breakout_close_strong_pass if is_strict_guard else True),
        )
    )
    used_map["Guard"].append(
        rule_item(
            key="Guard_vol_not_crazy",
            rule="量能不失控（Vol/VolMA20 <= 2.0）",
            value=float(vol_ratio_20_latest) if not pd.isna(vol_ratio_20_latest) else None,
            threshold="<= 2.0",
            passed=(not_crazy_volume_pass if is_strict_guard else True),
        )
    )
    used_map["Guard"].append(
        rule_item(
            key="Guard_not_too_hot",
            rule="乖離不貼近上限（Bias20 <= 9.5）",
            value=float(bias_20_val) if bias_20_val is not None and not pd.isna(bias_20_val) else None,
            threshold="<= 9.5",
            passed=(not_too_hot_pass if is_strict_guard else True),
        )
    )
    used_map["Guard"].append(
        rule_item(
            key="Guard_avwap_support",
            rule="成本線（AVWAP）不失守（Close >= AVWAP or AVWAP is NA）",
            value=float(avwap_val) if (avwap_val is not None and not pd.isna(avwap_val)) else "NA",
            threshold="Close >= AVWAP",
            passed=avwap_support_pass,
        )
    )

    used_map["Chip Notes"].append(
        rule_item(
            key="chip_divergence",
            rule="籌碼背離（股價上漲但外資近 3 日連賣）",
            value="True" if foreign_divergence_warning else "False",
            threshold="foreign_divergence_warning == False",
            passed=(not foreign_divergence_warning),
            note="若 True，Gate 可能降級為 WATCH",
        )
    )
    if foreign_net_latest is not None:
        used_map["Chip Notes"].append(
            rule_item(
                key="foreign_net_latest",
                rule="外資單日買賣超",
                value=float(foreign_net_latest),
                threshold="> 0（偏多）",
                passed=bool(foreign_net_latest > 0),
            )
        )
    if foreign_3d_net is not None:
        used_map["Chip Notes"].append(
            rule_item(
                key="foreign_3d_net",
                rule="外資 3 日累計買賣超",
                value=float(foreign_3d_net),
                threshold=">= 0（籌碼未散）",
                passed=bool(foreign_3d_net >= 0),
            )
        )
    if trust_3d_net is not None:
        used_map["Chip Notes"].append(
            rule_item(
                key="trust_3d_net",
                rule="投信 3 日累計買賣超",
                value=float(trust_3d_net),
                threshold="> 0（偏多）",
                passed=bool(trust_3d_net > 0),
            )
        )

    return used_map


def build_fail_lines_from_used_map(
    used_map: dict,
    *,
    max_lines: int = 5,
) -> List[str]:
    """與個股頁 LINE 推播：只列 FAIL，最多 max_lines 條；依 CATEGORY_PRIORITY 排序。"""
    fail_lines: List[str] = []
    for cat, it in iter_fail_items_ordered(used_map):
        rule = str(it.get("rule", "") or "").strip()
        note = str(it.get("note", "") or "").strip()
        fail_lines.append(f"- {cat}｜{rule}{('｜' + note) if note else ''}")
        if len(fail_lines) >= max_lines:
            break
    return fail_lines


def build_fail_lines_short_from_used_map(
    used_map: dict,
    *,
    max_lines: int = 5,
) -> List[str]:
    """
    短版風險句（僅口語化後文案，無 `- 分類｜` 前綴）。
    流程：依 FAIL 順序去重 → 依 RISK_PHRASE_ENTRIES 語氣層級排序 → 口語化 → 取前 max_lines 條。
    供精簡 LINE 尾段「風險：」濃縮顯示。
    """
    pairs = ordered_fail_norm_and_category(used_map)
    return [phrase_for_short_display(n) for n, _ in pairs][:max_lines]
