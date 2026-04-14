"""
群組查股：載入行情 → ResolvedDecision → PlainLanguageSummary → build_line_push_payload。
多檔：精簡一行列示（與單檔精簡一句話＋主風險融合同源）。
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from diagnosis_snapshot import (
    LineDiagnosisSnapshot,
    build_line_diagnosis_snapshot,
    resolve_decision_for_position_mode,
)
from final_decision_resolver import ACTION_UI, FinalAction
from indicator_used_map import (
    build_fail_lines_from_used_map,
    build_fail_lines_short_from_used_map,
    build_used_map_for_signals,
    get_primary_risk_category,
)
from line_group_query_bot import GroupQueryCommand
from line_push_formatter import (
    COMPACT_SHORT_RISK_MAX,
    PRIMARY_RISK_CATEGORY_SHORT_LABEL,
    build_line_push_payload,
    fuse_one_line_verdict_with_primary_risk,
)
from line_quick_reply import display_code_for_quick_reply
from plain_language_narrator import (
    PlainLanguageNarratorInput,
    PlainLanguageSummary,
    build_plain_language_summary,
    can_buy_the_dip,
)
from tomorrow_guard_price import calc_tomorrow_guard
from utils import to_scalar

# 多檔列示：最嚴重優先（與群組「先看風險」一致）
MULTI_MAX_TICKERS = 5
_MULTI_ACTION_SORT_KEY = {
    FinalAction.EXIT: 0,
    FinalAction.REDUCE: 1,
    FinalAction.WATCH: 2,
    FinalAction.HOLD: 3,
}
_MULTI_SUMMARY_ORDER = (
    FinalAction.EXIT,
    FinalAction.REDUCE,
    FinalAction.WATCH,
    FinalAction.HOLD,
)
_PRIMARY_CATEGORY_TIEBREAK_ORDER = {
    "Guard": 0,
    "Gate": 1,
    "Trigger": 2,
    "Chip Notes": 3,
}


def _summary_bias(counter: Counter) -> str:
    """依檔數占比與 HOLD/EXIT 比，回傳偏多／偏空／中性；無檔回空字串。
    純觀望（無續抱）不判偏多，避免「整體：偏多（3 觀望）」不合理語感。
    """
    total = sum(counter.values())
    if total == 0:
        return ""
    n_exit = counter.get(FinalAction.EXIT, 0)
    n_reduce = counter.get(FinalAction.REDUCE, 0)
    n_hold = counter.get(FinalAction.HOLD, 0)
    risk = n_exit + n_reduce
    if risk / total >= 0.5:
        return "偏空"
    if n_hold > 0 and n_hold >= n_exit:
        return "偏多"
    return "中性"


def _multi_summary_line(counts: Counter) -> str:
    parts: List[str] = []
    for act in _MULTI_SUMMARY_ORDER:
        n = counts.get(act, 0)
        if n:
            parts.append(f"{n} {ACTION_UI[act]['label']}")
    inner = "・".join(parts)
    if not inner:
        return "整體：無有效判斷"
    bias = _summary_bias(counts)
    if bias:
        return f"整體：{bias}（{inner}）"
    return f"整體：{inner}"


def _dominant_primary_category(results: List[dict]) -> Optional[str]:
    """回傳主風險分類鍵；同數量時依 Guard→Gate→Trigger→Chip 穩定決勝。"""
    cat_counter: Counter = Counter(
        str(r.get("primary_cat"))
        for r in results
        if str(r.get("primary_cat") or "") in PRIMARY_RISK_CATEGORY_SHORT_LABEL
    )
    if not cat_counter:
        return None

    def _cat_sort_key(cat: str) -> tuple[int, int, str]:
        return (
            -int(cat_counter.get(cat, 0)),
            _PRIMARY_CATEGORY_TIEBREAK_ORDER.get(cat, 99),
            cat,
        )

    return sorted(cat_counter.keys(), key=_cat_sort_key)[0]


def _chip_3d_net(series: Optional[pd.Series]) -> Optional[float]:
    if series is None or len(series) < 3:
        return None
    return float(series.sort_index().tail(3).sum())


def _guard_and_defense(
    snap: LineDiagnosisSnapshot,
    *,
    pos_exit_style: str = "波段守五日線",
):
    _defense = snap.ema20 if pos_exit_style == "長線守月線" else snap.ema5
    _defense_name = "EMA20（月線）" if pos_exit_style == "長線守月線" else "EMA5（五日線）"
    guard = None
    if _defense is not None:
        guard = calc_tomorrow_guard(
            ema_today=float(_defense),
            window=20 if pos_exit_style == "長線守月線" else 5,
            buffer_pct=1.5,
        )
    return guard, _defense_name


def _decision_narr_pl(
    snap: LineDiagnosisSnapshot, *, position_mode: bool
) -> Tuple[Any, PlainLanguageNarratorInput, PlainLanguageSummary]:
    decision = resolve_decision_for_position_mode(snap, position_mode=position_mode)
    if decision is None:
        decision = snap.page_resolved_decision

    latest = snap.latest
    narr_in = PlainLanguageNarratorInput(
        close=float(snap.close_price),
        ema5=float(snap.ema5) if snap.ema5 is not None else None,
        ema20=float(snap.ema20) if snap.ema20 is not None else None,
        gate_ok=snap.gate_ok,
        trigger_ok=snap.trigger_ok,
        guard_ok=bool(snap.exec_guard_ok),
        trigger_type=str(snap.trigger_type),
        bottom_status=(
            str(snap.bottom_now.get("status", "NA"))
            if isinstance(snap.bottom_now, dict)
            else None
        ),
        top_status=(
            str(snap.top_now.get("status", "NA"))
            if isinstance(snap.top_now, dict)
            else None
        ),
        has_position=position_mode,
        risk_alert=bool(latest.get("Is_Dangerous_Volume", False))
        or bool(snap.foreign_divergence_warning),
    )
    pl_summary = build_plain_language_summary(decision, narr_in)
    return decision, narr_in, pl_summary


def _bracket_code(raw: str, normalized: str) -> str:
    ru = (raw or "").strip().upper().replace(".TW", "")
    if re.fullmatch(r"\d{4}", ru):
        return ru
    m = re.search(r"(\d{4})", (normalized or "").upper())
    return m.group(1) if m else "????"


def _fused_line_and_final_action(
    cmd: GroupQueryCommand, *, time_range: str
) -> Tuple[str, Optional[FinalAction], Optional[str]]:
    """單檔精簡一句、裁決 action、主風險分類鍵；dip 僅回文字、action／category 為 None。"""
    if len(cmd.tickers_normalized) != 1:
        raise ValueError("expected single ticker")
    snap = build_line_diagnosis_snapshot(cmd.ticker_normalized, time_range=time_range)
    position_mode = bool(cmd.has_position_mode)
    decision, narr_in, pl_summary = _decision_narr_pl(snap, position_mode=position_mode)

    if cmd.action == "dip":
        dip = can_buy_the_dip(
            close=float(snap.close_price),
            ema5=narr_in.ema5,
            trigger_type=str(snap.trigger_type),
            guard_ok=bool(snap.exec_guard_ok),
            risk_alert=narr_in.risk_alert,
            near_support_red_bar=narr_in.near_support_red_bar,
        )
        return dip.explanation.strip(), None, None

    prev_close = to_scalar(snap.df["Close"].iloc[-2]) if len(snap.df) >= 2 else None
    bias_20_val = to_scalar(snap.latest.get("Bias20", np.nan))
    sma20_val = to_scalar(snap.latest.get("SMA20", np.nan))
    latest_volume = to_scalar(snap.latest.get("Volume"))
    vol_ma20_val = to_scalar(snap.latest.get("VolMA20", np.nan))

    fn = snap.foreign_net_series
    foreign_net_latest = float(fn.iloc[-1]) if fn is not None and not fn.empty else None
    foreign_3d_net = _chip_3d_net(fn)
    trust_3d_net = _chip_3d_net(snap.trust_net_series)

    used_map = build_used_map_for_signals(
        snap.df,
        snap.latest,
        close_price=float(snap.close_price),
        prev_close=prev_close,
        bias_20_val=bias_20_val,
        sma20_val=sma20_val,
        latest_volume=latest_volume,
        vol_ma20_val=vol_ma20_val,
        trigger_type=str(snap.trigger_type),
        foreign_divergence_warning=snap.foreign_divergence_warning,
        foreign_net_latest=foreign_net_latest,
        foreign_3d_net=foreign_3d_net,
        trust_3d_net=trust_3d_net,
    )
    fail_lines_short = build_fail_lines_short_from_used_map(
        used_map, max_lines=COMPACT_SHORT_RISK_MAX
    )
    primary = get_primary_risk_category(used_map)
    verdict = (pl_summary.one_line_verdict or "").strip()
    fused = fuse_one_line_verdict_with_primary_risk(
        verdict,
        fail_lines_short or None,
        risk_category=primary,
        show_risk_category=True,
    )
    return fused, decision.action, primary


def _fused_head_or_dip_line(cmd: GroupQueryCommand, *, time_range: str) -> str:
    text, _, _ = _fused_line_and_final_action(cmd, time_range=time_range)
    return text


def _run_multi_stock_summary(
    cmd: GroupQueryCommand,
    *,
    time_range: str,
    multi_ultra_show_category: bool = False,
) -> str:
    total_n = len(cmd.tickers_normalized)
    norms = cmd.tickers_normalized[:MULTI_MAX_TICKERS]
    raws = cmd.tickers_raw[:MULTI_MAX_TICKERS]
    over_n = max(0, total_n - MULTI_MAX_TICKERS)

    results: List[dict] = []
    for nt, rw in zip(norms, raws):
        sub = replace(
            cmd,
            tickers_normalized=(nt,),
            tickers_raw=(rw,),
        )
        code = _bracket_code(rw, nt)
        sub_eval = (
            replace(sub, action="compact") if cmd.action == "multi_ultra" else sub
        )
        body, fin_act, primary_cat = _fused_line_and_final_action(
            sub_eval, time_range=time_range
        )
        results.append(
            {
                "code": code,
                "line": f"【{code}】{body}",
                "action": fin_act,
                "primary_cat": primary_cat,
            }
        )

    if cmd.action == "dip" or all(r["action"] is None for r in results):
        lines_out = [r["line"] for r in results]
        out = "\n".join(lines_out)
    else:
        counts: Counter = Counter(
            r["action"] for r in results if r["action"] is not None
        )
        summary_line = _multi_summary_line(counts)
        dominant_cat = _dominant_primary_category(results)
        if dominant_cat:
            summary_line += (
                f"｜主因：{PRIMARY_RISK_CATEGORY_SHORT_LABEL[dominant_cat]}"
            )

        def _row_sort_key(row: dict) -> tuple[int, str]:
            act = row["action"]
            assert isinstance(act, FinalAction)
            return (_MULTI_ACTION_SORT_KEY.get(act, 99), str(row["code"]))

        results.sort(key=_row_sort_key)
        if cmd.action == "multi_ultra":
            ultra_lines: List[str] = []
            for r in results:
                act = r["action"]
                if isinstance(act, FinalAction):
                    act_zh = ACTION_UI[act]["label"]
                    cat_key = r.get("primary_cat")
                    cat_zh = (
                        PRIMARY_RISK_CATEGORY_SHORT_LABEL.get(str(cat_key or ""), "")
                        or ""
                    )
                    if multi_ultra_show_category and cat_zh:
                        ultra_lines.append(f"{r['code']} → {act_zh}｜{cat_zh}")
                    else:
                        ultra_lines.append(f"{r['code']} → {act_zh}")
            out = "\n".join([summary_line, ""] + ultra_lines)
        else:
            out = "\n".join([summary_line, ""] + [r["line"] for r in results])

    if cmd.action == "full":
        ex = display_code_for_quick_reply(raws[0] if raws else "")
        out += f"\n\n📌 單檔完整分析：完整 {ex}"

    if over_n > 0:
        out += f"\n\n（共 {total_n} 檔，僅顯示前 {MULTI_MAX_TICKERS} 檔）"
    return out


def _run_single_stock_query(cmd: GroupQueryCommand, *, time_range: str) -> str:
    snap = build_line_diagnosis_snapshot(cmd.ticker_normalized, time_range=time_range)

    position_mode = bool(cmd.has_position_mode)
    decision, narr_in, pl_summary = _decision_narr_pl(snap, position_mode=position_mode)

    if cmd.action == "dip":
        dip = can_buy_the_dip(
            close=float(snap.close_price),
            ema5=narr_in.ema5,
            trigger_type=str(snap.trigger_type),
            guard_ok=bool(snap.exec_guard_ok),
            risk_alert=narr_in.risk_alert,
            near_support_red_bar=narr_in.near_support_red_bar,
        )
        return (
            "撿便宜判斷\n"
            f"{dip.explanation}\n"
            f"{pl_summary.one_line_verdict}\n"
            f"{pl_summary.what_to_wait_for}"
        ).strip()

    mode = "full" if cmd.action == "full" else "compact"

    guard, defense_name = _guard_and_defense(snap)

    _sum_fb = (
        (pl_summary.what_to_wait_for or "").strip()
        or (pl_summary.current_state or "").strip()
    )

    prev_close = to_scalar(snap.df["Close"].iloc[-2]) if len(snap.df) >= 2 else None
    bias_20_val = to_scalar(snap.latest.get("Bias20", np.nan))
    sma20_val = to_scalar(snap.latest.get("SMA20", np.nan))
    latest_volume = to_scalar(snap.latest.get("Volume"))
    vol_ma20_val = to_scalar(snap.latest.get("VolMA20", np.nan))

    fn = snap.foreign_net_series
    foreign_net_latest = float(fn.iloc[-1]) if fn is not None and not fn.empty else None
    foreign_3d_net = _chip_3d_net(fn)
    trust_3d_net = _chip_3d_net(snap.trust_net_series)

    used_map = build_used_map_for_signals(
        snap.df,
        snap.latest,
        close_price=float(snap.close_price),
        prev_close=prev_close,
        bias_20_val=bias_20_val,
        sma20_val=sma20_val,
        latest_volume=latest_volume,
        vol_ma20_val=vol_ma20_val,
        trigger_type=str(snap.trigger_type),
        foreign_divergence_warning=snap.foreign_divergence_warning,
        foreign_net_latest=foreign_net_latest,
        foreign_3d_net=foreign_3d_net,
        trust_3d_net=trust_3d_net,
    )
    fail_lines = build_fail_lines_from_used_map(used_map, max_lines=5)
    _short_max = COMPACT_SHORT_RISK_MAX if mode == "compact" else 5
    fail_lines_short = build_fail_lines_short_from_used_map(
        used_map, max_lines=_short_max
    )
    _primary_cat = get_primary_risk_category(used_map)

    return build_line_push_payload(
        mode=mode,
        ticker=str(snap.effective_symbol),
        name=str(snap.name or ""),
        close_price=float(snap.close_price),
        score=int(snap.score),
        has_position=position_mode,
        decision=decision,
        one_line_verdict=(pl_summary.one_line_verdict or "").strip(),
        summary_fallback=_sum_fb,
        bottom_now=snap.bottom_now,
        top_now=snap.top_now,
        guard=guard,
        defense_name=defense_name,
        fail_lines=fail_lines,
        fail_lines_short=fail_lines_short,
        expert_msg=str(snap.expert_msg or ""),
        merge_primary_risk_into_verdict=True,
        primary_risk_category=_primary_cat,
        merge_primary_risk_show_category=True,
    )


def run_group_stock_query(
    cmd: GroupQueryCommand,
    *,
    time_range: str = "1y",
    multi_ultra_show_category: bool = False,
) -> str:
    if len(cmd.tickers_normalized) > 1 or cmd.action == "multi_ultra":
        return _run_multi_stock_summary(
            cmd,
            time_range=time_range,
            multi_ultra_show_category=multi_ultra_show_category,
        )
    return _run_single_stock_query(cmd, time_range=time_range)
