"""
LINE 推播：精簡／完整尾段與總長度保護（與頁面結論同源的前段由 build_compact_line_diagnosis 負責）。

精簡模式（compact）行數規格（尾段）：
- 補充：固定 1 行（bottom／top 合併）
- 防守：最多 1 行（有 guard 時）
- 風險：最多 2 行（無則省略）
- 不含「專家：」段落

前段由 final_decision_resolver.build_compact_line_diagnosis 產出（代碼、收盤、標題｜標籤、一句話、說明、AI）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from final_decision_resolver import ResolvedDecision, build_compact_line_diagnosis

LINE_PUSH_MAX_CHARS = 4800

# 精簡模式尾段「風險：」最多列幾條短句（與 append_line_push_tail 一致）
COMPACT_SHORT_RISK_MAX = 2

PRIMARY_RISK_CATEGORY_SHORT_LABEL: Dict[str, str] = {
    "Guard": "防守",
    "Gate": "結構",
    "Trigger": "動能",
    "Chip Notes": "籌碼",
}


def fuse_one_line_verdict_with_primary_risk(
    verdict: str,
    fail_lines_short: Optional[List[str]],
    *,
    risk_category: Optional[str] = None,
    show_risk_category: bool = False,
) -> str:
    """
    將第一條短版風險併入一句話（不修改 PlainLanguageNarrator，僅 LINE 前段展示用）。
    若一句話已含該風險字樣則不重複。
    show_risk_category：為 True 時附「｜分類」標籤（僅併入一句話時）。
    """
    if not fail_lines_short:
        return (verdict or "").strip()
    first = (fail_lines_short[0] or "").strip()
    if not first:
        return (verdict or "").strip()
    v = (verdict or "").strip()
    if first in v:
        return v if v else first
    if not v:
        return first
    if show_risk_category and risk_category:
        label = PRIMARY_RISK_CATEGORY_SHORT_LABEL.get(
            risk_category, risk_category
        )
        return f"{v}（{first}｜{label}）"
    return f"{v}（{first}）"


def ultra_compact_one_line(
    verdict: str,
    fail_lines_short: Optional[List[str]],
) -> str:
    """
    通知預覽／極短標題用：「主風險 → 結論」，不影響既有 fuse 路徑。
    無風險短句時退回一句話；皆空則「—」。
    """
    v = (verdict or "").strip()
    if not fail_lines_short:
        return v or "—"
    first = (fail_lines_short[0] or "").strip()
    if not first:
        return v or "—"
    if not v:
        return first
    return f"{first} → {v}"


def _turn_status_score_txt(tr: Any, denom: int) -> str:
    if not isinstance(tr, dict):
        return "NA"
    st = tr.get("status")
    try:
        sc = int(tr.get("score", 0))
    except (TypeError, ValueError):
        sc = 0
    return f"{st} {sc}/{denom}" if st is not None else "NA"


def truncate_line_push(text: str, max_chars: int = LINE_PUSH_MAX_CHARS) -> str:
    """避免超過 LINE 單則實務上限；過長時保留前段並標註截斷。"""
    if len(text) <= max_chars:
        return text
    cut = max(0, max_chars - 40)
    return text[:cut].rstrip() + "\n…（訊息過長，以下截斷）"


def append_line_push_tail(
    lines: List[str],
    *,
    compact: bool,
    bottom_txt: str,
    top_txt: str,
    guard: Optional[Any],
    defense_name: str,
    fail_lines: List[str],
    expert_msg: str,
    fail_lines_short: Optional[List[str]] = None,
) -> None:
    """
    固定標籤：補充（TURN）→ 防守 → 風險 → 專家（僅完整模式）。
    精簡模式：不附專家全文；風險最多 2 條；防守單行濃縮。
    """
    lines.append(f"補充：TURN bottom｜{bottom_txt}；TURN top｜{top_txt}")
    if guard is not None:
        if compact:
            lines.append(
                f"防守：明日保命 {float(guard.break_close):.2f}｜"
                f"警戒 {float(guard.guard_close):.2f}（{defense_name}）"
            )
        else:
            lines.append(
                f"防守：明日保命價 {float(guard.break_close):.2f}｜"
                f"保守警戒(+{float(guard.buffer_pct):.1f}%) {float(guard.guard_close):.2f}"
            )
            lines.append(f"防守（參考）：{defense_name}={float(guard.ema_today):.2f}")
    if fail_lines:
        if compact:
            if fail_lines_short:
                first = (fail_lines_short[0] or "").strip()
                lines.append(f"風險：{first}")
                if len(fail_lines_short) > 1:
                    second = (fail_lines_short[1] or "").strip()
                    lines.append(f"風險（續）：{second}")
            else:
                first = fail_lines[0].lstrip("- ").strip()
                lines.append(f"風險：{first}")
                if len(fail_lines) > 1:
                    second = fail_lines[1].lstrip("- ").strip()
                    lines.append(f"風險（續）：{second}")
        else:
            lines.append("風險：")
            lines.extend(fail_lines)
    if not compact:
        em = (expert_msg or "").strip()
        if em:
            lines.append("專家：")
            lines.append(em)


def build_line_push_payload(
    *,
    mode: Literal["compact", "full"],
    ticker: str,
    name: str = "",
    close_price: float,
    score: int,
    has_position: bool,
    decision: Optional[ResolvedDecision],
    one_line_verdict: str = "",
    summary_fallback: str = "",
    bottom_now: Any = None,
    top_now: Any = None,
    guard: Optional[Any] = None,
    defense_name: str = "",
    fail_lines: Optional[List[str]] = None,
    fail_lines_short: Optional[List[str]] = None,
    expert_msg: str = "",
    max_chars: int = LINE_PUSH_MAX_CHARS,
    merge_primary_risk_into_verdict: bool = False,
    primary_risk_category: Optional[str] = None,
    merge_primary_risk_show_category: bool = False,
    ultra_compact_head: bool = False,
) -> str:
    """
    單一入口：前段（結論契約）＋後段（補充／防守／風險／專家）＋截斷。
    app 層只需蒐集資料、選 mode、發送。
    fail_lines_short：精簡模式時可只傳 rule 短句（無 `- 分類｜`）；完整模式仍用 fail_lines。
    merge_primary_risk_into_verdict：為 True 時，將 fail_lines_short 第一條併入前段一句話（僅輸出層）。
    精簡模式會將 fail_lines_short 截成最多 COMPACT_SHORT_RISK_MAX 條再顯示／融合。
    ultra_compact_head：精簡模式且為 True 時，前段一句話改為「主風險 → 結論」（略過 fuse／category 融合）。
    """
    bottom_txt = _turn_status_score_txt(bottom_now, 4)
    top_txt = _turn_status_score_txt(top_now, 5)
    eff_short = [(x or "").strip() for x in (fail_lines_short or [])]
    eff_short = [x for x in eff_short if x]
    if mode == "compact":
        eff_short = eff_short[:COMPACT_SHORT_RISK_MAX]
    verdict_for_head = one_line_verdict
    if ultra_compact_head and mode == "compact" and eff_short:
        verdict_for_head = ultra_compact_one_line(one_line_verdict, eff_short)
    elif merge_primary_risk_into_verdict and eff_short:
        verdict_for_head = fuse_one_line_verdict_with_primary_risk(
            one_line_verdict,
            eff_short,
            risk_category=primary_risk_category,
            show_risk_category=merge_primary_risk_show_category,
        )
    head = build_compact_line_diagnosis(
        ticker=ticker,
        name=name,
        close_price=close_price,
        score=score,
        has_position=has_position,
        decision=decision,
        one_line_verdict=verdict_for_head,
        summary_fallback=summary_fallback,
    )
    lines = head.split("\n")
    append_line_push_tail(
        lines,
        compact=(mode == "compact"),
        bottom_txt=bottom_txt,
        top_txt=top_txt,
        guard=guard,
        defense_name=defense_name,
        fail_lines=list(fail_lines or []),
        expert_msg=expert_msg,
        fail_lines_short=eff_short if mode == "compact" else fail_lines_short,
    )
    return truncate_line_push("\n".join(lines), max_chars=max_chars)
