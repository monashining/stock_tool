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

import re
from typing import Any, Dict, List, Literal, Optional

from diagnosis_scoring import SCORING_VERSION
from final_decision_resolver import (
    ACTION_UI,
    ResolvedDecision,
    build_compact_line_diagnosis,
    get_status_bar_label,
    get_status_bar_title,
)
from position_advice import PositionAdvice, build_exit_guide_push_text

LINE_PUSH_MAX_CHARS = 4800

# 一般版 LINE 固定章節名（全站／群組／網頁推播一致，勿混用其他別名）
READER_LINE_HEADING_EXIT = "📍 下車指南"
READER_LINE_HEADING_EXPERT = "咸魚翻身｜AI 專家診斷"

# 精簡模式尾段「風險：」最多列幾條短句（與 append_line_push_tail 一致）
COMPACT_SHORT_RISK_MAX = 2

PRIMARY_RISK_CATEGORY_SHORT_LABEL: Dict[str, str] = {
    "Guard": "防守",
    "Gate": "結構",
    "Trigger": "上漲力道",
    "Chip Notes": "籌碼",
}


def interpret_diagnosis_score_tier(score: int) -> str:
    """診斷總分（0–100）粗分層，供一般版 LINE 一句話解讀。"""
    s = int(score)
    if s >= 85:
        return "強勢多頭"
    if s >= 70:
        return "中等偏多"
    if s >= 55:
        return "中性偏多"
    return "偏弱"


def interpret_diagnosis_score_mood(*, bias20_pct: Optional[float]) -> str:
    """搭配 Bias20（%）的第二語，避免每則訊息過長。"""
    if bias20_pct is None:
        return "趨勢穩定"
    try:
        b = float(bias20_pct)
    except (TypeError, ValueError):
        return "趨勢穩定"
    if b != b or abs(b) == float("inf"):
        return "趨勢穩定"
    if b > 10:
        return "動能偏熱"
    if b < -8:
        return "動能偏冷"
    return "趨勢穩定"


def format_line_reader_diagnosis_score_line(
    score: int, *, bias20_pct: Optional[float]
) -> str:
    tier = interpret_diagnosis_score_tier(score)
    mood = interpret_diagnosis_score_mood(bias20_pct=bias20_pct)
    return (
        f"診斷分數：{int(score)}（{tier}，{mood}）（模型 {SCORING_VERSION}）"
    )


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


def strip_markdown_for_line_push(text: str) -> str:
    """LINE 純文字：去掉常見 Markdown 粗體符號。"""
    return (text or "").replace("**", "").strip()


def strip_redundant_stock_name_from_line_expert_text(text: str, name: str) -> str:
    """
    LINE 一般版「專家診斷」：主結論標題已含股名時，刪除講評內重複的同一顯示名稱，
    例如「觀望保守：ChipMOS TECHNOLOGIES INC. 評分 44」→「觀望保守：評分 44」。
    僅處理「：／:」後緊接完整名稱再接後文之典型句式，避免誤傷他處字串。
    """
    s = (text or "").strip()
    n = (name or "").strip()
    if len(n) < 2 or n not in s:
        return s
    esc = re.escape(n)
    s = re.sub(rf"([：:])\s*{esc}\s+", r"\1", s)
    while "：：" in s:
        s = s.replace("：：", "：")
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()


def scrub_line_push_engineering_terms(text: str) -> str:
    """
    一般版 LINE：將殘留工程語改為人話（下車指南／專家段皆可套）。
    僅替換已知片語，避免誤傷公司名等；改規則集中在此。
    """
    if not (text or "").strip():
        return (text or "").strip()
    s = text
    pairs = [
        ("top 轉弱/過熱", "漲多轉弱"),
        ("top 風險升溫", "高檔風險升溫"),
        ("top 風險", "高檔風險"),
        ("top 轉弱", "漲多轉弱"),
        ("+ top 轉弱", "漲多轉弱"),
        (" + top 轉弱", "漲多轉弱"),
        ("且 top 風險升溫", "且高檔風險升溫"),
        ("且 top ", "且高檔"),
        ("bottom 分數不足", "進場訊號偏弱"),
        ("Profit >", "獲利逾"),
        ("TURN ", ""),
        (" TURN", ""),
    ]
    for a, b in pairs:
        s = s.replace(a, b)
    for token in (
        "[STRUCTURE]",
        "[PRICE]",
        "[GUARD]",
        "[RISK]",
        "[AI]",
        "[BUY]",
    ):
        s = s.replace(token, "")
    s = s.replace("bottom =", "進場=").replace("top =", "高檔=")
    s = re.sub(r"\bALLOW\b", "綠燈", s, flags=re.IGNORECASE)
    s = re.sub(r"\bWATCH\b", "黃燈", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBLOCK\b", "紅燈", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTrigger\b", "點火條件", s, flags=re.IGNORECASE)
    s = re.sub(r"\bGuard\b", "執行保護", s, flags=re.IGNORECASE)
    s = re.sub(r"\bGate\b", "進場門檻", s, flags=re.IGNORECASE)
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()


def _finite_pct(x: Any) -> bool:
    try:
        v = float(x)
        return v == v and abs(v) != float("inf")
    except (TypeError, ValueError):
        return False


def normalize_entry_reason(reason_key: str) -> str:
    """
    建倉一句話「括號內」用語：內部鍵 → 結果導向、同一語氣層；改文案只改此 mapping。
    reason_key：如 gate_fail、overheat、allow_default、allow_continuation …
    """
    mapping = {
        # BLOCK
        "gate_fail": "進場條件不足",
        "overheat": "過熱",
        "volume_spike": "爆量",
        # WATCH（Guard 未過 → 不寫工程詞「護欄」）
        "guard_fail": "風險偏高",
        "trigger_fail": "上漲力道不足",
        "watch_fallback": "條件未到",
        # ALLOW
        "allow_pullback": "回檔",
        "allow_breakout": "突破",
        "allow_continuation": "漲勢延續",
        "allow_default": "條件到位",
    }
    k = (reason_key or "").strip()
    return mapping.get(k, k or "—")


def build_entry_advice_one_line_for_push(
    *,
    gate_ok: bool,
    trigger_ok: bool,
    guard_ok: bool,
    trigger_type: str = "",
    bias20_pct: Optional[float] = None,
    volume_spike: bool = False,
) -> str:
    """
    建倉一句話（僅 LINE 一般版）：接在主結論後，不含 Gate 表或整區建倉文案。
    guard_ok 對應頁面 EXEC_GUARD（與 BUY_SIGNAL 敘述一致）。
    """
    overall = bool(gate_ok and trigger_ok and guard_ok)
    overheat_bias = bool(
        bias20_pct is not None
        and _finite_pct(bias20_pct)
        and abs(float(bias20_pct)) > 10.0
    )

    if overall:
        tt = (trigger_type or "").strip().upper()
        rkey = {
            "PULLBACK": "allow_pullback",
            "BREAKOUT": "allow_breakout",
            "CONTINUATION": "allow_continuation",
        }.get(tt, "allow_default")
        reason = normalize_entry_reason(rkey)
        return f"🚩 建倉建議：條件偏多，可考慮分批進場（{reason}）"

    if not gate_ok:
        if overheat_bias:
            return (
                "🚩 建倉建議：目前過熱，暫不建議進場（"
                f"{normalize_entry_reason('overheat')}）"
            )
        if volume_spike:
            return (
                "🚩 建倉建議：暫不建議進場（"
                f"{normalize_entry_reason('volume_spike')}）"
            )
        return (
            "🚩 建倉建議：暫不建議進場（"
            f"{normalize_entry_reason('gate_fail')}）"
        )

    if not trigger_ok:
        return (
            "🚩 建倉建議：條件尚未成熟，建議等待（"
            f"{normalize_entry_reason('trigger_fail')}）"
        )

    if not guard_ok:
        return (
            "🚩 建倉建議：條件尚未成熟，建議等待（"
            f"{normalize_entry_reason('guard_fail')}）"
        )

    return (
        "🚩 建倉建議：條件尚未成熟，建議等待（"
        f"{normalize_entry_reason('watch_fallback')}）"
    )


def build_main_conclusion_push_text(
    *,
    ticker: str,
    name: str = "",
    close_price: float,
    score: int,
    has_position: bool,
    decision: Optional[ResolvedDecision],
) -> str:
    """對齊網頁「主結論」卡：標的列＋燈號標題＋一句說明（不含 reason_code／trace）。"""
    parts = [f"【{ticker}】"]
    if name:
        parts.append(str(name))
    lines = [" ".join(parts).strip(), f"收盤參考：{close_price:.2f}", ""]
    if decision is None:
        lines.append("資料不足，尚無法產生主結論。")
        lines.append(f"診斷分數（參考）：{score}（模型 {SCORING_VERSION}）")
        return "\n".join(lines)
    ui = ACTION_UI[decision.action]
    title = get_status_bar_title(has_position)
    lab = get_status_bar_label(decision, has_position)
    lines.append(f"{ui['emoji']} {ui['label']}｜{decision.summary_title}")
    lines.append(f"{title}：{lab}")
    st = (decision.summary_text or "").strip()
    if st:
        lines.append(st)
    ea = strip_markdown_for_line_push(decision.expert_action_line or "")
    if ea and ea not in st:
        lines.append(ea)
    lines.append(f"診斷分數（參考）：{score}（模型 {SCORING_VERSION}）")
    return "\n".join(lines)


def split_reader_plain_main_header_and_verdict(main: str) -> tuple[str, str]:
    """
    將 build_main_conclusion_push_text 產出拆成：
    - header：【代碼】名稱 + 收盤參考（兩行）
    - verdict：燈號標題／標的狀態／說明／診斷分數等其餘主結論內容
    """
    lines = (main or "").split("\n")
    if len(lines) < 2:
        return ((main or "").strip(), "")
    header = "\n".join(lines[:2]).strip()
    rest = lines[2:]
    while rest and not (rest[0] or "").strip():
        rest = rest[1:]
    verdict = "\n".join(rest).strip()
    return header, verdict


def strip_embedded_verdict_block_from_expert_plain(
    expert_plain: str, verdict: str
) -> str:
    """若專家全文內誤含與主結論 verdict 相同整段，移除（保留專家講評本體）。"""
    s = (expert_plain or "").strip()
    v = (verdict or "").strip()
    if len(v) < 12 or not s:
        return s
    if v in s:
        s = s[: s.find(v)].rstrip()
    return s.strip()


def strip_trailing_duplicate_expert_action_from_plain(
    expert_plain: str, verdict: str
) -> str:
    """
    generate_expert_advice 常在文末附「---」+ 行動句；主結論 verdict 已含同一句時刪除專家段尾重複。
    """
    s = (expert_plain or "").strip()
    v = (verdict or "").strip()
    if not s or not v:
        return s
    for sep in ("\n\n---\n", "\n---\n", "\n\n---", "\n---"):
        if sep not in s:
            continue
        head, tail = s.rsplit(sep, 1)
        tail = tail.strip()
        if not tail:
            s = head.strip()
            continue
        if tail in v or any(line.strip() == tail for line in v.split("\n") if line.strip()):
            s = head.strip()
            break
    return s.strip()


def build_line_push_reader_plain(
    *,
    ticker: str,
    name: str = "",
    close_price: float,
    score: int,
    has_position: bool,
    decision: Optional[ResolvedDecision],
    expert_msg: str,
    avg_cost: float,
    exit_style: str,
    ema5: Optional[float],
    ema20: Optional[float],
    bottom_result: Optional[Dict[str, Any]] = None,
    top_result: Optional[Dict[str, Any]] = None,
    turn_result: Optional[Dict[str, Any]] = None,
    position_advice: PositionAdvice,
    entry_gate_ok: bool,
    entry_trigger_ok: bool,
    entry_guard_ok: bool,
    entry_trigger_type: str = "",
    entry_bias20_pct: Optional[float] = None,
    entry_volume_spike: bool = False,
    max_chars: int = LINE_PUSH_MAX_CHARS,
) -> str:
    """
    一般使用者版 LINE：標的列＋收盤 → 咸魚翻身｜AI 專家診斷（置頂）→ 診斷分數一行
    → 建倉一句話 → 下車指南；不含 TURN 代碼列、used_map 風險條、精準診斷儀表板全文。
    不含「燈號＋續抱/減碼｜主結論標題」整段（與專家敘述重疊，已由產品決定省略）。
    """
    main = build_main_conclusion_push_text(
        ticker=ticker,
        name=name,
        close_price=close_price,
        score=score,
        has_position=has_position,
        decision=decision,
    )
    header, verdict = split_reader_plain_main_header_and_verdict(main)
    entry_line = build_entry_advice_one_line_for_push(
        gate_ok=bool(entry_gate_ok),
        trigger_ok=bool(entry_trigger_ok),
        guard_ok=bool(entry_guard_ok),
        trigger_type=str(entry_trigger_type or ""),
        bias20_pct=entry_bias20_pct,
        volume_spike=bool(entry_volume_spike),
    )
    exit_txt = scrub_line_push_engineering_terms(
        build_exit_guide_push_text(
            close_last=float(close_price),
            avg_cost=float(avg_cost or 0.0),
            exit_style=str(exit_style or "波段守五日線"),
            ema5=ema5,
            ema20=ema20,
            bottom_result=bottom_result,
            top_result=top_result,
            turn_result=turn_result,
            advice=position_advice,
            section_heading=READER_LINE_HEADING_EXIT,
        )
    )
    expert_plain = strip_markdown_for_line_push(expert_msg)
    expert_plain = scrub_line_push_engineering_terms(expert_plain)
    expert_plain = strip_redundant_stock_name_from_line_expert_text(
        expert_plain, name=str(name or "").strip()
    )
    expert_plain = strip_embedded_verdict_block_from_expert_plain(expert_plain, verdict)
    expert_plain = strip_trailing_duplicate_expert_action_from_plain(expert_plain, verdict)
    expert_plain = (expert_plain or "").strip()
    parts_out: List[str] = [
        header,
        "",
        READER_LINE_HEADING_EXPERT,
        expert_plain or "（無）",
        "",
        format_line_reader_diagnosis_score_line(
            int(score), bias20_pct=entry_bias20_pct
        ),
    ]
    parts_out.extend(["", entry_line, "", exit_txt])
    body = "\n".join(parts_out)
    return truncate_line_push(body, max_chars=max_chars)
