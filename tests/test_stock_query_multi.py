"""多檔列示：整體摘要、依 FinalAction 排序、最多 5 檔（mock 資料層）。"""

from __future__ import annotations

from collections import Counter
from unittest.mock import patch

from final_decision_resolver import FinalAction
from line_group_query_bot import GroupQueryCommand

import stock_query_service as sqs


def test_multi_summary_line_empty_counter():
    assert sqs._multi_summary_line(Counter()) == "整體：無有效判斷"


def test_multi_summary_line_counts():
    c = Counter({FinalAction.EXIT: 1, FinalAction.WATCH: 2})
    # risk/total=1/3<0.5；HOLD(0)>=EXIT(1) 為假 → 中性
    assert sqs._multi_summary_line(c) == "整體：中性（1 出場・2 觀望）"


def test_multi_summary_line_bearish():
    c = Counter(
        {FinalAction.EXIT: 1, FinalAction.REDUCE: 1, FinalAction.WATCH: 1}
    )
    assert sqs._multi_summary_line(c) == "整體：偏空（1 出場・1 減碼・1 觀望）"


def test_multi_summary_line_bullish():
    c = Counter({FinalAction.HOLD: 2, FinalAction.WATCH: 1})
    # 內層順序固定：出場→減碼→觀望→續抱
    assert sqs._multi_summary_line(c) == "整體：偏多（1 觀望・2 續抱）"


def test_multi_summary_line_omits_zero_actions():
    c = Counter({FinalAction.REDUCE: 1, FinalAction.WATCH: 2})
    s = sqs._multi_summary_line(c)
    assert "0 " not in s
    assert "1 減碼・2 觀望" in s
    assert "中性" in s


def test_multi_summary_only_watch_is_neutral():
    c = Counter({FinalAction.WATCH: 3})
    assert sqs._multi_summary_line(c) == "整體：中性（3 觀望）"


def test_run_multi_summary_sorts_and_headers():
    cmd = GroupQueryCommand(
        action="compact",
        tickers_normalized=("3037.TW", "3189.TW", "2367.TW"),
        tickers_raw=("3037", "3189", "2367"),
        has_position_mode=False,
    )
    with patch.object(
        sqs,
        "_fused_line_and_final_action",
        side_effect=[
            ("觀望（防守｜防守）", FinalAction.WATCH, "Guard"),
            ("減碼（結構｜結構）", FinalAction.REDUCE, "Gate"),
            ("出場（上漲力道｜上漲力道）", FinalAction.EXIT, "Trigger"),
        ],
    ):
        text = sqs._run_multi_stock_summary(cmd, time_range="1y")
    assert text.startswith("整體：偏空（1 出場・1 減碼・1 觀望）｜主因：防守\n\n")
    assert text.index("【2367】") < text.index("【3189】") < text.index("【3037】")


def test_run_multi_truncates_at_five():
    cmd = GroupQueryCommand(
        action="compact",
        tickers_normalized=tuple(f"{i:04d}.TW" for i in range(3001, 3008)),
        tickers_raw=tuple(str(i) for i in range(3001, 3008)),
        has_position_mode=False,
    )
    with patch.object(
        sqs,
        "_fused_line_and_final_action",
        return_value=("x", FinalAction.WATCH, None),
    ):
        text = sqs._run_multi_stock_summary(cmd, time_range="1y")
    assert "僅顯示前 5 檔" in text
    assert text.count("【") == 5


def test_run_multi_ultra_compact_lines():
    cmd = GroupQueryCommand(
        action="multi_ultra",
        tickers_normalized=("3037.TW", "3189.TW", "2367.TW"),
        tickers_raw=("3037", "3189", "2367"),
        has_position_mode=False,
    )
    with patch.object(
        sqs,
        "_fused_line_and_final_action",
        side_effect=[
            ("觀望（防守｜防守）", FinalAction.WATCH, "Guard"),
            ("減碼（結構｜結構）", FinalAction.REDUCE, "Gate"),
            ("出場（上漲力道｜上漲力道）", FinalAction.EXIT, "Trigger"),
        ],
    ):
        text = sqs._run_multi_stock_summary(cmd, time_range="1y")
    assert "整體：偏空（1 出場・1 減碼・1 觀望）｜主因：防守" in text
    assert "2367 → 出場" in text
    assert "3189 → 減碼" in text
    assert "3037 → 觀望" in text
    assert "【3037】" not in text
    assert "→ 觀望｜" not in text


def test_run_multi_ultra_optional_primary_category():
    cmd = GroupQueryCommand(
        action="multi_ultra",
        tickers_normalized=("3037.TW", "3189.TW", "2367.TW"),
        tickers_raw=("3037", "3189", "2367"),
        has_position_mode=False,
    )
    with patch.object(
        sqs,
        "_fused_line_and_final_action",
        side_effect=[
            ("x", FinalAction.WATCH, "Guard"),
            ("x", FinalAction.REDUCE, "Gate"),
            ("x", FinalAction.EXIT, "Trigger"),
        ],
    ):
        text = sqs._run_multi_stock_summary(
            cmd, time_range="1y", multi_ultra_show_category=True
        )
    assert "2367 → 出場｜上漲力道" in text
    assert "3189 → 減碼｜結構" in text
    assert "3037 → 觀望｜防守" in text


def test_dominant_primary_category_by_count_then_tiebreak():
    rows = [
        {"primary_cat": "Gate"},
        {"primary_cat": "Guard"},
        {"primary_cat": "Gate"},
        {"primary_cat": "Guard"},
    ]
    # 同分時 Guard 先於 Gate
    assert sqs._dominant_primary_category(rows) == "Guard"


def test_run_multi_dip_skips_summary():
    cmd = GroupQueryCommand(
        action="dip",
        tickers_normalized=("3037.TW", "3189.TW"),
        tickers_raw=("3037", "3189"),
        has_position_mode=False,
    )
    with patch.object(
        sqs,
        "_fused_line_and_final_action",
        return_value=("撿便宜說明句。", None, None),
    ):
        text = sqs._run_multi_stock_summary(cmd, time_range="1y")
    assert not text.startswith("整體：")
