from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import math

import numpy as np


def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _fmt_wan(x: Any) -> str:
    v = _to_float(x)
    if v is None:
        return "NA"
    if abs(v) >= 10000:
        return f"{v/10000:,.0f} 萬"
    return f"{v:,.0f}"


# 精準診斷「行動」內部仍用英文枚舉；對使用者輸出改為白話（避免 ALLOW 等術語）
_PRECISION_ACTION_USER_ZH: dict[str, str] = {
    "ALLOW": "可依策略執行（籌碼與位階未亮紅燈）",
    "WATCH": "建議保守應對（先觀望或分批）",
    "REDUCE": "建議降低持股",
    "EXIT": "建議出場或嚴控部位",
}


@dataclass(frozen=True)
class PrecisionDiagnosis:
    """
    精準診斷（用來避免「技術面看多」與「籌碼面看空」互相打架）
    """

    level: str  # "ok" | "warning" | "danger"
    action: str  # "ALLOW" | "WATCH" | "REDUCE" | "EXIT"
    headline: str
    bullets: list[str]
    # 常用衍生數字（便於 UI/腳本顯示）
    total_inst_3d: Optional[float] = None
    showdown_ratio: Optional[float] = None  # 外資賣壓/投信承接（%）
    inst_participation_pct: Optional[float] = None  # 法人參與度（%）

    def as_one_liner(self) -> str:
        prefix = "✅" if self.level == "ok" else "⚠️" if self.level == "warning" else "🚨"
        act_zh = _PRECISION_ACTION_USER_ZH.get(
            str(self.action).strip().upper(), str(self.action)
        )
        return f"{prefix} {self.headline}｜建議：{act_zh}"


def diagnose_precision(
    *,
    foreign_3d_net: Any = None,
    trust_3d_net: Any = None,
    vol_sum_3d_lot: Any = None,
    bias20: Any = None,
    beta: Any = None,
    foreign_sell_3d: Optional[bool] = None,
) -> PrecisionDiagnosis:
    """
    你要的「力道天平」：用籌碼量級 + 股價位階，輸出不矛盾的結論。

    - foreign_3d_net / trust_3d_net：近 3 日（張）
    - vol_sum_3d_lot：近 3 日成交量（張）
    - bias20：SMA20 乖離（%）
    """

    f3 = _to_float(foreign_3d_net)
    t3 = _to_float(trust_3d_net)
    vol3 = _to_float(vol_sum_3d_lot)
    b20 = _to_float(bias20)
    b = _to_float(beta)

    bullets: list[str] = []
    level = "ok"
    action = "ALLOW"

    total_net = None
    gross = None
    participation = None
    showdown_ratio = None

    if f3 is not None and t3 is not None:
        total_net = float(f3) + float(t3)
        gross = abs(float(f3)) + abs(float(t3))

        lead_txt = (
            "空方微幅領先"
            if total_net < 0
            else "多方微幅領先"
            if total_net > 0
            else "勢均力敵"
        )
        bullets.append(f"法人合力淨值（外+投，3日）：{total_net:+,.0f} 張（{lead_txt}）")

        # 土洋對峙強弱比：外資賣壓/投信承接（只在外賣投買時有意義）
        if float(f3) < 0 and float(t3) > 0 and float(t3) != 0:
            showdown_ratio = abs(float(f3)) / abs(float(t3)) * 100.0
            lead = "外資賣壓領先投信" if showdown_ratio > 100.0 else "投信承接仍壓得住外資"
            bullets.append(f"土洋對峙：外資賣壓強度 {showdown_ratio:.0f}%（{lead}）")

        if gross is not None and vol3 is not None and vol3 > 0:
            participation = (float(gross) / float(vol3)) * 100.0
            bullets.append(
                f"法人參與度（對作/成交量，近3日）：{participation:.1f}%（越低越像散戶在玩）"
            )

    # 乖離位階（過熱濾網）
    if b20 is not None:
        bullets.append(f"乖離（Bias20）：{b20:.2f}%")

    if b is not None:
        bullets.append(f"Beta：{b:.2f}")

    # 進階：外資連賣（若未提供就用 f3 < 0 當作弱代理）
    if foreign_sell_3d is None:
        foreign_sell_3d = bool(f3 is not None and f3 < 0)

    # ---- 核心判斷（避免矛盾） ----
    # 1) 高檔派發：Bias20>15 + 外資連賣 → 直接高風險
    distribution = bool(b20 is not None and b20 > 15.0 and foreign_sell_3d)
    if distribution:
        level = "danger"
        action = "WATCH"
        headline = "高檔派發中：乖離過熱 + 外資偏空（避免追高）"
        bullets.insert(0, "解讀：外資趁強撤退，投信可能在承接撐盤；上漲純度不足，容易急殺洗盤。")

    # 2) 土洋對峙：外資賣壓 > 投信承接（>100%）→ 降級觀望
    if not distribution and showdown_ratio is not None and showdown_ratio > 100.0:
        level = "warning"
        action = "WATCH"
        headline = "土洋對峙偏空：外資賣壓領先投信（建議降級觀望/分批）"

    # 3) 法人參與度過低：散戶盤味（容易假強勢）
    if not distribution and participation is not None and participation < 10.0:
        # 參與度低通常要降級
        if level == "ok":
            level = "warning"
            action = "WATCH"
            headline = "法人參與度偏低：疑似散戶盤（避免追價）"

    # 4) Beta 偏高 + 籌碼偏空：風險加乘
    if (
        (level in ["warning", "danger"])
        and b is not None
        and b >= 1.5
        and (f3 is not None and f3 < 0)
    ):
        bullets.insert(
            0,
            "解讀：波動放大 + 籌碼偏空 → 回檔速度可能快於大盤，請嚴守防守線/分批。",
        )

    if headline := locals().get("headline"):
        pass
    else:
        headline = "籌碼/位階未見重大衝突（可依策略執行）"

    return PrecisionDiagnosis(
        level=level,
        action=action,
        headline=headline,
        bullets=bullets,
        total_inst_3d=total_net,
        showdown_ratio=showdown_ratio,
        inst_participation_pct=participation,
    )


def get_precision_diagnosis(
    f3d: Any = None,
    t3d: Any = None,
    vol: Any = None,
    bias20: Any = None,
) -> str:
    """
    為了對齊你原本腳本的呼叫型態，保留最簡單版：回傳一句話結論。
    """
    d = diagnose_precision(
        foreign_3d_net=f3d,
        trust_3d_net=t3d,
        vol_sum_3d_lot=vol,
        bias20=bias20,
    )
    return d.as_one_liner()

