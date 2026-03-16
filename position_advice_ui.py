from __future__ import annotations

from typing import Any, Optional

from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from data_sources import fetch_last_price, fetch_fundamental_snapshot
from position_advice import get_position_advice
from tomorrow_guard_price import calc_tomorrow_guard
from analysis import estimate_target_range, analyze_chip_flow, compute_volume_sum_3d
from risk_verification import risk_verification_from_data, DynamicStopAdvice


def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v
    except Exception:
        return None


def _get_last(df: pd.DataFrame, col: str) -> Optional[float]:
    if df is None or df.empty or col not in df.columns:
        return None
    try:
        return float(pd.to_numeric(df[col].iloc[-1], errors="coerce"))
    except Exception:
        return None


def _ema_fallback(close: pd.Series, span: int) -> Optional[float]:
    try:
        s = pd.to_numeric(close, errors="coerce")
        return float(s.ewm(span=span, adjust=False).mean().iloc[-1])
    except Exception:
        return None


def render_position_advice_panel(
    df: pd.DataFrame,
    *,
    symbol: str | None = None,
    avg_cost: float,
    qty: int = 0,
    exit_style: str = "波段守五日線",
    turn_result: Optional[dict] = None,
    bottom_result: Optional[dict] = None,
    top_result: Optional[dict] = None,
    current_price: float | None = None,
    guard_buffer_pct: float = 1.5,
    trailing_stop_pct: float = 10.0,
    foreign_net_series: Optional[pd.Series] = None,
    trust_net_series: Optional[pd.Series] = None,
):
    """
    持倉決策面板（白話文 + 儀表板）

    - 你可以只傳 turn_result（例如 top 模式結果）
    - 或同時傳 bottom_result / top_result（更完整）
    """
    if df is None or df.empty:
        st.info("資料不足：沒有價格資料")
        return

    # 兼容：只給一個 turn_result 時，自動判斷放在哪邊
    if isinstance(turn_result, dict) and (bottom_result is None and top_result is None):
        mode = str(turn_result.get("mode", "top"))
        if mode == "bottom":
            bottom_result = turn_result
        elif mode == "top":
            top_result = turn_result
        else:
            top_result = turn_result

    close_last = _get_last(df, "Close")
    if close_last is None:
        st.info("資料不足：缺少 Close")
        return

    ema5 = _get_last(df, "EMA5")
    ema20 = _get_last(df, "EMA20")
    if ema5 is None:
        ema5 = _ema_fallback(df["Close"], 5)
    if ema20 is None:
        ema20 = _ema_fallback(df["Close"], 20)

    # 防守線（依風格）
    if exit_style == "長線守月線":
        defense_name = "EMA20（月線）"
        defense = ema20
    else:
        defense_name = "EMA5（五日線）"
        defense = ema5

    # 先算持倉建議（供實戰副駕駛與持倉診斷一致）
    advice = get_position_advice(
        current_price=close_last,
        avg_cost=avg_cost,
        qty=qty,
        ema_defense=defense,
        bottom_result=bottom_result,
        top_result=top_result,
    )
    advice_suggests_reduce = (
        advice.level in ("warning", "error")
        and ("減碼" in advice.headline or "分批" in advice.headline or "落袋" in advice.headline)
    )

    # ---------------- 🛡️ 實戰副駕駛系統（決策層） ----------------
    # 計算保命價（以防守線 EMA + buffer 作為「保守警戒」）
    buffer_pct = float(guard_buffer_pct or 0.0)
    w = 20 if exit_style == "長線守月線" else 5
    guard = calc_tomorrow_guard(ema_today=defense, window=w, buffer_pct=buffer_pct)

    # 盤中自動更新（收盤前尤其重要；避免你在 13:00–13:35 被震盪洗掉）
    run_every = None
    try:
        _t0 = datetime.now(ZoneInfo("Asia/Taipei"))
        if _t0.weekday() < 5 and dt_time(9, 0) <= _t0.time() <= dt_time(13, 35):
            run_every = "20s"
    except Exception:
        run_every = None

    @st.fragment(run_every=run_every)
    def _render_dynamic_co_pilot():
        st.markdown("### 🛡️ 實戰副駕駛系統")
        if guard is None or guard.guard_close <= 0:
            st.info("資料不足：無法計算保命防線（缺少均線）")
            return

        # 現在時間（台北）
        tw_now = datetime.now(ZoneInfo("Asia/Taipei"))
        is_weekday = tw_now.weekday() < 5
        is_closing_time = (
            is_weekday
            and (tw_now.time() >= dt_time(13, 0))
            and (tw_now.time() <= dt_time(13, 35))
        )
        is_market_time = (
            is_weekday
            and (tw_now.time() >= dt_time(9, 0))
            and (tw_now.time() <= dt_time(13, 35))
        )

        # 取得即時價（收盤前最重要；其他時間用 Close 即可）
        cp = None
        try:
            cp = float(current_price) if current_price is not None else None
        except Exception:
            cp = None
        if cp is None and symbol and (is_closing_time or is_market_time):
            try:
                live = fetch_last_price(symbol)
                if live is not None:
                    cp = float(live)
            except Exception:
                cp = None
        if cp is None:
            cp = float(close_last)

        guard_price = float(guard.guard_close)  # 保守警戒（EMA + buffer）
        break_price = float(guard.break_close)  # 破線（EMA）
        buf_guard = ((float(cp) / float(guard_price)) - 1.0) * 100.0 if guard_price else 0.0

        # 三段式：綠（高於保守警戒）/ 黃（低於保守警戒但未破線）/ 紅（破線）
        if float(cp) >= float(guard_price):
            status_color = "#dcfce7"  # 淺綠
            text_color = "#166534"
            action_text = "✅ 趨勢安全：目前高於保守警戒價"
            if advice_suggests_reduce:
                instruction = "指令：**趨勢安全但風險升溫，建議參照下方持倉診斷分批減碼，剩餘守防線。**"
            else:
                instruction = "指令：**無視盤中震盪，安心抱過夜。**"
        elif float(cp) >= float(break_price):
            status_color = "#fef9c3"  # 淺黃
            text_color = "#854d0e"
            action_text = "🟡 靠近防線：已低於保守警戒，但尚未破線"
            instruction = f"指令：**先冷靜。收盤前盯住是否跌破 {defense_name}，未破線不急著動。**"
        else:
            status_color = "#fee2e2"  # 淺紅
            text_color = "#991b1b"
            action_text = "🚨 破線警戒：股價已跌破防守線"
            instruction = f"指令：**收盤前若站不回 {defense_name}，請執行減碼/出場。**"

        # TURN 狀態（簡短帶過，避免資訊過載）
        b = bottom_result or {}
        t = top_result or {}
        b_txt = (
            f"{str(b.get('status', 'NA'))} {int(b.get('score', 0) or 0)}/4"
            if isinstance(b, dict)
            else "NA"
        )
        t_txt = (
            f"{str(t.get('status', 'NA'))} {int(t.get('score', 0) or 0)}/5"
            if isinstance(t, dict)
            else "NA"
        )

        closing_html = (
            '<p style="background:yellow; padding:6px 8px; border-radius:6px; margin:10px 0 0 0;">'
            "⚠️ <b>注意：現在是收盤決策時間（13:00–13:35）！</b> 以「收盤是否守住防線」為最高權重。"
            "</p>"
            if is_closing_time
            else ""
        )
        auto_html = (
            '<p style="margin:6px 0 0 0; font-size:12px; color: #334155;">'
            "⏱️ 盤中自動更新：每 20 秒刷新一次（即時價快取約 60 秒）。"
            "</p>"
            if (is_closing_time or is_market_time) and symbol
            else ""
        )

        st.markdown(
            f"""
<div style="background-color:{status_color}; padding:16px 18px; border-radius:12px; border: 2px solid {text_color};">
  <h3 style="color:{text_color}; margin:0;">{action_text}</h3>
  <p style="font-size:16px; color:{text_color}; margin:10px 0 0 0;">
    目前價格：<b>{float(cp):.2f}</b> ｜ 保守警戒：<b>{float(guard_price):.2f}</b>（{defense_name} × (1+{float(buffer_pct):.1f}%））
    <br>
    容許緩衝（到保守警戒）：<span style="font-size:22px;"><b>{float(buf_guard):+.2f}%</b></span> ｜ 緩衝歸零價：<b>{float(guard_price):.2f}</b> 元
  </p>
  <p style="font-size:14px; color:{text_color}; margin:6px 0 0 0;">
    破線價（收盤 ≥ 今日EMA）：<b>{float(break_price):.2f}</b> ｜ TURN bottom：<b>{b_txt}</b> ｜ TURN top：<b>{t_txt}</b>
  </p>
  <hr style="border: 0.5px solid {text_color}; margin:12px 0;">
  <div style="font-size:16px; color:{text_color};">{instruction}</div>
  {closing_html}
</div>
{auto_html}
            """,
            unsafe_allow_html=True,
        )

        # 強勢股專屬：強勢回歸（3 日新高）→ 取消減碼念頭 / 可小量補回
        try:
            top_status = str((top_result or {}).get("status", ""))
            if top_status in ["WATCH", "BLOCK"]:
                hi = pd.to_numeric(df.get("High"), errors="coerce") if "High" in df.columns else None
                if hi is not None and hi.dropna().shape[0] >= 5:
                    prev_3d_high = hi.shift(1).rolling(3).max().iloc[-1]
                    if pd.notna(prev_3d_high) and float(cp) > float(prev_3d_high):
                        st.info(
                            f"⚠️ 偵測到強勢回歸：價格突破近 3 日高點（{float(prev_3d_high):.2f}）。"
                            "若你因 WATCH 已減碼，通常代表『洗盤後續攻』；可先取消再減碼念頭，改以防守線控風險。"
                        )
        except Exception:
            pass

    _render_dynamic_co_pilot()

    # -------- 買入／減碼價位（無需均價，精準數字） --------
    if guard is not None and defense is not None:
        break_p = float(guard.break_close)
        guard_p = float(guard.guard_close)
        b_status = str((bottom_result or {}).get("status", "NA"))
        buy_zone = ""
        if b_status == "ALLOW":
            buy_zone = f"回踩 **{defense:.2f}** 元（{defense_name}）可買｜或現價 **{close_last:.2f}** 元附近分批"
        else:
            buy_zone = f"等 TURN bottom 轉 ALLOW 後，回踩 **{defense:.2f}** 元可買"
        st.markdown("### 🎯 買入／減碼價位（無需均價）")
        st.markdown(
            f"**可買**：{buy_zone}\n\n"
            f"**減碼（賣一半）**：收盤跌破 **{guard_p:.2f}** 元（保守警戒）\n\n"
            f"**出清**：收盤跌破 **{break_p:.2f}** 元（{defense_name}）"
        )
        st.caption("以上為依防守線計算的精準價位，與持股均價無關。")

    # -------- 籌碼流向分析（緊貼買入／減碼價位，方便判斷） --------
    if foreign_net_series is not None or trust_net_series is not None:
        chip_advice = analyze_chip_flow(df, foreign_net_series, trust_net_series)
        st.markdown("### 籌碼流向分析")
        try:
            foreign_net_latest = (
                float(foreign_net_series.iloc[-1])
                if foreign_net_series is not None and not foreign_net_series.empty
                else None
            )
        except Exception:
            foreign_net_latest = None
        try:
            trust_net_latest = (
                float(trust_net_series.iloc[-1])
                if trust_net_series is not None and not trust_net_series.empty
                else None
            )
        except Exception:
            trust_net_latest = None
        try:
            foreign_3d_net_show = (
                float(foreign_net_series.dropna().tail(3).sum())
                if foreign_net_series is not None and foreign_net_series.dropna().shape[0] >= 3
                else None
            )
        except Exception:
            foreign_3d_net_show = None
        try:
            trust_3d_net_show = (
                float(trust_net_series.dropna().tail(3).sum())
                if trust_net_series is not None and trust_net_series.dropna().shape[0] >= 3
                else None
            )
        except Exception:
            trust_3d_net_show = None

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("外資單日（張）", f"{foreign_net_latest:,.0f}" if foreign_net_latest is not None else "NA")
        with c2:
            st.metric("外資 3 日（張）", f"{foreign_3d_net_show:,.0f}" if foreign_3d_net_show is not None else "NA")
        with c3:
            st.metric("投信單日（張）", f"{trust_net_latest:,.0f}" if trust_net_latest is not None else "NA")
        with c4:
            st.metric("投信 3 日（張）", f"{trust_3d_net_show:,.0f}" if trust_3d_net_show is not None else "NA")

        try:
            vol_sum_3d_lot = compute_volume_sum_3d(df)
        except Exception:
            vol_sum_3d_lot = None
        try:
            total_inst_3d = (
                float(foreign_3d_net_show) + float(trust_3d_net_show)
                if foreign_3d_net_show is not None and trust_3d_net_show is not None
                else None
            )
        except Exception:
            total_inst_3d = None
        try:
            gross_inst_3d = (
                abs(float(foreign_3d_net_show)) + abs(float(trust_3d_net_show))
                if foreign_3d_net_show is not None and trust_3d_net_show is not None
                else None
            )
        except Exception:
            gross_inst_3d = None

        ratio_txt = None
        try:
            if (
                foreign_3d_net_show is not None
                and trust_3d_net_show is not None
                and float(foreign_3d_net_show) < 0
                and float(trust_3d_net_show) > 0
                and float(trust_3d_net_show) != 0
            ):
                r = abs(float(foreign_3d_net_show)) / abs(float(trust_3d_net_show)) * 100.0
                lead = "外資賣壓領先投信" if r > 100 else "投信承接仍壓得住外資"
                ratio_txt = f"{r:.0f}%（{lead}）"
        except Exception:
            ratio_txt = None

        part_txt = None
        try:
            if (
                gross_inst_3d is not None
                and vol_sum_3d_lot is not None
                and float(vol_sum_3d_lot) > 0
            ):
                part = float(gross_inst_3d) / float(vol_sum_3d_lot) * 100.0
                part_txt = f"{part:.1f}%"
        except Exception:
            part_txt = None

        if total_inst_3d is not None:
            lead_txt = "空方微幅領先" if float(total_inst_3d) < 0 else "多方微幅領先" if float(total_inst_3d) > 0 else "勢均力敵"
            st.caption(f"法人合力淨值（外+投，3日）：{float(total_inst_3d):+,.0f} 張（{lead_txt}）")
        if ratio_txt:
            st.caption(f"土洋對峙強弱比（外資賣壓/投信承接）：{ratio_txt}")
        if part_txt:
            st.caption(f"法人參與度（對作/成交量，近3日）：{part_txt}（越低越像散戶在玩）")

        if "散戶接刀" in chip_advice:
            st.error(chip_advice)
        elif "大戶承接" in chip_advice or "法人鎖碼" in chip_advice:
            st.success(chip_advice)
        else:
            st.info(chip_advice)

        with st.expander("📉 查看外資/投信近 10 日明細（可收起）", expanded=False):
            try:
                if foreign_net_series is not None and not foreign_net_series.empty:
                    st.caption("外資近 10 日（張）")
                    st.bar_chart(foreign_net_series.tail(10))
                else:
                    st.caption("外資：資料不足")
            except Exception:
                st.caption("外資：資料不足")
            try:
                if trust_net_series is not None and not trust_net_series.empty:
                    st.caption("投信近 10 日（張）")
                    st.bar_chart(trust_net_series.tail(10))
                else:
                    st.caption("投信：資料不足")
            except Exception:
                st.caption("投信：資料不足")

    # -------- 明日保命價 + 目標價（收盤防線，緊貼籌碼流向方便判斷） --------
    st.markdown("---")
    st.subheader("📌 明日保命價與目標價")
    try:
        tp_range = estimate_target_range(df, symbol or "") if symbol else None
        gap_pct = None
        if guard is not None and float(close_last) > 0:
            try:
                gap_pct = (float(close_last) - float(guard.guard_close)) / float(close_last) * 100.0
            except Exception:
                pass

        c1, c2, c3 = st.columns(3)
        with c1:
            if guard is not None:
                st.metric("明日不破線（收盤 ≥ 今日EMA）", f"{float(guard.break_close):.2f}")
                st.caption(f"今日 {defense_name}：{float(guard.ema_today):.2f}")
            else:
                st.metric("明日不破線", "N/A")
                st.caption("資料不足")

        with c2:
            if guard is not None:
                delta_txt = f"{float(gap_pct):+.1f}% 緩衝" if gap_pct is not None else None
                st.metric(
                    f"保守警戒（+{float(buffer_pct):.1f}%）",
                    f"{float(guard.guard_close):.2f}",
                    delta_txt,
                )
            else:
                st.metric("保守警戒", "N/A")

        with c3:
            if tp_range is not None and tp_range.get("tp_high") is not None:
                tp_high = float(tp_range["tp_high"])
                tp_mid = float(tp_range["tp_mid"]) if tp_range.get("tp_mid") else tp_high
                dist_pct = ((tp_high - float(close_last)) / float(close_last) * 100) if float(close_last) > 0 else None
                delta_tp = f"{dist_pct:+.1f}% 空間" if dist_pct is not None else None
                st.metric("目標價（TP High）", f"{tp_high:.2f}", delta_tp)
                st.caption(f"TP Mid：{tp_mid:.2f}")
            else:
                st.metric("目標價（TP High）", "N/A")
                st.caption("資料不足")
    except Exception:
        st.caption("明日保命價計算失敗（已略過）")

    # -------- 風險驗證（動態止損建議） --------
    try:
        fundamental = fetch_fundamental_snapshot(symbol) if symbol else None
        stop_advice = risk_verification_from_data(
            df, fundamental, symbol=symbol, current_stop_pct=float(trailing_stop_pct or 10),
        )
        if stop_advice.is_high_valuation_risk:
            st.warning(
                f"🛠️ **風險驗證**：{stop_advice.reason}\n\n"
                f"（P/E {stop_advice.pe_vs_threshold}｜"
                f"建議止損收緊至 **{stop_advice.suggested_stop_pct:.0f}%**）"
            )
        elif stop_advice.suggested_stop_pct < float(trailing_stop_pct or 10):
            bias_txt = f"{stop_advice.bias_20_pct:.1f}%" if stop_advice.bias_20_pct is not None else "N/A"
            st.info(
                f"🛠️ **風險驗證**：{stop_advice.reason}\n\n"
                f"（乖離 {bias_txt} 過熱｜建議止損 **{stop_advice.suggested_stop_pct:.0f}%**）"
            )
    except Exception:
        pass

    # -------- 白話文診斷（泛用：底/頂都可） --------
    if float(avg_cost or 0.0) <= 0:
        st.caption("💡 輸入持股均價可顯示損益與分批減碼建議；上方價位無需均價。")

    st.markdown("### 💼 持倉診斷")
    if advice.level == "success":
        st.success(advice.headline)
    elif advice.level == "warning":
        st.warning(advice.headline)
    elif advice.level == "error":
        st.error(advice.headline)
    else:
        st.info(advice.headline)
    st.markdown("\n".join([f"- {x}" for x in advice.bullets]))

    # -------- 下車指南（偏 top 模式） --------
    st.markdown("---")
    st.subheader("🏁 決策輔助：該如何下車？")

    # 取 top 結果（沒有就用 turn_result/bottom_result 的資訊退化）
    tr = top_result or turn_result or {}
    score = int(tr.get("score", 0) or 0)
    status = str(tr.get("status", "NA"))
    mode = str(tr.get("mode", "top"))
    bias_hit = bool((tr.get("conditions") or {}).get("bias"))

    profit_ratio = None
    try:
        profit_ratio = float(close_last) / float(avg_cost) - 1.0
    except Exception:
        profit_ratio = None

    bias_ema5 = None
    if ema5 is not None and float(ema5) > 0:
        try:
            bias_ema5 = float(close_last) / float(ema5) - 1.0
        except Exception:
            bias_ema5 = None

    # 風格預設（可再參數化）
    if exit_style == "積極分批止盈":
        overheat_thr = 0.10  # 乖離 10%
        partial_text = "先賣出 1/2（或至少 1/3）"
        score_trigger = 2
    elif exit_style == "長線守月線":
        overheat_thr = 0.12
        partial_text = "先賣出 1/3"
        score_trigger = 3
    else:
        overheat_thr = 0.10
        partial_text = "先賣出 1/3"
        score_trigger = 3

    overheat = bool(bias_hit) or (
        bias_ema5 is not None and float(bias_ema5) > float(overheat_thr)
    )

    if mode == "top":
        if (
            score >= int(score_trigger)
            and defense is not None
            and float(defense) > 0
            and float(close_last) < float(defense)
        ):
            st.error(
                f"🚨 **【全數/大比例獲利了結】** 系統分數 {score} 且已跌破 {defense_name}（{float(defense):.2f}）。"
                "這通常是波段轉折點。"
            )
        elif overheat:
            bias_txt = f"{(float(bias_ema5) * 100.0):.1f}%" if bias_ema5 is not None else "NA"
            st.warning(
                f"⚠️ **【強勢減碼】** 乖離（相對 EMA5）約 {bias_txt}，且燈號={status}、分數={score}。"
                f"建議{partial_text}，剩餘持股守 {defense_name}。"
            )
        elif (
            exit_style == "積極分批止盈"
            and status in ["WATCH", "BLOCK"]
            and profit_ratio is not None
            and float(profit_ratio) > 0
            and defense is not None
            and float(defense) > 0
            and float(close_last) >= float(defense)
        ):
            pr_txt = f"{(float(profit_ratio) * 100.0):.1f}%"
            st.warning(
                f"🟡 **【二階段下車：先減碼】** 燈號={status}（score={score}），目前仍守住 {defense_name}（{float(defense):.2f}）。"
                f"目前獲利 {pr_txt}；建議先賣出 1/2，剩餘持股跌破 {defense_name} 再全出。"
            )
        elif score >= 2:
            pr_txt = f"{(float(profit_ratio) * 100.0):.1f}%" if profit_ratio is not None else "NA"
            st.info(
                f"🟡 **【高位監控】** 燈號已轉為 {status}（score={score}）。目前獲利 {pr_txt}。"
                f"建議將出場點設為收盤跌破 {defense_name}。"
            )
        else:
            st.success("💎 **【獲利奔跑中】** 目前尚未觸發減碼/下車訊號，可續抱。")
    else:
        st.info("目前不是 top 模式結果；下車指南以 top 風險控管為主，建議切到 top 或 both 觀察。")

    # -------- 儀表板（數據可視化） --------
    tp_range = estimate_target_range(df, symbol or "") if symbol else None  # 上方明日保命價已用過，此處供儀表板
    st.markdown("### 📊 數據儀表板")
    col1, col2, col3, col4 = st.columns(4)
    if profit_ratio is not None:
        col1.metric("未實現損益", f"{(float(profit_ratio) * 100.0):.2f}%", f"{close_last - float(avg_cost):.2f}")
    else:
        col1.metric("未實現損益", "NA")
    if defense is not None:
        col2.metric(defense_name, f"{float(defense):.2f}", f"{close_last - float(defense):.2f}")
    else:
        col2.metric(defense_name, "NA")
    col3.metric("系統危險分（top）", f"{int(score)}/5")
    if tp_range and tp_range.get("tp_high") and float(close_last) > 0:
        tp_high = float(tp_range["tp_high"])
        dist_pct = ((tp_high - float(close_last)) / float(close_last)) * 100
        dist_price = tp_high - float(close_last)
        col4.metric("目標價（TP High）", f"{tp_high:.2f}", f"{dist_pct:+.1f}%（+{dist_price:.2f} 元）")
    else:
        col4.metric("目標價", "N/A")