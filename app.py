import os
from datetime import date, datetime, time as dt_time
from zoneinfo import ZoneInfo
from copy import deepcopy

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
import pandas_ta_classic as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    BroadcastRequest,
    TextMessage,
)
from analysis import (
    check_global_buy_strategy,
    check_volume_risk,
    compute_buy_signals,
    compute_indicators,
    compute_risk_metrics,
    compute_volume_sum_3d,
    compute_weighted_score,
    estimate_target_range,
)
from data_sources import (
    FINMIND_AVAILABLE,
    _load_data_raw,
    fetch_chip_net_series,
    fetch_foreign_net_series,
    fetch_institutional_raw,
    fetch_last_price,
    fetch_last_price_batch,
    fetch_ticker_name,
    fetch_trust_net_series,
    get_weekly_trend,
    load_data,
    load_data_batch,
    load_market_index,
)
from utils import (
    align_by_date,
    align_net_series_to_price,
    build_net_series,
    detect_net_unit_tag,
    parse_portfolio_lines,
    to_scalar,
)
from position_advice import get_position_advice
from position_advice_ui import render_position_advice_panel
from turn_check_ui import render_turn_check_panel
from turn_check_engine import (
    backtest_turn_signals,
    grid_search_optimization,
    get_all_turn_details,
    get_all_turn_statuses,
    load_turn_config,
    run_turn_check,
)
from tomorrow_guard_price import calc_tomorrow_guard
from precision_diagnosis import diagnose_precision
from price_prediction_ui import render_price_prediction_panel
from portfolio_journal import (
    NEAR_LOCKED_TARGET_THRESHOLD_PCT,
    close_trade,
    create_trade,
    last_bar_date_from_ohlcv_df,
    list_open_trades,
    load_journal,
    prepare_df_for_journal,
    summarize_open_trades_for_ui,
    update_open_trades_daily,
)

# =========================
# Indicator taxonomy (分類字典)
# =========================
INDICATOR_TAXONOMY = {
    "籌碼（Chip）": [
        ("foreign_net_latest", "外資單日買賣超"),
        ("foreign_3d_net", "外資3日累計買賣超"),
        ("trust_net_latest", "投信單日買賣超"),
        ("trust_3d_net", "投信3日累計買賣超"),
        ("chip_divergence", "籌碼背離訊號"),
    ],
    "技術（Technical）": [
        ("SMA20", "SMA20"),
        ("SMA60", "SMA60"),
        ("EMA5", "EMA5"),
        ("EMA20", "EMA20"),
        ("RSI14", "RSI14"),
        ("ATR14", "ATR14"),
        ("bias_sma20_pct", "SMA20乖離率（%）"),
        ("bias_ema20_pct", "EMA20乖離率（%）"),
        ("Volume", "成交量"),
        ("VolMA20", "20日均量"),
        ("Vol_Avg_5", "5日均量"),
        ("VWAP", "VWAP"),
        ("AVWAP", "Anchored VWAP"),
        ("Is_Dangerous_Volume", "高檔爆量黑K"),
    ],
    "風險（Risk Metrics）": [
        ("beta", "Beta（相對大盤）"),
        ("volatility_annual", "年化波動率"),
        ("sharpe", "Sharpe"),
        ("max_drawdown", "Max Drawdown"),
    ],
    "市場狀態（Regime）": [
        ("market_trend", "大盤趨勢狀態"),
        ("rs_20d_vs_market", "相對強弱（20日）"),
    ],
}
# 1. 網頁配置（讓它在手機上看起來像 App）
st.set_page_config(page_title="Cursor 股票工具", layout="wide")

load_dotenv()

title_placeholder = st.empty()
expert_placeholder = st.empty()
latest_metrics_placeholder = st.empty()
diagnosis_summary = st.empty()
dashboard_placeholder = st.empty()

channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")


def send_line_message(message):
    if not channel_access_token:
        return False, "缺少 LINE 環境變數"
    try:
        configuration = Configuration(access_token=channel_access_token)
        with ApiClient(configuration) as api_client:
            api_instance = MessagingApi(api_client)
            api_instance.broadcast(
                BroadcastRequest(
                    messages=[TextMessage(text=message)],
                )
            )
        return True, "已群發給所有好友"
    except Exception as exc:
        return False, f"推播失敗：{exc}"


def format_actionable_summary(
    ticker,
    close_price,
    ema20,
    ema5,
    bias20,
    rsi,
    vol,
    vol5,
    foreign_net,
    score,
    stop_loss,
    take_profit,
    event,
    decision,
):
    lines = []
    lines.append(f"【策略通知】{ticker}｜{event}")
    lines.append(f"建議：{decision}")
    lines.append(f"收盤：{close_price:.2f}")
    if ema20 is not None and not pd.isna(ema20):
        lines.append(f"EMA20：{ema20:.2f}（{'站上' if close_price > ema20 else '跌破'}）")
    if ema5 is not None and ema20 is not None and not pd.isna(ema5) and not pd.isna(ema20):
        lines.append(
            f"EMA5/EMA20：{ema5:.2f}/{ema20:.2f}（{'多' if ema5 > ema20 else '弱'}）"
        )
    if bias20 is not None and not pd.isna(bias20):
        if bias20 > 6:
            bias_note = "偏熱"
        elif -3 <= bias20 <= 3:
            bias_note = "正常/貼線"
        else:
            bias_note = "偏冷"
        lines.append(f"20日乖離：{bias20:.2f}%（{bias_note}）")
    if rsi is not None and not pd.isna(rsi):
        lines.append(f"RSI14：{rsi:.1f}")
    if vol is not None and vol5 is not None and not pd.isna(vol5) and vol5 > 0:
        lines.append(f"量能：{int(vol):,}（5日均量 {int(vol5):,}｜{vol/vol5:.2f}x）")
    if foreign_net is not None and not pd.isna(foreign_net):
        lines.append(f"外資買賣超：{int(foreign_net):,}")
    if score is not None:
        lines.append(f"診斷分數：{score}")
    if stop_loss is not None and not pd.isna(stop_loss):
        lines.append(f"停損參考：{stop_loss:.2f}")
    if take_profit is not None and not pd.isna(take_profit):
        lines.append(f"停利參考：{take_profit:.2f}")
    return "\n".join(lines)


def send_with_cooldown(event_key, message, cooldown_minutes=30):
    now = datetime.now(ZoneInfo("Asia/Taipei"))
    cache_key = f"last_push_{event_key}"
    last_time = st.session_state.get(cache_key)
    if last_time and (now - last_time).total_seconds() < cooldown_minutes * 60:
        return False, "冷卻時間內，已略過重複推播"
    ok, msg = send_line_message(message)
    if ok:
        st.session_state[cache_key] = now
    return ok, msg


def load_prepared_df_for_journal(symbol: str, time_range: str = "1y") -> pd.DataFrame:
    """載入 OHLCV 並計算指標與 BUY_*，供持倉日誌更新使用。"""
    sym = (symbol or "").strip()
    if not sym:
        return pd.DataFrame()
    raw = load_data(sym, time_range)
    if raw is None or raw.empty:
        raw = _load_data_raw(sym, time_range)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [
            "_".join([str(x) for x in col if x is not None]) for col in raw.columns
        ]
    raw.columns = raw.columns.str.strip()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return prepare_df_for_journal(raw)


def is_taiwan_market_open(now):
    if now.weekday() >= 5:
        return False
    return dt_time(9, 0) <= now.time() <= dt_time(13, 30)


tw_now = datetime.now(ZoneInfo("Asia/Taipei"))
if is_taiwan_market_open(tw_now):
    st.success("即時監控中")


def generate_expert_advice(
    df, ticker_name, score, risk_metrics, is_chip_divergence=False
):
    if df is None or df.empty or len(df) < 2:
        return "資料不足，無法產生專家講評。"

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    k_range = latest["High"] - latest["Low"]
    upper_shadow = latest["High"] - max(latest["Open"], latest["Close"])
    lower_shadow = min(latest["Open"], latest["Close"]) - latest["Low"]
    body = abs(latest["Close"] - latest["Open"])

    analysis = []
    display_name = ticker_name or "此標的"

    is_dangerous_vol = bool(latest.get("Is_Dangerous_Volume"))
    is_pulling_out = is_dangerous_vol and is_chip_divergence

    # 過熱判斷（避免「強力多頭」與「籌碼背離」同時出現造成矛盾）
    bias20 = None
    try:
        b = latest.get("Bias20")
        bias20 = float(b) if (b is not None and not pd.isna(b)) else None
    except Exception:
        bias20 = None
    is_overheat = bias20 is not None and float(bias20) > 15.0
    chip_distribution = bool(is_chip_divergence and is_overheat)

    if chip_distribution:
        analysis.append(
            f"**高檔派發中**：{display_name} 評分 {score}，但出現「乖離過熱（Bias20>15%）+ 籌碼背離」。"
            "這種狀態常見於『外資趁強撤退、投信苦撐』，短線容易急殺洗盤。"
        )
    elif score >= 80:
        analysis.append(f"**強力多頭配置**：{display_name} 評分高達 {score}，動能強勁。")
    elif score >= 60:
        if is_pulling_out:
            analysis.append(
                f"**高度警戒**：{display_name} 評分 {score}，但偵測到「拉高出貨」徵兆（爆量黑K + 籌碼背離）。"
            )
        else:
            analysis.append(
                f"**趨勢穩定**：{display_name} 評分 {score}，屬於標準多方形態。"
            )
    else:
        analysis.append(
            f"**觀望保守**：{display_name} 評分 {score}，建議等待更明確的點火訊號。"
        )

    if body > 0 and lower_shadow > body * 1.5:
        analysis.append("**下方支撐強勁**：長下影線顯示回測後有買盤承接。")
    if body > 0 and upper_shadow > body * 1.5 and latest.get("BUY_TRIGGER_TYPE") == "BREAKOUT":
        analysis.append("**假突破風險**：創高但上影線過長，須防範套牢壓力。")
    if latest.get("Is_Dangerous_Volume"):
        analysis.append(
            "**注意：高檔放量收黑**。雖然有人接手，但目前空方力道大於多方，建議觀察 3 天。"
        )
    volume_risk = check_volume_risk(df)
    if volume_risk:
        analysis.append(volume_risk)

    if (
        risk_metrics
        and risk_metrics.get("beta", 1) is not None
        and risk_metrics.get("beta", 1) > 1.5
    ):
        analysis.append(
            f"**波動警告**：Beta 偏高 ({risk_metrics['beta']:.2f})，操作需嚴守停損。"
        )

    if chip_distribution:
        advice = (
            "**行動：減碼觀望**。乖離過熱且籌碼背離，建議先落袋一部分（或至少不再加碼），"
            "以 EMA5/EMA20 作為防守線，收盤守住再談續抱。"
        )
    elif is_pulling_out:
        advice = "**行動：果斷減碼**。偵測到籌碼與量價同步轉惡，建議減碼至 2 成以下或清倉。"
    elif is_dangerous_vol:
        advice = "**行動：減碼觀望**。高檔爆量收黑，守住今日低點再考慮續抱。"
    elif latest.get("BUY_SIGNAL"):
        advice = (
            f"**行動：符合進場標準 ({latest.get('BUY_TRIGGER_TYPE')})**。"
            "建議在平盤附近分批佈局。"
        )
    elif latest.get("Close") < latest.get("SMA20"):
        advice = "**行動：減碼觀望**。跌破 SMA20，在站回前不宜貿然接刀。"
    elif bias20 is not None and abs(float(bias20)) > 10:
        advice = (
            "**行動：持倉可續抱，建倉暫觀望**。結構未破壞但乖離過熱（Bias20≈{:.1f}%），"
            "追高風險大；建倉請等乖離回落。".format(float(bias20))
        )
    else:
        advice = "**行動：持股續抱**。目前雖無新訊號，但結構未被破壞。"

    return "\n\n".join(analysis) + "\n\n---\n" + advice


def render_conflict_warning(is_chip_divergence, is_dangerous_vol):
    if is_chip_divergence and is_dangerous_vol:
        st.error("警示：籌碼面與技術面嚴重背離（大戶在跑，股價在撐）。")
    elif is_chip_divergence:
        st.warning("警示：籌碼背離（股價漲但大戶賣），上漲動能可能枯竭。")


def render_risk_metrics_panel(risk):
    st.subheader("投資人風險（洗掉你風險）")
    with st.expander("穩健 Gate 白話說明"):
        st.markdown(
            """
**Beta ≤ 1.2**：不要比大盤更容易被甩。  
Beta 代表相對大盤的敏感度，1.2 意味著大盤動 1%，你可能動 1.2%。  
這條是「允許比大盤多抖一點，但不要放大太多」。

**年化波動率 ≤ 35%**：避免每天坐雲霄飛車。  
波動越大，越容易被正常回檔洗出去；35% 是較寬鬆上限。

**Max Drawdown ≥ -30%**：過去最慘不要跌超過 30%。  
最大回撤抓的是歷史最深的那一刀，太深代表心理與風險壓力大。

**Sharpe ≥ 0.3**：風險換報酬至少合理。  
承擔波動後，歷史上有給出對等報酬；過低代表承受的風險不值得。
            """
        )
    if risk is None:
        st.info("資料不足，無法計算 Beta / 波動率 / 夏普。")
        return
    if "note" in risk:
        st.info(risk["note"])
        return

    c1, c2, c3, c4 = st.columns(4)
    beta = risk["beta"]
    vol_annual = risk["vol_annual"]
    sharpe = risk["sharpe"]
    mdd = risk["max_drawdown"]

    c1.metric("Beta (vs 大盤)", f"{beta:.2f}" if beta is not None else "N/A")
    c2.metric(
        "年化波動率",
        f"{vol_annual*100:.1f}%" if vol_annual is not None else "N/A",
    )
    c3.metric("Max Drawdown", f"{mdd*100:.1f}%" if mdd is not None else "N/A")
    c4.metric("Sharpe", f"{sharpe:.2f}" if sharpe is not None else "N/A")

    warnings = []
    if beta is not None and beta > 1.2:
        warnings.append(f"Beta 偏高（{beta:.2f}）：相對大盤更容易被甩。")
    if vol_annual is not None and vol_annual > 0.35:
        warnings.append(f"波動偏大（{vol_annual*100:.1f}%）：短線容易洗掉。")
    if sharpe is not None and sharpe < 0.3:
        warnings.append(f"Sharpe 偏低（{sharpe:.2f}）：風險調整後報酬不佳。")
    if mdd is not None and mdd < -0.30:
        warnings.append(f"最大回撤偏深（{mdd*100:.1f}%）：心理壓力大。")

    if warnings:
        st.warning("\n".join(warnings))
    else:
        st.success("風險結構相對穩健（Beta/波動/Sharpe/回撤未見明顯警訊）。")


def render_target_range_panel(df, ticker):
    st.subheader("目標價區間 (Target Range)")
    tp = estimate_target_range(df, ticker)
    if tp is None or tp["tp_mid"] is None:
        st.warning("資料不足，無法估算目標價區間。")
        return

    tp_low = tp["tp_low"]
    tp_mid = tp["tp_mid"]
    tp_high = tp["tp_high"]
    close_val = tp["debug"]["close"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TP Low", f"{tp_low:.2f}" if tp_low else "N/A")
    c2.metric("TP Mid", f"{tp_mid:.2f}" if tp_mid else "N/A")
    c3.metric("TP High", f"{tp_high:.2f}" if tp_high else "N/A")
    c4.metric("Confidence", tp["confidence"])

    flags = tp["model_flags"]
    st.caption(
        "Models: "
        f"Fundamental={'ON' if flags['fund_enabled'] else 'OFF'} | "
        f"Tech={'ON' if flags['tech_enabled'] else 'OFF'} | "
        f"ATR={'ON' if flags['atr_enabled'] else 'OFF'} | "
        f"MeasuredMove={'ON' if flags['mm_enabled'] else 'OFF'}"
    )

    if tp_low is not None and tp_high is not None and close_val is not None:
        position = (close_val - tp_low) / (tp_high - tp_low) if tp_high != tp_low else 0
        position = max(0, min(1, position))
        st.progress(position)
        st.caption(f"目前價格在區間內位置：{position * 100:.1f}%")

    with st.expander("🔎 目標價計算 / 推播（進階，可收起）", expanded=False):
        st.json(tp["debug"])

        if st.button("發送 Target Range 到 LINE"):
            msg_lines = [
                f"【{ticker} 目標價區間】",
                f"Close: {tp['debug']['close']:.2f}",
                f"TP Range: {tp['tp_low']:.2f} ~ {tp['tp_high']:.2f} (Mid {tp['tp_mid']:.2f})",
                f"Confidence: {tp['confidence']}",
                f"Models: Fundamental={'ON' if flags['fund_enabled'] else 'OFF'}, "
                f"Tech={'ON' if flags['tech_enabled'] else 'OFF'}",
            ]
            ok, msg = send_with_cooldown(
                f"{ticker}_target_range",
                "\n".join(msg_lines),
                cooldown_minutes=60,
            )
            if ok:
                st.success("已傳送 Target Range")
            else:
                st.warning(msg)


def render_score_overview(score, bias_20):
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("當前診斷總分", f"{score} 分")
    with col2:
        if bias_20 is not None:
            st.metric("SMA20乖離率", f"{bias_20:.2f}%")
        else:
            st.metric("SMA20乖離率", "資料不足")
    with col3:
        if score >= 70:
            st.success("趨勢強勁：建議持股續抱")
        elif score >= 40:
            st.warning("趨勢分歧：建議分批減碼")
        else:
            st.error("趨勢轉弱：建議果斷離場")


def render_gate_trigger_guard_reminders(
    risk,
    foreign_divergence_warning,
    atr14_val,
    close_price,
    prev_close,
    strategy_gate=None,
):
    st.subheader("Gate / Trigger / Guard 提醒")
    if not isinstance(risk, dict):
        st.info("Gate：WATCH｜原因：風險資料不足｜動作：先觀察")
        return

    beta = risk.get("beta")
    vol = risk.get("vol_annual")
    sharpe = risk.get("sharpe")
    mdd = risk.get("max_drawdown")
    window_days = risk.get("window_days")

    if strategy_gate is None:
        st.info("策略 Gate：資料不足（請稍後）")
    else:
        strategy_label = "通過" if strategy_gate else "未通過"
        strategy_msg = f"策略 Gate：{strategy_label}｜來源：BUY_GATE"
        if strategy_gate:
            st.success(strategy_msg)
        else:
            st.warning(strategy_msg)

    gate_level = "ALLOW"
    gate_reasons = []
    if mdd is not None and mdd < -0.25:
        gate_level = "BLOCK"
        gate_reasons.append("Max Drawdown < -25%（風險底線破）")
    elif foreign_divergence_warning:
        gate_level = "WATCH"
        gate_reasons.append("籌碼背離（價漲但外資連賣）")

    if window_days is not None and window_days < 60:
        gate_reasons.append("樣本天數不足 60（可能虛高）")

    gate_action = (
        "不動" if gate_level == "BLOCK" else "先觀察" if gate_level == "WATCH" else "可進入決策"
    )
    gate_reason_text = " + ".join(gate_reasons) if gate_reasons else "風險底線未破"
    gate_msg = f"風險 Gate：{gate_level}｜原因：{gate_reason_text}｜動作：{gate_action}"

    if gate_level == "BLOCK":
        st.error(gate_msg)
    elif gate_level == "WATCH":
        st.warning(gate_msg)
    else:
        st.success(gate_msg)

    if gate_level == "ALLOW":
        trigger_notes = []
        if sharpe is not None:
            if sharpe > 1.5:
                trigger_notes.append("Sharpe 高")
            elif sharpe < 1.0:
                trigger_notes.append("Sharpe 低")
        if mdd is not None and mdd > -0.15:
            trigger_notes.append("最痛那一刀不深")
        if foreign_divergence_warning:
            trigger_notes.append("籌碼背離（排序後移）")
        trigger_msg = "Trigger：" + (" + ".join(trigger_notes) + " → CP 值排序" if trigger_notes else "等待更佳時機")
        st.info(trigger_msg)

    beta_factor = 1.0
    if beta is not None:
        if beta > 1.6:
            beta_factor = 0.5
        elif beta > 1.2:
            beta_factor = 0.7

    vol_factor = 1.0
    if vol is not None:
        if vol > 0.40:
            vol_factor = 0.5
        elif vol > 0.25:
            vol_factor = 0.7

    position_factor = min(beta_factor, vol_factor)
    if beta is not None and beta > 1.6 and vol is not None and vol > 0.40:
        position_factor = min(position_factor, 0.5)

    stop_mult = 1.5
    if beta is not None and beta > 1.6:
        stop_mult = 2.0
    if vol is not None and vol > 0.40:
        stop_mult = 2.0

    chase_block = False
    if atr14_val is not None and not pd.isna(atr14_val) and prev_close is not None:
        if close_price is not None and prev_close != 0:
            if (close_price - prev_close) > 2 * atr14_val:
                chase_block = True
            if (close_price / prev_close - 1) > 0.05:
                chase_block = True
    if foreign_divergence_warning:
        chase_block = True

    guard_parts = [
        f"部位係數 ≤ {position_factor:.1f}",
        f"停損距離 = {stop_mult:.1f}×ATR" if atr14_val is not None and not pd.isna(atr14_val) else "停損：用 ATR 或前低",
        "不追高（等回檔）" if chase_block else "可進入",
    ]
    st.info("Guard：" + "；".join(guard_parts))


def build_metrics_snapshot(**kwargs) -> dict:
    return dict(kwargs)


def rule_item(key: str, rule: str, value, threshold: str, passed: bool, note: str = "") -> dict:
    return {
        "key": key,
        "rule": rule,
        "value": value,
        "threshold": threshold,
        "pass": passed,
        "note": note,
    }


def render_warning_wall(used_map: dict):
    """
    精簡版「警報牆」：只顯示 FAIL（避免資訊過載）
    used_map 來源：Gate / Trigger / Guard / Chip Notes 的規則清單
    """
    try:
        cats = ["Gate", "Trigger", "Guard", "Chip Notes"]
        warnings: list[str] = []
        for cat in cats:
            for it in (used_map or {}).get(cat, []) or []:
                try:
                    passed = it.get("pass", True)
                    if bool(passed):
                        continue
                    rule = str(it.get("rule", "") or "").strip()
                    note = str(it.get("note", "") or "").strip()
                    threshold = str(it.get("threshold", "") or "").strip()
                    value = it.get("value", None)
                    value_txt = ""
                    if value is not None and value != "" and value != "NA":
                        try:
                            value_txt = f"｜目前：{value}"
                        except Exception:
                            value_txt = ""
                    thr_txt = f"｜門檻：{threshold}" if threshold else ""
                    note_txt = f"｜{note}" if note else ""
                    warnings.append(f"⚠️ {cat}｜{rule}{thr_txt}{value_txt}{note_txt}")
                except Exception:
                    continue

        if warnings:
            st.warning("目前面臨的風險因素（只列 FAIL）")
            for w in warnings[:12]:
                st.write(w)
            if len(warnings) > 12:
                st.caption(f"（其餘 {len(warnings) - 12} 條已省略）")
        else:
            st.success("✅ 目前技術與籌碼結構健康（未見重大 FAIL）")
    except Exception:
        # 避免因資料結構異常導致主頁崩潰
        st.info("警報牆資料不足（已略過）")


def render_trade_brief(used_map: dict, metrics: dict, *, has_position: bool = False):
    """
    視覺化濃縮：三層級「狀態總覽 / 核心阻礙 / 籌碼異狀」
    - used_map：Gate / Trigger / Guard / Chip Notes 規則清單
    - metrics：快照數值（bias/volume/chip...）
    - has_position：若 True，持倉者請以持倉診斷為準（此為建倉觀點）
    """

    def _find(cat: str, key: str):
        for it in (used_map or {}).get(cat, []) or []:
            try:
                if str(it.get("key", "")) == key:
                    return it
            except Exception:
                continue
        return None

    def _fmt_wan(x):
        try:
            v = float(x)
            if not np.isfinite(v):
                return "NA"
            v_abs = abs(v)
            if v_abs >= 10000:
                return f"{v/10000:,.0f} 萬"
            return f"{v:,.0f}"
        except Exception:
            return "NA"

    gate = _find("Gate", "BUY_GATE")
    trig = _find("Trigger", "BUY_TRIGGER")
    guard = _find("Guard", "EXEC_GUARD")
    chip_div = _find("Chip Notes", "chip_divergence")

    gate_ok = bool((gate or {}).get("pass", False))
    trig_ok = bool((trig or {}).get("pass", False))
    guard_ok = bool((guard or {}).get("pass", False))
    overall_ok = bool(gate_ok and trig_ok and guard_ok)

    trigger_type = str((trig or {}).get("value", "NONE") or "NONE")

    # -------- 狀態總覽 --------
    st.markdown("### 🚩 建倉建議（該不該買）")
    st.caption("此區回答「現在適不適合買進」。持倉者請以「持倉診斷／下車指南」為準。")
    if overall_ok:
        st.success("✅ PASS（符合買入條件）")
        st.caption(f"狀態總覽：Gate=PASS｜Trigger=PASS（{trigger_type}）｜Guard=PASS")
    else:
        # 若 gate 失敗且過熱/爆量 → 直接用「高度過熱」語氣
        bias_v = metrics.get("bias_sma20_pct")
        vol_v = metrics.get("Volume")
        volma_v = metrics.get("VolMA20")
        overheat = False
        try:
            if bias_v is not None and np.isfinite(float(bias_v)) and abs(float(bias_v)) > 10:
                overheat = True
        except Exception:
            pass
        try:
            if vol_v is not None and volma_v is not None and np.isfinite(float(volma_v)) and float(volma_v) > 0:
                if float(vol_v) > 1.5 * float(volma_v):
                    overheat = True
        except Exception:
            pass
        headline = "高度過熱，暫不入場" if overheat else "暫不入場（未通過門檻）"
        st.warning(f"⚠️ 建倉：FAIL（不符合買入條件）｜{headline}")
        st.caption(
            f"狀態總覽：Gate={'PASS' if gate_ok else 'FAIL'}｜"
            f"Trigger={'PASS' if trig_ok else 'FAIL'}（{trigger_type}）｜"
            f"Guard={'PASS' if guard_ok else 'FAIL'}"
        )

    # -------- 核心阻礙（Top 3）--------
    st.markdown("### 2️⃣ 核心阻礙（Top 3）")
    blockers = []

    # 依優先順序抓：乖離 → 量能 → 動能（Trigger）
    key_priority = [
        ("Gate", "Bias20_gate", "乖離過大"),
        ("Gate", "Volume_gate", "成交爆量"),
        ("Trigger", "BUY_TRIGGER", "動能不足"),
        ("Guard", "EXEC_GUARD", "執行護欄未過"),
        ("Gate", "SMA20", "趨勢未站上"),
        ("Gate", "SMA20_up", "趨勢未走升"),
    ]
    seen = set()
    for cat, k, name in key_priority:
        it = _find(cat, k)
        if not isinstance(it, dict):
            continue
        if bool(it.get("pass", True)):
            continue
        if k in seen:
            continue
        seen.add(k)

        val = it.get("value")
        thr = str(it.get("threshold", "") or "")
        note = str(it.get("note", "") or "").strip()

        # 針對幾個核心項目，顯示更像你報告的格式
        if k == "Bias20_gate":
            cur = f"{float(val):.2f}%" if val is not None and not pd.isna(val) else "NA"
            thr = "≤ 10%"
            diag = "嚴重過熱，追高風險極高。"
        elif k == "Volume_gate":
            cur = _fmt_wan(val)
            try:
                volma = metrics.get("VolMA20")
                thr = f"≤ {_fmt_wan(1.5 * float(volma))}" if volma is not None else "≤ 1.5×VolMA20"
            except Exception:
                thr = "≤ 1.5×VolMA20"
            diag = "量能失控，非溫和攻擊量。"
        elif k == "BUY_TRIGGER":
            cur = str(val) if val is not None else "NONE"
            thr = "PULLBACK / BREAKOUT / CONTINUATION"
            diag = "未達點火標準（突破/回踩/續攻均未成）。"
        else:
            # fallback
            if isinstance(val, (int, float, np.floating)) and np.isfinite(float(val)):
                cur = f"{float(val):.2f}"
            else:
                cur = str(val) if val is not None else "NA"
            diag = note or "未達門檻"

        blockers.append(
            {
                "風險項目": name,
                "當前數值": cur,
                "門檻標準": thr,
                "診斷結果": diag,
            }
        )
        if len(blockers) >= 3:
            break

    if blockers:
        st.dataframe(pd.DataFrame(blockers), width="stretch", hide_index=True)
    else:
        st.success("✅ 目前沒有明顯阻礙（核心門檻皆通過）")

    # -------- 籌碼異狀 --------
    st.markdown("### 3️⃣ 籌碼異狀（Chip Notes）")
    chip_bad = False
    try:
        chip_bad = bool((chip_div or {}).get("value", False))
    except Exception:
        chip_bad = False

    f1 = metrics.get("foreign_net_latest")
    f3 = metrics.get("foreign_3d_net")
    t1 = metrics.get("trust_net_latest")
    t3 = metrics.get("trust_3d_net")
    vol3 = metrics.get("vol_sum_3d")

    # 衍生：法人合力淨值 / 土洋對峙強弱比 / 法人參與度
    total_net = None
    gross_flow = None
    inst_participation_pct = None
    showdown_ratio = None
    showdown_text = None
    try:
        if f3 is not None and t3 is not None and np.isfinite(float(f3)) and np.isfinite(float(t3)):
            total_net = float(f3) + float(t3)
            gross_flow = abs(float(f3)) + abs(float(t3))
    except Exception:
        total_net = None
        gross_flow = None
    try:
        if gross_flow is not None and vol3 is not None and np.isfinite(float(vol3)) and float(vol3) > 0:
            inst_participation_pct = (float(gross_flow) / float(vol3)) * 100.0
    except Exception:
        inst_participation_pct = None
    try:
        # 最常見的對峙：外資賣、投信買（看誰氣長）
        if (
            f3 is not None
            and t3 is not None
            and np.isfinite(float(f3))
            and np.isfinite(float(t3))
            and float(f3) < 0
            and float(t3) > 0
            and float(t3) != 0
        ):
            showdown_ratio = abs(float(f3)) / abs(float(t3)) * 100.0
            lead = "外資賣壓領先投信" if float(showdown_ratio) > 100.0 else "投信承接仍壓得住外資"
            showdown_text = f"土洋對峙：外資賣壓強度 {float(showdown_ratio):.0f}%（{lead}）"
    except Exception:
        showdown_ratio = None
        showdown_text = None

    if chip_bad:
        st.error("🚨 籌碼背離警報：股價上漲但外資偏空，屬於「虛漲」風險。")
    else:
        st.success("✅ 籌碼未見明顯背離（本工具判定）")

    st.caption(
        f"外資單日：{_fmt_wan(f1)}｜投信單日：{_fmt_wan(t1)}｜外資 3 日：{_fmt_wan(f3)}｜投信 3 日：{_fmt_wan(t3)}"
    )
    if total_net is not None:
        lead_txt = "空方微幅領先" if float(total_net) < 0 else "多方微幅領先" if float(total_net) > 0 else "勢均力敵"
        st.caption(f"法人合力淨值（外+投，3日）：{float(total_net):+,.0f} 張（{lead_txt}）")
    if showdown_text:
        st.caption(showdown_text)
    if inst_participation_pct is not None:
        st.caption(f"法人參與度（對作/成交量，近3日）：約 {float(inst_participation_pct):.1f}%（越低越像散戶在玩）")

    # -------- Action plan --------
    st.markdown("### 💡 濃縮建議（Action Plan）")
    if overall_ok:
        st.info("目前策略：ALLOW（可進入）。建議分批、以防守線控風險。")
        if chip_bad or (showdown_ratio is not None and float(showdown_ratio) > 100.0):
            st.warning("⚠️ 策略降級：技術面雖通過，但籌碼背離/對峙偏強，建議改以 WATCH 方式進場（等量縮或等外資止賣）。")
    else:
        st.info("目前策略：WATCH（觀望）。禁止追價進場。")

    bullets = []
    # 依 blockers 生成待觀察條件
    for b in blockers:
        if b["風險項目"] == "乖離過大":
            bullets.append("等待 Bias20 回落至 10% 以內（修正過熱）。")
        if b["風險項目"] == "成交爆量":
            bullets.append("等待成交量回到 VolMA20 附近（量能穩定）。")
        if b["風險項目"] == "動能不足":
            bullets.append("等待 Trigger 出現（PULLBACK / BREAKOUT / CONTINUATION 任一成立）。")
    if chip_bad:
        bullets.append("等待外資由賣轉買（外資 3 日轉正更佳）。")
    if showdown_ratio is not None and float(showdown_ratio) > 100.0:
        bullets.append("土洋對峙：外資賣壓 > 投信承接（強弱比 > 100%）→ 優先降級為觀望/分批。")

    if bullets:
        st.markdown("\n".join([f"- {x}" for x in bullets]))
    else:
        st.caption("（目前無額外待確認項）")


def render_indicator_panels(metrics: dict, used_map: dict, taxonomy: dict = None):
    taxonomy = taxonomy or INDICATOR_TAXONOMY
    with st.expander("📚 詳細：指標總覽 / 規則清單（平時可收起）", expanded=False):
        st.subheader("指標總覽（依分類）")
        tab_names = list(taxonomy.keys())
        tabs = st.tabs(tab_names)
        for tab, cat_name in zip(tabs, tab_names):
            with tab:
                rows = []
                for key, label in taxonomy[cat_name]:
                    if key in metrics and metrics[key] is not None and not pd.isna(metrics[key]):
                        rows.append({"指標Key": key, "指標名稱": label, "目前值": metrics[key]})
                if rows:
                    df_show = pd.DataFrame(rows)
                    df_show = df_show.fillna("").astype(str)
                    st.dataframe(df_show, width="stretch", hide_index=True)
                else:
                    st.info("此分類目前沒有可顯示的指標值。")

        st.subheader("本次判斷使用到的規則（Gate / Trigger / Guard）")
        gt_names = ["Gate", "Trigger", "Guard", "Chip Notes"]
        gt_tabs = st.tabs(gt_names)
        for tab, name in zip(gt_tabs, gt_names):
            with tab:
                items = used_map.get(name, [])
                if not items:
                    st.info(f"{name}：目前沒有填入規則清單。")
                    continue
                df_show = pd.DataFrame(items)
                if "pass" in df_show.columns:
                    df_show["pass"] = df_show["pass"].apply(lambda x: "PASS" if x else "FAIL")
                df_show = df_show.fillna("").astype(str)
                st.dataframe(df_show, width="stretch", hide_index=True)


def portfolio_csv_to_raw_lines(df_csv: pd.DataFrame) -> str:
    """
    將使用者上傳的 CSV（stock_id / buy_price / shares/quantity）轉成 text_area 可讀的格式：
    每行：symbol, avg_cost, shares
    - symbol 可以是 2330 / 2330.TW / 2330.TWO
    - shares 可省略
    """
    if df_csv is None or df_csv.empty:
        return ""
    df = df_csv.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def _normalize_symbol(x) -> str:
        """
        修正常見 CSV 型別陷阱：
        - iterrows() 會把整列 upcast 成 float，導致 3037 → 3037.0
        - 這會讓「代號是否為數字」判定失敗，進而抓不到資料
        """
        try:
            if x is None or (isinstance(x, float) and not np.isfinite(x)):
                return ""
        except Exception:
            pass

        # 先處理數值型
        try:
            if isinstance(x, (int, np.integer)):
                return str(int(x))
            if isinstance(x, (float, np.floating)):
                xf = float(x)
                if np.isfinite(xf) and float(int(xf)) == float(xf):
                    return str(int(xf))
                return str(xf)
        except Exception:
            pass

        s = str(x).strip()
        if not s:
            return ""

        # 若是像 "3037.0" 這種浮點字串，轉回整數字串
        try:
            if "." in s:
                ss = s.replace(",", "")
                # 只處理純數字浮點（避免 2330.TW 被誤判）
                if ss.replace(".", "", 1).isdigit():
                    f = float(ss)
                    if np.isfinite(f) and float(int(f)) == float(f):
                        return str(int(f))
        except Exception:
            pass

        if s.endswith(".0") and s[:-2].isdigit():
            return s[:-2]
        return s

    col_stock = None
    for c in ["stock_id", "symbol", "ticker", "code"]:
        if c in df.columns:
            col_stock = c
            break
    if col_stock is None:
        # 允許第一欄就是代號
        col_stock = df.columns[0]

    col_buy = None
    for c in ["buy_price", "avg_cost", "cost", "buy", "price"]:
        if c in df.columns:
            col_buy = c
            break

    col_shares = None
    for c in ["shares", "qty", "quantity", "amount"]:
        if c in df.columns:
            col_shares = c
            break

    lines: list[str] = []
    for _, r in df.iterrows():
        sid = _normalize_symbol(r.get(col_stock, ""))
        if not sid:
            continue
        buy = r.get(col_buy) if col_buy else None
        try:
            buy_f = float(str(buy).strip().replace(",", ""))
        except Exception:
            buy_f = None
        if buy_f is None or not np.isfinite(buy_f) or buy_f <= 0:
            # 成本可省略，但此工具是「成本＋籌碼」診斷，缺成本就跳過
            continue

        parts = [sid, f"{buy_f:g}"]
        if col_shares:
            sh = r.get(col_shares)
            try:
                sh_f = float(str(sh).strip().replace(",", ""))
                sh_i = int(sh_f) if np.isfinite(sh_f) else None
            except Exception:
                sh_i = None
            if sh_i is not None and sh_i > 0:
                parts.append(str(sh_i))
        lines.append(", ".join(parts))
    return "\n".join(lines)


# 2. 側邊欄設定
ticker = st.sidebar.text_input("輸入股票代碼", value="2330.TW")
name = None
resolved_symbol = None
if ticker.strip():
    name, resolved_symbol = fetch_ticker_name(ticker)
    if name:
        label = f"名稱：{name}"
        if resolved_symbol and resolved_symbol != ticker.strip():
            label = f"{label}（{resolved_symbol}）"
        st.sidebar.caption(label)
    else:
        st.sidebar.caption("名稱：查無資料")

title_suffix = ""
if name:
    title_suffix = f"｜{name}"
    if resolved_symbol and resolved_symbol != ticker.strip():
        title_suffix = f"{title_suffix}（{resolved_symbol}）"
effective_symbol = (resolved_symbol or ticker or "").strip()
if not effective_symbol:
    st.error("請輸入有效的股票代碼")
    st.stop()
time_range = st.sidebar.selectbox(
    "顯示範圍",
    ["1mo", "3mo", "6mo", "1y", "5y"],
    index=1,
)
st.sidebar.markdown("### 功能分區")
nav_zone = st.sidebar.radio(
    "模組",
    ["看盤與推播", "回測與持股"],
    index=0,
    key="sidebar_nav_zone",
    help="看盤與推播：即時圖表、策略面板與 LINE 廣播。回測與持股：歷史回測與部位／交易日誌。",
    label_visibility="visible",
)
st.sidebar.caption(
    "看盤與推播＝主控台、廣播　｜　回測與持股＝回測、持股清單"
)
if nav_zone == "看盤與推播":
    page = st.sidebar.radio(
        "頁面",
        ["主控台", "廣播"],
        index=0,
        horizontal=True,
        key="sidebar_page_watch",
    )
else:
    page = st.sidebar.radio(
        "頁面",
        ["回測", "持股清單"],
        index=0,
        horizontal=True,
        key="sidebar_page_research",
    )

# 標題：持股清單／回測／廣播與側邊欄單股無關，避免「標題台積電、表格別檔」的誤導
if page == "持股清單":
    title_placeholder.title("鹹魚翻身：策略監控站｜持股清單")
elif page == "回測":
    title_placeholder.title("鹹魚翻身：策略監控站｜策略回測")
elif page == "廣播":
    title_placeholder.title("鹹魚翻身：策略監控站｜廣播")
else:
    title_placeholder.title(f"鹹魚翻身：策略監控站{title_suffix}")

show_chip_unit_check = False
with st.sidebar.expander("🧩 進階/除錯", expanded=False):
    show_chip_unit_check = st.checkbox("顯示籌碼單位檢查", value=False)
    if st.button("🔄 清除資料快取", help="切換股票後若資料沒更新，點此強制重新抓取"):
        st.cache_data.clear()
        st.rerun()

# TURN 參數：側邊欄即時調整（只影響主控台）
turn_cfg_base = load_turn_config()
turn_sidebar_override = False
turn_bt_mode = "bottom"
turn_bt_hold_days = 10
turn_bt_win_threshold = 3.0
turn_grid_scores = [2, 3, 4]
turn_grid_dry_up_ratios = [0.4, 0.5, 0.6, 0.7, 0.8]
turn_grid_min_signals = 3
turn_grid_sort_by = "expectancy"
run_turn_grid_search = False

if page == "主控台":
    st.sidebar.markdown("### 🛠 TURN 轉折參數微調")
    turn_sidebar_override = st.sidebar.checkbox(
        "啟用 TURN 參數覆寫（即時影響圖表/面板）", value=True
    )
    with st.sidebar.expander("核心門檻設定", expanded=False):
        allow_score = st.slider(
            "Bottom｜ALLOW 門檻（需達幾分）",
            1,
            4,
            int(turn_cfg_base["decision"]["bottom"]["allow_score"]),
            1,
        )
        watch_score = st.slider(
            "Bottom｜WATCH 門檻（需達幾分）",
            1,
            3,
            int(turn_cfg_base["decision"]["bottom"]["watch_score"]),
            1,
        )
        block_score_top = st.slider(
            "Top｜BLOCK 門檻（需達幾分）",
            1,
            4,
            int(turn_cfg_base["decision"]["top"]["block_score"]),
            1,
        )
        watch_score_top = st.slider(
            "Top｜WATCH 門檻（需達幾分）",
            0,
            3,
            int(turn_cfg_base["decision"]["top"]["watch_score"]),
            1,
        )

    with st.sidebar.expander("結構 / 動能 / 量能 / 籌碼", expanded=False):
        lookback = st.number_input(
            "結構回看天數（Lookback）",
            min_value=10,
            max_value=80,
            value=int(turn_cfg_base["structure"]["lookback"]),
            step=1,
        )
        support_buffer = st.slider(
            "支撐緩衝（support_buffer）",
            0.0,
            0.02,
            float(turn_cfg_base["structure"].get("support_buffer", 0.0) or 0.0),
            0.001,
        )
        div_lookback = st.number_input(
            "背離回看（div_lookback）",
            min_value=3,
            max_value=20,
            value=int(turn_cfg_base["momentum"]["div_lookback"]),
            step=1,
        )
        rsi_oversold = st.slider(
            "RSI 超賣（bottom）",
            10,
            40,
            int(turn_cfg_base["momentum"].get("rsi_oversold", 30) or 30),
            1,
        )
        rsi_overbought = st.slider(
            "RSI 超買（top）",
            60,
            90,
            int(turn_cfg_base["momentum"].get("rsi_overbought", 70) or 70),
            1,
        )
        ma_window = st.number_input(
            "均量回看（ma_window）",
            min_value=5,
            max_value=60,
            value=int(turn_cfg_base["volume"].get("ma_window", 20) or 20),
            step=1,
        )
        compare_window = st.number_input(
            "量能比較窗（compare_window）",
            min_value=2,
            max_value=15,
            value=int(turn_cfg_base["volume"]["compare_window"]),
            step=1,
        )
        dry_up_ratio = st.slider(
            "窒息量比例（dry_up_ratio）",
            0.3,
            1.0,
            float(turn_cfg_base["volume"].get("dry_up_ratio", 0.6) or 0.6),
            0.05,
        )
        top_range_window = st.number_input(
            "Top｜跌幅參考窗（平均震幅 N 日；top_range_window）",
            min_value=2,
            max_value=30,
            value=int(turn_cfg_base["volume"].get("top_range_window", 5) or 5),
            step=1,
        )
        top_drop_mult = st.slider(
            "Top｜大跌確認倍數（跌幅 ≥ 平均震幅 × X；0=關閉｜top_drop_mult）",
            0.0,
            3.0,
            float(turn_cfg_base["volume"].get("top_drop_mult", 1.5) or 1.5),
            0.1,
        )
        trust_days = st.number_input(
            "投信連續性天數（trust_days）",
            min_value=1,
            max_value=10,
            value=int(turn_cfg_base["chip"].get("trust_days", 3) or 3),
            step=1,
        )
        require_both = st.checkbox(
            "籌碼需土洋同向（require_both）",
            value=bool(turn_cfg_base["chip"].get("require_both", False)),
        )

        # 乖離率（Bias）：top 模式過熱時 +1 分
        bias_ma_window = st.number_input(
            "乖離均線（bias_ma_window）",
            min_value=5,
            max_value=120,
            value=int(turn_cfg_base.get("bias", {}).get("ma_window", 20) or 20),
            step=1,
        )
        bias_overheat_pct_top = st.slider(
            "乖離過熱（top +1分，%）",
            0.0,
            20.0,
            float(turn_cfg_base.get("bias", {}).get("overheat_pct_top", 8.0) or 8.0),
            0.5,
        )
        top_shield_enabled = st.checkbox(
            "Top 防賣飛：BLOCK 需跌破均線（在 EMA 上方降級為 WATCH）",
            value=bool(turn_cfg_base.get("top_shield", {}).get("enabled", True)),
        )
        top_shield_ma_window = st.number_input(
            "Top 防守均線（EMA window）",
            min_value=3,
            max_value=20,
            value=int(turn_cfg_base.get("top_shield", {}).get("ma_window", 5) or 5),
            step=1,
        )
        top_trend_filter_enabled = st.checkbox(
            "Top 趨勢過濾：多頭時 BLOCK 更嚴格（Close > EMA → block_score +N）",
            value=bool(turn_cfg_base.get("top_trend_filter", {}).get("enabled", False)),
        )
        top_trend_ma_window = st.number_input(
            "Top 趨勢均線（EMA window）",
            min_value=10,
            max_value=120,
            value=int(turn_cfg_base.get("top_trend_filter", {}).get("ma_window", 20) or 20),
            step=1,
        )
        top_trend_block_score_add = st.slider(
            "Top 趨勢加嚴（block_score + N）",
            0,
            2,
            int(turn_cfg_base.get("top_trend_filter", {}).get("block_score_add", 1) or 1),
            1,
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 TURN 即時回測")
    st.sidebar.caption("一句話總結：")
    st.sidebar.caption("bottom：看「進場後能不能拉開上漲空間、回檔有多深」")
    st.sidebar.caption("top：看「出場後有沒有真的回檔、賣太早踏空的風險多大」")
    turn_bt_mode = st.sidebar.selectbox("回測模式", ["bottom", "top"], index=0)
    turn_bt_hold_days = st.sidebar.select_slider(
        "回測最長持有天數（出場＝狀態降級 / TS 先到）",
        options=[3, 5, 10, 20],
        value=10,
    )
    turn_bt_trailing_stop = st.sidebar.slider(
        "移動止損（回檔/反彈出場，%｜0=關閉）",
        0.0,
        15.0,
        3.0,
        0.5,
    )
    turn_bt_exit_ma_window = st.sidebar.select_slider(
        "二階段出場確認（bottom）：跌破 EMA N 日才出場（0=關閉）",
        options=[0, 3, 5, 10, 20],
        value=5,
    )
    turn_bt_win_threshold = st.sidebar.slider(
        "成功門檻（結案報酬，%）",
        1.0,
        10.0,
        3.0,
        0.5,
    )

    with st.sidebar.expander("🚀 參數自動優化（Grid Search）", expanded=False):
        turn_grid_scores = st.multiselect(
            "掃描門檻分數（bottom=ALLOW門檻 / top=BLOCK門檻）",
            options=[1, 2, 3, 4],
            default=[2, 3, 4],
        )
        turn_grid_dry_up_ratios = st.multiselect(
            "掃描窒息量比例（dry_up_ratio）",
            options=[0.4, 0.5, 0.6, 0.7, 0.8],
            default=[0.4, 0.5, 0.6, 0.7, 0.8],
        )
        turn_grid_min_signals = st.number_input(
            "至少訊號次數（避免樣本太少）",
            min_value=1,
            max_value=50,
            value=3,
            step=1,
        )
        turn_grid_sort_by = st.selectbox(
            "排名依據",
            ["expectancy", "win_rate", "avg_fav", "avg_final", "count"],
            index=0,
        )
        run_turn_grid_search = st.button("🚀 開始參數自動優化")

    if turn_sidebar_override:
        turn_cfg_runtime = deepcopy(turn_cfg_base)
        turn_cfg_runtime["decision"]["bottom"]["allow_score"] = int(allow_score)
        turn_cfg_runtime["decision"]["bottom"]["watch_score"] = int(watch_score)
        turn_cfg_runtime["decision"]["top"]["block_score"] = int(block_score_top)
        turn_cfg_runtime["decision"]["top"]["watch_score"] = int(watch_score_top)
        turn_cfg_runtime["structure"]["lookback"] = int(lookback)
        turn_cfg_runtime["structure"]["support_buffer"] = float(support_buffer)
        turn_cfg_runtime["momentum"]["div_lookback"] = int(div_lookback)
        turn_cfg_runtime["momentum"]["rsi_oversold"] = int(rsi_oversold)
        turn_cfg_runtime["momentum"]["rsi_overbought"] = int(rsi_overbought)
        turn_cfg_runtime["volume"]["ma_window"] = int(ma_window)
        turn_cfg_runtime["volume"]["compare_window"] = int(compare_window)
        turn_cfg_runtime["volume"]["dry_up_ratio"] = float(dry_up_ratio)
        turn_cfg_runtime["volume"]["top_range_window"] = int(top_range_window)
        turn_cfg_runtime["volume"]["top_drop_mult"] = float(top_drop_mult)
        turn_cfg_runtime["chip"]["trust_days"] = int(trust_days)
        turn_cfg_runtime["chip"]["require_both"] = bool(require_both)
        turn_cfg_runtime.setdefault("bias", {})
        turn_cfg_runtime["bias"]["ma_window"] = int(bias_ma_window)
        turn_cfg_runtime["bias"]["overheat_pct_top"] = float(bias_overheat_pct_top)
        turn_cfg_runtime.setdefault("top_shield", {})
        turn_cfg_runtime["top_shield"]["enabled"] = bool(top_shield_enabled)
        turn_cfg_runtime["top_shield"]["ma_window"] = int(top_shield_ma_window)
        turn_cfg_runtime.setdefault("top_trend_filter", {})
        turn_cfg_runtime["top_trend_filter"]["enabled"] = bool(top_trend_filter_enabled)
        turn_cfg_runtime["top_trend_filter"]["ma_window"] = int(top_trend_ma_window)
        turn_cfg_runtime["top_trend_filter"]["block_score_add"] = int(top_trend_block_score_add)
    else:
        turn_cfg_runtime = turn_cfg_base
else:
    turn_cfg_runtime = turn_cfg_base

# 3. 抓取資料（技術圖表使用 yfinance）

if page == "回測":
    from backtest_ui import render_backtest_page
    render_backtest_page(effective_symbol or ticker or "2330.TW", time_range)
    st.stop()

if page == "廣播":
    action = st.selectbox("動作", ["買入", "賣出"], index=0)
    default_symbol = resolved_symbol or ticker
    symbol = st.text_input("股票代碼", value=default_symbol)
    broadcast_name = None
    broadcast_resolved = None
    if symbol.strip():
        broadcast_name, broadcast_resolved = fetch_ticker_name(symbol)
    header_suffix = ""
    if broadcast_name:
        header_suffix = f"｜{broadcast_name}"
        if broadcast_resolved and broadcast_resolved != symbol.strip():
            header_suffix = f"{header_suffix}（{broadcast_resolved}）"
    st.header(f"廣播訊息{header_suffix}")
    price = st.number_input("價格", min_value=0.0, step=0.01, format="%.2f")
    date_value = st.date_input("日期")
    time_value = st.time_input("時間")
    note = st.text_area("補充說明（可留空）", height=80)
    if st.button("發送廣播"):
        if not symbol.strip():
            st.warning("請輸入股票代碼")
        elif price <= 0:
            st.warning("請輸入有效價格")
        else:
            dt_str = f"{date_value} {time_value.strftime('%H:%M')}"
            lines = [
                "【交易廣播】",
                f"動作：{action}",
                f"股票：{symbol.strip()} {broadcast_name or ''}".rstrip(),
                f"價格：{price:.2f}",
                f"時間：{dt_str}",
            ]
            if note.strip():
                lines.append(f"備註：{note.strip()}")
            ok, msg = send_with_cooldown(
                "broadcast_trade",
                "\n".join(lines),
                cooldown_minutes=0,
            )
            if ok:
                st.success("已發送廣播")
            else:
                st.warning(msg)
    st.stop()
elif page == "持股清單":
    st.header("持股清單")
    st.caption("輸入格式：股票代碼, 成本均價, 股數（股數以「股」為單位；成本/股數可省略）")
    st.caption("持股清單指標以 1y 日線資料計算（與主頁顯示範圍無關）。")

    st.subheader("Open Trades（交易日誌）")
    st.caption(
        "未結案交易來自 `data/portfolio/trades.csv`。"
        "無買賣的日子仍應定期按「更新 journal」：每個交易日會累積一筆持有中快照（價格、目標距離、損益、風險與趨勢欄位），非交易事件。"
        "下列最新收盤為 1y 資料之最後一根 bar；**持有天數**＝進場日至「行情最後 bar 日」之日曆天；"
        "**行情最後 bar 日**用來對照 **最後更新 bar 日**（journal）是否落後。"
        "距鎖定目標欄位已做成文字提示（已達標／接近／%）；**風險／市場狀態**取自最新 journal，未更新時為 —。"
        "**Journal 與行情**：比對「行情最後 bar 日」與「最後更新 bar 日」；落後時請先批次更新 journal 再判讀動態欄位。"
        "**已賣出要結案**：展開下方 **📓 真實持倉交易日誌（可回測用）** 的 **結案**，選 `trade_id`、填出場日／價後按 **✅ 結案（CLOSE）**（會寫入 `trades.csv` 並停止後續 OPEN journal）。"
    )

    with st.expander("📓 真實持倉交易日誌（可回測用）", expanded=True):
        st.caption(
            "寫入專案內 `data/portfolio/trades.csv` 與 `trade_daily_journal.csv`。"
            "：**建倉**只在你實際買進時按一次；**持有期間**請用本頁下方「批次更新 OPEN journal」為 OPEN 補上每日快照。"
            "建倉時會以你在表單填的 **標的** 抓取 1y 日線並計算指標後寫入進場快照（與側邊欄看盤代碼無關）。進場／結案在表單內送出。"
        )
        with st.form(key="portfolio_journal_pf_create_form"):
            pf_pj_sym = st.text_input(
                "建倉標的（含 .TW）",
                value="",
                key="portfolio_journal_pf_entry_symbol",
                placeholder="例：3037.TW",
                help="僅在此填寫要建檔的標的；送出後才依該代碼下載資料。",
            )
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                pf_pj_entry_date = st.date_input(
                    "進場日",
                    value=date.today(),
                    key="portfolio_journal_pf_entry_date",
                )
            with pc2:
                pf_pj_entry_price = st.number_input(
                    "進場價",
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                    key="portfolio_journal_pf_entry_price",
                )
            with pc3:
                pf_pj_shares = st.number_input(
                    "股數",
                    min_value=0,
                    value=0,
                    step=1,
                    key="portfolio_journal_pf_shares",
                )
            pf_pj_entry_reason = st.selectbox(
                "進場理由代碼",
                ["MANUAL", "BUY_SIGNAL", "IMPORT"],
                index=0,
                key="portfolio_journal_pf_entry_reason",
            )
            pf_pj_stop_loss = st.number_input(
                "停損參考價（0＝不記錄）",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                key="portfolio_journal_pf_stop",
            )
            pf_pj_notes = st.text_input(
                "備註（可空白）", key="portfolio_journal_pf_notes"
            )
            st.markdown(
                "**選填：覆寫鎖定目標**（留白則用進場當下動態目標區間）"
            )
            pl1, pl2 = st.columns(2)
            with pl1:
                pf_pj_lock_lo = st.number_input(
                    "locked 低",
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                    key="portfolio_journal_pf_lock_lo",
                )
            with pl2:
                pf_pj_lock_hi = st.number_input(
                    "locked 高",
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                    key="portfolio_journal_pf_lock_hi",
                )
            pf_pj_submit_create = st.form_submit_button(
                "➕ 建立 OPEN 交易並寫入進場日 Journal",
            )

        if pf_pj_submit_create:
            _sym_c = (pf_pj_sym or "").strip()
            _df_c = (
                load_prepared_df_for_journal(_sym_c, "1y")
                if _sym_c
                else pd.DataFrame()
            )
            if not _sym_c:
                st.warning("請填建倉標的。")
            elif _df_c.empty:
                st.warning(f"無法載入 `{_sym_c}` 的資料，請確認代碼。")
            elif pf_pj_entry_price <= 0 or pf_pj_shares <= 0:
                st.warning("請填進場價與股數（須大於 0）。")
            else:
                if pf_pj_entry_reason == "BUY_SIGNAL" and not bool(
                    _df_c.iloc[-1].get("BUY_SIGNAL")
                ):
                    st.info(
                        "提醒：該標的最後一日並無 BUY_SIGNAL；理由仍會依你選擇存成 BUY_SIGNAL。"
                    )
                _tid_pf = create_trade(
                    _sym_c,
                    pf_pj_entry_date.isoformat(),
                    float(pf_pj_entry_price),
                    shares=int(pf_pj_shares),
                    entry_reason=str(pf_pj_entry_reason),
                    df_at_entry=_df_c.copy(),
                    locked_tp_low=float(pf_pj_lock_lo) if pf_pj_lock_lo > 0 else None,
                    locked_tp_high=float(pf_pj_lock_hi) if pf_pj_lock_hi > 0 else None,
                    stop_loss_price=float(pf_pj_stop_loss)
                    if pf_pj_stop_loss > 0
                    else None,
                    notes=str(pf_pj_notes or ""),
                )
                st.success(f"已建立 trade_id = `{_tid_pf}`")
                st.rerun()

        _pj_pf_open_journal = list_open_trades()
        _pj_trade_entry_snapshot_cols = (
            "dynamic_tp_low_at_entry",
            "dynamic_tp_high_at_entry",
            "valuation_low_at_entry",
            "valuation_high_at_entry",
        )
        if not _pj_pf_open_journal.empty:
            st.markdown("**目前 OPEN**")
            st.caption(
                "主表以 **locked_tp_*** 為這筆的計畫目標；"
                "進場當下的 **dynamic / valuation at entry** 為快照，"
                "目前版本常與 locked 數字相同（見進階說明）。**trades.csv 仍保留完整欄位。**"
            )
            _pj_open_main_cols = [
                c
                for c in _pj_pf_open_journal.columns
                if c not in _pj_trade_entry_snapshot_cols
            ]
            st.dataframe(
                _pj_pf_open_journal[_pj_open_main_cols],
                hide_index=True,
                width="stretch",
            )
            with st.expander(
                "進階：進場目標快照（dynamic / valuation at entry）", expanded=False
            ):
                st.caption(
                    "**dynamic_tp_*_at_entry**：進場那一刻系統算出的動態目標快照，寫入後不改。"
                    " **valuation_*_at_entry**：預留給未來獨立估值帶；"
                    "Phase A 仍來自同一組 target bundle，故常與 dynamic 相同。"
                    "未手填 locked 時，locked 預設等於當時 dynamic。"
                )
                _pj_snap_disp_cols = [
                    "trade_id",
                    "symbol",
                ] + [
                    c
                    for c in _pj_trade_entry_snapshot_cols
                    if c in _pj_pf_open_journal.columns
                ]
                st.dataframe(
                    _pj_pf_open_journal[_pj_snap_disp_cols],
                    hide_index=True,
                    width="stretch",
                )
            st.markdown("**結案**")
            _pj_pf_ids_j = _pj_pf_open_journal["trade_id"].astype(str).tolist()
            pf_pj_close_id = st.selectbox(
                "trade_id",
                _pj_pf_ids_j,
                key="portfolio_journal_pf_close_pick",
            )
            _m_sym_row = _pj_pf_open_journal["trade_id"].astype(str) == str(
                pf_pj_close_id
            )
            _close_sym = (
                str(_pj_pf_open_journal.loc[_m_sym_row, "symbol"].iloc[0])
                if _m_sym_row.any()
                else str(_pj_pf_open_journal.iloc[0]["symbol"])
            )
            _df_exit_pf = load_prepared_df_for_journal(_close_sym, "1y")
            _exit_px_def = (
                float(pd.to_numeric(_df_exit_pf["Close"], errors="coerce").iloc[-1])
                if _df_exit_pf is not None
                and not _df_exit_pf.empty
                and "Close" in _df_exit_pf.columns
                else 0.0
            )
            with st.form(key="portfolio_journal_pf_close_form"):
                cx2, cx3 = st.columns(2)
                with cx2:
                    pf_pj_exit_date = st.date_input(
                        "出場日",
                        value=date.today(),
                        key="portfolio_journal_pf_exit_date",
                    )
                with cx3:
                    pf_pj_exit_price = st.number_input(
                        "出場價（預設：該標的最後收盤）",
                        min_value=0.0,
                        value=float(_exit_px_def or 0.0),
                        step=0.01,
                        format="%.2f",
                        key="portfolio_journal_pf_exit_price",
                    )
                pf_pj_exit_reason = st.text_input(
                    "出場原因",
                    key="portfolio_journal_pf_exit_reason",
                )
                pf_pj_submit_close = st.form_submit_button("✅ 結案（CLOSE）")
            if pf_pj_submit_close:
                if pf_pj_exit_price <= 0:
                    st.warning("出場價須大於 0。")
                else:
                    _ok_pf = close_trade(
                        pf_pj_close_id,
                        pf_pj_exit_date.isoformat(),
                        float(pf_pj_exit_price),
                        str(pf_pj_exit_reason or ""),
                    )
                    if _ok_pf:
                        st.success("已結案。")
                        st.rerun()
                    else:
                        st.warning("結案失敗（找不到、或該筆非 OPEN）。")
        else:
            st.caption("尚無 OPEN 交易時僅能建倉；結案欄位會在建立 OPEN 後出現。")

        _pj_pf_j = load_journal()
        if not _pj_pf_j.empty:
            _pj_pf_jfilter = st.text_input(
                "Journal 檢視標的（空白＝全部最近）",
                value="",
                key="portfolio_journal_pf_jfilter",
                placeholder="例：3037.TW",
            )
            _pj_pf_jsym = str(_pj_pf_jfilter or "").strip()
            if _pj_pf_jsym:
                _pj_pf_jview = _pj_pf_j[
                    _pj_pf_j["symbol"].astype(str).str.strip() == _pj_pf_jsym
                ].tail(30)
            else:
                _pj_pf_jview = _pj_pf_j.tail(30)
            st.markdown(
                f"**Journal 檢視（`{_pj_pf_jsym or '全部最近'}`，最多 30 列）** · 全檔共 {len(_pj_pf_j)} 列"
            )
            st.caption(
                "Journal 的 `date` 欄＝該列對應的日線 bar 日期（持有中每日快照，非買賣事件）。"
            )
            st.dataframe(_pj_pf_jview, hide_index=True, width="stretch")

    _pj_pf_open = list_open_trades()
    if _pj_pf_open.empty:
        st.info("尚無 OPEN 交易。可在上方 📓 真實持倉交易日誌建立持倉紀錄。")
    else:
        _pj_pf_syms = sorted(
            set(_pj_pf_open["symbol"].astype(str).str.strip().tolist())
        )
        _pj_pf_batch = load_data_batch(_pj_pf_syms, period="1y")
        _pj_pf_close: dict[str, float | None] = {}
        _pj_pf_mkt_bar: dict[str, str | None] = {}
        for _s in _pj_pf_syms:
            _dh = _pj_pf_batch.get(_s)
            if _dh is not None and not _dh.empty and "Close" in _dh.columns:
                try:
                    _pj_pf_close[_s] = float(
                        pd.to_numeric(_dh["Close"], errors="coerce").iloc[-1]
                    )
                    _pj_pf_mkt_bar[_s] = last_bar_date_from_ohlcv_df(_dh)
                except Exception:
                    _pj_pf_close[_s] = None
                    _pj_pf_mkt_bar[_s] = None
            else:
                try:
                    _dh2 = load_data(_s, "1y")
                    if (
                        _dh2 is not None
                        and not _dh2.empty
                        and "Close" in _dh2.columns
                    ):
                        _pj_pf_close[_s] = float(
                            pd.to_numeric(_dh2["Close"], errors="coerce").iloc[-1]
                        )
                        _pj_pf_mkt_bar[_s] = last_bar_date_from_ohlcv_df(_dh2)
                    else:
                        _pj_pf_close[_s] = None
                        _pj_pf_mkt_bar[_s] = None
                except Exception:
                    _pj_pf_close[_s] = None
                    _pj_pf_mkt_bar[_s] = None

        _pj_sum = summarize_open_trades_for_ui(
            _pj_pf_close,
            market_last_bar_date_by_symbol=_pj_pf_mkt_bar,
        )
        _pj_disp = _pj_sum.rename(
            columns={
                "entry_date": "進場日",
                "days_held": "持有天數",
                "market_last_bar_date": "行情最後 bar 日",
                "entry_price": "進場價",
                "shares": "股數",
                "latest_close": "最新收盤（最後 bar）",
                "unrealized_pnl": "未實現損益",
                "return_pct": "報酬%",
                "locked_tp_high": "鎖定目標價",
                "dynamic_tp_high": "動態目標（參考）",
                "distance_to_locked_high_pct": "距鎖定目標 (%)",
                "distance_to_dynamic_high_pct": "距動態目標 (%)（參考）",
                "drawdown_from_peak_pct": "自高點回撤 (%)",
                "trend_ok": "趨勢狀態",
                "risk_ok": "風險狀態",
                "regime_type": "市場狀態",
                "last_journal_bar_date": "最後更新 bar 日",
                "journal_stale_flag": "Journal 與行情",
            }
        )

        def _pj_fmt_journal_stale(v) -> str:
            try:
                if v is None or pd.isna(v):
                    return "無法比對"
            except Exception:
                return "無法比對"
            if v == True:  # noqa: E712
                return "待補寫（落後行情）"
            if v == False:  # noqa: E712
                return "已同步"
            return "無法比對"

        def _pj_fmt_dist_locked(v) -> str:
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "—"
                x = float(v)
            except Exception:
                return "—"
            if x <= 0:
                return "已達標"
            if x <= 3:
                return f"接近（{x:.2f}%）"
            return f"{x:.2f}%"

        def _pj_fmt_drawdown(v) -> str:
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "—"
                x = float(v)
            except Exception:
                return "—"
            if x <= -8:
                return f"高回撤（{x:.2f}%）"
            if x <= -5:
                return f"留意（{x:.2f}%）"
            return f"{x:.2f}%"

        def _pj_fmt_trend(s) -> str:
            t = str(s).strip() if s is not None else ""
            if t == "否":
                return "否 · 留意"
            return t if t else "—"

        def _pj_target_status_label(tr, nr, d_lock) -> str:
            t_na = pd.isna(tr)
            n_na = pd.isna(nr)
            if t_na and n_na:
                return "—"
            if not t_na and bool(tr):
                return "已達鎖定"
            if not n_na and bool(nr):
                try:
                    if pd.notna(d_lock) and np.isfinite(float(d_lock)):
                        return f"接近（{float(d_lock):.2f}%）"
                except (TypeError, ValueError):
                    pass
                return "接近"
            return ""

        def _pj_status_sort_priority(label: str) -> int:
            if label == "已達鎖定":
                return 0
            if isinstance(label, str) and label.startswith("接近"):
                return 1
            if label == "—":
                return 3
            return 2

        _pj_disp["距鎖定目標 (%)"] = _pj_sum["distance_to_locked_high_pct"].map(
            _pj_fmt_dist_locked
        )
        _pj_disp["距動態目標 (%)（參考）"] = _pj_sum[
            "distance_to_dynamic_high_pct"
        ].map(_pj_fmt_dist_locked)
        _pj_disp["自高點回撤 (%)"] = _pj_sum["drawdown_from_peak_pct"].map(
            _pj_fmt_drawdown
        )
        _pj_disp["趨勢狀態"] = _pj_sum["trend_ok"].map(_pj_fmt_trend)
        _pj_disp["風險狀態"] = _pj_sum["risk_ok"].map(_pj_fmt_trend)
        _pj_disp["Journal 與行情"] = _pj_sum["journal_stale_flag"].map(
            _pj_fmt_journal_stale
        )
        _pj_disp["目標狀態"] = [
            _pj_target_status_label(tr, nr, dlk)
            for tr, nr, dlk in zip(
                _pj_sum["target_reached_flag"],
                _pj_sum["near_locked_target_flag"],
                _pj_sum["distance_to_locked_high_pct"],
            )
        ]

        _pj_col_order = [
            "symbol",
            "trade_id",
            "進場日",
            "持有天數",
            "行情最後 bar 日",
            "進場價",
            "股數",
            "最新收盤（最後 bar）",
            "報酬%",
            "未實現損益",
            "鎖定目標價",
            "距鎖定目標 (%)",
            "目標狀態",
            "動態目標（參考）",
            "距動態目標 (%)（參考）",
            "自高點回撤 (%)",
            "趨勢狀態",
            "風險狀態",
            "市場狀態",
            "最後更新 bar 日",
            "Journal 與行情",
        ]
        _pj_disp = _pj_disp[[c for c in _pj_col_order if c in _pj_disp.columns]]
        for _pj_dc in ("行情最後 bar 日", "最後更新 bar 日"):
            if _pj_dc in _pj_disp.columns:
                _pj_disp[_pj_dc] = _pj_disp[_pj_dc].replace("", "—")

        _pj_disp = _pj_disp.assign(
            _sort_p=_pj_disp["目標狀態"].map(_pj_status_sort_priority),
            _sort_d=pd.to_numeric(
                _pj_sum["distance_to_locked_high_pct"], errors="coerce"
            ),
        ).sort_values(
            by=["_sort_p", "_sort_d"],
            ascending=[True, True],
            na_position="last",
        )
        _pj_disp = _pj_disp.drop(columns=["_sort_p", "_sort_d"])

        st.caption(
            "**持股決策表**：主軸為 **鎖定目標／距鎖定**（交易計畫，來自 trades 主檔）。"
            "**排序**：已達鎖定 → 接近（距鎖定由小到大）→ 其餘。**接近**欄位會附帶實際距離%。"
            f"**目標狀態**：**已達鎖定**＝最新收盤 ≥ 鎖定目標上緣；**接近**＝尚未已達且距鎖定目標在 **(0%，{NEAR_LOCKED_TARGET_THRESHOLD_PCT:g}%]**（其餘為實質未達，欄位留空）。"
            "**動態（參考）** 取自最新 journal，代表模型當下重算的延伸空間，非紀律主目標。"
            "**進場快照**（dynamic/valuation at entry）不在此表，請見 📓「進階：進場目標快照」。"
        )
        st.dataframe(
            _pj_disp,
            hide_index=True,
            width="stretch",
            column_config={
                "鎖定目標價": st.column_config.NumberColumn(
                    format="%.2f",
                    help="計畫目標（locked_tp_high）。",
                ),
                "動態目標（參考）": st.column_config.NumberColumn(
                    format="%.2f",
                    help="最新 journal 之 dynamic_tp_high，每日更新，僅供對照。",
                ),
                "持有天數": st.column_config.NumberColumn(format="%.0f"),
                "進場價": st.column_config.NumberColumn(format="%.2f"),
                "最新收盤（最後 bar）": st.column_config.NumberColumn(format="%.2f"),
                "報酬%": st.column_config.NumberColumn(format="%.2f"),
                "未實現損益": st.column_config.NumberColumn(format="%.0f"),
            },
        )

        with st.expander("批次更新 OPEN journal", expanded=True):
            _pj_ms = st.multiselect(
                "要更新的 symbol（留空＝全部）",
                options=_pj_pf_syms,
                default=_pj_pf_syms,
                key="portfolio_page_pj_multiselect",
            )
            _pj_only_miss = st.checkbox(
                "僅更新「尚缺目前最後一根 K 線日期」之 journal",
                value=False,
                key="portfolio_page_pj_only_miss",
            )
            _pj_show_detail = st.checkbox(
                "更新後顯示成功／跳過／失敗明細",
                value=True,
                key="portfolio_page_pj_show_detail",
            )
            if st.button("🔄 依選取更新 OPEN journal", key="portfolio_page_pj_update_btn"):
                _pj_sym_arg: list[str] | None
                if not _pj_ms or set(_pj_ms) == set(_pj_pf_syms):
                    _pj_sym_arg = None
                else:
                    _pj_sym_arg = list(_pj_ms)
                _pj_rep_pf = update_open_trades_daily(
                    fetch_df=lambda s: load_prepared_df_for_journal(s, "1y"),
                    symbols=_pj_sym_arg,
                    only_missing_today=_pj_only_miss,
                )
                st.success(
                    f"成功 {len(_pj_rep_pf.updated)} 筆；"
                    f"跳過 {len(_pj_rep_pf.skipped)}；失敗 {len(_pj_rep_pf.failed)}。"
                )
                if _pj_show_detail:
                    st.markdown("**已更新 trade_id**")
                    st.write(
                        ", ".join(_pj_rep_pf.updated)
                        if _pj_rep_pf.updated
                        else "（無）"
                    )
                    if _pj_rep_pf.skipped:
                        st.markdown("**跳過**")
                        st.dataframe(
                            pd.DataFrame(_pj_rep_pf.skipped),
                            hide_index=True,
                            width="stretch",
                        )
                    if _pj_rep_pf.failed:
                        st.markdown("**失敗（資料不足或 snapshot 錯誤）**")
                        st.dataframe(
                            pd.DataFrame(_pj_rep_pf.failed),
                            hide_index=True,
                            width="stretch",
                        )
                st.rerun()

    st.markdown("---")

    use_live_price = st.checkbox("使用即時價（可能較慢）", value=False)
    DEFAULT_PORTFOLIO_TEXT = "2330.TW, 610, 1000\n2317.TW, 98.5, 2000\n2882.TW"
    if "portfolio_raw_symbols" not in st.session_state:
        # 預設先吃資料夾內的 my_portfolio.csv（若存在且可解析），否則用內建範例
        default_csv_path = os.path.join(os.path.dirname(__file__), "my_portfolio.csv")
        default_raw = ""
        if os.path.exists(default_csv_path):
            try:
                df_csv_default = pd.read_csv(default_csv_path)
                default_raw = portfolio_csv_to_raw_lines(df_csv_default)
            except Exception:
                default_raw = ""
        st.session_state["portfolio_raw_symbols"] = (
            default_raw.strip() if default_raw.strip() else DEFAULT_PORTFOLIO_TEXT
        )

    with st.expander("📥 匯入持股 CSV（自動補股票名稱/診斷用）", expanded=False):
        st.caption("CSV 最簡欄位：stock_id,buy_price,shares/quantity（股數可省略）")
        default_csv_path = os.path.join(os.path.dirname(__file__), "my_portfolio.csv")
        if os.path.exists(default_csv_path):
            st.caption(f"已偵測到專案內預設檔：`{default_csv_path}`（首次進頁面會自動載入）")
            if st.button("🔄 重新讀取本機 my_portfolio.csv（覆蓋下方輸入框）"):
                try:
                    df_csv_default = pd.read_csv(default_csv_path)
                    raw_default = portfolio_csv_to_raw_lines(df_csv_default)
                    if not raw_default.strip():
                        st.warning("本機 my_portfolio.csv 讀到了，但沒有解析出有效持股列（請檢查欄位/成本）。")
                    else:
                        st.session_state["portfolio_raw_symbols"] = raw_default
                        st.rerun()
                except Exception as exc:
                    st.warning(f"讀取本機 my_portfolio.csv 失敗：{exc}")
        up = st.file_uploader("上傳 my_portfolio.csv", type=["csv"])
        if up is not None:
            try:
                df_csv = pd.read_csv(up)
                raw_from_csv = portfolio_csv_to_raw_lines(df_csv)
                if not raw_from_csv.strip():
                    st.warning("CSV 讀到了，但沒有成功解析出任何有效持股列（請檢查欄位/成本）。")
                else:
                    st.success("CSV 已解析完成。按下「套用」會覆蓋下方輸入框。")
                    st.code(raw_from_csv, language="text")
                    if st.button("✅ 套用 CSV 到下方持股輸入框"):
                        st.session_state["portfolio_raw_symbols"] = raw_from_csv
                        st.rerun()
            except Exception as exc:
                st.warning(f"CSV 解析失敗：{exc}")

    raw_symbols = st.text_area(
        "我的持股（可選：代碼, 成本均價, 股數）",
        key="portfolio_raw_symbols",
        height=200,
    )
    portfolio_items = parse_portfolio_lines(raw_symbols)
    seen = set()
    items = []
    for it in portfolio_items:
        sym = it["symbol"].strip()
        if sym and sym not in seen:
            seen.add(sym)
            items.append(it)
    symbols = [it["symbol"] for it in items]

    # 籌碼資料開關（FinMind）：持股多時可先關閉，速度會差很多
    if "portfolio_enable_chip" not in st.session_state:
        st.session_state["portfolio_enable_chip"] = bool(FINMIND_AVAILABLE)
    enable_chip = st.checkbox(
        "啟用籌碼資料（FinMind，可能較慢）",
        value=bool(st.session_state.get("portfolio_enable_chip", bool(FINMIND_AVAILABLE))),
        key="portfolio_enable_chip",
        disabled=not bool(FINMIND_AVAILABLE),
    )
    if not FINMIND_AVAILABLE:
        st.caption("FinMind 未啟用：持股清單的籌碼欄位將顯示空白。")

    MAX_SYMBOLS = 80
    if len(symbols) > MAX_SYMBOLS:
        st.warning(
            f"你輸入了 {len(symbols)} 檔，為避免資料源限流，先只分析前 {MAX_SYMBOLS} 檔。"
        )
        symbols = symbols[:MAX_SYMBOLS]

    rows = []
    batch_missing_fixed = []
    still_missing = []
    resolved_map = {}
    name_map = {}
    resolved_list = []
    turn_cfg = load_turn_config()
    turn_mode_default = turn_cfg.get("mode_default", "bottom")

    for it in items:
        sym = it["symbol"]
        display_name, resolved = fetch_ticker_name(sym)
        resolved_sym = (resolved or sym).strip()
        it["resolved_sym"] = resolved_sym
        it["display_name"] = display_name or "N/A"
        resolved_map[sym] = resolved_sym
        name_map[resolved_sym] = display_name or "N/A"
        if resolved_sym not in resolved_list:
            resolved_list.append(resolved_sym)

    # 為了讓 cache key 穩定（避免輸入順序造成不必要的重抓），先排序
    resolved_list = sorted(resolved_list)
    data_dict = load_data_batch(resolved_list, period="1y")

    live_price_map = {}
    if use_live_price and resolved_list:
        try:
            live_price_map = fetch_last_price_batch(resolved_list) or {}
        except Exception:
            live_price_map = {}

    # 指標快取（避免你只改排序/勾選就重算 80 檔指標）
    ind_cache = st.session_state.setdefault("portfolio_ind_cache", {})
    try:
        for k in list(ind_cache.keys()):
            if k not in resolved_list:
                ind_cache.pop(k, None)
    except Exception:
        pass

    # 進度條（持股多時才顯示，避免小清單閃爍）
    progress_ph = st.empty()
    status_ph = st.empty()
    bar = None
    total_n = len(items)
    if total_n >= 5:
        bar = progress_ph.progress(0)

    for it in items:
        resolved_sym = it.get("resolved_sym") or resolved_map.get(it["symbol"], it["symbol"])
        display_name = it.get("display_name") or name_map.get(resolved_sym, "N/A")
        if bar is not None and total_n > 0:
            try:
                i_now = len(rows) + 1
                status_ph.caption(f"分析中：{resolved_sym}（{i_now}/{total_n}）")
                bar.progress(min(100, int(i_now / float(total_n) * 100)))
            except Exception:
                pass

        df_hold = data_dict.get(resolved_sym)
        # yfinance 偶發批次下載漏資料：若 batch 抓不到，退回單檔抓取補齊
        batch_missing = df_hold is None or df_hold.empty
        if batch_missing:
            try:
                df_hold = load_data(resolved_sym, "1y")
            except Exception:
                pass
            if df_hold is not None and not df_hold.empty:
                batch_missing_fixed.append(resolved_sym)
        if df_hold is None or df_hold.empty:
            if batch_missing:
                still_missing.append(resolved_sym)
            rows.append(
                {
                    "股票": resolved_sym,
                    "名稱": display_name,
                    "TURN": "",
                    "TURN 分數": np.nan,
                    "籌碼燈號": "",
                    "籌碼行動": "",
                    "籌碼診斷": "",
                    "法人合力3日(張)": np.nan,
                    "土洋強弱比%": np.nan,
                    "法人參與度%": np.nan,
                    "外資3日(張)": np.nan,
                    "投信3日(張)": np.nan,
                    "股數(股)": it.get("shares") if it.get("shares") is not None else np.nan,
                    "成本均價": it.get("avg_cost") if it.get("avg_cost") is not None else np.nan,
                    "目前價": np.nan,
                    "投入成本": np.nan,
                    "未實現損益": np.nan,
                    "未實現損益%": np.nan,
                    "目標價高": np.nan,
                    "達標可賺": np.nan,
                    "達標報酬%": np.nan,
                    "每張達標可賺": np.nan,
                    "距離達標%": np.nan,
                    "AI 分數": np.nan,
                }
            )
            continue

        # 指標計算（有快取）
        raw_as_of = None
        try:
            raw_as_of = df_hold.index[-1]
        except Exception:
            raw_as_of = None
        cached = ind_cache.get(resolved_sym) if isinstance(ind_cache, dict) else None
        if (
            isinstance(cached, dict)
            and cached.get("as_of") == raw_as_of
            and int(cached.get("n") or 0) == int(len(df_hold))
            and isinstance(cached.get("df"), pd.DataFrame)
            and not cached.get("df").empty
        ):
            df_hold = cached.get("df")
        else:
            df_hold = compute_indicators(df_hold, include_vwap=False)
            df_hold = df_hold.dropna(subset=["Close"]).sort_index()
            try:
                ind_cache[resolved_sym] = {"as_of": raw_as_of, "n": int(len(df_hold)), "df": df_hold}
            except Exception:
                pass

        latest_row = df_hold.iloc[-1]
        close_price = to_scalar(latest_row.get("Close"))
        if use_live_price:
            live_price = live_price_map.get(resolved_sym)
            if live_price is not None and np.isfinite(live_price):
                close_price = live_price
        ema20 = to_scalar(latest_row.get("EMA20"))
        ema5 = to_scalar(latest_row.get("EMA5"))
        bias_20_val = to_scalar(latest_row.get("Bias20", np.nan))

        avg_vol_5 = (
            to_scalar(df_hold["Volume"].rolling(5).mean().iloc[-1])
            if "Volume" in df_hold.columns
            else np.nan
        )
        current_vol = (
            to_scalar(latest_row.get("Volume"))
            if "Volume" in df_hold.columns
            else np.nan
        )

        foreign_series = None
        trust_series = None
        if enable_chip:
            try:
                foreign_series, trust_series = fetch_chip_net_series(resolved_sym)
            except Exception:
                foreign_series, trust_series = None, None
        foreign_series_aligned = align_net_series_to_price(df_hold, foreign_series)
        trust_series_aligned = align_net_series_to_price(df_hold, trust_series)
        foreign_for_score = (
            foreign_series_aligned if foreign_series_aligned is not None else foreign_series
        )
        trust_for_score = (
            trust_series_aligned if trust_series_aligned is not None else trust_series
        )

        foreign_used = (
            foreign_series_aligned if foreign_series_aligned is not None else foreign_series
        )
        trust_used = trust_series_aligned if trust_series_aligned is not None else trust_series
        foreign_3d_net = (
            foreign_used.sort_index().tail(3).sum()
            if foreign_used is not None and len(foreign_used) >= 3
            else None
        )
        trust_3d_net = (
            trust_used.sort_index().tail(3).sum()
            if trust_used is not None and len(trust_used) >= 3
            else None
        )
        # 若籌碼資料未啟用/不足，改用 NaN 避免 TURN chip 條件被「默認通過」
        foreign_3d_for_turn = (
            float(foreign_3d_net)
            if foreign_3d_net is not None and np.isfinite(foreign_3d_net)
            else np.nan
        )
        trust_3d_for_turn = (
            float(trust_3d_net)
            if trust_3d_net is not None and np.isfinite(trust_3d_net)
            else np.nan
        )
        foreign_sell_3d = None
        try:
            if foreign_used is not None and len(foreign_used) >= 3:
                f3 = pd.to_numeric(foreign_used.sort_index().tail(3), errors="coerce").dropna()
                if len(f3) >= 3:
                    foreign_sell_3d = bool((f3 < 0).all())
        except Exception:
            foreign_sell_3d = None

        vol_sum_3d = compute_volume_sum_3d(df_hold)
        df_turn = df_hold.copy()
        if "RSI" not in df_turn.columns and "RSI14" in df_turn.columns:
            df_turn["RSI"] = df_turn["RSI14"]
        turn_result = run_turn_check(
            df_turn,
            foreign_3d_net=foreign_3d_for_turn,
            trust_3d_net=trust_3d_for_turn,
            mode=turn_mode_default,
            cfg=turn_cfg,
        )

        weighted_score, _, _ = compute_weighted_score(
            ema20=ema20,
            ema5=ema5,
            close_price=close_price,
            current_vol=current_vol,
            avg_vol_5=avg_vol_5,
            bias_20_val=bias_20_val,
            foreign_net_series=foreign_for_score,
            trust_net_series=trust_for_score,
            vol_sum_3d=vol_sum_3d,
            is_dangerous_vol=bool(latest_row.get("Is_Dangerous_Volume", False)),
        )

        # 精準診斷（力道天平）：土洋對峙 / 法人參與度 / 高檔派發
        precision = None
        if enable_chip:
            try:
                precision = diagnose_precision(
                    foreign_3d_net=foreign_3d_net,
                    trust_3d_net=trust_3d_net,
                    vol_sum_3d_lot=vol_sum_3d,
                    bias20=bias_20_val,
                    beta=None,
                    foreign_sell_3d=foreign_sell_3d,
                )
            except Exception:
                precision = None
        chip_light = ""
        chip_action = ""
        chip_headline = ""
        total_inst_3d = None
        showdown_ratio = None
        inst_part = None
        if precision is not None:
            chip_light = "🚨" if precision.level == "danger" else "⚠️" if precision.level == "warning" else "✅"
            chip_action = precision.action
            chip_headline = precision.headline
            total_inst_3d = precision.total_inst_3d
            showdown_ratio = precision.showdown_ratio
            inst_part = precision.inst_participation_pct
        elif not enable_chip:
            chip_light = "⏸️"
            chip_headline = "籌碼未啟用（可勾選上方開關）"

        tp = estimate_target_range(df_hold, resolved_sym)
        tp_high = tp.get("tp_high") if tp is not None else None

        avg_cost = it.get("avg_cost")
        shares_qty = it.get("shares")

        invest = None
        mkt_value = None
        upl = None
        upl_pct = None
        tp_profit = None
        tp_profit_pct = None
        tp_profit_per_lot = None
        distance_to_tp_pct = None

        if close_price is not None and not pd.isna(close_price) and shares_qty:
            mkt_value = close_price * shares_qty

        if avg_cost is not None and shares_qty:
            invest = avg_cost * shares_qty

        if invest is not None and mkt_value is not None:
            upl = mkt_value - invest
            upl_pct = (upl / invest) * 100 if invest != 0 else None

        base_price = avg_cost if avg_cost is not None else close_price
        if tp_high is not None and shares_qty and base_price is not None:
            tp_profit = (tp_high - base_price) * shares_qty
            if base_price != 0:
                tp_profit_pct = ((tp_high / base_price) - 1) * 100
        if tp_high is not None and base_price is not None:
            tp_profit_per_lot = (tp_high - base_price) * 1000

        if tp_high is not None and close_price is not None and not pd.isna(close_price):
            distance_to_tp_pct = ((tp_high - close_price) / close_price) * 100 if close_price != 0 else None

        rows.append(
            {
                "股票": resolved_sym,
                "名稱": display_name,
                "TURN": turn_result.get("status"),
                "TURN 分數": turn_result.get("score"),
                "籌碼燈號": chip_light,
                "籌碼行動": chip_action,
                "籌碼診斷": chip_headline,
                "法人合力3日(張)": total_inst_3d,
                "土洋強弱比%": showdown_ratio,
                "法人參與度%": inst_part,
                "外資3日(張)": foreign_3d_net,
                "投信3日(張)": trust_3d_net,
                "股數(股)": shares_qty if shares_qty is not None else np.nan,
                "成本均價": avg_cost if avg_cost is not None else np.nan,
                "目前價": close_price,
                "投入成本": invest,
                "未實現損益": upl,
                "未實現損益%": upl_pct,
                "目標價高": tp_high,
                "達標可賺": tp_profit,
                "達標報酬%": tp_profit_pct,
                "每張達標可賺": tp_profit_per_lot,
                "距離達標%": distance_to_tp_pct,
                "AI 分數": weighted_score,
            }
        )

    if rows:
        if bar is not None:
            try:
                bar.progress(100)
            except Exception:
                pass
            progress_ph.empty()
            status_ph.empty()
        if batch_missing_fixed:
            st.caption(
                f"⚙️ 批次抓價偶發漏資料：{', '.join(batch_missing_fixed)}（已自動改用單檔補抓）"
            )
        if still_missing:
            st.warning(
                f"以下標的仍抓不到日線資料（可能是資料源暫時失敗或代號不支援）：{', '.join(still_missing)}"
            )
        sort_mode = st.selectbox(
            "進階排序模式",
            ["資金效率（達標報酬%・未扣費）", "可達性（距離達標%）", "TURN Gate（依 TURN 狀態）"],
            index=0,
        )
        df_list = pd.DataFrame(rows)
        # 資料源偶發失敗時，可能會因為所有 df_hold 都是空而缺欄位，排序會 KeyError。
        # 先補齊關鍵欄位，確保排序/顯示穩定。
        _text_defaults = {"TURN": "", "籌碼燈號": "", "籌碼行動": "", "籌碼診斷": ""}
        _num_defaults = [
            "TURN 分數",
            "AI 分數",
            "達標報酬%",
            "距離達標%",
            "每張達標可賺",
            "法人合力3日(張)",
            "土洋強弱比%",
            "法人參與度%",
            "外資3日(張)",
            "投信3日(張)",
        ]
        for _k, _v in _text_defaults.items():
            if _k not in df_list.columns:
                df_list[_k] = _v
        for _k in _num_defaults:
            if _k not in df_list.columns:
                df_list[_k] = np.nan
        if not df_list.empty:
            turn_order = {"ALLOW": 0, "WATCH": 1, "BLOCK": 2}
            if turn_mode_default == "top":
                turn_order = {"BLOCK": 0, "WATCH": 1, "ALLOW": 2}
            df_list["_TURN_RANK"] = (
                df_list["TURN"].map(turn_order).fillna(99)
                if "TURN" in df_list.columns
                else 99
            )
            df_list["_TURN_SCORE_NUM"] = pd.to_numeric(
                df_list.get("TURN 分數"), errors="coerce"
            )
            df_list["_AI_SCORE_NUM"] = pd.to_numeric(
                df_list.get("AI 分數"), errors="coerce"
            )
            best_idx = (
                df_list.sort_values(
                    by=["_TURN_RANK", "_TURN_SCORE_NUM", "_AI_SCORE_NUM"],
                    ascending=[True, False, False],
                    na_position="last",
                )
                .head(1)
                .index
            )
            df_list["最佳"] = ""
            best_label = "⭐較佳進場" if turn_mode_default == "bottom" else "⭐較佳出場"
            if len(best_idx) > 0:
                df_list.loc[best_idx[0], "最佳"] = best_label
            df_list = df_list.drop(
                columns=["_TURN_RANK", "_TURN_SCORE_NUM", "_AI_SCORE_NUM"]
            )
        preferred_order = [
            "股票",
            "名稱",
            "最佳",
            "TURN",
            "TURN 分數",
            "AI 分數",
            "籌碼燈號",
            "籌碼行動",
            "籌碼診斷",
            "法人合力3日(張)",
            "土洋強弱比%",
            "法人參與度%",
            "外資3日(張)",
            "投信3日(張)",
            "股數(股)",
            "成本均價",
            "目前價",
            "投入成本",
            "未實現損益",
            "未實現損益%",
            "目標價高",
            "達標可賺",
            "達標報酬%",
            "每張達標可賺",
            "距離達標%",
        ]
        existing_order = [c for c in preferred_order if c in df_list.columns]
        remaining_cols = [c for c in df_list.columns if c not in existing_order]
        df_list = df_list[existing_order + remaining_cols]
        if sort_mode == "資金效率（達標報酬%・未扣費）":
            df_list = df_list.sort_values(
                by=["達標報酬%", "距離達標%", "AI 分數"],
                ascending=[False, True, False],
                na_position="last",
            )
        elif sort_mode == "可達性（距離達標%）":
            df_list = df_list.sort_values(
                by=["距離達標%", "達標報酬%", "AI 分數"],
                ascending=[True, False, False],
                na_position="last",
            )
        else:
            turn_order = {"ALLOW": 0, "WATCH": 1, "BLOCK": 2}
            if turn_mode_default == "top":
                turn_order = {"BLOCK": 0, "WATCH": 1, "ALLOW": 2}
            df_list["_TURN_ORDER"] = (
                df_list["TURN"].map(turn_order).fillna(99)
                if "TURN" in df_list.columns
                else 99
            )
            df_list = df_list.sort_values(
                by=["_TURN_ORDER", "TURN 分數", "AI 分數"],
                ascending=[True, False, False],
                na_position="last",
            ).drop(columns=["_TURN_ORDER"])

        money_cols = ["投入成本", "未實現損益", "達標可賺", "每張達標可賺"]
        pct_cols = ["未實現損益%", "達標報酬%", "距離達標%"]
        chip_pct_cols = ["土洋強弱比%", "法人參與度%"]
        chip_int_cols = ["法人合力3日(張)", "外資3日(張)", "投信3日(張)"]

        for c in money_cols:
            if c in df_list.columns:
                df_list[c] = df_list[c].map(
                    lambda x: f"{x:,.0f}" if pd.notna(x) and x != "" else ""
                )

        for c in pct_cols:
            if c in df_list.columns:
                df_list[c] = df_list[c].map(
                    lambda x: f"{x:.1f}%" if pd.notna(x) and x != "" else ""
                )
        for c in chip_pct_cols:
            if c in df_list.columns:
                df_list[c] = df_list[c].map(
                    lambda x: f"{x:.1f}%" if pd.notna(x) and x != "" else ""
                )
        for c in chip_int_cols:
            if c in df_list.columns:
                df_list[c] = df_list[c].map(
                    lambda x: f"{x:+,.0f}" if pd.notna(x) and x != "" else ""
                )
        if "股數(股)" in df_list.columns:
            df_list["股數(股)"] = pd.to_numeric(df_list["股數(股)"], errors="coerce")
        st.subheader("持股排序（目標差優先、分數次之）")
        st.caption("達標可賺 / 達標報酬% 為未扣手續費與交易稅的估算值")
        st.dataframe(
            df_list,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("請先輸入至少一檔股票代碼。")

    st.stop()

df = load_data(effective_symbol, time_range)
if df is None or df.empty:
    df = _load_data_raw(effective_symbol, time_range)

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if x is not None]) for col in df.columns
        ]
    df.columns = df.columns.str.strip()

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"找不到必要的 OHLCV 欄位：{', '.join(missing)}，現有欄位：{df.columns.tolist()}")
        st.stop()
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. 計算技術指標與買入訊號
    df = compute_indicators(df)
    df = compute_buy_signals(df)

    # 4.0.0 投資人風險指標（使用較長期間資料）
    df_risk = load_data(effective_symbol, "1y")
    if df_risk is None or df_risk.empty:
        df_risk = df
    mkt_df, idx_symbol = load_market_index(effective_symbol)
    risk = compute_risk_metrics(df_risk, mkt_df, rf_annual=0.015)

    foreign_net_series = None
    trust_net_series = None
    foreign_net_latest = None
    foreign_net_date = None
    foreign_divergence_warning = False
    try:
        foreign_net_series = fetch_foreign_net_series(effective_symbol)
        trust_net_series = fetch_trust_net_series(effective_symbol)
        foreign_series_aligned = align_net_series_to_price(df, foreign_net_series)
        trust_series_aligned = align_net_series_to_price(df, trust_net_series)
        if foreign_series_aligned is not None:
            foreign_net_series = foreign_series_aligned
        if trust_series_aligned is not None:
            trust_net_series = trust_series_aligned
        if foreign_net_series is not None and not foreign_net_series.empty:
            foreign_net_latest = foreign_net_series.iloc[-1]
            foreign_net_date = foreign_net_series.index[-1]
            aligned = align_by_date(df, foreign_net_series)
            if len(aligned) >= 3:
                recent = aligned.tail(3)
                if (recent["net"] < 0).all() and recent["Close"].iloc[-1] > recent["Close"].iloc[0]:
                    foreign_divergence_warning = True
    except Exception:
        foreign_net_series = None
        trust_net_series = None

    # 4.0 最新價量
    latest_row = df.iloc[-1]
    latest_close = to_scalar(latest_row["Close"])
    prev_close = to_scalar(df["Close"].iloc[-2]) if len(df) >= 2 else None
    latest_volume = to_scalar(latest_row["Volume"])
    latest_vol_avg_5 = np.nan
    if effective_symbol.endswith(".TW") or effective_symbol.endswith(".TWO"):
        vol_avg_5_series = df["Volume"].rolling(5).mean()
        latest_vol_avg_5 = to_scalar(vol_avg_5_series.iloc[-1])

    # 4.0.2 建倉建議
    st.subheader("建倉建議")
    st.write(
        "Gate 條件：價格 > SMA20、SMA20 走升、20日乖離 <= 10%、成交量 <= 1.5x 20日均量。"
    )
    st.write(
        "Trigger 條件：回踩買（乖離 -1%~+3% 且收盤 > 昨收）、突破買（創 20 日新高且量能放大但不爆量），或延續買（乖離 3%~10% 且收盤創 5 日新高）。"
    )
    with st.expander("Gate / Trigger 白話說明"):
        st.markdown(
            """
**Gate = 允許進場的基本安全門檻**  
回答的是：現在這個位置，值不值得進入「考慮建倉」的狀態？

你定義的 Gate 條件：
- 價格 > SMA20：股價站在中期均線之上（趨勢至少不弱）
- SMA20 走升：中期趨勢往上（不是橫盤或下彎）
- 20 日乖離 ≤ 10%：不要追高過熱（避免被洗掉）
- 成交量 ≤ 1.5 × 20 日均量：不要爆量失控（避免高檔出貨/過熱）

**Trigger = 進場的點火訊號**  
回答的是：就算環境允許，什麼時機點進去比較合理？

你定義三種 Trigger：
- 回踩買（PULLBACK）：乖離 -1%～+3%，且收盤 > 昨收
- 突破買（BREAKOUT）：收盤創 20 日新高，且量能放大（>1.2×20 日均量）但不爆量
 - 延續買（CONTINUATION）：乖離 +3%～+10%，且收盤創 5 日新高

**為什麼 Gate AND Trigger 都要過？**  
Gate 避開不該碰的環境，Trigger 幫你找比較划算的進場點。  
所以 BUY_SIGNAL = Gate 通過 AND Trigger 通過。

**Execution Guard = 防假突破/追高風險的保護層**  
只在 BREAKOUT / CONTINUATION 類型時啟用，Pullback 不會被嚴格限制。  
它會檢查：
- 收盤位置要夠強（收在 K 線上緣附近，避免長上影假突破）
- 量能不要失控（放量可接受，但不超過 2.0× 20 日均量）
- 乖離不要貼近上限（避免過熱追高）
- 跌回 Anchored VWAP 下方（視為成本線失守，避免追高換手）
  
通過 Gate + Trigger 後，還要過 Guard 才會發出 BUY_SIGNAL。
            """
        )
    latest = df.iloc[-1]
    gate_ok = bool(latest.get("BUY_GATE", False))
    trigger_ok = bool(latest.get("BUY_TRIGGER", False))
    trigger_type = str(latest.get("BUY_TRIGGER_TYPE", "NONE"))
    exec_guard_ok = bool(latest.get("EXEC_GUARD", True))
    exec_block_reason = str(latest.get("EXEC_BLOCK_REASON", "")).strip()
    buy_flag = bool(latest.get("BUY_SIGNAL", False))
    buy_note = str(latest.get("BUY_NOTE", "")).strip()
    gate_label = "通過" if gate_ok else "未通過"
    trigger_label = "通過" if trigger_ok else "未通過"
    exec_guard_label = "通過" if exec_guard_ok else "未通過"
    signal_label = "買入" if buy_flag else "觀望"
    gate_color = "#16a34a" if gate_ok else "#6b7280"
    trigger_color = "#16a34a" if trigger_ok else "#6b7280"
    exec_guard_color = "#16a34a" if exec_guard_ok else "#ef4444"
    signal_color = "#16a34a" if buy_flag else "#f59e0b"
    st.markdown(
        f"""
<div style="padding:12px;border:1px solid #e5e7eb;border-radius:8px;">
  <div style="font-weight:600;margin-bottom:6px;">狀態總覽</div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;">
    <span style="background:{gate_color};color:white;padding:4px 8px;border-radius:999px;">Gate: {gate_label}</span>
    <span style="background:{trigger_color};color:white;padding:4px 8px;border-radius:999px;">Trigger: {trigger_label}</span>
    <span style="background:{exec_guard_color};color:white;padding:4px 8px;border-radius:999px;">Guard: {exec_guard_label}</span>
    <span style="background:{signal_color};color:white;padding:4px 8px;border-radius:999px;">信號: {signal_label}</span>
  </div>
  <div style="margin-top:6px;color:#6b7280;">Trigger 類型：{trigger_type}</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    if buy_note:
        st.info(f"買入訊號狀態：{buy_note}")
    if not exec_guard_ok:
        if exec_block_reason:
            st.warning(f"Guard 未通過：{exec_block_reason}")
        else:
            st.warning(
                "Guard 未通過：代表突破/續強型態風險偏高（可能假突破或情緒爆量），建議等待更乾淨的確認。"
            )

    bias_20_val = to_scalar(latest.get("Bias20", np.nan))
    vol_ma20_val = to_scalar(latest.get("VolMA20", np.nan))
    sma20_val = to_scalar(latest.get("SMA20", np.nan))
    sma60_val = to_scalar(latest.get("SMA60", np.nan))
    atr14_val = to_scalar(latest.get("ATR14", np.nan))
    stop_loss_price = None
    take_profit_price = None
    if not pd.isna(atr14_val):
        stop_loss_price = latest_close - (2 * atr14_val)
        take_profit_price = latest_close + (3 * atr14_val)
    metrics_row = st.columns(4)
    with metrics_row[0]:
        if not pd.isna(bias_20_val):
            st.metric("SMA20乖離率", f"{bias_20_val:.2f}%")
        else:
            st.metric("SMA20乖離率", "資料不足")
    with metrics_row[1]:
        if not pd.isna(sma20_val) and not pd.isna(sma60_val):
            st.metric("SMA20/SMA60", f"{sma20_val:.2f} / {sma60_val:.2f}")
        else:
            st.metric("SMA20/SMA60", "資料不足")
    with metrics_row[2]:
        if not pd.isna(vol_ma20_val):
            st.metric("20日均量", f"{vol_ma20_val:,.0f}")
        else:
            st.metric("20日均量", "資料不足")
    with metrics_row[3]:
        if not pd.isna(atr14_val):
            st.metric("ATR14", f"{atr14_val:.2f}")
        else:
            st.metric("ATR14", "資料不足")

    # 4.0.1 成交量分析
    df["Vol_Avg_5"] = df["Volume"].rolling(5).mean()
    current_vol = to_scalar(df["Volume"].iloc[-1])
    avg_vol_5 = to_scalar(df["Vol_Avg_5"].iloc[-1])

    st.subheader("成交量分析")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("今日成交量", f"{current_vol:,.0f}")
    with col2:
        if not pd.isna(avg_vol_5):
            vol_ratio = current_vol / avg_vol_5
            delta_label = f"{'爆量' if vol_ratio > 1.5 else '縮量'}({vol_ratio:.2f}x)"
            st.metric("5日均量", f"{avg_vol_5:,.0f}", delta=delta_label)
        else:
            st.metric("5日均量", "資料不足")


    # 4.1 今日收盤價與 EMA20 的關係
    st.subheader("收盤價與 EMA20")
    ema20 = None
    ema5 = None
    close_price = latest_close
    if "EMA20" in df.columns:
        ema_df = df.dropna(subset=["EMA20", "EMA5"])
    else:
        ema_df = pd.DataFrame()

    if not ema_df.empty:
        ema_latest = ema_df.iloc[-1]
        close_price = to_scalar(ema_latest["Close"])
        ema20 = to_scalar(ema_latest["EMA20"])
        ema5 = to_scalar(ema_latest["EMA5"])
        st.write(f"當前 EMA20: {ema20:.2f}")
        st.write(f"當前 EMA5: {ema5:.2f}")
        st.metric("收盤價", f"{close_price:.2f}", delta=f"{close_price - ema20:.2f} vs EMA20")

        def build_push_message(event, decision, score_value=None):
            rsi_latest = df["RSI14"].iloc[-1]
            return format_actionable_summary(
                ticker=ticker,
                close_price=close_price,
                ema20=ema20,
                ema5=ema5,
                bias20=bias_20_val,
                rsi=rsi_latest,
                vol=latest_volume,
                vol5=avg_vol_5,
                foreign_net=foreign_net_latest,
                score=score_value,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                event=event,
                decision=decision,
            )
        if len(ema_df) >= 2:
            ema_prev = ema_df.iloc[-2]
            crossed = ema_prev["Close"] <= ema_prev["EMA20"] and close_price > ema20
            prev_ema5 = to_scalar(ema_prev["EMA5"])
            prev_ema20 = to_scalar(ema_prev["EMA20"])
            golden_cross = prev_ema5 <= prev_ema20 and ema5 > ema20
            if crossed:
                st.success("股價剛突破 EMA20，推薦買入")
                st.balloons()
            elif close_price > ema20:
                st.info("股價高於 EMA20")
                st.write("短中線分水嶺：股價在 EMA20 之上，代表過去一個月買的人平均是賺錢的，屬於多頭強勢。")
                st.write("策略解釋：黃金交叉/站上均線，推薦買入。")
            else:
                st.info("股價低於 EMA20")
                st.write("短中線分水嶺：股價在 EMA20 之下，代表短線趨勢偏弱，屬於空頭整理。")
                st.write("策略解釋：跌破/弱勢整理，建議觀望。")

            # Strategy 區塊
            st.subheader("Strategy")
            if golden_cross:
                st.success("偵測到黃金交叉，屬於多頭訊號")
                if st.button("發送黃金交叉通知"):
                    ok, msg = send_with_cooldown(
                        f"{effective_symbol}_golden_cross",
                        build_push_message("黃金交叉", "買入"),
                        cooldown_minutes=60,
                    )
                    if ok:
                        st.success("已傳送黃金交叉通知")
                    else:
                        st.warning(msg)
            if crossed and avg_vol_5 and current_vol > avg_vol_5 * 1.5:
                st.info("交叉伴隨放量，訊號強度較高")
            if golden_cross and avg_vol_5 and current_vol > avg_vol_5 * 1.5:
                st.success("黃金交叉伴隨放量，屬於強訊號")
                if st.button("發送黃金交叉強訊號通知"):
                    ok, msg = send_with_cooldown(
                        f"{effective_symbol}_golden_cross_strong",
                        build_push_message("黃金交叉強訊號", "買入"),
                        cooldown_minutes=60,
                    )
                    if ok:
                        st.success("已傳送強訊號通知")
                    else:
                        st.warning(msg)
        else:
            st.info("資料不足，無法判斷是否突破 EMA20")
    else:
        st.warning("EMA20 計算失敗，請檢查資料長度是否足夠（至少 20 天）")

    # 4.1.1 去留診斷
    is_high_volume_sell = False
    if ema20 is not None and ema5 is not None and not pd.isna(avg_vol_5):
        is_high_price = close_price > ema20 * 1.05
        is_huge_vol = current_vol > avg_vol_5 * 2
        is_selling_pressure = close_price < to_scalar(df["Open"].iloc[-1])
        is_high_volume_sell = is_high_price and is_huge_vol and is_selling_pressure
    weighted_score, weighted_reasons, weighted_flags = compute_weighted_score(
        ema20=ema20,
        ema5=ema5,
        close_price=close_price,
        current_vol=current_vol,
        avg_vol_5=avg_vol_5,
        bias_20_val=bias_20_val,
        foreign_net_series=foreign_net_series,
        trust_net_series=trust_net_series,
        vol_sum_3d=compute_volume_sum_3d(df),
        is_dangerous_vol=bool(latest.get("Is_Dangerous_Volume", False)),
    )
    context_score = 0
    context_reasons = []
    ema20_up = None
    if ema20 is not None and len(ema_df) >= 2:
        ema20_up = to_scalar(ema_df["EMA20"].iloc[-1]) > to_scalar(
            ema_df["EMA20"].iloc[-2]
        )
    if ema20 is None or ema20_up is None:
        context_reasons.append("EMA20 資料不足，無法判斷趨勢")
    elif close_price > ema20 and ema20_up:
        context_score += 1
        context_reasons.append("股價在 EMA20 之上且 EMA20 走升，趨勢偏多")
    else:
        context_score -= 1
        context_reasons.append("股價跌破 EMA20 或 EMA20 走平/走弱")

    if ema5 is None or ema20 is None:
        context_reasons.append("EMA5/EMA20 資料不足，無法判斷交叉")
    elif ema5 > ema20:
        context_score += 1
        context_reasons.append("短期動能強於中期 (EMA5 > EMA20)")
    else:
        context_score -= 1
        context_reasons.append("短期動能轉弱 (EMA5 <= EMA20)")

    if not pd.isna(df["RSI14"].iloc[-1]):
        rsi_latest = to_scalar(df["RSI14"].iloc[-1])
        if rsi_latest >= 50:
            context_score += 1
            context_reasons.append("RSI 動能偏多 (>= 50)")
        else:
            context_reasons.append("RSI 動能偏弱 (< 50)")

    if foreign_net_latest is not None:
        if foreign_net_latest > 0:
            context_score += 1
            context_reasons.append("外資買超，籌碼偏多")
        elif foreign_net_latest < 0:
            context_reasons.append("外資賣超，籌碼偏空")

    trend_mid_ok = (
        not pd.isna(sma20_val)
        and not pd.isna(sma60_val)
        and sma20_val > sma60_val
    )
    if trend_mid_ok:
        context_score += 1
        context_reasons.append("SMA20 高於 SMA60，中期趨勢偏多")

    sma60_up = False
    if len(df) >= 6:
        sma60_now = df["SMA60"].iloc[-1]
        sma60_prev = df["SMA60"].iloc[-6]
        if not pd.isna(sma60_now) and not pd.isna(sma60_prev):
            sma60_up = sma60_now > sma60_prev
            if sma60_up:
                context_score += 1
                context_reasons.append("SMA60 走升，長週期動能偏多")

    vol_ratio_20 = None
    if not pd.isna(vol_ma20_val) and vol_ma20_val > 0:
        vol_ratio_20 = latest_volume / vol_ma20_val

    vol_breakout_ok = (
        trigger_type == "BREAKOUT"
        and vol_ratio_20 is not None
        and 1.2 <= vol_ratio_20 <= 1.8
    )
    if vol_breakout_ok:
        context_score += 1
        context_reasons.append("突破時放量但不爆量（量能健康）")

    vol_shrink_ok = False
    if len(df) >= 3:
        last2_vol = df["Volume"].tail(2)
        last2_vol_ma = df["VolMA20"].tail(2)
        last2_close = df["Close"].tail(2)
        last2_ema20 = df["EMA20"].tail(2)
        if (
            (last2_vol < last2_vol_ma).all()
            and (last2_close >= last2_ema20).all()
        ):
            vol_shrink_ok = True
            context_score += 1
            context_reasons.append("突破後量縮且價不破，換手健康")

    rs_ok = False
    if len(df) >= 21:
        ret_stock_20d = (df["Close"].iloc[-1] / df["Close"].iloc[-21]) - 1
        mkt, idx_symbol = load_market_index(effective_symbol)
        if len(mkt) >= 21:
            ret_mkt_20d = (mkt["Close"].iloc[-1] / mkt["Close"].iloc[-21]) - 1
            if ret_stock_20d > ret_mkt_20d:
                rs_ok = True
                context_score += 1
                context_reasons.append(f"相對強勢：近 20 日漲幅優於 {idx_symbol}")

    atr_pct_ok = False
    if atr14_val is not None and not pd.isna(atr14_val) and latest_close > 0:
        atr_pct = atr14_val / latest_close
        if 0.01 <= atr_pct <= 0.05:
            atr_pct_ok = True
            context_score += 1
            context_reasons.append("波動度適中（ATR% 在 1%~5%）")

    higher_low_ok = False
    if len(df) >= 20:
        low_recent = df["Low"].tail(10).min()
        low_prev = df["Low"].shift(10).tail(10).min()
        if not pd.isna(low_recent) and not pd.isna(low_prev) and low_recent > low_prev:
            higher_low_ok = True
            context_score += 1
            context_reasons.append("低點抬高（Higher Low）")

    weekly_status = get_weekly_trend(effective_symbol)
    if weekly_status == "多頭":
        context_score += 1
        context_reasons.append("週線趨勢偏多 (大環境保護小環境)")
    elif weekly_status == "空頭":
        context_score -= 1
        context_reasons.append("週線趨勢偏空 (短線反彈需謹慎)")
    else:
        context_reasons.append("週線趨勢未知")

    if len(ema_df) >= 3 and ema20 is not None:
        last3 = ema_df.tail(3)
        below_ema20 = (last3["Close"] < last3["EMA20"]).all()
        if below_ema20:
            context_score -= 1
            context_reasons.append("股價連續 3 天站不回 EMA20")
    else:
        below_ema20 = False

    prior_high = None
    if "High" in df.columns and len(df) >= 2:
        prior_high = to_scalar(df["High"].shift(1).rolling(20).max().iloc[-1])
        if not pd.isna(prior_high):
            if close_price > prior_high:
                context_score += 1
                context_reasons.append("股價突破前高 (近 20 日高點)")
            else:
                context_reasons.append("股價未突破前高 (近 20 日高點)")

    if ema5 is not None:
        if close_price < ema5:
            context_score -= 1
            context_reasons.append("股價跌破 EMA5")
        if len(ema_df) >= 3:
            last3 = ema_df.tail(3)
            below_ema5 = (last3["Close"] < last3["EMA5"]).all()
            if below_ema5:
                context_score -= 1
                context_reasons.append("股價連續 3 天站不回 EMA5")

    if is_high_volume_sell:
        context_score -= 3
        context_reasons.append("高檔爆大量收黑，可能有出貨壓力")

    is_dangerous_volume = bool(latest.get("Is_Dangerous_Volume", False))
    if is_dangerous_volume:
        context_score -= 2
        context_reasons.append("高檔放量收黑（疑似危險換手）")

    if bias_20_val is not None and not pd.isna(bias_20_val) and bias_20_val > 10:
        context_score -= 2
        context_reasons.append("乖離過熱（Bias20 > 10%）")

    upper_wick_bad = False
    if len(df) >= 2:
        last2 = df.tail(2)
        upper_wick = last2["High"] - last2[["Open", "Close"]].max(axis=1)
        k_range = last2["High"] - last2["Low"]
        wick_ratio = upper_wick / k_range.replace(0, np.nan)
        if (wick_ratio > 0.5).all():
            upper_wick_bad = True
            context_score -= 1
            context_reasons.append("連續 2 天上影線過長")

    if prior_high is not None:
        if latest["High"] > prior_high and close_price < prior_high:
            context_score -= 2
            context_reasons.append("突破前高後收回（可能假突破）")

    if ema20 is not None and vol_ratio_20 is not None:
        if close_price < ema20 and vol_ratio_20 > 1.3:
            context_score -= 2
            context_reasons.append("跌破 EMA20 且放量（趨勢破壞）")

    gap_risk = False
    if len(df) >= 2:
        prev_row = df.iloc[-2]
        if (
            latest["Open"] > prev_row["High"] * 1.01
            and latest["Low"] < prev_row["Close"]
        ):
            gap_risk = True
            context_score -= 1
            context_reasons.append("跳空上漲後回補缺口")

    market_bear = weekly_status == "空頭"
    if market_bear:
        context_score -= 2
        context_reasons.append("市場趨勢偏空，訊號降級")

    if foreign_divergence_warning:
        context_score -= 1
        context_reasons.append("籌碼背離：股價上漲但外資連續賣超")

    recent_foreign_sum = None
    if foreign_net_series is not None and len(foreign_net_series) >= 3:
        recent_foreign_sum = foreign_net_series.tail(3).sum()
        if recent_foreign_sum <= -5000:
            context_score -= 2
            context_reasons.append("外資近 3 天累積賣超超過 5000 張")

    # 合併分數（加權主分數 + 情境加減分）
    score = int(np.clip(weighted_score + context_score, 0, 100))
    reasons = weighted_reasons + context_reasons

    # --- Dashboard 顯示區塊 ---
    with dashboard_placeholder.container():
        st.divider()
        st.header("策略即時監控儀表板")
        if foreign_divergence_warning:
            st.warning("籌碼背離：股價上漲但外資連續賣超，可能有拉高出貨風險，建議分批離場。")

        # 精準診斷（力道天平）：把「成本 vs 籌碼現況」的結論濃縮到儀表板
        try:
            f_used = foreign_net_series
            t_used = trust_net_series
            f3d = (
                f_used.sort_index().tail(3).sum()
                if f_used is not None and len(f_used) >= 3
                else None
            )
            t3d = (
                t_used.sort_index().tail(3).sum()
                if t_used is not None and len(t_used) >= 3
                else None
            )
            foreign_sell_3d = None
            try:
                if f_used is not None and len(f_used) >= 3:
                    last3 = pd.to_numeric(f_used.sort_index().tail(3), errors="coerce").dropna()
                    if len(last3) >= 3:
                        foreign_sell_3d = bool((last3 < 0).all())
            except Exception:
                foreign_sell_3d = None

            risk_beta = risk.get("beta") if isinstance(risk, dict) else None
            vol_sum_3d = compute_volume_sum_3d(df)
            precision = diagnose_precision(
                foreign_3d_net=f3d,
                trust_3d_net=t3d,
                vol_sum_3d_lot=vol_sum_3d,
                bias20=bias_20_val,
                beta=risk_beta,
                foreign_sell_3d=foreign_sell_3d,
            )

            if precision.level == "danger":
                st.error(precision.as_one_liner())
            elif precision.level == "warning":
                st.warning(precision.as_one_liner())
            else:
                st.success(precision.as_one_liner())

            with st.expander("查看精準診斷細節（土洋對峙/參與度/派發）", expanded=False):
                if precision.bullets:
                    st.markdown("\n".join([f"- {x}" for x in precision.bullets]))
                else:
                    st.caption("（目前沒有額外細節）")
        except Exception:
            pass

        with st.expander("🧾 證據層：風險 / 目標價（可收起）", expanded=False):
            render_risk_metrics_panel(risk)
            render_target_range_panel(df, effective_symbol)

        with st.expander("📊 機率預測模組（GBM / 斐波那契 / 波動率區間）", expanded=False):
            render_price_prediction_panel(
                df,
                symbol=effective_symbol,
                current_price=close_price,
            )

        bias_sma20 = bias_20_val if bias_20_val is not None and not pd.isna(bias_20_val) else None
        bias_ema20 = None
        if (
            ema20 is not None
            and not pd.isna(ema20)
            and ema20 != 0
            and close_price is not None
            and not pd.isna(close_price)
        ):
            bias_ema20 = (close_price / ema20 - 1) * 100

        with st.expander("📚 幫助說明：指標分類總覽（可收起）", expanded=False):
            st.subheader("指標分類總覽")
            st.markdown(
                """
**籌碼指標（法人/大戶流向）**
- 外資買賣超：`foreign_net_series` / `foreign_net_latest`（FinMind）
- 投信買賣超：`trust_net_series`（FinMind）
- 土洋鎖碼：`compute_lock_thresholds(vol_sum_3d)`（以成交量推門檻）
- 籌碼背離：`foreign_divergence_warning`（股價上漲但外資連賣）

**技術指標（價格/成交量推導）**
- 趨勢：SMA20/SMA60、EMA5/EMA20、週線趨勢（週K EMA10）
- 動能：RSI14
- 波動：ATR14
- 乖離：SMA20/EMA20
- 量能：VolMA20、5日均量、Volume
- K線結構：上/下影線、實體比例、`Is_Dangerous_Volume`
- 訊號模組：Gate / Trigger / Guard / BUY_SIGNAL

**風險指標（投資人風控）**
- Beta（vs 大盤）
- 年化波動率
- Sharpe
- Max Drawdown
                """
            )

        render_score_overview(score, bias_sma20)
        st.caption("AI 分數 = 趨勢(40) + 量能(30) + 籌碼(20) + 乖離(10) ± 籌碼加分/危險扣分")
        with st.expander("🔎 詳細：Gate / Trigger / Guard / Debug（可收起）", expanded=False):
            render_gate_trigger_guard_reminders(
                risk,
                foreign_divergence_warning,
                atr14_val,
                close_price,
                prev_close,
                strategy_gate=bool(latest.get("BUY_GATE", False)),
            )

            if show_chip_unit_check and (effective_symbol.endswith(".TW") or effective_symbol.endswith(".TWO")):
                st.subheader("籌碼單位檢查（快速判定）")
                st.caption("用量級判斷是否為「股」或「張」，避免門檻與成交量單位不一致。")
                st.write("foreign_net_unit:", detect_net_unit_tag(foreign_net_series))
                st.write("trust_net_unit:", detect_net_unit_tag(trust_net_series))
                st.write(
                    "foreign_net_last5_raw:",
                    foreign_net_series.sort_index().tail(5)
                    if foreign_net_series is not None
                    else None,
                )
                st.write(
                    "foreign_net_3d_used:",
                    foreign_net_series.sort_index().tail(3)
                    if foreign_net_series is not None
                    else None,
                )
                st.write(
                    "trust_net_last5_raw:",
                    trust_net_series.sort_index().tail(5)
                    if trust_net_series is not None
                    else None,
                )
                st.write(
                    "trust_net_3d_used:",
                    trust_net_series.sort_index().tail(3)
                    if trust_net_series is not None
                    else None,
                )
                df_inst_debug = fetch_institutional_raw(effective_symbol)
                if df_inst_debug is not None and not df_inst_debug.empty:
                    st.write(
                        df_inst_debug.tail(5)[["date", "name", "buy", "sell"]]
                    )

        foreign_3d_net = (
            foreign_net_series.sort_index().tail(3).sum()
            if foreign_net_series is not None and len(foreign_net_series) >= 3
            else None
        )
        trust_3d_net = (
            trust_net_series.sort_index().tail(3).sum()
            if trust_net_series is not None and len(trust_net_series) >= 3
            else None
        )
        risk_vol_annual = risk.get("vol_annual") if isinstance(risk, dict) else None
        risk_beta = risk.get("beta") if isinstance(risk, dict) else None
        risk_sharpe = risk.get("sharpe") if isinstance(risk, dict) else None
        risk_mdd = risk.get("max_drawdown") if isinstance(risk, dict) else None

        metrics = build_metrics_snapshot(
            Close=close_price,
            SMA20=sma20_val,
            SMA60=sma60_val,
            EMA5=ema5,
            EMA20=ema20,
            RSI14=to_scalar(latest.get("RSI14", np.nan)),
            ATR14=atr14_val,
            bias_sma20_pct=bias_sma20,
            bias_ema20_pct=bias_ema20,
            Volume=latest_volume,
            VolMA20=vol_ma20_val,
            Vol_Avg_5=latest_vol_avg_5,
            VWAP=to_scalar(latest.get("VWAP", np.nan)),
            AVWAP=to_scalar(latest.get("AVWAP", np.nan)),
            Is_Dangerous_Volume=bool(latest.get("Is_Dangerous_Volume", False)),
            foreign_net_latest=foreign_net_latest,
            foreign_3d_net=foreign_3d_net,
            trust_net_latest=trust_net_series.iloc[-1] if trust_net_series is not None and not trust_net_series.empty else None,
            trust_3d_net=trust_3d_net,
            vol_sum_3d=compute_volume_sum_3d(df),
            beta=risk_beta,
            volatility_annual=risk_vol_annual,
            sharpe=risk_sharpe,
            max_drawdown=risk_mdd,
            market_trend=weekly_status,
            rs_20d_vs_market=rs_ok,
            chip_divergence=foreign_divergence_warning,
        )

        used_map = {"Gate": [], "Trigger": [], "Guard": [], "Chip Notes": []}
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
            if (prev_close is not None and bias_20_val is not None and not pd.isna(bias_20_val))
            else False
        )
        hhv20_prev = to_scalar(df["High"].rolling(20, min_periods=20).max().shift(1).iloc[-1]) if len(df) >= 21 else np.nan
        trigger_breakout_pass = bool(
            trigger_type == "BREAKOUT"
            and (close_price is not None and not pd.isna(hhv20_prev) and close_price > hhv20_prev)
            and (latest_volume is not None and vol_ma20_val is not None and not pd.isna(vol_ma20_val) and latest_volume > 1.2 * vol_ma20_val)
        )
        hhv5_prev = to_scalar(df["High"].rolling(5, min_periods=5).max().shift(1).iloc[-1]) if len(df) >= 6 else np.nan
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
                if (bias_20_val is not None and prev_close is not None and close_price is not None and not pd.isna(bias_20_val))
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
                if (close_price is not None and not pd.isna(hhv20_prev) and latest_volume is not None and vol_ma20_val is not None and not pd.isna(vol_ma20_val))
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
                if (bias_20_val is not None and close_price is not None and not pd.isna(hhv5_prev) and not pd.isna(bias_20_val))
                else None,
                threshold="(3<=Bias20<=10) & (Close>HHV5.shift(1))",
                passed=trigger_cont_pass,
            )
        )

        k_range_latest = float((latest["High"] - latest["Low"])) if ("High" in latest and "Low" in latest and (latest["High"] - latest["Low"]) != 0) else np.nan
        close_pos_latest = float((latest["Close"] - latest["Low"]) / k_range_latest) if (not pd.isna(k_range_latest)) else np.nan
        vol_ratio_20_latest = float(latest_volume / vol_ma20_val) if (latest_volume is not None and vol_ma20_val is not None and not pd.isna(vol_ma20_val) and vol_ma20_val != 0) else np.nan
        avwap_val = to_scalar(latest.get("AVWAP", np.nan))

        breakout_close_strong_pass = bool(close_pos_latest >= 0.6) if not pd.isna(close_pos_latest) else False
        not_crazy_volume_pass = bool(vol_ratio_20_latest <= 2.0) if not pd.isna(vol_ratio_20_latest) else False
        not_too_hot_pass = bool(bias_20_val <= 9.5) if (bias_20_val is not None and not pd.isna(bias_20_val)) else False
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

        _safe_symbol = "".join([c if c.isalnum() else "_" for c in (effective_symbol or "default")])
        _has_pos = float(st.session_state.get(f"pos_avg_cost_{_safe_symbol}", 0) or 0) > 0
        render_trade_brief(used_map, metrics, has_position=_has_pos)

        render_indicator_panels(metrics=metrics, used_map=used_map, taxonomy=INDICATOR_TAXONOMY)

        with st.expander("🔎 詳細：指標達成 / 教學 / 風險控管（平時可收起）", expanded=False):
            st.subheader("指標達成詳細狀況")
            left_col, right_col = st.columns(2)

            with left_col:
                st.write("加分項目 (多頭確認)")
                st.markdown("**趨勢強度**")
                st.checkbox(
                    "股價在 EMA20 之上且走升",
                    value=bool(ema20_up and ema20 is not None and close_price > ema20),
                    disabled=True,
                )
                st.checkbox(
                    "SMA20 高於 SMA60",
                    value=bool(trend_mid_ok),
                    disabled=True,
                )
                st.checkbox(
                    "SMA60 走升",
                    value=bool(sma60_up),
                    disabled=True,
                )
                st.checkbox(
                    "EMA5 > EMA20 (短線動能)",
                    value=bool(ema5 is not None and ema20 is not None and ema5 > ema20),
                    disabled=True,
                )
                st.markdown("**量能結構**")
                st.checkbox(
                    "突破時量能健康",
                    value=bool(vol_breakout_ok),
                    disabled=True,
                )
                st.checkbox(
                    "突破後量縮且價不破",
                    value=bool(vol_shrink_ok),
                    disabled=True,
                )
                st.markdown("**結構**")
                st.checkbox(
                    "突破近 20 日高點",
                    value=bool(prior_high is not None and close_price > prior_high),
                    disabled=True,
                )
                st.checkbox(
                    "低點抬高 (Higher Low)",
                    value=bool(higher_low_ok),
                    disabled=True,
                )
                st.markdown("**籌碼**")
                st.checkbox(
                    "外資今日買超",
                    value=bool(foreign_net_latest is not None and foreign_net_latest > 0),
                    disabled=True,
                )
                st.markdown("**動能 / 波動**")
                st.checkbox(
                    "RSI >= 50",
                    value=bool(
                        not pd.isna(df["RSI14"].iloc[-1]) and to_scalar(df["RSI14"].iloc[-1]) >= 50
                    ),
                    disabled=True,
                )
                st.checkbox(
                    "相對強勢（勝過大盤）",
                    value=bool(rs_ok),
                    disabled=True,
                )
                st.checkbox(
                    "波動度適中 (ATR%)",
                    value=bool(atr_pct_ok),
                    disabled=True,
                )

            with right_col:
                st.write("扣分項目 (風險警示)")
                st.markdown("**追高過熱**")
                st.checkbox(
                    "SMA20乖離過熱 (Bias20 > 10%)",
                    value=bool(
                        bias_20_val is not None and not pd.isna(bias_20_val) and bias_20_val > 10
                    ),
                    disabled=True,
                )
                st.checkbox(
                    "股價跌破 EMA20",
                    value=bool(ema20 is not None and close_price < ema20),
                    disabled=True,
                )
                st.checkbox(
                    "連續 3 天站在 EMA20 之下",
                    value=bool(below_ema20),
                    disabled=True,
                )
                st.markdown("**假突破與回落**")
                st.checkbox(
                    "上影線過長（連 2 日）",
                    value=bool(upper_wick_bad),
                    disabled=True,
                )
                st.checkbox(
                    "假突破回落",
                    value=bool(
                        prior_high is not None
                        and latest["High"] > prior_high
                        and close_price < prior_high
                    ),
                    disabled=True,
                )
                st.checkbox(
                    "跌破 EMA20 且放量",
                    value=bool(
                        ema20 is not None
                        and vol_ratio_20 is not None
                        and close_price < ema20
                        and vol_ratio_20 > 1.3
                    ),
                    disabled=True,
                )
                st.markdown("**缺口風險**")
                st.checkbox(
                    "外資近 3 天大賣 (>5000張)",
                    value=bool(recent_foreign_sum is not None and recent_foreign_sum <= -5000),
                    disabled=True,
                )
                st.checkbox(
                    "高檔放量收黑（危險換手）",
                    value=bool(is_dangerous_volume),
                    disabled=True,
                )
                st.checkbox(
                    "跳空缺口回補",
                    value=bool(gap_risk),
                    disabled=True,
                )
                st.markdown("**市場 / 籌碼風險**")
                st.checkbox(
                    "市場趨勢偏空",
                    value=bool(market_bear),
                    disabled=True,
                )
                st.checkbox(
                    "籌碼背離",
                    value=bool(foreign_divergence_warning),
                    disabled=True,
                )

            with st.expander("點擊查看指標定義與教學"):
                st.markdown(
                    """
| 指標名稱 | 判斷邏輯 | 策略意義 |
| :--- | :--- | :--- |
| **EMA20 趨勢** | 股價與月線的關係 | 站上代表中期趨勢偏多，跌破代表走弱。 |
| **SMA20乖離率** | 股價離SMA20的距離 | >10% 代表過熱易回檔；<-10% 代表超跌易反彈。 |
| **高檔爆大量** | 股價高位且量增 2 倍 | 通常是大戶倒貨給散戶的訊號。 |
| **外資籌碼** | 近 3 日累積買賣 | 追蹤法人大戶是否與股價走勢一致（避開背離）。 |
                    """
                )

            # 風險控管：動態出場參考
            atr_val = to_scalar(df["ATR14"].iloc[-1]) if "ATR14" in df.columns else np.nan
            if not pd.isna(atr_val):
                stop_loss_price = close_price - (2 * atr_val)
                take_profit_price = close_price + (3 * atr_val)
                rr_denominator = close_price - stop_loss_price
                risk_to_reward_ratio = (
                    (take_profit_price - close_price) / rr_denominator if rr_denominator != 0 else 0
                )

                st.divider()
                st.subheader("風險控管：動態出場參考")
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric(
                        "建議停損點 (ATR-2)",
                        f"{stop_loss_price:.2f}",
                        delta=f"{(stop_loss_price / close_price - 1) * 100:.1f}%",
                        delta_color="normal",
                    )
                    st.caption("跌破此位代表趨勢慣性徹底改變")
                with r2:
                    st.metric(
                        "建議停利點 (ATR+3)",
                        f"{take_profit_price:.2f}",
                        delta=f"{(take_profit_price / close_price - 1) * 100:.1f}%",
                        delta_color="normal",
                    )
                    st.caption("短線獲利目標空間")
                with r3:
                    st.metric("盈虧比 (R/R Ratio)", f"{risk_to_reward_ratio:.2f}")
                    if risk_to_reward_ratio > 2:
                        st.success("適合進場：潛在報酬大於風險")
                    else:
                        st.info("觀望為宜：目前風險回報比普通")

    with expert_placeholder.container():
        # 💼 持倉與下車指南設定（輸入均價）：放在圖表設定上方
        _safe_symbol = effective_symbol or "default"
        _safe_symbol = "".join([c if c.isalnum() else "_" for c in _safe_symbol])
        _key_avg = f"pos_avg_cost_{_safe_symbol}"
        _key_qty = f"pos_qty_{_safe_symbol}"
        _key_style = f"pos_exit_style_{_safe_symbol}"
        with st.expander("💼 持倉與下車指南設定（輸入均價）", expanded=True):
            st.caption("這一區改成「套用才更新」：你調數字不會立刻重跑整頁。")
            with st.form(key=f"pos_settings_form_{_safe_symbol}"):
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    pos_avg_cost = st.number_input(
                        "持股均價（計算損益用）",
                        min_value=0.0,
                        value=float(st.session_state.get(_key_avg, 0.0) or 0.0),
                        step=0.1,
                        key=_key_avg,
                        format="%.2f",
                    )
                with col_p2:
                    pos_qty = st.number_input(
                        "持有股數",
                        min_value=0,
                        value=int(st.session_state.get(_key_qty, 0) or 0),
                        step=1,
                        key=_key_qty,
                    )
                with col_p3:
                    style_options = ["波段守五日線", "積極分批止盈", "長線守月線"]
                    default_style = str(
                        st.session_state.get(_key_style, style_options[0]) or style_options[0]
                    )
                    style_index = (
                        style_options.index(default_style) if default_style in style_options else 0
                    )
                    pos_exit_style = st.selectbox(
                        "偏好下車風格",
                        style_options,
                        index=style_index,
                        key=_key_style,
                    )
                applied_pos = st.form_submit_button("✅ 套用設定（更新下車指令）")
            if applied_pos:
                st.success("已套用持倉設定（下車指令已更新）")

        st.caption(
            "📓 **交易日誌（建倉／結案）** 已移到左側「持股清單」頁統一操作。"
        )

        # 🏁 下車指南（白話文決策）：移到「圖表設定」上方
        # 先算 3 日籌碼（供 TURN 引擎使用）
        chip_foreign_3d_net = (
            float(foreign_net_series.dropna().tail(3).sum())
            if foreign_net_series is not None and foreign_net_series.dropna().shape[0] >= 3
            else None
        )
        chip_trust_3d_net = (
            float(trust_net_series.dropna().tail(3).sum())
            if trust_net_series is not None and trust_net_series.dropna().shape[0] >= 3
            else None
        )

        with st.expander("🏁 下車指南（白話文決策）", expanded=True):
            try:
                df_turn_advice = df.copy()
                if "RSI" not in df_turn_advice.columns and "RSI14" in df_turn_advice.columns:
                    df_turn_advice["RSI"] = df_turn_advice["RSI14"]
                if foreign_net_series is not None and not foreign_net_series.empty:
                    df_turn_advice["Foreign_Net"] = foreign_net_series.reindex(df_turn_advice.index)
                if trust_net_series is not None and not trust_net_series.empty:
                    df_turn_advice["Trust_Net"] = trust_net_series.reindex(df_turn_advice.index)

                top_now = run_turn_check(
                    df_turn_advice,
                    mode="top",
                    cfg=turn_cfg_runtime,
                    foreign_3d_net=chip_foreign_3d_net,
                    trust_3d_net=chip_trust_3d_net,
                )
                bottom_now = run_turn_check(
                    df_turn_advice,
                    mode="bottom",
                    cfg=turn_cfg_runtime,
                    foreign_3d_net=chip_foreign_3d_net,
                    trust_3d_net=chip_trust_3d_net,
                )
            except Exception:
                df_turn_advice = df
                top_now = None
                bottom_now = None

            # 持倉建議（供去留診斷與持倉區塊一致）
            _defense = ema20 if str(pos_exit_style) == "長線守月線" else ema5
            pos_advice = get_position_advice(
                current_price=close_price,
                avg_cost=float(pos_avg_cost or 0.0),
                qty=int(pos_qty or 0),
                ema_defense=_defense,
                bottom_result=bottom_now,
                top_result=top_now,
            )

            _trail = float(turn_bt_trailing_stop or 0.0)
            render_position_advice_panel(
                df_turn_advice,
                symbol=effective_symbol,
                avg_cost=float(pos_avg_cost or 0.0),
                qty=int(pos_qty or 0),
                exit_style=str(pos_exit_style),
                bottom_result=bottom_now,
                top_result=top_now,
                trailing_stop_pct=_trail if _trail > 0 else 10.0,
                foreign_net_series=foreign_net_series,
                trust_net_series=trust_net_series,
            )

        st.markdown("### 圖表設定")
        turn_cfg = turn_cfg_runtime
        turn_mode_default = turn_cfg.get("mode_default", "bottom")
        show_turn_both = st.checkbox("TURN 同時顯示 bottom + top（不需切換）", value=True)
        if show_turn_both:
            turn_mode = "both"
            st.caption("顯示：下半部=bottom（較佳進場），上半部=top（較佳出場／風險）")
        else:
            turn_mode = st.selectbox(
                "TURN 模式（K 線變色）",
                ["bottom", "top"],
                index=0 if turn_mode_default == "bottom" else 1,
            )
            st.caption("標註：bottom=較佳進場、top=較佳出場（依 TURN 狀態）")
        st.caption(
            "建議：短線/波段 40–60（約 2–3 個月日線）、中期 80–120（約 4–6 個月）。"
            "若作為反轉前風險濾網，60 最平衡；想快反應用 40，想穩一點用 80。"
        )
        turn_window = st.slider("TURN 變色區段長度", 20, 120, 60, 5)
        st.caption("⭐ 進場、⚡ 出場：狀態切換＋移動止損 TS（出場＝降級或 TS 先到）")

        # K 線 + KD（整合圖，放在圖表設定下方）

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.72, 0.28],
        )
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="K線",
                increasing_line_color="red",
                increasing_fillcolor="red",
                decreasing_line_color="green",
                decreasing_fillcolor="green",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA20"],
                line=dict(color="purple", width=2),
                name="EMA20",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA5"],
                line=dict(color="yellow", width=1),
                name="EMA5",
            ),
            row=1,
            col=1,
        )

        # TURN CHECK 變色區段（向量化：一次算完整段）
        df_turn_series = df.copy()
        if "RSI" not in df_turn_series.columns and "RSI14" in df_turn_series.columns:
            df_turn_series["RSI"] = df_turn_series["RSI14"]
        # 讓 TURN CHECK 可用「籌碼連續性」判斷（若有資料）
        if foreign_net_series is not None and not foreign_net_series.empty:
            df_turn_series["Foreign_Net"] = foreign_net_series.reindex(df_turn_series.index)
        if trust_net_series is not None and not trust_net_series.empty:
            df_turn_series["Trust_Net"] = trust_net_series.reindex(df_turn_series.index)

        # 側邊欄：即時回測（使用目前覆寫後的參數）
        if page == "主控台":
            try:
                bt_df = backtest_turn_signals(
                    df_turn_series,
                    mode=turn_bt_mode,
                    cfg=turn_cfg,
                    foreign_3d_net=chip_foreign_3d_net,
                    trust_3d_net=chip_trust_3d_net,
                    hold_days=int(turn_bt_hold_days),
                    trailing_stop_pct=float(turn_bt_trailing_stop or 0.0),
                    exit_ma_window=int(turn_bt_exit_ma_window or 0),
                )
            except Exception:
                bt_df = None
            if isinstance(bt_df, pd.DataFrame) and not bt_df.empty:
                # 結案勝率：以「結案報酬（final_pct；含狀態降級出場）」判定
                final_col = "final_pct" if "final_pct" in bt_df.columns else None
                if final_col is None and "return_pct" in bt_df.columns:
                    final_col = "return_pct"
                final_series = bt_df[final_col] if final_col else pd.Series(dtype=float)
                win_rate = (
                    (final_series >= float(turn_bt_win_threshold)).mean()
                    if isinstance(final_series, pd.Series) and not final_series.empty
                    else np.nan
                )
                avg_fav = bt_df["max_favorable_pct"].mean()
                avg_adv = bt_df["max_adverse_pct"].mean()
                avg_final = (
                    final_series.mean()
                    if isinstance(final_series, pd.Series) and not final_series.empty
                    else np.nan
                )
                avg_hold = (
                    bt_df["holding_days"].mean()
                    if "holding_days" in bt_df.columns and not bt_df["holding_days"].empty
                    else np.nan
                )
                avg_giveback = (
                    bt_df["giveback_ratio"].mean()
                    if "giveback_ratio" in bt_df.columns and not bt_df["giveback_ratio"].empty
                    else np.nan
                )
                rr = np.nan
                try:
                    if pd.notna(avg_adv) and float(avg_adv) != 0:
                        rr = abs(float(avg_fav) / float(avg_adv))
                except Exception:
                    rr = np.nan

                # Expectancy（真實）：平均結案報酬（%/次）
                expectancy = float(avg_final) if pd.notna(avg_final) else np.nan

                if pd.notna(win_rate):
                    st.sidebar.metric(
                        f"結案成功率（≥{float(turn_bt_win_threshold):.1f}%）",
                        f"{win_rate:.1%}",
                    )
                if pd.notna(avg_final):
                    st.sidebar.metric(
                        f"平均結案報酬（最多{int(turn_bt_hold_days)}天）",
                        f"{avg_final:.2f}%",
                    )
                st.sidebar.metric("平均最大有利幅度", f"{avg_fav:.2f}%")
                st.sidebar.metric("平均最大不利幅度", f"{avg_adv:.2f}%")
                if pd.notna(avg_hold):
                    st.sidebar.metric("平均持有天數", f"{avg_hold:.1f} 天")
                if pd.notna(avg_giveback):
                    st.sidebar.metric("平均獲利回吐比", f"{(float(avg_giveback) * 100.0):.0f}%")
                if pd.notna(rr):
                    st.sidebar.metric("風報酬比（|有利/不利|）", f"{rr:.2f}")
                if pd.notna(expectancy):
                    st.sidebar.info(f"參數期望值：{expectancy:.2f}% / 次（平均結案報酬）")
                    if expectancy > 2.0:
                        st.sidebar.success("目前參數組合具有良好正期望值")
                    elif expectancy < 0:
                        st.sidebar.error("目前參數組合長期可能虧損")
                st.sidebar.write(f"總共抓到 {len(bt_df)} 次轉折")
                with st.sidebar.expander("回測明細（最近 10 筆）", expanded=False):
                    st.dataframe(bt_df.tail(10), width="stretch")
            else:
                st.sidebar.warning("當前參數下沒有轉折訊號（可試著放寬門檻）")

            # 自動參數優化（Grid Search）
            if run_turn_grid_search:
                with st.spinner("正在掃描最佳參數組合..."):
                    gs_df = grid_search_optimization(
                        df_turn_series,
                        mode=turn_bt_mode,
                        base_cfg=turn_cfg,
                        foreign_3d_net=chip_foreign_3d_net,
                        trust_3d_net=chip_trust_3d_net,
                        hold_days=int(turn_bt_hold_days),
                        trailing_stop_pct=float(turn_bt_trailing_stop or 0.0),
                        exit_ma_window=int(turn_bt_exit_ma_window or 0),
                        win_threshold=float(turn_bt_win_threshold),
                        score_thresholds=[int(x) for x in turn_grid_scores],
                        dry_up_ratios=[float(x) for x in turn_grid_dry_up_ratios],
                        min_signals=int(turn_grid_min_signals),
                    )
                st.session_state["turn_grid_search_result"] = gs_df
                st.session_state["turn_grid_search_meta"] = {
                    "symbol": effective_symbol,
                    "time_range": time_range,
                    "mode": turn_bt_mode,
                    "hold_days": int(turn_bt_hold_days),
                    "win_threshold": float(turn_bt_win_threshold),
                }

            gs_df = st.session_state.get("turn_grid_search_result")
            if isinstance(gs_df, pd.DataFrame) and not gs_df.empty:
                # 依使用者選擇排序
                sort_col = turn_grid_sort_by if turn_grid_sort_by in gs_df.columns else "expectancy"
                if sort_col != "expectancy":
                    gs_df = gs_df.sort_values(
                        by=[sort_col, "expectancy", "count"],
                        ascending=[False, False, False],
                        na_position="last",
                    )

                with st.expander("🚀 TURN 參數自動優化結果（Grid Search）", expanded=False):
                    st.caption("提示：樣本數太少時（count 很小）請勿過度相信結果。")
                    st.caption("本頁 win_rate 以「結案報酬（含狀態降級出場）」是否達門檻計算；expectancy 為平均結案報酬（%/次）。")
                    st.dataframe(gs_df.head(15), width="stretch", hide_index=True)

                    # 熱力圖（Expectancy）
                    try:
                        pivot = gs_df.pivot(
                            index="score_threshold",
                            columns="dry_up_ratio",
                            values="expectancy",
                        )
                        fig_hm = go.Figure(
                            data=go.Heatmap(
                                z=pivot.values,
                                x=[str(x) for x in pivot.columns],
                                y=[str(y) for y in pivot.index],
                                colorscale="RdYlGn",
                                colorbar=dict(title="Expectancy"),
                                hoverongaps=False,
                            )
                        )
                        fig_hm.update_layout(
                            height=360,
                            margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="dry_up_ratio",
                            yaxis_title="score_threshold",
                        )
                        st.plotly_chart(fig_hm, width="stretch")
                    except Exception:
                        st.info("熱力圖資料不足或格式不一致，已略過。")

        recent = df_turn_series.tail(turn_window)
        colors = {
            "ALLOW": "rgba(0, 200, 0, 0.12)",
            "WATCH": "rgba(255, 200, 0, 0.12)",
            "BLOCK": "rgba(255, 0, 0, 0.12)",
        }

        if turn_mode == "both":
            # 下半部：bottom；上半部：top
            details_all_bottom = get_all_turn_details(
                df_turn_series,
                mode="bottom",
                cfg=turn_cfg,
                foreign_3d_net=chip_foreign_3d_net,
                trust_3d_net=chip_trust_3d_net,
            )
            details_recent_bottom = (
                details_all_bottom.tail(turn_window)
                if isinstance(details_all_bottom, pd.DataFrame)
                else pd.DataFrame()
            )
            status_recent_bottom = (
                details_recent_bottom["status"]
                if isinstance(details_recent_bottom, pd.DataFrame)
                else pd.Series(dtype=object)
            )
            details_all_top = get_all_turn_details(
                df_turn_series,
                mode="top",
                cfg=turn_cfg,
                foreign_3d_net=chip_foreign_3d_net,
                trust_3d_net=chip_trust_3d_net,
            )
            details_recent_top = (
                details_all_top.tail(turn_window)
                if isinstance(details_all_top, pd.DataFrame)
                else pd.DataFrame()
            )
            status_recent_top = (
                details_recent_top["status"]
                if isinstance(details_recent_top, pd.DataFrame)
                else pd.Series(dtype=object)
            )

            if not status_recent_bottom.empty:
                seg_id = (status_recent_bottom != status_recent_bottom.shift(1)).cumsum()
                for _, seg in status_recent_bottom.groupby(seg_id):
                    status = seg.iloc[0]
                    color = colors.get(status, "rgba(200, 200, 200, 0.08)")
                    fig.add_vrect(
                        x0=seg.index[0],
                        x1=seg.index[-1],
                        y0=0.00,
                        y1=0.48,
                        fillcolor=color,
                        opacity=0.35,
                        line_width=0,
                        layer="below",
                        row=1,
                        col=1,
                    )

            if not status_recent_top.empty:
                seg_id = (status_recent_top != status_recent_top.shift(1)).cumsum()
                for _, seg in status_recent_top.groupby(seg_id):
                    status = seg.iloc[0]
                    color = colors.get(status, "rgba(200, 200, 200, 0.08)")
                    fig.add_vrect(
                        x0=seg.index[0],
                        x1=seg.index[-1],
                        y0=0.52,
                        y1=1.00,
                        fillcolor=color,
                        opacity=0.35,
                        line_width=0,
                        layer="below",
                        row=1,
                        col=1,
                    )

            # ⭐/⚡ bottom：進場/出場（出場＝ALLOW 降級）
            if len(status_recent_bottom) >= 2:
                best_status = "ALLOW"
                entry_mask = (status_recent_bottom == best_status) & (
                    status_recent_bottom.shift(1) != best_status
                )
                exit_mask = (status_recent_bottom != best_status) & (
                    status_recent_bottom.shift(1) == best_status
                )
                entry_mask.iloc[0] = False
                exit_mask.iloc[0] = False

                entry_pos_arr = np.where(entry_mask.fillna(False).values)[0]
                exit_pos_arr = np.where(exit_mask.fillna(False).values)[0]

                entry_pos = None
                exit_pos = None
                fallback_entry = False

                last_entry_pos = int(entry_pos_arr[-1]) if entry_pos_arr.size > 0 else None
                last_exit_pos = int(exit_pos_arr[-1]) if exit_pos_arr.size > 0 else None

                if last_entry_pos is not None and (
                    last_exit_pos is None or last_entry_pos > last_exit_pos
                ):
                    # 目前仍在最佳狀態：只標最後一次進場
                    entry_pos = last_entry_pos
                elif last_exit_pos is not None:
                    # 目前非最佳狀態：標最後一次出場，並補上該次出場前最近的進場
                    exit_pos = last_exit_pos
                    if entry_pos_arr.size > 0:
                        entry_before = entry_pos_arr[entry_pos_arr < exit_pos]
                        if entry_before.size > 0:
                            entry_pos = int(entry_before[-1])

                # fallback：區間內沒有進場切換，但目前為 ALLOW
                if (
                    entry_pos is None
                    and len(status_recent_bottom) > 0
                    and str(status_recent_bottom.iloc[-1]) == best_status
                ):
                    entry_pos = len(status_recent_bottom) - 1
                    fallback_entry = True

                if entry_pos is not None:
                    entry_idx = status_recent_bottom.index[int(entry_pos)]
                    new_s = str(status_recent_bottom.iloc[int(entry_pos)])
                    old_s = (
                        str(status_recent_bottom.iloc[int(entry_pos) - 1])
                        if int(entry_pos) - 1 >= 0
                        else ""
                    )
                    label = (
                        "⭐bottom 持有中（ALLOW）"
                        if fallback_entry
                        else (f"⭐bottom 進場 {old_s}→{new_s}" if old_s else "⭐bottom 進場（ALLOW）")
                    )
                    text = label
                    if (
                        isinstance(details_recent_bottom, pd.DataFrame)
                        and not details_recent_bottom.empty
                        and entry_idx in details_recent_bottom.index
                    ):
                        drow = details_recent_bottom.loc[entry_idx]
                        hits = [
                            f"結構{'✅' if bool(drow.get('structure')) else '❌'}",
                            f"動能{'✅' if bool(drow.get('momentum')) else '❌'}",
                            f"量能{'✅' if bool(drow.get('volume')) else '❌'}",
                            f"籌碼{'✅' if bool(drow.get('chip')) else '❌'}",
                        ]
                        try:
                            sc = int(drow.get("score"))
                        except Exception:
                            sc = None
                        if sc is not None:
                            text = f"{label}<br>Score {sc}/4 | {' '.join(hits)}"
                    if entry_idx in recent.index:
                        fig.add_annotation(
                            x=entry_idx,
                            y=float(recent.loc[entry_idx, "Low"]),
                            text=text,
                            showarrow=True,
                            arrowhead=2,
                            ax=20,
                            ay=-30,
                            arrowcolor="rgba(0, 150, 0, 0.85)",
                            bgcolor="rgba(210, 255, 210, 0.85)",
                            bordercolor="rgba(0, 150, 0, 0.85)",
                            row=1,
                            col=1,
                        )

                # 追加：移動止損（TS）出場（若比狀態降級更早）
                exit_is_ts = False
                ts_stop = None
                ts_tag = None
                trail_pct = float(turn_bt_trailing_stop or 0.0)
                if entry_pos is not None and (not fallback_entry) and trail_pct > 0:
                    try:
                        trail = float(trail_pct) / 100.0
                    except Exception:
                        trail = 0.0
                    if trail > 0:
                        try:
                            peak = float(
                                pd.to_numeric(
                                    df_turn_series.loc[entry_idx, "High"], errors="coerce"
                                )
                            )
                        except Exception:
                            peak = np.nan
                        if not np.isfinite(peak) or peak <= 0:
                            try:
                                peak = float(
                                    pd.to_numeric(
                                        df_turn_series.loc[entry_idx, "Close"],
                                        errors="coerce",
                                    )
                                )
                            except Exception:
                                peak = np.nan

                        # 只允許「更早」的 TS 覆蓋（避免同一天重複定義）
                        search_end = (
                            int(exit_pos) - 1
                            if exit_pos is not None and int(exit_pos) > 0
                            else (len(status_recent_bottom) - 1)
                        )
                        start_j = int(entry_pos) + 1
                        if (
                            search_end >= start_j
                            and np.isfinite(peak)
                            and float(peak) > 0
                        ):
                            for j in range(start_j, int(search_end) + 1):
                                idx_j = status_recent_bottom.index[int(j)]
                                stop = float(peak) * (1.0 - float(trail))
                                try:
                                    o = float(
                                        pd.to_numeric(
                                            df_turn_series.loc[idx_j, "Open"],
                                            errors="coerce",
                                        )
                                    )
                                    h = float(
                                        pd.to_numeric(
                                            df_turn_series.loc[idx_j, "High"],
                                            errors="coerce",
                                        )
                                    )
                                    l = float(
                                        pd.to_numeric(
                                            df_turn_series.loc[idx_j, "Low"],
                                            errors="coerce",
                                        )
                                    )
                                except Exception:
                                    o, h, l = np.nan, np.nan, np.nan

                                if np.isfinite(o) and float(o) <= float(stop):
                                    exit_pos = int(j)
                                    exit_is_ts = True
                                    ts_stop = float(stop)
                                    ts_tag = "TS_GAP"
                                    break
                                if np.isfinite(l) and float(l) <= float(stop):
                                    exit_pos = int(j)
                                    exit_is_ts = True
                                    ts_stop = float(stop)
                                    ts_tag = "TS"
                                    break
                                if np.isfinite(h):
                                    peak = max(float(peak), float(h))

                if exit_pos is not None:
                    exit_idx = status_recent_bottom.index[int(exit_pos)]
                    if exit_is_ts:
                        label = f"⚡bottom {ts_tag or 'TS'} 出場（回檔{trail_pct:.1f}%）"
                        if ts_stop is not None and np.isfinite(ts_stop):
                            label = f"{label}<br>stop≈{float(ts_stop):.2f}"
                        text = label
                    else:
                        new_s = str(status_recent_bottom.iloc[int(exit_pos)])
                        old_s = (
                            str(status_recent_bottom.iloc[int(exit_pos) - 1])
                            if int(exit_pos) - 1 >= 0
                            else ""
                        )
                        label = (
                            f"⚡bottom 出場 {old_s}→{new_s}"
                            if old_s
                            else f"⚡bottom 出場（→{new_s}）"
                        )
                        text = label
                    if (
                        isinstance(details_recent_bottom, pd.DataFrame)
                        and not details_recent_bottom.empty
                        and exit_idx in details_recent_bottom.index
                    ):
                        drow = details_recent_bottom.loc[exit_idx]
                        hits = [
                            f"結構{'✅' if bool(drow.get('structure')) else '❌'}",
                            f"動能{'✅' if bool(drow.get('momentum')) else '❌'}",
                            f"量能{'✅' if bool(drow.get('volume')) else '❌'}",
                            f"籌碼{'✅' if bool(drow.get('chip')) else '❌'}",
                        ]
                        try:
                            sc = int(drow.get("score"))
                        except Exception:
                            sc = None
                        if sc is not None:
                            text = f"{label}<br>Score {sc}/4 | {' '.join(hits)}"
                    if exit_idx in recent.index:
                        fig.add_annotation(
                            x=exit_idx,
                            y=float(recent.loc[exit_idx, "High"]),
                            text=text,
                            showarrow=True,
                            arrowhead=2,
                            ax=20,
                            ay=30,
                            arrowcolor="rgba(200, 120, 0, 0.90)",
                            bgcolor="rgba(255, 200, 120, 0.90)",
                            bordercolor="rgba(200, 120, 0, 0.90)",
                            row=1,
                            col=1,
                        )

            # ⭐/⚡ top：出場/解除警戒（解除＝BLOCK 降級）
            if len(status_recent_top) >= 2:
                best_status = "BLOCK"
                entry_mask = (status_recent_top == best_status) & (
                    status_recent_top.shift(1) != best_status
                )
                exit_mask = (status_recent_top != best_status) & (
                    status_recent_top.shift(1) == best_status
                )
                entry_mask.iloc[0] = False
                exit_mask.iloc[0] = False

                entry_pos_arr = np.where(entry_mask.fillna(False).values)[0]
                exit_pos_arr = np.where(exit_mask.fillna(False).values)[0]

                entry_pos = None
                exit_pos = None
                fallback_entry = False

                last_entry_pos = int(entry_pos_arr[-1]) if entry_pos_arr.size > 0 else None
                last_exit_pos = int(exit_pos_arr[-1]) if exit_pos_arr.size > 0 else None

                if last_entry_pos is not None and (
                    last_exit_pos is None or last_entry_pos > last_exit_pos
                ):
                    entry_pos = last_entry_pos
                elif last_exit_pos is not None:
                    exit_pos = last_exit_pos
                    if entry_pos_arr.size > 0:
                        entry_before = entry_pos_arr[entry_pos_arr < exit_pos]
                        if entry_before.size > 0:
                            entry_pos = int(entry_before[-1])

                # fallback：區間內沒有出場切換，但目前為 BLOCK
                if (
                    entry_pos is None
                    and len(status_recent_top) > 0
                    and str(status_recent_top.iloc[-1]) == best_status
                ):
                    entry_pos = len(status_recent_top) - 1
                    fallback_entry = True

                if entry_pos is not None:
                    entry_idx = status_recent_top.index[int(entry_pos)]
                    new_s = str(status_recent_top.iloc[int(entry_pos)])
                    old_s = (
                        str(status_recent_top.iloc[int(entry_pos) - 1])
                        if int(entry_pos) - 1 >= 0
                        else ""
                    )
                    label = (
                        "⭐top 目前較佳出場（狀態未切換）"
                        if fallback_entry
                        else (f"⭐top 出場 {old_s}→{new_s}" if old_s else "⭐top 出場（BLOCK）")
                    )
                    text = label
                    if (
                        isinstance(details_recent_top, pd.DataFrame)
                        and not details_recent_top.empty
                        and entry_idx in details_recent_top.index
                    ):
                        drow = details_recent_top.loc[entry_idx]
                        hits = [
                            f"結構{'✅' if bool(drow.get('structure')) else '❌'}",
                            f"動能{'✅' if bool(drow.get('momentum')) else '❌'}",
                            f"量能{'✅' if bool(drow.get('volume')) else '❌'}",
                            f"籌碼{'✅' if bool(drow.get('chip')) else '❌'}",
                            f"乖離{'✅' if bool(drow.get('bias')) else '❌'}",
                        ]
                        try:
                            sc = int(drow.get("score"))
                        except Exception:
                            sc = None
                        if sc is not None:
                            text = f"{label}<br>Score {sc}/5 | {' '.join(hits)}"
                    if entry_idx in recent.index:
                        fig.add_annotation(
                            x=entry_idx,
                            y=float(recent.loc[entry_idx, "High"]),
                            text=text,
                            showarrow=True,
                            arrowhead=2,
                            ax=20,
                            ay=30,
                            arrowcolor="rgba(200, 0, 0, 0.85)",
                            bgcolor="rgba(255, 215, 215, 0.85)",
                            bordercolor="rgba(200, 0, 0, 0.85)",
                            row=1,
                            col=1,
                        )

                # 追加：移動止損（TS）解除警戒（若比狀態降級更早）
                exit_is_ts = False
                ts_stop = None
                ts_tag = None
                trail_pct = float(turn_bt_trailing_stop or 0.0)
                if entry_pos is not None and (not fallback_entry) and trail_pct > 0:
                    try:
                        trail = float(trail_pct) / 100.0
                    except Exception:
                        trail = 0.0
                    if trail > 0:
                        try:
                            trough = float(
                                pd.to_numeric(
                                    df_turn_series.loc[entry_idx, "Low"], errors="coerce"
                                )
                            )
                        except Exception:
                            trough = np.nan
                        if not np.isfinite(trough) or trough <= 0:
                            try:
                                trough = float(
                                    pd.to_numeric(
                                        df_turn_series.loc[entry_idx, "Close"],
                                        errors="coerce",
                                    )
                                )
                            except Exception:
                                trough = np.nan

                        search_end = (
                            int(exit_pos) - 1
                            if exit_pos is not None and int(exit_pos) > 0
                            else (len(status_recent_top) - 1)
                        )
                        start_j = int(entry_pos) + 1
                        if (
                            search_end >= start_j
                            and np.isfinite(trough)
                            and float(trough) > 0
                        ):
                            for j in range(start_j, int(search_end) + 1):
                                idx_j = status_recent_top.index[int(j)]
                                stop = float(trough) * (1.0 + float(trail))
                                try:
                                    o = float(
                                        pd.to_numeric(
                                            df_turn_series.loc[idx_j, "Open"],
                                            errors="coerce",
                                        )
                                    )
                                    h = float(
                                        pd.to_numeric(
                                            df_turn_series.loc[idx_j, "High"],
                                            errors="coerce",
                                        )
                                    )
                                    l = float(
                                        pd.to_numeric(
                                            df_turn_series.loc[idx_j, "Low"],
                                            errors="coerce",
                                        )
                                    )
                                except Exception:
                                    o, h, l = np.nan, np.nan, np.nan

                                if np.isfinite(o) and float(o) >= float(stop):
                                    exit_pos = int(j)
                                    exit_is_ts = True
                                    ts_stop = float(stop)
                                    ts_tag = "TS_GAP"
                                    break
                                if np.isfinite(h) and float(h) >= float(stop):
                                    exit_pos = int(j)
                                    exit_is_ts = True
                                    ts_stop = float(stop)
                                    ts_tag = "TS"
                                    break
                                if np.isfinite(l):
                                    trough = min(float(trough), float(l))

                if exit_pos is not None:
                    exit_idx = status_recent_top.index[int(exit_pos)]
                    if exit_is_ts:
                        label = f"⚡top {ts_tag or 'TS'} 解除（反彈{trail_pct:.1f}%）"
                        if ts_stop is not None and np.isfinite(ts_stop):
                            label = f"{label}<br>stop≈{float(ts_stop):.2f}"
                        text = label
                    else:
                        new_s = str(status_recent_top.iloc[int(exit_pos)])
                        label = f"⚡top 解除警戒（BLOCK→{new_s}）"
                        text = label
                    if (
                        isinstance(details_recent_top, pd.DataFrame)
                        and not details_recent_top.empty
                        and exit_idx in details_recent_top.index
                    ):
                        drow = details_recent_top.loc[exit_idx]
                        hits = [
                            f"結構{'✅' if bool(drow.get('structure')) else '❌'}",
                            f"動能{'✅' if bool(drow.get('momentum')) else '❌'}",
                            f"量能{'✅' if bool(drow.get('volume')) else '❌'}",
                            f"籌碼{'✅' if bool(drow.get('chip')) else '❌'}",
                            f"乖離{'✅' if bool(drow.get('bias')) else '❌'}",
                        ]
                        try:
                            sc = int(drow.get("score"))
                        except Exception:
                            sc = None
                        if sc is not None:
                            text = f"{label}<br>Score {sc}/5 | {' '.join(hits)}"
                    if exit_idx in recent.index:
                        fig.add_annotation(
                            x=exit_idx,
                            y=float(recent.loc[exit_idx, "Low"]),
                            text=text,
                            showarrow=True,
                            arrowhead=2,
                            ax=20,
                            ay=-30,
                            arrowcolor="rgba(200, 120, 0, 0.90)",
                            bgcolor="rgba(255, 200, 120, 0.90)",
                            bordercolor="rgba(200, 120, 0, 0.90)",
                            row=1,
                            col=1,
                        )

            # 買點/賣點建議（以最新一根狀態為準；即使沒有反轉⭐也會顯示）
            if isinstance(details_recent_bottom, pd.DataFrame) and not details_recent_bottom.empty:
                b_status = str(details_recent_bottom["status"].iloc[-1])
                b_hint = (
                    "較佳進場" if b_status == "ALLOW" else "觀察" if b_status == "WATCH" else "先保守"
                )
            else:
                b_status, b_hint = "NA", "資料不足"
            if isinstance(details_recent_top, pd.DataFrame) and not details_recent_top.empty:
                t_status = str(details_recent_top["status"].iloc[-1])
                t_hint = (
                    "較佳出場" if t_status == "BLOCK" else "留意風險" if t_status == "WATCH" else "風險較低"
                )
            else:
                t_status, t_hint = "NA", "資料不足"
            st.caption(f"買點（bottom）：{b_hint}（{b_status}）｜賣點（top）：{t_hint}（{t_status}）")
        else:
            details_all = get_all_turn_details(
                df_turn_series,
                mode=turn_mode,
                cfg=turn_cfg,
                foreign_3d_net=chip_foreign_3d_net,
                trust_3d_net=chip_trust_3d_net,
            )
            details_recent = (
                details_all.tail(turn_window)
                if isinstance(details_all, pd.DataFrame)
                else pd.DataFrame()
            )
            status_recent = (
                details_recent["status"]
                if isinstance(details_recent, pd.DataFrame)
                else pd.Series(dtype=object)
            )

            if not status_recent.empty:
                seg_id = (status_recent != status_recent.shift(1)).cumsum()
                for _, seg in status_recent.groupby(seg_id):
                    status = seg.iloc[0]
                    color = colors.get(status, "rgba(200, 200, 200, 0.08)")
                    fig.add_vrect(
                        x0=seg.index[0],
                        x1=seg.index[-1],
                        fillcolor=color,
                        opacity=0.35,
                        line_width=0,
                        layer="below",
                        row=1,
                        col=1,
                    )

            # ⭐/⚡ 單一模式：進場/出場（出場＝狀態降級）
            if len(status_recent) >= 2:
                best_status = "ALLOW" if turn_mode == "bottom" else "BLOCK"
                entry_mask = (status_recent == best_status) & (status_recent.shift(1) != best_status)
                exit_mask = (status_recent != best_status) & (status_recent.shift(1) == best_status)
                entry_mask.iloc[0] = False
                exit_mask.iloc[0] = False

                entry_pos_arr = np.where(entry_mask.fillna(False).values)[0]
                exit_pos_arr = np.where(exit_mask.fillna(False).values)[0]

                entry_pos = None
                exit_pos = None
                fallback_entry = False

                last_entry_pos = int(entry_pos_arr[-1]) if entry_pos_arr.size > 0 else None
                last_exit_pos = int(exit_pos_arr[-1]) if exit_pos_arr.size > 0 else None

                if last_entry_pos is not None and (
                    last_exit_pos is None or last_entry_pos > last_exit_pos
                ):
                    entry_pos = last_entry_pos
                elif last_exit_pos is not None:
                    exit_pos = last_exit_pos
                    if entry_pos_arr.size > 0:
                        entry_before = entry_pos_arr[entry_pos_arr < exit_pos]
                        if entry_before.size > 0:
                            entry_pos = int(entry_before[-1])

                # fallback：區間內沒有進場切換，但目前為最佳狀態
                if (
                    entry_pos is None
                    and len(status_recent) > 0
                    and str(status_recent.iloc[-1]) == best_status
                ):
                    entry_pos = len(status_recent) - 1
                    fallback_entry = True

                # 標註座標（依模式）
                entry_y_col = "Low" if turn_mode == "bottom" else "High"
                entry_ay = -30 if turn_mode == "bottom" else 30
                exit_y_col = "High" if turn_mode == "bottom" else "Low"
                exit_ay = 30 if turn_mode == "bottom" else -30

                # entry ⭐
                if entry_pos is not None:
                    entry_idx = status_recent.index[int(entry_pos)]
                    new_s = str(status_recent.iloc[int(entry_pos)])
                    old_s = str(status_recent.iloc[int(entry_pos) - 1]) if int(entry_pos) - 1 >= 0 else ""
                    if fallback_entry:
                        label = (
                            "⭐目前較佳進場（狀態未切換）"
                            if turn_mode == "bottom"
                            else "⭐目前較佳出場（狀態未切換）"
                        )
                    else:
                        label = (
                            f"⭐進場 {old_s}→{new_s}"
                            if turn_mode == "bottom"
                            else f"⭐出場 {old_s}→{new_s}"
                        )
                    text = label
                    if (
                        isinstance(details_recent, pd.DataFrame)
                        and not details_recent.empty
                        and entry_idx in details_recent.index
                    ):
                        drow = details_recent.loc[entry_idx]
                        hits = [
                            f"結構{'✅' if bool(drow.get('structure')) else '❌'}",
                            f"動能{'✅' if bool(drow.get('momentum')) else '❌'}",
                            f"量能{'✅' if bool(drow.get('volume')) else '❌'}",
                            f"籌碼{'✅' if bool(drow.get('chip')) else '❌'}",
                        ]
                        if turn_mode == "top":
                            hits.append(f"乖離{'✅' if bool(drow.get('bias')) else '❌'}")
                        try:
                            sc = int(drow.get("score"))
                        except Exception:
                            sc = None
                        if sc is not None:
                            denom = 5 if turn_mode == "top" else 4
                            text = f"{label}<br>Score {sc}/{denom} | {' '.join(hits)}"
                    if entry_idx in recent.index:
                        if turn_mode == "bottom":
                            bgcolor = "rgba(210, 255, 210, 0.85)"
                            bordercolor = "rgba(0, 150, 0, 0.85)"
                            arrowcolor = "rgba(0, 150, 0, 0.85)"
                        else:
                            bgcolor = "rgba(255, 215, 215, 0.85)"
                            bordercolor = "rgba(200, 0, 0, 0.85)"
                            arrowcolor = "rgba(200, 0, 0, 0.85)"
                        fig.add_annotation(
                            x=entry_idx,
                            y=float(recent.loc[entry_idx, entry_y_col]),
                            text=text,
                            showarrow=True,
                            arrowhead=2,
                            ax=20,
                            ay=entry_ay,
                            arrowcolor=arrowcolor,
                            bgcolor=bgcolor,
                            bordercolor=bordercolor,
                            row=1,
                            col=1,
                        )

                # 追加：移動止損（TS）出場（若比狀態降級更早）
                exit_is_ts = False
                ts_stop = None
                ts_tag = None
                trail_pct = float(turn_bt_trailing_stop or 0.0)
                if entry_pos is not None and (not fallback_entry) and trail_pct > 0:
                    try:
                        trail = float(trail_pct) / 100.0
                    except Exception:
                        trail = 0.0
                    if trail > 0:
                        search_end = (
                            int(exit_pos) - 1
                            if exit_pos is not None and int(exit_pos) > 0
                            else (len(status_recent) - 1)
                        )
                        start_j = int(entry_pos) + 1
                        if search_end >= start_j:
                            if turn_mode == "bottom":
                                try:
                                    peak = float(
                                        pd.to_numeric(
                                            df_turn_series.loc[entry_idx, "High"],
                                            errors="coerce",
                                        )
                                    )
                                except Exception:
                                    peak = np.nan
                                if not np.isfinite(peak) or peak <= 0:
                                    try:
                                        peak = float(
                                            pd.to_numeric(
                                                df_turn_series.loc[entry_idx, "Close"],
                                                errors="coerce",
                                            )
                                        )
                                    except Exception:
                                        peak = np.nan

                                if np.isfinite(peak) and float(peak) > 0:
                                    for j in range(start_j, int(search_end) + 1):
                                        idx_j = status_recent.index[int(j)]
                                        stop = float(peak) * (1.0 - float(trail))
                                        try:
                                            o = float(
                                                pd.to_numeric(
                                                    df_turn_series.loc[idx_j, "Open"],
                                                    errors="coerce",
                                                )
                                            )
                                            h = float(
                                                pd.to_numeric(
                                                    df_turn_series.loc[idx_j, "High"],
                                                    errors="coerce",
                                                )
                                            )
                                            l = float(
                                                pd.to_numeric(
                                                    df_turn_series.loc[idx_j, "Low"],
                                                    errors="coerce",
                                                )
                                            )
                                        except Exception:
                                            o, h, l = np.nan, np.nan, np.nan

                                        if np.isfinite(o) and float(o) <= float(stop):
                                            exit_pos = int(j)
                                            exit_is_ts = True
                                            ts_stop = float(stop)
                                            ts_tag = "TS_GAP"
                                            break
                                        if np.isfinite(l) and float(l) <= float(stop):
                                            exit_pos = int(j)
                                            exit_is_ts = True
                                            ts_stop = float(stop)
                                            ts_tag = "TS"
                                            break
                                        if np.isfinite(h):
                                            peak = max(float(peak), float(h))
                            else:
                                try:
                                    trough = float(
                                        pd.to_numeric(
                                            df_turn_series.loc[entry_idx, "Low"],
                                            errors="coerce",
                                        )
                                    )
                                except Exception:
                                    trough = np.nan
                                if not np.isfinite(trough) or trough <= 0:
                                    try:
                                        trough = float(
                                            pd.to_numeric(
                                                df_turn_series.loc[entry_idx, "Close"],
                                                errors="coerce",
                                            )
                                        )
                                    except Exception:
                                        trough = np.nan

                                if np.isfinite(trough) and float(trough) > 0:
                                    for j in range(start_j, int(search_end) + 1):
                                        idx_j = status_recent.index[int(j)]
                                        stop = float(trough) * (1.0 + float(trail))
                                        try:
                                            o = float(
                                                pd.to_numeric(
                                                    df_turn_series.loc[idx_j, "Open"],
                                                    errors="coerce",
                                                )
                                            )
                                            h = float(
                                                pd.to_numeric(
                                                    df_turn_series.loc[idx_j, "High"],
                                                    errors="coerce",
                                                )
                                            )
                                            l = float(
                                                pd.to_numeric(
                                                    df_turn_series.loc[idx_j, "Low"],
                                                    errors="coerce",
                                                )
                                            )
                                        except Exception:
                                            o, h, l = np.nan, np.nan, np.nan

                                        if np.isfinite(o) and float(o) >= float(stop):
                                            exit_pos = int(j)
                                            exit_is_ts = True
                                            ts_stop = float(stop)
                                            ts_tag = "TS_GAP"
                                            break
                                        if np.isfinite(h) and float(h) >= float(stop):
                                            exit_pos = int(j)
                                            exit_is_ts = True
                                            ts_stop = float(stop)
                                            ts_tag = "TS"
                                            break
                                        if np.isfinite(l):
                                            trough = min(float(trough), float(l))

                # exit ⚡
                if exit_pos is not None:
                    exit_idx = status_recent.index[int(exit_pos)]
                    if exit_is_ts:
                        label = (
                            f"⚡{ts_tag or 'TS'} 出場（回檔{trail_pct:.1f}%）"
                            if turn_mode == "bottom"
                            else f"⚡{ts_tag or 'TS'} 解除（反彈{trail_pct:.1f}%）"
                        )
                        if ts_stop is not None and np.isfinite(ts_stop):
                            label = f"{label}<br>stop≈{float(ts_stop):.2f}"
                        text = label
                    else:
                        new_s = str(status_recent.iloc[int(exit_pos)])
                        label = (
                            f"⚡出場（{best_status}→{new_s}）"
                            if turn_mode == "bottom"
                            else f"⚡解除警戒（{best_status}→{new_s}）"
                        )
                        text = label
                    if (
                        isinstance(details_recent, pd.DataFrame)
                        and not details_recent.empty
                        and exit_idx in details_recent.index
                    ):
                        drow = details_recent.loc[exit_idx]
                        hits = [
                            f"結構{'✅' if bool(drow.get('structure')) else '❌'}",
                            f"動能{'✅' if bool(drow.get('momentum')) else '❌'}",
                            f"量能{'✅' if bool(drow.get('volume')) else '❌'}",
                            f"籌碼{'✅' if bool(drow.get('chip')) else '❌'}",
                        ]
                        if turn_mode == "top":
                            hits.append(f"乖離{'✅' if bool(drow.get('bias')) else '❌'}")
                        try:
                            sc = int(drow.get("score"))
                        except Exception:
                            sc = None
                        if sc is not None:
                            denom = 5 if turn_mode == "top" else 4
                            text = f"{label}<br>Score {sc}/{denom} | {' '.join(hits)}"
                    if exit_idx in recent.index:
                        fig.add_annotation(
                            x=exit_idx,
                            y=float(recent.loc[exit_idx, exit_y_col]),
                            text=text,
                            showarrow=True,
                            arrowhead=2,
                            ax=20,
                            ay=exit_ay,
                            arrowcolor="rgba(200, 120, 0, 0.90)",
                            bgcolor="rgba(255, 200, 120, 0.90)",
                            bordercolor="rgba(200, 120, 0, 0.90)",
                            row=1,
                            col=1,
                        )

            # 買點/賣點建議（以最新一根狀態為準；即使沒有反轉⭐也會顯示）
            if isinstance(details_recent, pd.DataFrame) and not details_recent.empty:
                cur_status = str(details_recent["status"].iloc[-1])
                if turn_mode == "bottom":
                    hint = "較佳進場" if cur_status == "ALLOW" else "觀察" if cur_status == "WATCH" else "先保守"
                    st.caption(f"買點（bottom）：{hint}（{cur_status}）")
                else:
                    hint = "較佳出場" if cur_status == "BLOCK" else "留意風險" if cur_status == "WATCH" else "風險較低"
                    st.caption(f"賣點（top）：{hint}（{cur_status}）")
            else:
                st.caption("買/賣點建議：資料不足")

        # KD（下方子圖）
        stoch = ta.stoch(df["High"], df["Low"], df["Close"])
        kd_ok = stoch is not None and not stoch.empty
        if kd_ok:
            k_col = stoch.columns[0]
            d_col = stoch.columns[1] if len(stoch.columns) > 1 else None
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=stoch[k_col],
                    line=dict(color="red", width=1),
                    name="KD-K",
                ),
                row=2,
                col=1,
            )
            if d_col:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=stoch[d_col],
                        line=dict(color="green", width=1),
                        name="KD-D",
                    ),
                    row=2,
                    col=1,
                )
            fig.update_yaxes(range=[0, 100], row=2, col=1)

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=700,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        st.plotly_chart(fig, width="stretch")
        if not kd_ok:
            st.info("KD 資料不足")

        # TURN CHECK（反轉 Gate）：放在「圖表設定」下方（圖表之後）
        st.markdown("---")
        turn_result = render_turn_check_panel(
            df_turn_series,
            foreign_3d_net=chip_foreign_3d_net,
            trust_3d_net=chip_trust_3d_net,
            mode=turn_mode,
            symbol=effective_symbol,
            position_avg_cost=float(locals().get("pos_avg_cost", 0.0) or 0.0),
            position_qty=int(locals().get("pos_qty", 0) or 0),
            position_exit_style=str(locals().get("pos_exit_style", "波段守五日線")),
            show_position_settings=False,
            show_exit_guide=False,
            config_path="turn_check_config.json",
            cfg_override=turn_cfg_runtime,
            allow_param_tuning=not bool(turn_sidebar_override),
        )

        # 反彈 vs 反轉結論卡（緊貼 TURN CHECK）
        if isinstance(turn_result, dict):
            if (
                turn_result.get("mode") == "both"
                and "bottom" in turn_result
                and "top" in turn_result
            ):
                col_b, col_t = st.columns(2)
                with col_b:
                    st.markdown("**結論（bottom｜較佳進場）**")
                    r = turn_result.get("bottom") or {}
                    c = r.get("conditions", {})
                    status = r.get("status")
                    if status == "ALLOW" and c.get("structure") and c.get("momentum"):
                        st.success("反轉機率較高：結構與動能同時轉正。")
                    elif status in ["ALLOW", "WATCH"]:
                        st.warning("偏反彈：條件尚未齊備，先觀察是否站穩與量能延續。")
                    else:
                        st.info("偏反彈或轉弱：目前屬於風險區，先保守觀察。")

                with col_t:
                    st.markdown("**結論（top｜較佳出場／風險）**")
                    r = turn_result.get("top") or {}
                    c = r.get("conditions", {})
                    status = r.get("status")
                    if status == "BLOCK" and c.get("structure") and c.get("momentum"):
                        st.warning("轉弱風險較高：頭部結構與動能同時轉弱，避免追高。")
                    elif status in ["BLOCK", "WATCH"]:
                        st.warning("偏轉弱／高檔震盪：先保守，留意量價背離與籌碼轉空。")
                    else:
                        st.info("風險較低：頭部 Gate 尚未亮紅，但仍建議配合趨勢與停損。")
            else:
                mode_now = turn_result.get("mode", "bottom")
                c = turn_result.get("conditions", {})
                status = turn_result.get("status")
                if mode_now == "top":
                    if status == "BLOCK" and c.get("structure") and c.get("momentum"):
                        st.warning("轉弱風險較高：頭部結構與動能同時轉弱，避免追高。")
                    elif status in ["BLOCK", "WATCH"]:
                        st.warning("偏轉弱／高檔震盪：先保守，留意量價背離與籌碼轉空。")
                    else:
                        st.info("風險較低：頭部 Gate 尚未亮紅，但仍建議配合趨勢與停損。")
                else:
                    if status == "ALLOW" and c.get("structure") and c.get("momentum"):
                        st.success("反轉機率較高：結構與動能同時轉正。")
                    elif status in ["ALLOW", "WATCH"]:
                        st.warning("偏反彈：條件尚未齊備，先觀察是否站穩與量能延續。")
                    else:
                        st.info("偏反彈或轉弱：目前屬於風險區，先保守觀察。")

        st.markdown("### 咸魚翻身｜AI 專家診斷")
        render_conflict_warning(
            foreign_divergence_warning, bool(latest.get("Is_Dangerous_Volume", False))
        )
        expert_msg = generate_expert_advice(
            df, name, score, risk, is_chip_divergence=foreign_divergence_warning
        )
        if latest.get("BUY_SIGNAL"):
            st.success(expert_msg)
        else:
            st.info(expert_msg)
        st.caption("分數門檻：≥70 留、40~69 減碼、<40 去")

        # 持倉模式：以持倉診斷為準，避免與「安心抱/分批減碼」矛盾
        pos_cost = float(locals().get("pos_avg_cost", 0) or 0)
        if pos_cost > 0 and "pos_advice" in locals():
            pa = locals()["pos_advice"]
            if pa.level == "error" or ("減碼" in pa.headline or "落袋" in pa.headline):
                diagnosis_summary.warning("去留診斷：建議分批減碼")
            elif pa.level == "warning":
                diagnosis_summary.warning("去留診斷：減碼觀望")
            elif pa.level == "success":
                diagnosis_summary.success("去留診斷：持股續抱")
            else:
                diagnosis_summary.info("去留診斷：續抱/觀察")
        elif score >= 70:
            diagnosis_summary.success("去留診斷：持股續抱")
        elif score >= 40:
            diagnosis_summary.warning("去留診斷：減碼觀望")
        else:
            diagnosis_summary.error("去留診斷：考慮離場")
        if st.button("一鍵傳送完整診斷到 LINE"):
            try:
                if pos_cost > 0 and "pos_advice" in locals():
                    pa = locals()["pos_advice"]
                    if pa.level == "error" or ("減碼" in pa.headline or "落袋" in pa.headline):
                        decision_text = "建議分批減碼"
                    elif pa.level == "warning":
                        decision_text = "減碼觀望"
                    elif pa.level == "success":
                        decision_text = "續抱"
                    else:
                        decision_text = "續抱/觀察"
                else:
                    decision_text = "續抱" if score >= 70 else "減碼觀望" if score >= 40 else "考慮離場"

                # 防守線：依使用者下車風格（若取不到就用 EMA5）
                _exit_style = str(locals().get("pos_exit_style", "") or "")
                _defense_name = "EMA20（月線）" if _exit_style == "長線守月線" else "EMA5（五日線）"
                _defense = ema20 if _exit_style == "長線守月線" else ema5
                guard = calc_tomorrow_guard(
                    ema_today=_defense,
                    window=20 if _exit_style == "長線守月線" else 5,
                    buffer_pct=1.5,
                )

                # TURN（若可取得）
                _bottom_now = locals().get("bottom_now")
                _top_now = locals().get("top_now")
                bottom_txt = (
                    f"{_bottom_now.get('status')} {int(_bottom_now.get('score', 0))}/4"
                    if isinstance(_bottom_now, dict)
                    else "NA"
                )
                top_txt = (
                    f"{_top_now.get('status')} {int(_top_now.get('score', 0))}/5"
                    if isinstance(_top_now, dict)
                    else "NA"
                )

                # 風險警報（只列 FAIL，最多 5 條）
                fail_lines = []
                for cat in ["Gate", "Trigger", "Guard", "Chip Notes"]:
                    for it in (used_map or {}).get(cat, []) or []:
                        if it.get("pass", True) is False:
                            rule = str(it.get("rule", "") or "").strip()
                            note = str(it.get("note", "") or "").strip()
                            fail_lines.append(f"- {cat}｜{rule}{('｜' + note) if note else ''}")
                        if len(fail_lines) >= 5:
                            break
                    if len(fail_lines) >= 5:
                        break

                msg_lines = []
                msg_lines.append(f"【{ticker} 完整診斷】")
                if name:
                    msg_lines.append(f"名稱：{name}")
                msg_lines.append(f"收盤：{close_price:.2f}")
                msg_lines.append(f"AI 分數：{score}｜建議：{decision_text}")
                msg_lines.append(f"TURN bottom：{bottom_txt}｜TURN top：{top_txt}")
                if guard is not None:
                    msg_lines.append(
                        f"明日保命價：{guard.break_close:.2f}｜保守警戒(+{guard.buffer_pct:.1f}%) {guard.guard_close:.2f}"
                    )
                    msg_lines.append(f"防守線：{_defense_name}={guard.ema_today:.2f}")
                if fail_lines:
                    msg_lines.append("⚠️ 風險警報（FAIL）：")
                    msg_lines.extend(fail_lines)
                msg_lines.append("---")
                msg_lines.append(expert_msg)

                ok, msg = send_with_cooldown(
                    f"{effective_symbol}_full_diagnosis",
                    "\n".join(msg_lines),
                    cooldown_minutes=5,
                )
                if ok:
                    st.success("已推播完整診斷到 LINE")
                else:
                    st.warning(msg)
            except Exception as exc:
                st.warning(f"推播失敗：{exc}")

    with latest_metrics_placeholder.container():
        with st.expander("📊 最新價量（可收起）", expanded=False):
            st.subheader("最新價量")
            c1, c2, c3 = st.columns(3)
            c1.metric("最新收盤價", f"{latest_close:.2f}")
            c2.metric("最新成交量", f"{int(latest_volume):,}")
            if not pd.isna(latest_vol_avg_5):
                c3.metric("5日均量", f"{latest_vol_avg_5:,.0f}")
            else:
                c3.metric("5日均量", "資料不足")
            if not pd.isna(latest_vol_avg_5):
                if latest_volume > latest_vol_avg_5:
                    st.success("當天成交量高於 5 日均量")
                else:
                    st.info("當天成交量低於 5 日均量")
            else:
                st.info("資料不足，無法計算 5 日均量")

    with st.expander("🔔 LINE / 指標明細（進階工具，可收起）", expanded=False):
        # 4.2 LINE 通知（進階）
        st.subheader("LINE 通知")
        auto_push = st.checkbox("EMA20 突破時自動推播", value=False)
        if st.button("測試推播"):
            ok, msg = send_line_message(f"【{ticker}】測試訊息：LINE Messaging API 已連線")
            if ok:
                st.success(msg)
            else:
                st.warning(msg)

        if st.button("執行分析並傳送 LINE 通知"):
            if ema20 is not None and close_price > ema20:
                event = "站上均線"
                decision = "買入"
            else:
                event = "跌破均線"
                decision = "觀望"
            ok, msg = send_with_cooldown(
                f"{effective_symbol}_manual_analysis",
                build_push_message(event, decision, score_value=score),
                cooldown_minutes=5,
            )
            if ok:
                st.success("已傳送 LINE 通知")
            else:
                st.warning(msg)

        if st.button("發送分析報告"):
            if not pd.isna(avg_vol_5):
                msg_lines = [
                    f"【{ticker} 報告】",
                    f"今日成交：{current_vol:,.0f}",
                    f"5日均量：{avg_vol_5:,.0f}",
                ]
                if current_vol > avg_vol_5 * 1.5:
                    msg_lines.append("注意：今日成交量異常放大")
                ok, msg = send_line_message("\n".join(msg_lines))
                if ok:
                    st.success("已傳送分析報告")
                else:
                    st.warning(msg)
            else:
                st.warning("資料不足，無法發送分析報告")

        if auto_push and ema20 is not None and len(ema_df) >= 2:
            prev = ema_df.iloc[-2]
            crossed = prev["Close"] <= prev["EMA20"] and close_price > ema20
            if crossed:
                ok, msg = send_with_cooldown(
                    f"{effective_symbol}_ema20_cross",
                    build_push_message("EMA20 突破", "買入", score_value=score),
                    cooldown_minutes=30,
                )
                if ok:
                    st.success("已推播至 LINE")
                else:
                    st.warning(msg)

        # 6. 顯示 RSI 指標（進階）
        st.subheader("RSI 指標")
        st.line_chart(df["RSI14"])

        # 7. 顯示外資買賣超 (台股；進階)
        if effective_symbol.endswith(".TW") or effective_symbol.endswith(".TWO"):
            st.subheader("外資買賣超")
            if FINMIND_AVAILABLE:
                try:
                    if foreign_net_series is not None and not foreign_net_series.empty:
                        st.bar_chart(foreign_net_series)
                    else:
                        st.info("找不到外資資料")
                except Exception as exc:
                    st.warning(f"FinMind 抓取失敗：{exc}")
            else:
                st.warning("未安裝 FinMind，無法抓取外資資料")

else:
    st.error("找不到資料，請確認代碼（例如台股要加 .TW）")
