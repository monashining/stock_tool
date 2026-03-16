"""
回測 UI：策略回測、參數調校、結果比較
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import compute_indicators
from backtest_engine import run_backtest, BacktestSummary
from data_sources import load_data, fetch_chip_net_series, fetch_fundamental_snapshot
from turn_check_engine import load_turn_config
from risk_stress import (
    trailing_stop_monte_carlo_stress,
    compute_friction_adjusted_return,
    get_signal_date_range,
    ROUND_TRIP_FRICTION_PCT,
)
from risk_verification import get_dynamic_stop_advice


STRATEGY_LABELS = {
    "buy_signal": "BUY_SIGNAL（Gate/Trigger/Guard）",
    "turn_bottom": "TURN bottom（較佳進場）",
    "turn_top": "TURN top（較佳出場）",
}


def render_backtest_page(
    ticker: str,
    time_range: str = "1y",
):
    """回測頁面：選擇策略、參數、執行、檢視結果"""
    st.title("📈 策略回測")
    st.caption("回測不同策略的歷史表現，調參後可即時比較，逐步提升精準度。")

    if not ticker or not ticker.strip():
        st.info("請在側邊欄輸入股票代碼")
        return

    # 回測區間（可覆寫側邊欄）
    range_opts = ["6mo", "1y", "2y", "5y"]
    try:
        idx = range_opts.index(time_range) if time_range in range_opts else 1
    except ValueError:
        idx = 1
    bt_range = st.selectbox(
        "回測區間",
        range_opts,
        index=idx,
        help="區間越長訊號越多，但市場結構可能變化",
    )

    # 載入資料（含技術指標，供風險驗證用）
    with st.spinner("載入歷史資料..."):
        df = load_data(ticker, bt_range)
    if df is None or df.empty or len(df) < 60:
        st.warning("資料不足（需至少 60 個交易日），請換較長區間或另一檔股票")
        return
    df = compute_indicators(df)

    # 籌碼（TURN 用）
    foreign_3d = trust_3d = None
    if ticker.endswith(".TW") or ticker.endswith(".TWO"):
        try:
            f, t = fetch_chip_net_series(ticker)
            if f is not None and not f.empty:
                foreign_3d = float(f.tail(3).sum())
            if t is not None and not t.empty:
                trust_3d = float(t.tail(3).sum())
        except Exception:
            pass

    # 策略選擇
    st.markdown("### 1️⃣ 選擇策略")
    strategy = st.radio(
        "策略",
        list(STRATEGY_LABELS.keys()),
        format_func=lambda k: STRATEGY_LABELS[k],
        horizontal=True,
    )

    # 建議參數（依 P/E、乖離自動計算）
    _fund = fetch_fundamental_snapshot(ticker) if ticker else None
    _pe = (_fund.get("pe_ttm") if isinstance(_fund, dict) else None) if _fund else None
    _bias = None
    if "Bias20" in df.columns and not df.empty:
        try:
            _bias = float(df["Bias20"].iloc[-1])
        except Exception:
            pass
    _stop_advice = get_dynamic_stop_advice(_pe, _bias, current_stop_pct=10.0)
    suggested_stop = _stop_advice.suggested_stop_pct
    suggested_hold = 10 if not _stop_advice.is_high_valuation_risk else 7

    _key = f"bt_params_{ticker}_{strategy}"
    defaults = st.session_state.get(_key, {})
    # 預設用建議值，若曾套用則用套用值
    hold_default = int(defaults.get("hold_days") or suggested_hold)
    stop_default = float(defaults.get("trailing_stop_pct") or suggested_stop)
    ema_default = int(defaults.get("exit_ema_window") or 0)
    win_default = float(defaults.get("win_threshold_pct") or 0.0)

    st.markdown("### 2️⃣ 回測參數")
    with st.expander("💡 建議參數（一鍵套用）", expanded=True):
        st.caption(
            f"依本檔 P/E、乖離自動建議：移動止損 **{suggested_stop:.0f}%**｜"
            f"持有天數 **{suggested_hold}** 天"
        )
        if st.button("✅ 套用建議參數", key=f"apply_suggested_{ticker}"):
            st.session_state[_key] = {
                "hold_days": suggested_hold,
                "trailing_stop_pct": suggested_stop,
                "exit_ema_window": 0,
                "win_threshold_pct": 0.0,
            }
            st.rerun()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        hold_days = st.slider("最長持有天數", 3, 30, hold_default or 10)
    with c2:
        trailing_stop_pct = st.slider("移動止損（%）", 0.0, 15.0, stop_default or 10.0, 0.5)
    with c3:
        exit_ema_window = st.slider("出場 EMA 確認（0=關閉）", 0, 20, ema_default or 0)
    with c4:
        win_threshold_pct = st.slider("勝率門檻（%）", -5.0, 10.0, win_default or 0.0, 0.5)

    st.markdown("### 3️⃣ 執行回測")
    if st.button("執行回測", type="primary"):
        turn_cfg = load_turn_config() if strategy.startswith("turn_") else None
        result = run_backtest(
            df,
            strategy=strategy,
            symbol=ticker,
            hold_days=hold_days,
            trailing_stop_pct=trailing_stop_pct,
            exit_ema_window=exit_ema_window,
            win_threshold_pct=win_threshold_pct,
            foreign_3d_net=foreign_3d,
            trust_3d_net=trust_3d,
            turn_cfg=turn_cfg,
        )
        st.session_state["backtest_result"] = result
        st.session_state["backtest_params"] = {
            "strategy": strategy,
            "hold_days": hold_days,
            "trailing_stop_pct": trailing_stop_pct,
            "exit_ema_window": exit_ema_window,
        }

    result: Optional[BacktestSummary] = st.session_state.get("backtest_result")
    params = st.session_state.get("backtest_params", {})
    if result is not None:
        _render_backtest_result(result, win_threshold_pct, df, params)
    else:
        st.info("設定參數後點擊「執行回測」")


def _render_backtest_result(
    result: BacktestSummary,
    win_threshold_pct: float,
    df: pd.DataFrame,
    params: dict,
):
    """渲染回測結果"""
    st.markdown("---")
    st.markdown("### 4️⃣ 回測結果")

    # 指標卡片
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("訊號數", result.n_signals)
        if result.n_signals < 10 and result.n_signals > 0:
            st.caption("⚠️ 樣本少，建議放寬 Gate 或延長區間")
    with col2:
        st.metric(
            f"勝率（≥{win_threshold_pct:.1f}%）",
            f"{result.win_rate:.1%}" if result.n_signals > 0 else "N/A",
        )
    with col3:
        st.metric("平均報酬", f"{result.avg_return_pct:.2f}%")
    with col4:
        st.metric("期望值", f"{result.expectancy:.2f}%")
    with col5:
        st.metric("最大回撤", f"{result.max_drawdown_pct:.1f}%")

    col6, col7, col8 = st.columns(3)
    with col6:
        st.metric("平均持有天數", f"{result.avg_holding_days:.1f}")
    with col7:
        st.metric("Profit Factor", f"{result.profit_factor:.2f}" if result.profit_factor < 100 else "∞")
    with col8:
        sharpe_txt = f"{result.sharpe_approx:.2f}" if result.sharpe_approx < 100 else "N/A"
        st.metric("Sharpe（近似）", sharpe_txt)

    # 期望值解讀
    if result.n_signals > 0:
        if result.expectancy > 2:
            st.success("✅ 目前參數組合具有良好正期望值，可持續追蹤實盤表現")
        elif result.expectancy > 0:
            st.info("🟡 期望值為正但偏低，可嘗試調整參數或結合其他濾網")
        else:
            st.warning("⚠️ 期望值為負，建議放寬門檻或更換策略")

    # 5️⃣ 風險壓力測試（系統驗證）
    st.markdown("---")
    st.markdown("### 5️⃣ 風險壓力測試（系統驗證）")
    hold_days = int(params.get("hold_days", 10) or 10)
    trailing_stop_pct = float(params.get("trailing_stop_pct", 0) or 0)

    # 摩擦成本
    friction_adj = compute_friction_adjusted_return(result.avg_return_pct)
    st.caption(
        f"**摩擦成本**：證交稅 0.3% + 手續費約 0.29% ≈ {ROUND_TRIP_FRICTION_PCT:.2f}% 來回。"
        f" 扣後平均報酬：**{friction_adj:.2f}%**（原 {result.avg_return_pct:.2f}%）"
    )

    # 倖存者偏差檢查：訊號日期區間
    date_range = get_signal_date_range(result.trades_df)
    if date_range:
        st.caption(
            f"**訊號時序**：{date_range[0]} ～ {date_range[1]}。"
            "請確認是否集中在特定行情（如 AI 噴發期），空頭/盤整時可能無訊號。"
        )

    # 動態止損建議（P/E、乖離檢查）
    _trail_for_advice = trailing_stop_pct if trailing_stop_pct > 0 else 10.0
    _fund = fetch_fundamental_snapshot(result.symbol) if result.symbol else None
    _pe = (_fund.get("pe_ttm") if isinstance(_fund, dict) else None) if _fund else None
    _bias = None
    if df is not None and not df.empty:
        if "Bias20" in df.columns:
            try:
                _bias = float(df["Bias20"].iloc[-1])
            except Exception:
                pass
        elif "SMA20" in df.columns and "Close" in df.columns:
            try:
                c = float(df["Close"].iloc[-1])
                s = float(df["SMA20"].iloc[-1])
                if s and s > 0:
                    _bias = (c / s - 1.0) * 100.0
            except Exception:
                pass
    _stop_advice = get_dynamic_stop_advice(_pe, _bias, current_stop_pct=_trail_for_advice)
    if _stop_advice.is_high_valuation_risk or _stop_advice.suggested_stop_pct < _trail_for_advice:
        st.warning(
            f"🛠️ **動態止損建議**：{_stop_advice.reason}\n\n"
            f"（P/E {_stop_advice.pe_vs_threshold}｜建議止損 **{_stop_advice.suggested_stop_pct:.0f}%**）"
        )

    # 蒙地卡羅壓力測試（僅當有移動止損時）
    if trailing_stop_pct > 0 and df is not None and not df.empty:
        vol_shock = st.slider(
            "波動率壓力情境（σ 增加 %）",
            0, 50, 20, 5,
            help="若未來波動率增加 X%，策略破功（虧損）的機率？",
        )
        if st.button("執行蒙地卡羅壓力測試"):
            stress = trailing_stop_monte_carlo_stress(
                df,
                trailing_stop_pct=trailing_stop_pct,
                hold_days=hold_days,
                vol_shock_pct=float(vol_shock),
                n_sim=10000,
            )
            st.session_state["stress_result"] = stress
        stress = st.session_state.get("stress_result")
        if stress is not None:
            st.markdown(f"**波動率 +{stress.vol_shock_pct:.0f}% 情境**（{stress.n_sim:,} 次模擬）")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("破功機率（虧損）", f"{stress.pct_loss:.1f}%")
            with col2:
                st.metric("模擬平均報酬", f"{stress.avg_return_pct:.2f}%")
            with col3:
                st.metric("模擬中位數報酬", f"{stress.median_return_pct:.2f}%")
            if stress.pct_loss > 30:
                st.warning(
                    f"⚠️ 波動率增加 {stress.vol_shock_pct:.0f}% 時，約 {stress.pct_loss:.0f}% 路徑會虧損。"
                    "實盤前建議縮小止損或加入波動率濾網。"
                )
            elif stress.pct_loss < 15:
                st.success(
                    f"✅ 壓力情境下破功機率僅 {stress.pct_loss:.1f}%，策略具一定韌性。"
                )
            else:
                st.info(f"🟡 壓力情境破功機率 {stress.pct_loss:.1f}%，需留意波動放大時風險。")

    # 權益曲線
    trades = result.trades_df
    if trades is not None and len(trades) >= 2:
        ret = pd.to_numeric(trades["return_pct"], errors="coerce").dropna()
        cum = (1 + ret / 100.0).cumprod()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(cum))),
                y=cum.values,
                mode="lines",
                name="累積報酬",
                line=dict(color="steelblue", width=2),
            )
        )
        fig.update_layout(
            title="累積報酬曲線",
            xaxis_title="交易次數",
            yaxis_title="累積倍數",
            height=280,
            margin=dict(l=50, r=50, t=40, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    # 明細表
    with st.expander("📋 交易明細", expanded=False):
        st.dataframe(trades, use_container_width=True)
