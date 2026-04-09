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
from data_sources import load_data, load_data_batch, fetch_chip_net_series, fetch_fundamental_snapshot
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

MAX_BACKTEST_SYMBOLS = 25


def _normalize_bt_symbol(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s.isdigit():
        return f"{s}.TW"
    return s


def _parse_bt_symbols_text(text: str) -> list[str]:
    """多行或逗號分隔；去重、保序。"""
    out: list[str] = []
    seen: set[str] = set()
    for raw_line in (text or "").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for part in line.split(","):
            p = part.strip()
            if not p:
                continue
            n = _normalize_bt_symbol(p)
            if n and n not in seen:
                seen.add(n)
                out.append(n)
    return out


def _render_power_metrics(result: BacktestSummary, win_threshold_pct: float, params: dict):
    """分頁一：戰力評估（勝率、期望值、回撤等）"""
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

    if result.n_signals > 0:
        if result.expectancy > 2:
            st.success("✅ 目前參數組合具有良好正期望值，可持續追蹤實盤表現")
        elif result.expectancy > 0:
            st.info("🟡 期望值為正但偏低，可嘗試調整參數或結合其他濾網")
        else:
            st.warning("⚠️ 期望值為負，建議放寬門檻或更換策略")

    # 摩擦成本
    friction_adj = compute_friction_adjusted_return(result.avg_return_pct)
    st.caption(
        f"**摩擦成本**：證交稅 0.3% + 手續費約 0.29% ≈ {ROUND_TRIP_FRICTION_PCT:.2f}% 來回。"
        f" 扣後平均報酬：**{friction_adj:.2f}%**（原 {result.avg_return_pct:.2f}%）"
    )
    date_range = get_signal_date_range(result.trades_df)
    if date_range:
        st.caption(
            f"**訊號時序**：{date_range[0]} ～ {date_range[1]}。"
            "請確認是否集中在特定行情（如 AI 噴發期），空頭/盤整時可能無訊號。"
        )

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

    with st.expander("📋 交易明細", expanded=False):
        st.dataframe(trades, use_container_width=True)


def _render_live_guidance(result: BacktestSummary, df: Optional[pd.DataFrame], ts_pct: float):
    """分頁二：實戰指引（現況診斷、保命價）"""
    status = getattr(result, "current_status", None)
    if status:
        st.markdown("#### 現況診斷")
        col1, col2 = st.columns(2)
        with col1:
            action = status.get("suggested_action", "N/A")
            if action == "HOLD":
                st.success(f"**建議動作**：續抱")
            else:
                st.error(f"**建議動作**：下車")
            st.caption(f"收盤 {status.get('last_close', 0):.2f} vs EMA5 {status.get('ema5', 0):.2f}")
        with col2:
            is_hot = status.get("is_hot", False)
            if is_hot:
                st.warning(f"⚠️ 乖離過熱（Bias20={status.get('bias20', 0):.1f}%）")
            else:
                st.info(f"乖離 {status.get('bias20', 0):.1f}%")
        st.caption(f"最近一筆出場原因：{status.get('last_exit_reason', 'N/A')}")

    if df is not None and not df.empty and ts_pct > 0:
        try:
            current_high = float(df["High"].iloc[-1])
            real_time_stop = current_high * (1 - ts_pct / 100)
            st.markdown("#### 今日移動止損參考")
            st.metric("保命價（最高 × (1 - 止損%)）", f"{real_time_stop:.2f}")
            st.info(f"💡 根據回測邏輯，若收盤跌破 **{real_time_stop:.2f}** 應果斷下車。")
        except Exception:
            pass
    elif ts_pct <= 0:
        st.info("未啟用移動止損，請在回測參數設定「移動止損 %」後重新執行。")


def _render_strategy_live_diagnostic(result: BacktestSummary, df: Optional[pd.DataFrame], ts_pct: float):
    """策略實戰連動診斷：回測參數與下車指南一體化"""
    st.subheader("🚀 策略實戰連動診斷")
    is_strategy_reliable = (result.expectancy > 0) and (result.n_signals >= 5)

    if is_strategy_reliable:
        st.success("✅ 策略驗證通過：歷史期望值為正，具備實戰參考價值。")
        if df is not None and not df.empty and ts_pct > 0:
            try:
                current_high = float(df["High"].iloc[-1])
                real_time_stop = current_high * (1 - ts_pct / 100)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("今日移動止損參考價", f"{real_time_stop:.2f}")
                with col2:
                    st.metric("預估持有天數", f"{result.avg_holding_days:.1f} 天")
                st.info(f"💡 根據回測邏輯，若明日收盤跌破 **{real_time_stop:.2f}** 應果斷下車。")
            except Exception:
                pass
    else:
        st.warning("⚠️ 策略品質警示：回測樣本不足或期望值為負。")
        st.error("此參數組合『不建議』用於實戰。請嘗試放寬 Gate 或調整移動止損。")


def render_backtest_page(
    ticker: str,
    time_range: str = "1y",
):
    """回測頁面：支援多檔標的、同一組參數批次回測並彙總比較。"""
    st.title("📈 策略回測")
    st.caption(
        "可一次回測多檔持股／關注清單，共用同一策略與參數；彙總表比較期望值與勝率，再選單檔看圖表與壓測。"
    )

    default_ticker = (ticker or "2330.TW").strip()
    if "bt_multi_symbols_text" not in st.session_state:
        st.session_state["bt_multi_symbols_text"] = default_ticker

    st.markdown("### 0️⃣ 回測標的（可多檔）")
    st.text_area(
        "股票代號（每行一列或逗號分隔；純數字＝台股 .TW）",
        height=110,
        key="bt_multi_symbols_text",
        help="例：3037.TW、2367、2330.TW 分行列出。最多 25 檔以免逾時。",
        label_visibility="collapsed",
    )
    symbols = _parse_bt_symbols_text(st.session_state.get("bt_multi_symbols_text", ""))
    b0, b1 = st.columns(2)
    with b0:
        if st.button("帶入側邊欄目前代號", key="bt_fill_sidebar_symbol"):
            st.session_state["bt_multi_symbols_text"] = default_ticker
            st.rerun()
    with b1:
        if st.button("載入 OPEN trades 全部代號", key="bt_fill_open_trade_symbols"):
            try:
                from portfolio_journal import list_open_trades

                ot = list_open_trades()
                if ot is None or ot.empty:
                    st.warning("目前沒有 OPEN trade。")
                else:
                    syms = sorted(set(ot["symbol"].astype(str).str.strip().tolist()))
                    st.session_state["bt_multi_symbols_text"] = "\n".join(syms)
                    st.rerun()
            except Exception as exc:
                st.warning(f"載入失敗：{exc}")

    if not symbols:
        st.info("請至少輸入一檔股票代號（或從側邊欄／OPEN trades 帶入）。")
        return
    if len(symbols) > MAX_BACKTEST_SYMBOLS:
        st.warning(
            f"最多同時回測 {MAX_BACKTEST_SYMBOLS} 檔，已截取前 {MAX_BACKTEST_SYMBOLS} 檔。"
        )
        symbols = symbols[:MAX_BACKTEST_SYMBOLS]

    primary_sym = symbols[0]

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

    df_primary: Optional[pd.DataFrame] = None
    with st.spinner(f"載入 **{primary_sym}** 報價（供建議參數）..."):
        _raw_p = load_data(primary_sym, bt_range)
    if (
        _raw_p is not None
        and not _raw_p.empty
        and len(_raw_p) >= 60
    ):
        df_primary = compute_indicators(_raw_p)
    else:
        st.warning(
            f"第一檔 **{primary_sym}** 資料不足（需至少 60 交易日）時，建議參數可能僅供參考；"
            "其餘標的仍可於批次回測中各自載入。"
        )

    st.markdown("### 1️⃣ 選擇策略")
    strategy = st.radio(
        "策略",
        list(STRATEGY_LABELS.keys()),
        format_func=lambda k: STRATEGY_LABELS[k],
        horizontal=True,
    )

    _fund = fetch_fundamental_snapshot(primary_sym) if primary_sym else None
    _pe = (_fund.get("pe_ttm") if isinstance(_fund, dict) else None) if _fund else None
    _bias = None
    if (
        df_primary is not None
        and not df_primary.empty
        and "Bias20" in df_primary.columns
    ):
        try:
            _bias = float(df_primary["Bias20"].iloc[-1])
        except Exception:
            pass
    _stop_advice = get_dynamic_stop_advice(_pe, _bias, current_stop_pct=10.0)
    suggested_stop = _stop_advice.suggested_stop_pct
    suggested_hold = 10 if not _stop_advice.is_high_valuation_risk else 7

    _key = f"bt_params_multi_{strategy}"
    defaults = st.session_state.get(_key, {})
    hold_default = int(defaults.get("hold_days") or suggested_hold)
    stop_default = float(defaults.get("trailing_stop_pct") or suggested_stop)
    ema_default = int(defaults.get("exit_ema_window") or 0)
    win_default = float(defaults.get("win_threshold_pct") or 0.0)

    st.markdown("### 2️⃣ 回測參數（全部標的共用）")
    with st.expander("💡 建議參數（依第一檔 P/E、乖離）", expanded=True):
        st.caption(
            f"第一檔 **{primary_sym}**：移動止損建議 **{suggested_stop:.0f}%**｜"
            f"持有天數 **{suggested_hold}** 天"
        )
        if st.button("✅ 套用建議參數", key=f"apply_suggested_multi_{strategy}"):
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
    st.caption(
        f"即將批次回測 **{len(symbols)}** 檔：{', '.join(symbols[:8])}"
        f"{'…' if len(symbols) > 8 else ''}"
    )
    if st.button("執行批次回測", type="primary"):
        turn_cfg = load_turn_config() if strategy.startswith("turn_") else None
        results: dict[str, BacktestSummary] = {}
        dfs_out: dict[str, pd.DataFrame] = {}
        errors: list[tuple[str, str]] = []
        with st.spinner("下載報價資料中…"):
            batch_raw = load_data_batch(symbols, bt_range)
        progress = st.progress(0)
        n_sym = len(symbols)
        for i, sym in enumerate(symbols):
            progress.progress(int((i + 1) / max(n_sym, 1) * 100))
            df_sym = batch_raw.get(sym) if batch_raw else None
            if df_sym is None or df_sym.empty:
                df_sym = load_data(sym, bt_range)
            if df_sym is None or df_sym.empty or len(df_sym) < 60:
                errors.append((sym, "資料不足（<60 交易日）"))
                continue
            df_sym = compute_indicators(df_sym)
            foreign_3d = trust_3d = None
            if sym.endswith(".TW") or sym.endswith(".TWO"):
                try:
                    f, t = fetch_chip_net_series(sym)
                    if f is not None and not f.empty:
                        foreign_3d = float(f.tail(3).sum())
                    if t is not None and not t.empty:
                        trust_3d = float(t.tail(3).sum())
                except Exception:
                    pass
            result = run_backtest(
                df_sym,
                strategy=strategy,
                symbol=sym,
                hold_days=hold_days,
                trailing_stop_pct=trailing_stop_pct,
                exit_ema_window=exit_ema_window,
                win_threshold_pct=win_threshold_pct,
                foreign_3d_net=foreign_3d,
                trust_3d_net=trust_3d,
                turn_cfg=turn_cfg,
            )
            if result is None:
                errors.append((sym, "回測無結果"))
                continue
            results[sym] = result
            dfs_out[sym] = df_sym
        progress.empty()

        st.session_state["backtest_batch_results"] = results
        st.session_state["backtest_batch_dfs"] = dfs_out
        st.session_state["backtest_batch_errors"] = errors
        st.session_state["backtest_last_symbols"] = [s for s in symbols if s in results]
        st.session_state["backtest_params"] = {
            "strategy": strategy,
            "hold_days": hold_days,
            "trailing_stop_pct": trailing_stop_pct,
            "exit_ema_window": exit_ema_window,
        }
        st.session_state.pop("backtest_result", None)
        st.session_state.pop("backtest_df", None)
        st.session_state.pop("stress_result", None)
        st.rerun()

    batch_results: dict[str, BacktestSummary] = st.session_state.get(
        "backtest_batch_results", {}
    ) or {}
    batch_dfs: dict[str, pd.DataFrame] = st.session_state.get("backtest_batch_dfs", {}) or {}
    params = st.session_state.get("backtest_params", {})
    batch_errors: list = st.session_state.get("backtest_batch_errors", [])

    if batch_results:
        order = st.session_state.get("backtest_last_symbols") or list(batch_results.keys())
        order = [s for s in order if s in batch_results]

        st.markdown("---")
        st.markdown("### 4️⃣ 回測結果（多檔彙總）")
        cmp_rows = []
        for sym in order:
            r = batch_results[sym]
            cmp_rows.append(
                {
                    "symbol": sym,
                    "訊號數": r.n_signals,
                    "勝率": f"{r.win_rate:.1%}" if r.n_signals > 0 else "N/A",
                    "期望值%": round(r.expectancy, 2),
                    "平均報酬%": round(r.avg_return_pct, 2),
                    "平均持有天": round(r.avg_holding_days, 1),
                    "最大回撤%": round(r.max_drawdown_pct, 1),
                    "PF": round(r.profit_factor, 2) if r.profit_factor < 100 else None,
                }
            )
        st.dataframe(pd.DataFrame(cmp_rows), hide_index=True, width="stretch")
        if batch_errors:
            with st.expander(f"未納入彙總（{len(batch_errors)}）", expanded=False):
                st.dataframe(
                    pd.DataFrame(batch_errors, columns=["symbol", "原因"]),
                    hide_index=True,
                    width="stretch",
                )

        pick = st.selectbox(
            "檢視單檔細節（圖表、實戰指引、壓測）",
            options=order,
            index=0,
            key="bt_batch_detail_pick",
        )
        result = batch_results[pick]
        _df_detail = batch_dfs.get(pick)
        if _df_detail is None or _df_detail.empty:
            _df_detail = df_primary if df_primary is not None else pd.DataFrame()
        _render_backtest_result(
            result,
            win_threshold_pct,
            _df_detail,
            params,
            heading=f"### 單檔：{pick}",
        )
    else:
        st.info("設定參數後點擊「執行批次回測」。")
        if batch_errors and not batch_results:
            with st.expander("上次執行錯誤", expanded=True):
                st.dataframe(
                    pd.DataFrame(batch_errors, columns=["symbol", "原因"]),
                    hide_index=True,
                    width="stretch",
                )


def _render_backtest_result(
    result: BacktestSummary,
    win_threshold_pct: float,
    df: pd.DataFrame,
    params: dict,
    *,
    heading: str = "### 4️⃣ 回測結果",
):
    """渲染回測結果（分頁：戰力評估 + 實戰指引）"""
    st.markdown("---")
    st.markdown(heading)

    ts_pct = float(getattr(result, "trailing_stop_pct", None) or params.get("trailing_stop_pct", 0) or 0)

    tab1, tab2 = st.tabs(["📊 戰力評估", "🏁 實戰指引"])

    with tab1:
        _render_power_metrics(result, win_threshold_pct, params)

    with tab2:
        _render_live_guidance(result, df, ts_pct)

    # 策略實戰連動診斷（回測參數與下車指南一體化）
    st.markdown("---")
    _render_strategy_live_diagnostic(result, df, ts_pct)

    # 5️⃣ 風險壓力測試（系統驗證）
    st.markdown("---")
    st.markdown("### 5️⃣ 風險壓力測試（系統驗證）")
    hold_days = int(params.get("hold_days", 10) or 10)
    trailing_stop_pct = float(params.get("trailing_stop_pct", 0) or 0)

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
