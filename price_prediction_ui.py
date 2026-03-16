"""
機率預測模組 UI：GBM 蒙地卡羅、斐波那契、波動率區間
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from price_prediction import (
    fibonacci_from_df,
    gbm_monte_carlo_full,
    volatility_range_from_df,
)


def render_price_prediction_panel(
    df: pd.DataFrame,
    *,
    symbol: str | None = None,
    current_price: float | None = None,
):
    """
    機率預測面板：GBM 蒙地卡羅、斐波那契、波動率區間
    """
    if df is None or df.empty:
        st.info("資料不足：無法進行機率預測")
        return

    st.markdown("### 📊 機率預測模組")
    st.caption(
        "數學模型：GBM 蒙地卡羅、斐波那契回撤/擴展、波動率區間。"
        "僅供參考，市場具隨機性，不構成投資建議。"
    )

    # 參數設定
    col1, col2, col3 = st.columns(3)
    with col1:
        n_days = st.slider("預測天數", 5, 60, 21, help="模擬未來 N 個交易日")
    with col2:
        n_sim = st.selectbox(
            "蒙地卡羅模擬次數",
            [1000, 5000, 10000, 20000],
            index=2,
            help="模擬次數越多，分佈越穩定",
        )
    with col3:
        fib_lookback = st.slider("斐波那契區間（日）", 20, 120, 60)

    close_last = float(pd.to_numeric(df["Close"].iloc[-1], errors="coerce"))
    s0 = float(current_price) if current_price and current_price > 0 else close_last

    # -------- 1. GBM 蒙地卡羅 --------
    st.markdown("#### 1️⃣ GBM 蒙地卡羅模擬")
    gbm = gbm_monte_carlo_full(df, s0=s0, n_days=n_days, n_sim=n_sim)
    if gbm is not None:
        st.caption(
            f"參數：μ（年化報酬）≈ {gbm.mu*100:.2f}%｜σ（年化波動）≈ {gbm.sigma*100:.2f}%｜"
            f"起點 S₀ = {gbm.s0:.2f}"
        )
        # 畫 100 條路徑 + 95% 置信區間
        n_show = min(100, gbm.paths.shape[0])
        fig = go.Figure()
        for i in range(n_show):
            fig.add_trace(
                go.Scatter(
                    x=list(range(n_days)),
                    y=gbm.paths[i, :],
                    mode="lines",
                    line=dict(color="rgba(100,100,100,0.15)", width=1),
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=list(range(n_days)),
                y=gbm.percentile_5,
                mode="lines",
                line=dict(color="blue", width=2, dash="dash"),
                name="5% 分位",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(n_days)),
                y=gbm.percentile_50,
                mode="lines",
                line=dict(color="green", width=2),
                name="中位數（50%）",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(n_days)),
                y=gbm.percentile_95,
                mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                name="95% 分位",
            )
        )
        fig.update_layout(
            title="蒙地卡羅股價路徑（5% / 50% / 95% 分位）",
            xaxis_title="交易日",
            yaxis_title="股價",
            height=350,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        # 終值分佈摘要
        final_prices = gbm.paths[:, -1]
        p5_final = float(np.percentile(final_prices, 5))
        p50_final = float(np.percentile(final_prices, 50))
        p95_final = float(np.percentile(final_prices, 95))
        st.markdown(
            f"**{n_days} 日後股價分佈**：5% 分位 **{p5_final:.2f}**｜"
            f"中位數 **{p50_final:.2f}**｜95% 分位 **{p95_final:.2f}**"
        )
    else:
        st.warning("GBM 模擬失敗（資料不足）")

    # -------- 2. 斐波那契 --------
    st.markdown("#### 2️⃣ 斐波那契回撤與擴展")
    fib = fibonacci_from_df(df, lookback=fib_lookback)
    if fib is not None:
        st.caption(f"區間：高 {fib.high:.2f}｜低 {fib.low:.2f}｜幅度 {fib.range_val:.2f}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**回撤位（支撐）**")
            for r, price in sorted(fib.retracement.items()):
                st.write(f"- {r*100:.1f}%：**{price:.2f}**")
        with c2:
            st.markdown("**擴展位（目標）**")
            for r, price in sorted(fib.extension.items()):
                st.write(f"- {r}x：**{price:.2f}**")
    else:
        st.warning("斐波那契計算失敗")

    # -------- 3. 波動率區間 --------
    st.markdown("#### 3️⃣ 波動率區間（約 68% 機率）")
    vol_range = volatility_range_from_df(df, stock_price=s0, dte=n_days)
    if vol_range is not None:
        st.caption(
            f"公式：S ± (S × σ × √(DTE/365))｜σ ≈ {vol_range.sigma*100:.2f}%｜"
            f"DTE = {vol_range.dte} 日"
        )
        st.markdown(
            f"**區間**：**{vol_range.lower:.2f}** ～ **{vol_range.upper:.2f}**"
        )
        if s0 > 0:
            pct_low = (vol_range.lower / s0 - 1) * 100
            pct_high = (vol_range.upper / s0 - 1) * 100
            st.caption(f"相對現價：{pct_low:+.1f}% ～ {pct_high:+.1f}%")
    else:
        st.warning("波動率區間計算失敗")
