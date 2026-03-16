import pandas as pd
import streamlit as st
import copy

from turn_check_engine import (
    get_all_turn_statuses,
    load_turn_config,
    run_turn_check,
    validate_turn_cfg,
)
from position_advice_ui import render_position_advice_panel


def _status_badge(status: str) -> str:
    # Streamlit markdown 小膠囊
    if status == "ALLOW":
        return "🟢 **ALLOW**"
    if status == "WATCH":
        return "🟡 **WATCH**"
    return "🔴 **BLOCK**"


def _status_label(status: str) -> str:
    if status == "ALLOW":
        return "🟢 ALLOW"
    if status == "WATCH":
        return "🟡 WATCH"
    return "🔴 BLOCK"


def _ensure_turn_cfg_schema(cfg: dict) -> dict:
    """
    UI 層保護：避免 cfg 缺 key 就 crash。
    只補「常用且 UI 會讀到」的欄位，並保留既有值。
    """
    if not isinstance(cfg, dict):
        cfg = {}

    cfg.setdefault("mode_default", "bottom")

    cfg.setdefault("structure", {})
    cfg["structure"].setdefault("lookback", 20)
    cfg["structure"].setdefault("swing_window", 5)
    cfg["structure"].setdefault("support_buffer", 0.005)

    cfg.setdefault("momentum", {})
    cfg["momentum"].setdefault("rsi_period", 14)
    cfg["momentum"].setdefault("div_lookback", 5)
    cfg["momentum"].setdefault("rsi_oversold", 30)
    cfg["momentum"].setdefault("rsi_overbought", 70)

    cfg.setdefault("volume", {})
    cfg["volume"].setdefault("ma_window", 20)
    cfg["volume"].setdefault("compare_window", 4)
    cfg["volume"].setdefault("dry_up_ratio", 0.6)
    cfg["volume"].setdefault("top_range_window", 5)
    cfg["volume"].setdefault("top_drop_mult", 1.5)

    cfg.setdefault("chip", {})
    cfg["chip"].setdefault("foreign_days", 3)
    cfg["chip"].setdefault("trust_days", 3)
    cfg["chip"].setdefault("require_both", False)

    cfg.setdefault("bias", {})
    cfg["bias"].setdefault("ma_window", 20)
    cfg["bias"].setdefault("overheat_pct_top", 8.0)

    cfg.setdefault("decision", {})
    cfg["decision"].setdefault("bottom", {})
    cfg["decision"]["bottom"].setdefault("allow_score", 3)
    cfg["decision"]["bottom"].setdefault("watch_score", 2)
    cfg["decision"].setdefault("top", {})
    cfg["decision"]["top"].setdefault("block_score", 4)
    cfg["decision"]["top"].setdefault("watch_score", 2)

    cfg.setdefault("top_shield", {})
    cfg["top_shield"].setdefault("enabled", True)
    cfg["top_shield"].setdefault("ma_window", 5)

    cfg.setdefault("top_trend_filter", {})
    cfg["top_trend_filter"].setdefault("enabled", False)
    cfg["top_trend_filter"].setdefault("ma_window", 20)
    cfg["top_trend_filter"].setdefault("block_score_add", 1)

    return cfg


def _deep_update(base: dict, override: dict) -> dict:
    """
    深層合併（override 覆蓋 base；dict 會遞迴合併）。
    用途：cfg_override 只給部分 key 時，仍保留 JSON 內的其餘設定。
    """
    if not isinstance(base, dict):
        base = {}
    if not isinstance(override, dict):
        return base
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base.get(k, {}), v)
        else:
            base[k] = v
    return base


@st.cache_data(show_spinner=False, ttl=60)
def _run_turn_check_cached(
    df: pd.DataFrame,
    *,
    foreign_3d_net: float | None,
    trust_3d_net: float | None,
    mode: str,
    cfg: dict,
) -> dict:
    # 快取：降低（both 模式）重複運算成本
    return run_turn_check(
        df,
        foreign_3d_net=foreign_3d_net,
        trust_3d_net=trust_3d_net,
        mode=mode,
        cfg=cfg,
    )


@st.cache_data(show_spinner=False, ttl=60)
def _get_turn_statuses_cached(
    df: pd.DataFrame,
    *,
    foreign_3d_net: float | None,
    trust_3d_net: float | None,
    mode: str,
    cfg: dict,
) -> pd.Series:
    return get_all_turn_statuses(
        df,
        mode=mode,
        cfg=cfg,
        foreign_3d_net=foreign_3d_net,
        trust_3d_net=trust_3d_net,
    )


def _explain(status: str, mode: str, detail: dict) -> str:
    c = detail["conditions"]
    hits = [k for k, v in c.items() if v]
    miss = [k for k, v in c.items() if not v]

    if mode == "bottom":
        base = "底部反轉 Gate：只回答「能不能開始盯」"
    else:
        base = "頭部反轉 Gate：偏保守，用來擋追高風險"

    return (
        f"{base}\n\n"
        f"- 命中：{', '.join(hits) if hits else '（無）'}\n"
        f"- 未命中：{', '.join(miss) if miss else '（無）'}\n"
        f"- 結論：{status}\n"
    )


def explain_chip(foreign_3d_net, trust_3d_net):
    if foreign_3d_net is None or trust_3d_net is None:
        return "籌碼資料不足"

    if trust_3d_net > 0 and foreign_3d_net < 0:
        return "投信接、外資賣（常見於底部或區間）"
    if trust_3d_net > 0 and foreign_3d_net >= 0:
        return "投信＋外資同步偏多"
    if trust_3d_net < 0 and foreign_3d_net < 0:
        return "投信＋外資同步偏空（高風險）"
    if trust_3d_net <= 0 and foreign_3d_net >= 0:
        return "外資撐、投信觀望"

    return "籌碼中性"


def render_turn_check_panel(
    df: pd.DataFrame,
    *,
    foreign_3d_net: float | None = None,
    trust_3d_net: float | None = None,
    mode: str | None = None,
    symbol: str | None = None,
    position_avg_cost: float | None = None,
    position_qty: int | None = None,
    position_exit_style: str | None = None,
    show_position_settings: bool = True,
    show_exit_guide: bool = True,
    config_path: str = "turn_check_config.json",
    cfg_override: dict | None = None,
    allow_param_tuning: bool = True,
):
    st.subheader("TURN CHECK（反轉 Gate）")

    base_cfg = load_turn_config(config_path)
    if isinstance(cfg_override, dict):
        cfg = _deep_update(copy.deepcopy(base_cfg), copy.deepcopy(cfg_override))
    else:
        cfg = copy.deepcopy(base_cfg)
    cfg = _ensure_turn_cfg_schema(cfg)

    # 參數健檢（不阻擋運行，但會提示可能的矛盾/異常）
    try:
        cfg_warns = validate_turn_cfg(cfg)
    except Exception:
        cfg_warns = []
    if cfg_warns:
        with st.expander("⚠️ 參數檢查提醒（建議修正）", expanded=False):
            st.markdown("\n".join([f"- {w}" for w in cfg_warns]))

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if mode is None:
            mode = st.selectbox("模式", ["both", "bottom", "top"], index=0)
        else:
            if mode not in ["bottom", "top", "both"]:
                mode = cfg.get("mode_default", "bottom")
            if mode == "both":
                st.caption("模式：bottom + top（與圖表設定同步）")
            else:
                st.caption(f"模式：{mode}（與圖表設定同步）")
    with col2:
        st.caption("分數越高代表條件命中越多")
    with col3:
        st.caption("輸出：ALLOW / WATCH / BLOCK")

    # ---------------- 持倉與風險設定（下車指南用） ----------------
    # 以 symbol 當作 session key，讓每檔股票各自記住
    safe_symbol = symbol or "default"
    safe_symbol = "".join([c if c.isalnum() else "_" for c in safe_symbol])
    key_avg = f"pos_avg_cost_{safe_symbol}"
    key_qty = f"pos_qty_{safe_symbol}"
    key_style = f"pos_exit_style_{safe_symbol}"

    # 從外部傳入（例如放在圖表設定上方）優先；否則讀取 session_state；最後再用預設值補齊
    pos_avg_cost = (
        float(position_avg_cost)
        if position_avg_cost is not None
        else float(st.session_state.get(key_avg, 0.0) or 0.0)
    )
    pos_qty = (
        int(position_qty)
        if position_qty is not None
        else int(st.session_state.get(key_qty, 0) or 0)
    )
    pos_exit_style = (
        str(position_exit_style)
        if position_exit_style is not None
        else str(st.session_state.get(key_style, "波段守五日線") or "波段守五日線")
    )

    if show_position_settings:
        with st.expander("💼 持倉與風險設定（在這裡輸入均價）", expanded=True):
            st.caption("這一區改成「套用才更新」：你調數字不會立刻重跑整頁。")
            with st.form(key=f"pos_settings_form_{safe_symbol}"):
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    pos_avg_cost = st.number_input(
                        "持股均價（計算損益用）",
                        min_value=0.0,
                        value=float(pos_avg_cost or 0.0),
                        step=0.1,
                        key=key_avg,
                        format="%.2f",
                    )
                with col_p2:
                    pos_qty = st.number_input(
                        "持有股數",
                        min_value=0,
                        value=int(pos_qty or 0),
                        step=1,
                        key=key_qty,
                    )
                with col_p3:
                    style_options = ["波段守五日線", "積極分批止盈", "長線守月線"]
                    default_style = str(pos_exit_style or style_options[0])
                    style_index = (
                        style_options.index(default_style)
                        if default_style in style_options
                        else 0
                    )
                    pos_exit_style = st.selectbox(
                        "偏好下車風格",
                        style_options,
                        index=style_index,
                        key=key_style,
                    )

                applied = st.form_submit_button("✅ 套用設定（更新下車指令）")
            if applied:
                st.success("已套用持倉設定（下車指令已更新）")
    else:
        st.caption("持倉均價/股數/下車風格：已移到「圖表設定」上方的持倉設定區塊。")

    if allow_param_tuning:
        with st.expander("參數（可調）", expanded=False):
            # 你要最常調的那幾個：lookback / div_lookback / compare_window
            cfg["structure"]["lookback"] = st.slider(
                "Structure lookback",
                10,
                80,
                int(cfg["structure"]["lookback"]),
                1,
            )
            cfg["structure"]["support_buffer"] = st.slider(
                "Support buffer（避免蹭底/貼頂；0.005=0.5%）",
                0.0,
                0.02,
                float(cfg["structure"].get("support_buffer", 0.0) or 0.0),
                0.001,
            )
            cfg["momentum"]["div_lookback"] = st.slider(
                "Divergence lookback",
                3,
                20,
                int(cfg["momentum"]["div_lookback"]),
                1,
            )
            cfg["momentum"]["rsi_oversold"] = st.slider(
                "RSI oversold（底部）",
                10,
                40,
                int(cfg["momentum"].get("rsi_oversold", 30) or 30),
                1,
            )
            cfg["momentum"]["rsi_overbought"] = st.slider(
                "RSI overbought（頭部）",
                60,
                90,
                int(cfg["momentum"].get("rsi_overbought", 70) or 70),
                1,
            )
            cfg["volume"]["ma_window"] = st.slider(
                "Volume MA window（窒息量均量）",
                5,
                60,
                int(cfg["volume"].get("ma_window", 20) or 20),
                1,
            )
            cfg["volume"]["compare_window"] = st.slider(
                "Volume compare window",
                2,
                15,
                int(cfg["volume"]["compare_window"]),
                1,
            )
            cfg["volume"]["dry_up_ratio"] = st.slider(
                "Dry-up ratio（窒息量；0.6=低於均量60%）",
                0.3,
                1.0,
                float(cfg["volume"].get("dry_up_ratio", 0.6) or 0.6),
                0.05,
            )
            cfg["volume"]["top_range_window"] = st.slider(
                "Top｜跌幅參考窗（平均震幅 N 日）",
                2,
                30,
                int(cfg["volume"].get("top_range_window", 5) or 5),
                1,
            )
            cfg["volume"]["top_drop_mult"] = st.slider(
                "Top｜大跌確認倍數（跌幅 ≥ 平均震幅 × X；0=關閉）",
                0.0,
                3.0,
                float(cfg["volume"].get("top_drop_mult", 1.5) or 1.5),
                0.1,
            )
            cfg["chip"]["require_both"] = st.checkbox(
                "Require both（需土洋同向才算籌碼命中）",
                value=bool(cfg["chip"].get("require_both", False)),
            )
            cfg["bias"]["ma_window"] = st.slider(
                "Bias MA window（乖離參考均線）",
                5,
                120,
                int(cfg["bias"].get("ma_window", 20) or 20),
                1,
            )
            cfg["bias"]["overheat_pct_top"] = st.slider(
                "Bias overheat（top +1分，%）",
                0.0,
                20.0,
                float(cfg["bias"].get("overheat_pct_top", 8.0) or 8.0),
                0.5,
            )
            cfg["top_shield"]["enabled"] = st.checkbox(
                "Top 防賣飛：BLOCK 需跌破均線（在 EMA 上方降級為 WATCH）",
                value=bool(cfg["top_shield"].get("enabled", True)),
            )
            cfg["top_shield"]["ma_window"] = st.slider(
                "Top 防守均線（EMA window）",
                3,
                20,
                int(cfg["top_shield"].get("ma_window", 5) or 5),
                1,
            )
            cfg["top_trend_filter"]["enabled"] = st.checkbox(
                "Top 趨勢過濾：多頭時 BLOCK 更嚴格（Close > EMA → block_score +N）",
                value=bool(cfg.get("top_trend_filter", {}).get("enabled", False)),
            )
            cfg["top_trend_filter"]["ma_window"] = st.slider(
                "Top 趨勢均線（EMA window）",
                10,
                60,
                int(cfg.get("top_trend_filter", {}).get("ma_window", 20) or 20),
                1,
            )
            cfg["top_trend_filter"]["block_score_add"] = st.slider(
                "Top 趨勢加嚴（block_score + N）",
                0,
                2,
                int(cfg.get("top_trend_filter", {}).get("block_score_add", 1) or 1),
                1,
            )

            st.info("這裡改的是「本次運行的參數」。要永久保存到 JSON，再跟我說我幫你加保存按鈕。")

    foreign_text = f"{foreign_3d_net:+.0f}" if foreign_3d_net is not None else "NA"
    trust_text = f"{trust_3d_net:+.0f}" if trust_3d_net is not None else "NA"
    st.caption(
        f"籌碼解讀：{explain_chip(foreign_3d_net, trust_3d_net)} "
        f"(外資3日={foreign_text}, 投信3日={trust_text})"
    )

    def _render_one(result: dict, mode_label: str, mode_key: str):
        status = str(result.get("status", "BLOCK"))
        score = result.get("score", 0)
        denom = 5 if mode_key == "top" else 4
        title = f"{mode_label}｜{_status_label(status)}｜Score {score} / {denom}"

        # 標題視覺（狀態色彩一致）
        if status == "ALLOW":
            st.success(title)
        elif status == "WATCH":
            st.warning(title)
        else:
            st.error(title)

        # 決策輔助：距離上次訊號天數（狀態維持）與距離上次「最佳狀態」訊號
        try:
            s = _get_turn_statuses_cached(
                df,
                foreign_3d_net=foreign_3d_net,
                trust_3d_net=trust_3d_net,
                mode=mode_key,
                cfg=cfg,
            )
        except Exception:
            s = pd.Series(dtype=object)

        days_in_state = None
        last_change_idx = None
        last_change_text = None
        days_since_best = None
        last_best_idx = None
        best_status = "ALLOW" if mode_key == "bottom" else "BLOCK"
        if isinstance(s, pd.Series) and not s.empty:
            s = s.astype(str)
            if len(s) >= 2:
                change_mask = (s != s.shift(1)).fillna(False)
                try:
                    change_mask.iloc[0] = False
                except Exception:
                    pass
                change_pos = [i for i, v in enumerate(change_mask.values) if bool(v)]
                if change_pos:
                    last_pos = int(change_pos[-1])
                    days_in_state = int(len(s) - 1 - last_pos)
                    last_change_idx = s.index[last_pos]
                    old_s = str(s.iloc[last_pos - 1]) if last_pos - 1 >= 0 else ""
                    new_s = str(s.iloc[last_pos])
                    last_change_text = f"{old_s}→{new_s}" if old_s else f"→{new_s}"
                else:
                    days_in_state = int(len(s) - 1)
                    last_change_idx = s.index[0]
                    last_change_text = "（區間內無切換）"

            best_mask = ((s == best_status) & (s.shift(1) != best_status)).fillna(False)
            try:
                best_mask.iloc[0] = False
            except Exception:
                pass
            best_pos = [i for i, v in enumerate(best_mask.values) if bool(v)]
            if best_pos:
                last_best_pos = int(best_pos[-1])
                days_since_best = int(len(s) - 1 - last_best_pos)
                last_best_idx = s.index[last_best_pos]

        # 乖離率（Bias）資訊（top 模式會加分）
        bias_cfg = cfg.get("bias", {}) if isinstance(cfg, dict) else {}
        bias_ma_window = bias_cfg.get("ma_window", 20)
        try:
            bias_ma_window = int(bias_ma_window) if bias_ma_window is not None else None
        except Exception:
            bias_ma_window = None
        bias_thr = bias_cfg.get("overheat_pct_top")
        bias_pct = None
        try:
            ind = result.get("indicators") or {}
            if ind.get("bias_pct") is not None:
                bias_pct = float(ind.get("bias_pct"))
        except Exception:
            bias_pct = None

        c_age, c_sig, c_bias = st.columns(3)
        with c_age:
            st.metric(
                "狀態維持（交易日）",
                f"{days_in_state}" if days_in_state is not None else "NA",
            )
            if last_change_idx is not None:
                st.caption(f"自 {str(last_change_idx)[:10]}｜{last_change_text}")
        with c_sig:
            st.metric(
                f"距離上次 {best_status} 訊號",
                f"{days_since_best}" if days_since_best is not None else "NA",
            )
            if last_best_idx is not None:
                st.caption(f"{str(last_best_idx)[:10]}")
        with c_bias:
            ma_txt = f"MA{bias_ma_window}" if bias_ma_window else "MA?"
            if bias_pct is None:
                st.metric(f"乖離率 Bias（{ma_txt}）", "NA")
            else:
                st.metric(f"乖離率 Bias（{ma_txt}）", f"{bias_pct:.1f}%")
            if mode_key == "top" and bias_thr is not None:
                try:
                    st.caption(f"過熱門檻：{float(bias_thr):.1f}%（觸發時 top +1 分）")
                except Exception:
                    st.caption("過熱門檻：NA（觸發時 top +1 分）")

        # 使用者體感：今天剛轉 WATCH vs 已經 WATCH 很久
        if status == "WATCH" and days_in_state == 0:
            st.caption("提示：狀態剛轉 WATCH（今天第一天），警覺性可拉高。")
        if status == "WATCH" and days_in_state is not None and days_in_state >= 10:
            st.caption("提示：WATCH 已維持較久，訊號偏舊，建議搭配均線/量能再判讀。")

        # 分項條件（用 dataframe 更緊湊且可滾動）
        conds = result.get("conditions") or {}
        rows = [{"訊號項目": k, "是否命中": ("✅" if bool(v) else "❌")} for k, v in conds.items()]
        with st.expander("分項條件（structure / momentum / volume / chip / bias）", expanded=False):
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.info(_explain(status, mode_key, result))
        with st.expander(f"Debug detail（{mode_key}）", expanded=False):
            st.json(result)

    def _render_position_exit_guide(
        *,
        result_bottom: dict,
        result_top: dict,
        avg_cost: float,
        qty: int,
        exit_style: str,
    ):
        """
        持倉診斷 + 下車指南（不寫死個人持股；由使用者輸入均價/股數/風格）
        """
        with st.expander("🏁 持倉診斷與下車指南", expanded=True):
            st.caption(
                f"目前設定：均價={float(avg_cost or 0.0):.2f}｜股數={int(qty or 0)}｜風格={exit_style}"
            )
            render_position_advice_panel(
                df,
                symbol=symbol,
                avg_cost=float(avg_cost or 0.0),
                qty=int(qty or 0),
                exit_style=str(exit_style),
                bottom_result=result_bottom,
                top_result=result_top,
            )

    if mode == "both":
        result_bottom = _run_turn_check_cached(
            df,
            foreign_3d_net=foreign_3d_net,
            trust_3d_net=trust_3d_net,
            mode="bottom",
            cfg=cfg,
        )
        result_top = _run_turn_check_cached(
            df,
            foreign_3d_net=foreign_3d_net,
            trust_3d_net=trust_3d_net,
            mode="top",
            cfg=cfg,
        )
        c1, c2 = st.columns(2)
        with c1:
            _render_one(result_bottom, "bottom（較佳進場）", "bottom")
        with c2:
            _render_one(result_top, "top（較佳出場／風險）", "top")
        if show_exit_guide:
            _render_position_exit_guide(
                result_bottom=result_bottom,
                result_top=result_top,
                avg_cost=float(pos_avg_cost or 0.0),
                qty=int(pos_qty or 0),
                exit_style=str(pos_exit_style),
            )
        else:
            st.caption("🏁 下車指南：已移到「圖表設定」上方。")

        with st.expander("架構示意圖（TURN CHECK 流程）", expanded=False):
            st.markdown(
                """
```text
OHLCV + RSI (+ Foreign_Net/Trust_Net)
  ↓
_compute_turn_conditions_vectorized()
  ├─ structure（底: 守支撐｜頂: 不創高）
  ├─ momentum（RSI 背離 / 超賣反彈 / 超買轉弱）
  ├─ volume（量縮/攻擊量｜跌有量）
  └─ chip（外資/投信連續性/同向性）
  └─ bias（乖離率；top 過熱時 +1 分）
  ↓
score = bottom 0~4；top 0~5（含 bias）
  ↓
turn_check_decision()（含「結構 gating」）
  ↓
status = ALLOW / WATCH / BLOCK
```
                """
            )
        return {"mode": "both", "bottom": result_bottom, "top": result_top}

    # single mode
    if mode not in ["bottom", "top"]:
        mode = cfg.get("mode_default", "bottom")
    result = _run_turn_check_cached(
        df,
        foreign_3d_net=foreign_3d_net,
        trust_3d_net=trust_3d_net,
        mode=mode,
        cfg=cfg,
    )
    _render_one(result, f"{mode}（單一模式）", mode)
    # 單一模式也補上完整下車指南（會自動補算另一側模式）
    if mode == "bottom":
        result_bottom = result
        result_top = _run_turn_check_cached(
            df,
            foreign_3d_net=foreign_3d_net,
            trust_3d_net=trust_3d_net,
            mode="top",
            cfg=cfg,
        )
    else:
        result_top = result
        result_bottom = _run_turn_check_cached(
            df,
            foreign_3d_net=foreign_3d_net,
            trust_3d_net=trust_3d_net,
            mode="bottom",
            cfg=cfg,
        )
    if show_exit_guide:
        _render_position_exit_guide(
            result_bottom=result_bottom,
            result_top=result_top,
            avg_cost=float(pos_avg_cost or 0.0),
            qty=int(pos_qty or 0),
            exit_style=str(pos_exit_style),
        )
    else:
        st.caption("🏁 下車指南：已移到「圖表設定」上方。")
    return result
