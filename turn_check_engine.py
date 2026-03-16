import json
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

import pandas as pd
import numpy as np
import copy


# -------------------------
# Config loader
# -------------------------
def load_turn_config(path: str = "turn_check_config.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Checks
# -------------------------
def structure_bottom_check(
    df: pd.DataFrame,
    lookback: int = 20,
    support_buffer: float = 0.0,
) -> bool:
    if len(df) < lookback + 2:
        return False
    recent_low = df["Low"].iloc[-lookback:-1].min()
    last_low = df["Low"].iloc[-1]
    # support_buffer: 避免「只是在底部蹭底」就過關（要求更明顯的抬高低點）
    # 例如 0.005 代表低點需高於區間低點 0.5% 以上
    return last_low >= (recent_low * (1.0 + float(support_buffer or 0.0)))


def structure_top_check(
    df: pd.DataFrame,
    lookback: int = 20,
    support_buffer: float = 0.0,
) -> bool:
    if len(df) < lookback + 2:
        return False
    recent_high = df["High"].iloc[-lookback:-1].max()
    last_high = df["High"].iloc[-1]
    # support_buffer: 避免「只是貼著區間高點」就被判成轉弱（要求更明顯的走弱／無法創高）
    return last_high <= (recent_high * (1.0 - float(support_buffer or 0.0)))


def momentum_divergence_check(
    close: pd.Series,
    rsi: pd.Series,
    div_lookback: int = 5,
    mode: str = "bottom",
    rsi_oversold: Optional[float] = None,
    rsi_overbought: Optional[float] = None,
) -> bool:
    if len(close) < div_lookback + 2 or len(rsi) < div_lookback + 2:
        return False

    price_now = close.iloc[-1]
    price_prev = close.iloc[-(div_lookback + 1)]
    rsi_now = rsi.iloc[-1]
    rsi_prev = rsi.iloc[-(div_lookback + 1)]

    if mode == "bottom":
        ok = (price_now <= price_prev) and (rsi_now > rsi_prev)
        if rsi_oversold is not None:
            thr = float(rsi_oversold)
            ok = ok and ((rsi_now <= thr) or (rsi_prev <= thr))
        return ok
    else:
        ok = (price_now >= price_prev) and (rsi_now < rsi_prev)
        if rsi_overbought is not None:
            thr = float(rsi_overbought)
            ok = ok and ((rsi_now >= thr) or (rsi_prev >= thr))
        return ok


def volume_turn_check(
    df: pd.DataFrame,
    compare_window: int = 4,
    mode: str = "bottom",
    ma_window: Optional[int] = None,
    dry_up_ratio: Optional[float] = None,
    top_range_window: Optional[int] = None,
    top_drop_mult: Optional[float] = None,
) -> bool:
    # compare_window: 平均前N天量（不含今天）
    if len(df) < compare_window + 3:
        return False

    vol_now = df["Volume"].iloc[-1]
    vol_prev = df["Volume"].iloc[-(compare_window + 1):-1].mean()

    price_change = df["Close"].iloc[-1] - df["Close"].iloc[-2]

    if mode == "bottom":
        # 跌不動（轉紅）或 量縮不再殺；可選擇加入「窒息量」(dry_up_ratio)
        dry_ok = False
        if ma_window and dry_up_ratio is not None and len(df) >= int(ma_window) + 2:
            vol_ma = df["Volume"].iloc[-(int(ma_window) + 1):-1].mean()
            if pd.notna(vol_ma):
                dry_ok = vol_now <= (float(vol_ma) * float(dry_up_ratio))
        return (price_change >= 0) or (vol_now < vol_prev) or dry_ok
    else:
        # 漲無量/動能弱 + 跌有量（出貨味）
        # 強化：避免「小跌小量」就被當成出貨（可選：跌幅需 > 過去 N 日平均震幅 * X）
        big_drop_ok = True
        try:
            if top_range_window is not None and top_drop_mult is not None:
                rw = int(top_range_window)
                mult = float(top_drop_mult)
                if rw >= 2 and mult > 0 and len(df) >= (rw + 2):
                    prev_close = float(pd.to_numeric(df["Close"].iloc[-2], errors="coerce"))
                    curr_close = float(pd.to_numeric(df["Close"].iloc[-1], errors="coerce"))
                    if np.isfinite(prev_close) and prev_close > 0 and np.isfinite(curr_close):
                        drop_pct = (prev_close - curr_close) / prev_close
                        drop_pct = max(0.0, float(drop_pct))
                        hi = pd.to_numeric(df["High"].iloc[-(rw + 1) : -1], errors="coerce")
                        lo = pd.to_numeric(df["Low"].iloc[-(rw + 1) : -1], errors="coerce")
                        cl = pd.to_numeric(df["Close"].iloc[-(rw + 1) : -1], errors="coerce")
                        valid = cl.notna() & (cl != 0) & hi.notna() & lo.notna()
                        if bool(valid.any()):
                            rng = ((hi - lo).abs() / cl).where(valid)
                            avg_rng = float(rng.mean())
                            if np.isfinite(avg_rng) and avg_rng > 0:
                                big_drop_ok = drop_pct >= (avg_rng * mult)
        except Exception:
            big_drop_ok = True
        return (price_change < 0) and (vol_now > vol_prev) and bool(big_drop_ok)


def chip_turn_check(
    foreign_net: Optional[float],
    trust_net: Optional[float],
    mode: str = "bottom",
    require_both: bool = False,
) -> bool:
    foreign_net = foreign_net or 0.0
    trust_net = trust_net or 0.0

    if mode == "bottom":
        # 投信偏多 or 外資賣壓趨緩
        if require_both:
            # 土洋同買才算「確認」
            return (trust_net > 0) and (foreign_net > 0)
        return (trust_net > 0) or (foreign_net >= 0)
    else:
        # 外資偏空 or 投信轉空
        if require_both:
            # 土洋同賣才算「確認」
            return (foreign_net < 0) and (trust_net < 0)
        return (foreign_net < 0) or (trust_net < 0)


def turn_check_decision(
    conditions: Dict[str, bool],
    mode: str,
    decision_cfg: Dict[str, Any],
) -> Tuple[str, int]:
    score = sum(1 for v in conditions.values() if v)

    if mode == "bottom":
        allow_score = decision_cfg["bottom"]["allow_score"]
        watch_score = decision_cfg["bottom"]["watch_score"]
        if score >= allow_score:
            status = "ALLOW"
        elif score >= watch_score:
            status = "WATCH"
        else:
            status = "BLOCK"

        # 結構是基石：若結構未成立，即使其他條件很強，也不直接給 ALLOW
        if status == "ALLOW" and not conditions.get("structure", False):
            status = "WATCH"
        return status, score

    if mode == "top":
        block_score = decision_cfg["top"]["block_score"]
        watch_score = decision_cfg["top"]["watch_score"]
        if score >= block_score:
            status = "BLOCK"
        elif score >= watch_score:
            status = "WATCH"
        else:
            status = "ALLOW"

        # 結構是基石：若結構未成立，避免過早判定 BLOCK（轉為 WATCH）
        if status == "BLOCK" and not conditions.get("structure", False):
            status = "WATCH"
        return status, score

    raise ValueError("mode must be 'bottom' or 'top'")


def _pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_rsi_series(df: pd.DataFrame, *, rsi_period: int) -> pd.Series | None:
    rsi_col = _pick_first_existing_col(df, ["RSI", f"RSI{rsi_period}", "RSI14"])
    return df[rsi_col] if rsi_col else None


def _get_ema_series(df: pd.DataFrame, *, window: int) -> pd.Series:
    """
    取得 EMA 序列：
    - 優先使用 df 內既有欄位（例如 EMA5/EMA20）
    - 若不存在則用 Close 自行計算
    """
    try:
        w = max(1, int(window))
    except Exception:
        w = 5
    col = _pick_first_existing_col(df, [f"EMA{w}", f"ema{w}"])
    if col:
        return pd.to_numeric(df[col], errors="coerce")
    close_n = pd.to_numeric(df["Close"], errors="coerce")
    return close_n.ewm(span=w, adjust=False).mean()


def _compute_turn_conditions_vectorized(
    df: pd.DataFrame,
    *,
    mode: str,
    cfg: Dict[str, Any],
    foreign_3d_net: Optional[float] = None,
    trust_3d_net: Optional[float] = None,
) -> Dict[str, pd.Series]:
    """
    以向量化方式計算整段條件序列，並盡量對齊 run_turn_check 的 len 門檻行為。
    回傳：{"structure":..., "momentum":..., "volume":..., "chip":..., "bias":...}
    """
    if df is None or df.empty:
        empty = pd.Series(dtype=bool)
        return {
            "structure": empty,
            "momentum": empty,
            "volume": empty,
            "chip": empty,
            "bias": empty,
        }

    if mode not in ["bottom", "top"]:
        mode = cfg.get("mode_default", "bottom")

    structure_cfg = cfg.get("structure", {})
    lookback = int(structure_cfg.get("lookback", 20))
    support_buffer = float(structure_cfg.get("support_buffer", 0.0) or 0.0)

    momentum_cfg = cfg.get("momentum", {})
    div_lookback = int(momentum_cfg.get("div_lookback", 5))
    rsi_period = int(momentum_cfg.get("rsi_period", 14))
    rsi_oversold = momentum_cfg.get("rsi_oversold")
    rsi_overbought = momentum_cfg.get("rsi_overbought")

    volume_cfg = cfg.get("volume", {})
    compare_window = int(volume_cfg.get("compare_window", 4))
    ma_window = volume_cfg.get("ma_window")
    ma_window = int(ma_window) if ma_window is not None else None
    dry_up_ratio = volume_cfg.get("dry_up_ratio")
    top_range_window = volume_cfg.get("top_range_window", 5)
    top_drop_mult = volume_cfg.get("top_drop_mult", 1.5)
    try:
        top_range_window = int(top_range_window) if top_range_window is not None else None
    except Exception:
        top_range_window = None
    try:
        top_drop_mult = float(top_drop_mult) if top_drop_mult is not None else None
    except Exception:
        top_drop_mult = None

    chip_cfg = cfg.get("chip", {})
    trust_days = int(chip_cfg.get("trust_days", 3))
    foreign_days = int(chip_cfg.get("foreign_days", 3))
    require_both = bool(chip_cfg.get("require_both", False))

    bias_cfg = cfg.get("bias", {})
    bias_ma_window = bias_cfg.get("ma_window", 20)
    try:
        bias_ma_window = int(bias_ma_window) if bias_ma_window is not None else None
    except Exception:
        bias_ma_window = None
    bias_overheat_pct_top = bias_cfg.get("overheat_pct_top")

    # 數值化：避免遇到 object/字串導致 rolling/diff 出錯
    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")
    rsi = _get_rsi_series(df, rsi_period=rsi_period)
    if rsi is not None:
        rsi = pd.to_numeric(rsi, errors="coerce")

    pos = np.arange(len(df))

    # ---------------- structure ----------------
    struct_valid = pos >= (lookback + 1)  # len >= lookback+2
    struct_win = max(1, lookback - 1)  # match iloc[-lookback:-1] -> lookback-1 elements
    if mode == "bottom":
        recent_min = low.shift(1).rolling(struct_win).min()
        cond_struct = (low >= (recent_min * (1.0 + support_buffer))) & struct_valid
    else:
        recent_max = high.shift(1).rolling(struct_win).max()
        cond_struct = (high <= (recent_max * (1.0 - support_buffer))) & struct_valid

    # ---------------- momentum ----------------
    mom_div_valid = pos >= (div_lookback + 1)  # len >= div_lookback+2
    if rsi is None:
        cond_mom = pd.Series(False, index=df.index)
    else:
        price_prev = close.shift(div_lookback)
        rsi_prev = rsi.shift(div_lookback)
        if mode == "bottom":
            div_ok = (close <= price_prev) & (rsi > rsi_prev) & mom_div_valid
            bounce_ok = pd.Series(False, index=df.index)
            if rsi_oversold is not None:
                thr = float(rsi_oversold)
                bounce_ok = (rsi <= thr) & (rsi > rsi.shift(1)) & (pos >= 1)
            cond_mom = div_ok | bounce_ok
        else:
            div_ok = (close >= price_prev) & (rsi < rsi_prev) & mom_div_valid
            turn_ok = pd.Series(False, index=df.index)
            if rsi_overbought is not None:
                thr = float(rsi_overbought)
                turn_ok = (rsi >= thr) & (rsi < rsi.shift(1)) & (pos >= 1)
            cond_mom = div_ok | turn_ok

    # ---------------- volume ----------------
    vol_valid = pos >= (compare_window + 2)  # len >= compare_window+3
    vol_prev_mean = volume.shift(1).rolling(compare_window).mean()
    dry_ok = pd.Series(False, index=df.index)
    if ma_window and dry_up_ratio is not None:
        vol_ma = volume.shift(1).rolling(ma_window).mean()
        dry_ok = volume <= (vol_ma * float(dry_up_ratio))
    attack_ok = (close > close.shift(1)) & (volume > volume.shift(1))

    if mode == "bottom":
        # 量縮（不再殺）/窒息量/攻擊量
        cond_vol = ((volume < vol_prev_mean) | dry_ok | attack_ok) & vol_valid
    else:
        # 跌有量（出貨味）+（可選）大跌確認：跌幅需 >= 過去 N 日平均震幅 * X，避免小震盪被誤判
        big_drop_ok = pd.Series(True, index=df.index)
        if (
            top_range_window
            and int(top_range_window) >= 2
            and top_drop_mult is not None
            and float(top_drop_mult) > 0
        ):
            prev_close = close.shift(1)
            valid_prev = prev_close.notna() & (prev_close != 0)
            drop_pct = ((prev_close - close) / prev_close).where(valid_prev)
            drop_pct = drop_pct.where(drop_pct > 0, 0.0)
            range_pct = ((high - low).abs() / prev_close).where(valid_prev)
            avg_range = range_pct.shift(1).rolling(int(top_range_window)).mean()
            big_drop_ok = (
                valid_prev
                & avg_range.notna()
                & (avg_range > 0)
                & drop_pct.notna()
                & (drop_pct >= (avg_range * float(top_drop_mult)))
            )
        cond_vol = ((close.diff() < 0) & (volume > vol_prev_mean) & big_drop_ok) & vol_valid

    # ---------------- chip ----------------
    trust_col = _pick_first_existing_col(df, ["Trust_Net"])
    foreign_col = _pick_first_existing_col(df, ["Foreign_Net"])

    if trust_col:
        trust_s = pd.to_numeric(df[trust_col], errors="coerce").fillna(0.0)
    else:
        trust_s = None
    if foreign_col:
        foreign_s = pd.to_numeric(df[foreign_col], errors="coerce").fillna(0.0)
    else:
        foreign_s = None

    if mode == "bottom":
        if trust_s is None:
            trust_ok = pd.Series(bool((trust_3d_net or 0.0) > 0), index=df.index)
        else:
            tpos = trust_s > 0
            if trust_days <= 1:
                tconsec = tpos
            else:
                tconsec = (tpos.rolling(trust_days).sum() == trust_days).fillna(False)
            trust_ok = tconsec | tpos

        if foreign_s is None:
            foreign_ok_nonneg = pd.Series(bool((foreign_3d_net or 0.0) >= 0), index=df.index)
            foreign_ok_pos = pd.Series(bool((foreign_3d_net or 0.0) > 0), index=df.index)
        else:
            foreign_ok_nonneg = foreign_s >= 0
            foreign_ok_pos = foreign_s > 0

        cond_chip = (trust_ok & foreign_ok_pos) if require_both else (trust_ok | foreign_ok_nonneg)
    else:
        if trust_s is None:
            trust_ok = pd.Series(bool((trust_3d_net or 0.0) < 0), index=df.index)
        else:
            tneg = trust_s < 0
            if trust_days <= 1:
                tconsec = tneg
            else:
                tconsec = (tneg.rolling(trust_days).sum() == trust_days).fillna(False)
            trust_ok = tconsec | tneg

        if foreign_s is None:
            foreign_ok = pd.Series(bool((foreign_3d_net or 0.0) < 0), index=df.index)
        else:
            foreign_ok = foreign_s < 0

        cond_chip = (trust_ok & foreign_ok) if require_both else (trust_ok | foreign_ok)

    # ---------------- bias ----------------
    # 乖離率（Bias）= (Close - MA) / MA * 100%
    # 需求：Bias 過高時，在 top 模式加 1 分（以條件命中形式實作）
    cond_bias = pd.Series(False, index=df.index)
    if mode == "top" and bias_ma_window and bias_overheat_pct_top is not None:
        thr = float(bias_overheat_pct_top)
        close_n = pd.to_numeric(close, errors="coerce")
        ma = close_n.shift(1).rolling(int(bias_ma_window)).mean()
        valid = ma.notna() & (ma != 0) & close_n.notna()
        bias_pct = ((close_n - ma) / ma) * 100.0
        cond_bias = valid & (bias_pct >= thr)

    return {
        "structure": cond_struct.fillna(False).astype(bool),
        "momentum": cond_mom.fillna(False).astype(bool),
        "volume": cond_vol.fillna(False).astype(bool),
        "chip": cond_chip.fillna(False).astype(bool),
        "bias": cond_bias.fillna(False).astype(bool),
    }


def run_turn_check(
    df: pd.DataFrame,
    *,
    foreign_3d_net: Optional[float] = None,
    trust_3d_net: Optional[float] = None,
    mode: str = "bottom",
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Expect df columns: Open/High/Low/Close/Volume, plus RSI column if you already computed.
    If df doesn't have RSI, you should compute it in your pipeline before calling.
    """
    if cfg is None:
        cfg = load_turn_config()

    if df is None or df.empty:
        status = "BLOCK" if mode == "bottom" else "ALLOW"
        return {
            "mode": mode,
            "status": status,
            "score": 0,
            "conditions": {
                "structure": False,
                "momentum": False,
                "volume": False,
                "chip": False,
                "bias": False,
            },
            "inputs": {
                "foreign_3d_net": foreign_3d_net,
                "trust_3d_net": trust_3d_net,
            },
            "config_used": {},
        }

    conds_series = _compute_turn_conditions_vectorized(
        df,
        mode=mode,
        cfg=cfg,
        foreign_3d_net=foreign_3d_net,
        trust_3d_net=trust_3d_net,
    )
    conditions = {k: bool(v.iloc[-1]) for k, v in conds_series.items()}

    # ---------------- decision（支援 top 趨勢過濾：多頭時更難觸發 BLOCK） ----------------
    score = sum(1 for v in conditions.values() if bool(v))
    top_trend_enabled = False
    top_trend_ma_window = None
    top_trend_block_score_add = 0
    top_trend_ma_last = None
    top_trend_up_last = False
    top_trend_block_score_eff = None

    if mode == "bottom":
        status, _ = turn_check_decision(conditions, mode=mode, decision_cfg=cfg["decision"])
    else:
        decision_cfg = cfg.get("decision", {})
        block_score = int(decision_cfg.get("top", {}).get("block_score", 4))
        watch_score = int(decision_cfg.get("top", {}).get("watch_score", 2))

        # 趨勢過濾：Close > EMA(N) 時，BLOCK 門檻 +K（避免強勢股小震盪就被判成頂部）
        top_trend_cfg = cfg.get("top_trend_filter", {})
        top_trend_enabled = bool(top_trend_cfg.get("enabled", False))
        try:
            top_trend_ma_window = int(top_trend_cfg.get("ma_window", 20) or 20)
        except Exception:
            top_trend_ma_window = None
        try:
            top_trend_block_score_add = int(top_trend_cfg.get("block_score_add", 1) or 0)
        except Exception:
            top_trend_block_score_add = 0

        block_score_eff = block_score
        if (
            top_trend_enabled
            and top_trend_ma_window
            and int(top_trend_ma_window) >= 2
            and int(top_trend_block_score_add) > 0
        ):
            try:
                close_n = pd.to_numeric(df["Close"], errors="coerce")
                ma_s = _get_ema_series(df, window=int(top_trend_ma_window))
                c_last = close_n.iloc[-1]
                m_last = ma_s.iloc[-1]
                if pd.notna(m_last):
                    top_trend_ma_last = float(m_last)
                if pd.notna(c_last) and pd.notna(m_last) and float(c_last) >= float(m_last):
                    top_trend_up_last = True
                    block_score_eff = min(5, int(block_score) + int(top_trend_block_score_add))
            except Exception:
                top_trend_up_last = False

        top_trend_block_score_eff = int(block_score_eff)
        if score >= int(block_score_eff):
            status = "BLOCK"
        elif score >= int(watch_score):
            status = "WATCH"
        else:
            status = "ALLOW"

        # 結構是基石：若結構未成立，避免過早判定 BLOCK（轉為 WATCH）
        if status == "BLOCK" and not conditions.get("structure", False):
            status = "WATCH"

    # top：避免「賣飛」的保護（強勢股在短均線上方時，不直接判 BLOCK）
    top_shield_cfg = cfg.get("top_shield", {})
    top_shield_enabled = bool(top_shield_cfg.get("enabled", True))
    top_shield_ma_window = top_shield_cfg.get("ma_window", 5)
    try:
        top_shield_ma_window = int(top_shield_ma_window) if top_shield_ma_window is not None else None
    except Exception:
        top_shield_ma_window = None
    top_shield_ma_last = None
    top_shield_adjusted = False
    if (
        mode == "top"
        and status == "BLOCK"
        and top_shield_enabled
        and top_shield_ma_window
        and int(top_shield_ma_window) >= 2
    ):
        try:
            close_n = pd.to_numeric(df["Close"], errors="coerce")
            ma_s = _get_ema_series(df, window=int(top_shield_ma_window))
            c_last = close_n.iloc[-1]
            m_last = ma_s.iloc[-1]
            if pd.notna(m_last):
                top_shield_ma_last = float(m_last)
            if pd.notna(c_last) and pd.notna(m_last) and float(c_last) >= float(m_last):
                status = "WATCH"
                top_shield_adjusted = True
        except Exception:
            top_shield_ma_last = None
            top_shield_adjusted = False

    structure_cfg = cfg.get("structure", {})
    lookback = int(structure_cfg.get("lookback", 20))
    support_buffer = float(structure_cfg.get("support_buffer", 0.0) or 0.0)

    momentum_cfg = cfg.get("momentum", {})
    div_lookback = int(momentum_cfg.get("div_lookback", 5))
    rsi_period = int(momentum_cfg.get("rsi_period", 14))
    rsi_oversold = momentum_cfg.get("rsi_oversold")
    rsi_overbought = momentum_cfg.get("rsi_overbought")

    volume_cfg = cfg.get("volume", {})
    compare_window = int(volume_cfg.get("compare_window", 4))
    ma_window = volume_cfg.get("ma_window")
    ma_window = int(ma_window) if ma_window is not None else None
    dry_up_ratio = volume_cfg.get("dry_up_ratio")

    chip_cfg = cfg.get("chip", {})
    require_both = bool(chip_cfg.get("require_both", False))

    bias_cfg = cfg.get("bias", {})
    bias_ma_window = bias_cfg.get("ma_window", 20)
    try:
        bias_ma_window = int(bias_ma_window) if bias_ma_window is not None else None
    except Exception:
        bias_ma_window = None
    bias_overheat_pct_top = bias_cfg.get("overheat_pct_top")
    bias_pct_last = None
    try:
        if bias_ma_window:
            close = pd.to_numeric(df["Close"], errors="coerce")
            ma = close.shift(1).rolling(int(bias_ma_window)).mean()
            ma_last = ma.iloc[-1]
            c_last = close.iloc[-1]
            if pd.notna(ma_last) and float(ma_last) != 0 and pd.notna(c_last):
                bias_pct_last = float(((float(c_last) - float(ma_last)) / float(ma_last)) * 100.0)
    except Exception:
        bias_pct_last = None

    return {
        "mode": mode,
        "status": status,
        "score": score,
        "conditions": conditions,
        "indicators": {
            "bias_pct": bias_pct_last,
            "top_shield_ma": top_shield_ma_last,
            "top_shield_adjusted": top_shield_adjusted,
            "top_trend_ma": top_trend_ma_last,
            "top_trend_up": bool(top_trend_up_last),
            "top_trend_block_score_eff": top_trend_block_score_eff,
        },
        "inputs": {
            "foreign_3d_net": foreign_3d_net,
            "trust_3d_net": trust_3d_net,
        },
        "config_used": {
            "structure_lookback": lookback,
            "structure_support_buffer": support_buffer,
            "momentum_rsi_period": rsi_period,
            "momentum_div_lookback": div_lookback,
            "momentum_rsi_oversold": rsi_oversold,
            "momentum_rsi_overbought": rsi_overbought,
            "volume_ma_window": ma_window,
            "volume_compare_window": compare_window,
            "volume_dry_up_ratio": dry_up_ratio,
            "chip_require_both": require_both,
            "bias_ma_window": bias_ma_window,
            "bias_overheat_pct_top": bias_overheat_pct_top,
            "top_shield_enabled": top_shield_enabled,
            "top_shield_ma_window": top_shield_ma_window,
            "top_trend_filter_enabled": top_trend_enabled,
            "top_trend_filter_ma_window": top_trend_ma_window,
            "top_trend_filter_block_score_add": top_trend_block_score_add,
        },
    }


def get_all_turn_statuses(
    df: pd.DataFrame,
    *,
    mode: str = "bottom",
    cfg: Dict[str, Any] | None = None,
    foreign_3d_net: Optional[float] = None,
    trust_3d_net: Optional[float] = None,
) -> pd.Series:
    """
    一次性計算整段 K 線的 TURN 狀態（向量化），避免逐根呼叫 run_turn_check。
    - 邏輯會盡量對齊 run_turn_check（含原本的 len 門檻行為）
    """
    if cfg is None:
        cfg = load_turn_config()
    if df is None or df.empty:
        return pd.Series(dtype=object)
    if mode not in ["bottom", "top"]:
        mode = cfg.get("mode_default", "bottom")

    conds = _compute_turn_conditions_vectorized(
        df,
        mode=mode,
        cfg=cfg,
        foreign_3d_net=foreign_3d_net,
        trust_3d_net=trust_3d_net,
    )
    cond_struct = conds["structure"]
    score = (
        conds["structure"].astype(int)
        + conds["momentum"].astype(int)
        + conds["volume"].astype(int)
        + conds["chip"].astype(int)
        + conds["bias"].astype(int)
    )

    if mode == "bottom":
        allow_score = int(cfg["decision"]["bottom"]["allow_score"])
        watch_score = int(cfg["decision"]["bottom"]["watch_score"])
        status = pd.Series("BLOCK", index=df.index)
        status = status.mask(score >= watch_score, "WATCH")
        status = status.mask(score >= allow_score, "ALLOW")
        status = status.mask((status == "ALLOW") & (~cond_struct), "WATCH")
        return status

    block_score = int(cfg["decision"]["top"]["block_score"])
    watch_score = int(cfg["decision"]["top"]["watch_score"])
    block_thr = pd.Series(int(block_score), index=df.index)

    # top：趨勢過濾（多頭時更難觸發 BLOCK；例如 Close>EMA20 → block_score+1）
    top_trend_cfg = cfg.get("top_trend_filter", {})
    top_trend_enabled = bool(top_trend_cfg.get("enabled", False))
    try:
        top_trend_ma_window = int(top_trend_cfg.get("ma_window", 20) or 20)
    except Exception:
        top_trend_ma_window = None
    try:
        top_trend_block_score_add = int(top_trend_cfg.get("block_score_add", 1) or 0)
    except Exception:
        top_trend_block_score_add = 0
    if (
        top_trend_enabled
        and top_trend_ma_window
        and int(top_trend_ma_window) >= 2
        and int(top_trend_block_score_add) > 0
    ):
        try:
            close_n = pd.to_numeric(df["Close"], errors="coerce")
            ma_s = _get_ema_series(df, window=int(top_trend_ma_window))
            trend_up = close_n.notna() & ma_s.notna() & (close_n >= ma_s)
            inc = min(5, int(block_score) + int(top_trend_block_score_add))
            block_thr = block_thr.mask(trend_up, int(inc))
        except Exception:
            pass
    status = pd.Series("ALLOW", index=df.index)
    status = status.mask(score >= watch_score, "WATCH")
    status = status.mask(score >= block_thr, "BLOCK")
    status = status.mask((status == "BLOCK") & (~cond_struct), "WATCH")

    # top：強勢股保護（在短均線上方時，不直接判 BLOCK）
    top_shield_cfg = cfg.get("top_shield", {})
    top_shield_enabled = bool(top_shield_cfg.get("enabled", True))
    top_shield_ma_window = top_shield_cfg.get("ma_window", 5)
    try:
        top_shield_ma_window = int(top_shield_ma_window) if top_shield_ma_window is not None else None
    except Exception:
        top_shield_ma_window = None
    if top_shield_enabled and top_shield_ma_window and int(top_shield_ma_window) >= 2:
        try:
            close_n = pd.to_numeric(df["Close"], errors="coerce")
            ma_s = _get_ema_series(df, window=int(top_shield_ma_window))
            valid = close_n.notna() & ma_s.notna()
            status = status.mask((status == "BLOCK") & valid & (close_n >= ma_s), "WATCH")
        except Exception:
            pass
    return status


def get_all_turn_details(
    df: pd.DataFrame,
    *,
    mode: str = "bottom",
    cfg: Dict[str, Any] | None = None,
    foreign_3d_net: Optional[float] = None,
    trust_3d_net: Optional[float] = None,
) -> pd.DataFrame:
    """
    回傳整段 TURN 明細（向量化）：
    - status: ALLOW/WATCH/BLOCK
    - score: bottom=0~4；top=0~5（含 bias 加分）
    - structure/momentum/volume/chip/bias: bool
    """
    if cfg is None:
        cfg = load_turn_config()
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["status", "score", "structure", "momentum", "volume", "chip", "bias"]
        )
    if mode not in ["bottom", "top"]:
        mode = cfg.get("mode_default", "bottom")

    conds = _compute_turn_conditions_vectorized(
        df,
        mode=mode,
        cfg=cfg,
        foreign_3d_net=foreign_3d_net,
        trust_3d_net=trust_3d_net,
    )
    cond_struct = conds["structure"]
    score = (
        conds["structure"].astype(int)
        + conds["momentum"].astype(int)
        + conds["volume"].astype(int)
        + conds["chip"].astype(int)
        + conds["bias"].astype(int)
    ).astype(int)

    if mode == "bottom":
        allow_score = int(cfg["decision"]["bottom"]["allow_score"])
        watch_score = int(cfg["decision"]["bottom"]["watch_score"])
        status = pd.Series("BLOCK", index=df.index)
        status = status.mask(score >= watch_score, "WATCH")
        status = status.mask(score >= allow_score, "ALLOW")
        status = status.mask((status == "ALLOW") & (~cond_struct), "WATCH")
    else:
        block_score = int(cfg["decision"]["top"]["block_score"])
        watch_score = int(cfg["decision"]["top"]["watch_score"])
        block_thr = pd.Series(int(block_score), index=df.index)

        # top：趨勢過濾（多頭時更難觸發 BLOCK；例如 Close>EMA20 → block_score+1）
        top_trend_cfg = cfg.get("top_trend_filter", {})
        top_trend_enabled = bool(top_trend_cfg.get("enabled", False))
        try:
            top_trend_ma_window = int(top_trend_cfg.get("ma_window", 20) or 20)
        except Exception:
            top_trend_ma_window = None
        try:
            top_trend_block_score_add = int(top_trend_cfg.get("block_score_add", 1) or 0)
        except Exception:
            top_trend_block_score_add = 0
        if (
            top_trend_enabled
            and top_trend_ma_window
            and int(top_trend_ma_window) >= 2
            and int(top_trend_block_score_add) > 0
        ):
            try:
                close_n = pd.to_numeric(df["Close"], errors="coerce")
                ma_s = _get_ema_series(df, window=int(top_trend_ma_window))
                trend_up = close_n.notna() & ma_s.notna() & (close_n >= ma_s)
                inc = min(5, int(block_score) + int(top_trend_block_score_add))
                block_thr = block_thr.mask(trend_up, int(inc))
            except Exception:
                pass
        status = pd.Series("ALLOW", index=df.index)
        status = status.mask(score >= watch_score, "WATCH")
        status = status.mask(score >= block_thr, "BLOCK")
        status = status.mask((status == "BLOCK") & (~cond_struct), "WATCH")
        # top：強勢股保護（在短均線上方時，不直接判 BLOCK）
        top_shield_cfg = cfg.get("top_shield", {})
        top_shield_enabled = bool(top_shield_cfg.get("enabled", True))
        top_shield_ma_window = top_shield_cfg.get("ma_window", 5)
        try:
            top_shield_ma_window = int(top_shield_ma_window) if top_shield_ma_window is not None else None
        except Exception:
            top_shield_ma_window = None
        if top_shield_enabled and top_shield_ma_window and int(top_shield_ma_window) >= 2:
            try:
                close_n = pd.to_numeric(df["Close"], errors="coerce")
                ma_s = _get_ema_series(df, window=int(top_shield_ma_window))
                valid = close_n.notna() & ma_s.notna()
                status = status.mask((status == "BLOCK") & valid & (close_n >= ma_s), "WATCH")
            except Exception:
                pass

    out = pd.DataFrame(
        {
            "status": status,
            "score": score,
            "structure": conds["structure"],
            "momentum": conds["momentum"],
            "volume": conds["volume"],
            "chip": conds["chip"],
            "bias": conds["bias"],
        },
        index=df.index,
    )
    return out


def backtest_turn_signals(
    df: pd.DataFrame,
    *,
    mode: str = "bottom",
    cfg: Dict[str, Any] | None = None,
    foreign_3d_net: Optional[float] = None,
    trust_3d_net: Optional[float] = None,
    hold_days: int = 10,
    trailing_stop_pct: float = 0.0,
    exit_ma_window: int = 0,
) -> pd.DataFrame:
    """
    回測（含「狀態降級」＋「移動止損」＋「二階段確認出場」）：

    - 進場訊號：狀態切換後進入「最佳狀態」
      - bottom: 變成 ALLOW（代表較佳進場）
      - top: 變成 BLOCK（代表較佳出場／風險升高）
    - 出場策略（誰先到就用誰）：
      1) 進場後第一個「不再是最佳狀態」的 bar（狀態降級）
      - bottom: ALLOW -> WATCH/BLOCK
      - top: BLOCK -> WATCH/ALLOW（解除警戒）
         若 exit_ma_window >= 2（僅 bottom 生效）：改為「狀態降級 且 收盤跌破 EMA(exit_ma_window)」才出場
      2) 移動止損（trailing stop；回檔/反彈 X% 強制出場）
         - trailing_stop_pct 以「百分比」輸入，例如 3.0 代表 3%
         - bottom（做多）：從進場後最高點回檔 X% 觸發
         - top（出場品質）：從訊號後最低點「反彈 X%」視為回補點（避免只看最低點造成過度樂觀）
      若在 hold_days 視窗內沒有降級/止損，則以視窗末端結案（TIMEOUT），或資料末端（HOLDING）。

    - entry: 訊號當天 Close
    - exit:
      - bottom：出場價格（狀態降級：當天 Close；止損：以 OHLC 模擬觸發價）
      - top：解除警戒/回補點（同上規則；正值代表賣出後有回檔，負值代表踏空）

    回傳每個訊號：
    - max_favorable_pct / max_adverse_pct：從進場後到出場期間的 MFE/MAE（%）
    - return_pct：
      - bottom：結案報酬（%）= exit/entry
      - top：出場品質（%）= entry/exit
        - >0：賣出後價格下跌（有回檔/避免回吐）
        - <0：賣出後繼續上漲（踏空/賣太早）
    - final_pct：為保持相容，等同於 return_pct（結案報酬）
    - holding_days：持有天數（以 bar 計；最小為 1）
    - giveback_ratio：獲利回吐比（(MFE - return) / MFE；MFE<=0 時為 NaN）
    """
    if cfg is None:
        cfg = load_turn_config()
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "entry_close",
                "exit_close",
                "exit_date",
                "exit_reason",
                "exit_stop",
                "holding_days",
                "max_favorable_pct",
                "max_adverse_pct",
                "return_pct",
                "final_pct",
                "giveback_pct",
                "giveback_ratio",
            ]
        )

    if mode not in ["bottom", "top"]:
        mode = cfg.get("mode_default", "bottom")

    try:
        hold_days = max(1, int(hold_days))
    except Exception:
        hold_days = 10

    try:
        trailing_stop_pct = float(trailing_stop_pct or 0.0)
    except Exception:
        trailing_stop_pct = 0.0
    trailing_stop_pct = max(0.0, float(trailing_stop_pct))
    trailing_stop_ratio = float(trailing_stop_pct) / 100.0

    try:
        exit_ma_window = int(exit_ma_window or 0)
    except Exception:
        exit_ma_window = 0
    exit_ma_window = max(0, int(exit_ma_window))

    details = get_all_turn_details(
        df,
        mode=mode,
        cfg=cfg,
        foreign_3d_net=foreign_3d_net,
        trust_3d_net=trust_3d_net,
    )
    if details is None or details.empty:
        return pd.DataFrame(
            columns=[
                "entry_close",
                "exit_close",
                "exit_date",
                "exit_reason",
                "exit_stop",
                "holding_days",
                "max_favorable_pct",
                "max_adverse_pct",
                "return_pct",
                "final_pct",
                "giveback_pct",
                "giveback_ratio",
            ]
        )

    best_status = "ALLOW" if mode == "bottom" else "BLOCK"
    status = details["status"].astype(str)
    signal_mask = (status == best_status) & (status != status.shift(1))
    if len(signal_mask) > 0:
        signal_mask.iloc[0] = False
    signal_pos = np.where(signal_mask.fillna(False).values)[0]

    # 預先轉成可運算的數列（避免 dtype/字串）
    close_s = pd.to_numeric(df["Close"], errors="coerce")
    high_s = pd.to_numeric(df["High"], errors="coerce")
    low_s = pd.to_numeric(df["Low"], errors="coerce")
    if "Open" in df.columns:
        open_s = pd.to_numeric(df["Open"], errors="coerce")
    else:
        open_s = close_s

    exit_ma_s = None
    if mode == "bottom" and exit_ma_window >= 2:
        try:
            exit_ma_s = _get_ema_series(df, window=int(exit_ma_window))
        except Exception:
            exit_ma_s = None

    rows: list[dict] = []
    for i in signal_pos:
        # 確保不超出 DataFrame 邊界（至少要有下一根）
        if i + 1 >= len(df):
            continue
        entry_close = float(close_s.iloc[i])
        if not np.isfinite(entry_close) or entry_close == 0:
            continue

        # 視窗末端位置（對應舊版：future window = 後續 hold_days 根，最後一根為 i+hold_days）
        max_exit_pos = min(len(df) - 1, i + int(hold_days))
        status_after = status.iloc[i + 1 : max_exit_pos + 1]

        # --- exit（基準）：狀態降級（或二階段確認） ---
        downgrade_pos = None
        downgrade_mask = (status_after != best_status).fillna(False).values
        if downgrade_mask.any():
            rel = int(np.argmax(downgrade_mask))  # first True
            downgrade_pos = int(i + 1 + rel)

        status_exit_pos = int(max_exit_pos)
        status_exit_reason = "HOLDING" if status_exit_pos == (len(df) - 1) else "TIMEOUT"

        # 二階段確認：僅 bottom 使用（避免強勢股小震盪就提前出場）
        if mode == "bottom" and exit_ma_s is not None and max_exit_pos >= (i + 1):
            try:
                close_after = close_s.iloc[i + 1 : max_exit_pos + 1]
                ma_after = pd.to_numeric(exit_ma_s.iloc[i + 1 : max_exit_pos + 1], errors="coerce")
                confirm = (
                    (status_after != best_status)
                    & close_after.notna()
                    & ma_after.notna()
                    & (close_after < ma_after)
                )
                if bool(confirm.any()):
                    rel = int(np.argmax(confirm.values))  # first True
                    status_exit_pos = int(i + 1 + rel)
                    status_exit_reason = (
                        f"DOWNGRADE+EMA{int(exit_ma_window)}_BREAK"
                        f"({str(status.iloc[status_exit_pos])})"
                    )
                else:
                    # 若有降級但未破線，代表「警戒但撐住」→ 以 TIMEOUT/HOLDING 結案（避免賣飛）
                    if downgrade_pos is not None:
                        base = "HOLDING" if status_exit_pos == (len(df) - 1) else "TIMEOUT"
                        status_exit_reason = f"DOWNGRADE_NO_EMA{int(exit_ma_window)}_BREAK_{base}"
            except Exception:
                # fallback：維持 TIMEOUT/HOLDING
                pass
        else:
            # 原版：第一個狀態降級就出場
            if downgrade_pos is not None:
                status_exit_pos = int(downgrade_pos)
                status_exit_reason = str(status.iloc[status_exit_pos])

        status_exit_price = float(close_s.iloc[status_exit_pos])
        if not np.isfinite(status_exit_price) or status_exit_price <= 0:
            continue

        # --- exit（trailing stop）：觸發則可能更早出場 ---
        ts_exit_pos: Optional[int] = None
        ts_exit_reason: Optional[str] = None
        ts_exit_price: Optional[float] = None
        ts_stop_price: Optional[float] = None

        if trailing_stop_ratio > 0 and max_exit_pos >= (i + 1):
            if mode == "bottom":
                # 重要：避免「同一根 K」偷看（entry 用 Close，不應用 entry bar 的 High/Low 初始化）
                peak = float(entry_close)
                for j in range(i + 1, max_exit_pos + 1):
                    stop = peak * (1.0 - float(trailing_stop_ratio))
                    o = float(open_s.iloc[j])
                    h = float(high_s.iloc[j])
                    l = float(low_s.iloc[j])
                    if np.isfinite(o) and o <= stop:
                        ts_exit_pos = int(j)
                        ts_exit_price = float(o)
                        ts_stop_price = float(stop)
                        ts_exit_reason = f"TS_GAP({trailing_stop_pct:.1f}%)"
                        break
                    if np.isfinite(l) and l <= stop:
                        ts_exit_pos = int(j)
                        ts_exit_price = float(stop)
                        ts_stop_price = float(stop)
                        ts_exit_reason = f"TS({trailing_stop_pct:.1f}%)"
                        break
                    if np.isfinite(h):
                        peak = max(float(peak), float(h))
            else:
                # 重要：避免「同一根 K」偷看（entry 用 Close，不應用 entry bar 的 High/Low 初始化）
                trough = float(entry_close)
                for j in range(i + 1, max_exit_pos + 1):
                    stop = trough * (1.0 + float(trailing_stop_ratio))
                    o = float(open_s.iloc[j])
                    h = float(high_s.iloc[j])
                    l = float(low_s.iloc[j])
                    if np.isfinite(o) and o >= stop:
                        ts_exit_pos = int(j)
                        ts_exit_price = float(o)
                        ts_stop_price = float(stop)
                        ts_exit_reason = f"TS_GAP({trailing_stop_pct:.1f}%)"
                        break
                    if np.isfinite(h) and h >= stop:
                        ts_exit_pos = int(j)
                        ts_exit_price = float(stop)
                        ts_stop_price = float(stop)
                        ts_exit_reason = f"TS({trailing_stop_pct:.1f}%)"
                        break
                    if np.isfinite(l):
                        trough = min(float(trough), float(l))

        # --- 最終選擇：較早的出場規則（狀態降級 vs TS） ---
        exit_pos = int(status_exit_pos)
        exit_reason = str(status_exit_reason)
        exit_price = float(status_exit_price)
        exit_stop = np.nan
        if (
            ts_exit_pos is not None
            and ts_exit_price is not None
            and np.isfinite(ts_exit_price)
            and int(ts_exit_pos) < int(status_exit_pos)
        ):
            exit_pos = int(ts_exit_pos)
            exit_reason = str(ts_exit_reason or "TS")
            exit_price = float(ts_exit_price)
            if ts_stop_price is not None and np.isfinite(ts_stop_price):
                exit_stop = float(ts_stop_price)

        if not np.isfinite(exit_price) or exit_price <= 0:
            continue
        holding_days = int(exit_pos - i)
        if holding_days <= 0:
            continue

        fut_high = high_s.iloc[i + 1 : exit_pos + 1].max()
        fut_low = low_s.iloc[i + 1 : exit_pos + 1].min()
        if pd.isna(fut_high) or pd.isna(fut_low):
            continue
        if fut_low <= 0:
            continue

        if mode == "bottom":
            max_fav = (float(fut_high) / entry_close - 1.0) * 100.0
            max_adv = (float(fut_low) / entry_close - 1.0) * 100.0  # 通常為負
            return_pct = (float(exit_price) / entry_close - 1.0) * 100.0
        else:
            # top: 賣出後希望下跌（favorable=跌幅），不利=繼續上漲（opportunity cost）
            max_fav = (entry_close / float(fut_low) - 1.0) * 100.0 if fut_low != 0 else 0.0
            max_adv = (float(fut_high) / entry_close - 1.0) * 100.0
            return_pct = (entry_close / float(exit_price) - 1.0) * 100.0

        final_pct = float(return_pct)  # 相容欄位：結案報酬
        giveback_pct = np.nan
        giveback_ratio = np.nan
        try:
            if np.isfinite(max_fav) and float(max_fav) > 0:
                giveback_pct = float(max_fav) - float(return_pct)
                giveback_ratio = float(giveback_pct) / float(max_fav)
        except Exception:
            giveback_pct = np.nan
            giveback_ratio = np.nan

        rows.append(
            {
                "date": df.index[i],
                "entry_close": entry_close,
                "exit_close": exit_price,
                "exit_date": df.index[exit_pos],
                "exit_reason": exit_reason,
                "exit_stop": exit_stop,
                "holding_days": holding_days,
                "max_favorable_pct": max_fav,
                "max_adverse_pct": max_adv,
                "return_pct": float(return_pct),
                "final_pct": final_pct,
                "giveback_pct": giveback_pct,
                "giveback_ratio": giveback_ratio,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "entry_close",
                "exit_close",
                "exit_date",
                "exit_reason",
                "exit_stop",
                "holding_days",
                "max_favorable_pct",
                "max_adverse_pct",
                "return_pct",
                "final_pct",
                "giveback_pct",
                "giveback_ratio",
            ]
        )

    out = pd.DataFrame(rows).set_index("date")
    return out


def grid_search_optimization(
    df: pd.DataFrame,
    *,
    mode: str = "bottom",
    base_cfg: Dict[str, Any] | None = None,
    foreign_3d_net: Optional[float] = None,
    trust_3d_net: Optional[float] = None,
    hold_days: int = 10,
    trailing_stop_pct: float = 0.0,
    exit_ma_window: int = 0,
    win_threshold: float = 3.0,
    score_thresholds: Optional[list[int]] = None,
    dry_up_ratios: Optional[list[float]] = None,
    min_signals: int = 1,
) -> pd.DataFrame:
    """
    暴力掃描參數組合，尋找較佳「甜蜜點」。
    - bottom: 掃描 decision.bottom.allow_score
    - top: 掃描 decision.top.block_score
    同時掃描 volume.dry_up_ratio。
    其他參數會固定使用 base_cfg。
    回傳欄位：score_threshold/dry_up_ratio/count/win_rate/avg_fav/avg_adv/avg_final/rr/expectancy

    - win_rate：以「結案報酬 final_pct（含狀態降級出場）」是否 ≥ win_threshold 計算
    - expectancy：使用平均結案報酬（avg_final）作為真實期望值（%/次）
    """
    if base_cfg is None:
        base_cfg = load_turn_config()
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "score_threshold",
                "dry_up_ratio",
                "count",
                "win_rate",
                "avg_fav",
                "avg_adv",
                "avg_final",
                "rr",
                "expectancy",
            ]
        )

    if mode not in ["bottom", "top"]:
        mode = base_cfg.get("mode_default", "bottom")

    if score_thresholds is None:
        score_thresholds = [2, 3, 4]
    if dry_up_ratios is None:
        dry_up_ratios = [0.4, 0.5, 0.6, 0.7, 0.8]

    rows: list[dict] = []
    for score_thr in score_thresholds:
        for vol_ratio in dry_up_ratios:
            cfg = copy.deepcopy(base_cfg)

            # 套用測試參數
            if mode == "bottom":
                cfg["decision"]["bottom"]["allow_score"] = int(score_thr)
                # 避免不合理門檻（WATCH > ALLOW）
                cfg["decision"]["bottom"]["watch_score"] = min(
                    int(cfg["decision"]["bottom"].get("watch_score", 2)),
                    int(score_thr),
                )
            else:
                cfg["decision"]["top"]["block_score"] = int(score_thr)
                cfg["decision"]["top"]["watch_score"] = min(
                    int(cfg["decision"]["top"].get("watch_score", 1)),
                    int(score_thr),
                )

            cfg.setdefault("volume", {})
            cfg["volume"]["dry_up_ratio"] = float(vol_ratio)

            bt_df = backtest_turn_signals(
                df,
                mode=mode,
                cfg=cfg,
                foreign_3d_net=foreign_3d_net,
                trust_3d_net=trust_3d_net,
                hold_days=int(hold_days),
                trailing_stop_pct=float(trailing_stop_pct or 0.0),
                exit_ma_window=int(exit_ma_window or 0),
            )

            count = int(len(bt_df)) if isinstance(bt_df, pd.DataFrame) else 0
            if count < int(min_signals):
                rows.append(
                    {
                        "score_threshold": int(score_thr),
                        "dry_up_ratio": float(vol_ratio),
                        "count": count,
                        "win_rate": np.nan,
                        "avg_fav": np.nan,
                        "avg_adv": np.nan,
                        "avg_final": np.nan,
                        "rr": np.nan,
                        "expectancy": np.nan,
                    }
                )
                continue

            win_rate = (bt_df["final_pct"] >= float(win_threshold)).mean()
            avg_fav = bt_df["max_favorable_pct"].mean()
            avg_adv = bt_df["max_adverse_pct"].mean()
            avg_final = bt_df["final_pct"].mean() if "final_pct" in bt_df.columns else np.nan

            rr = np.nan
            try:
                if pd.notna(avg_adv) and float(avg_adv) != 0:
                    rr = abs(float(avg_fav) / float(avg_adv))
            except Exception:
                rr = np.nan

            # 期望值（真實）：直接用平均結案報酬（%/次）
            expectancy = float(avg_final) if pd.notna(avg_final) else np.nan

            rows.append(
                {
                    "score_threshold": int(score_thr),
                    "dry_up_ratio": float(vol_ratio),
                    "count": count,
                    "win_rate": float(win_rate),
                    "avg_fav": float(avg_fav),
                    "avg_adv": float(avg_adv),
                    "avg_final": float(avg_final) if pd.notna(avg_final) else np.nan,
                    "rr": float(rr) if pd.notna(rr) else np.nan,
                    "expectancy": float(expectancy) if pd.notna(expectancy) else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        by=["expectancy", "win_rate", "count"],
        ascending=[False, False, False],
        na_position="last",
    )
    return out


def validate_turn_cfg(cfg: Dict[str, Any]) -> list[str]:
    """
    參數健檢（不改值，只回傳提醒）。
    用途：
    - UI 顯示警告（避免 slider 調出矛盾組合）
    - 腳本/回測在參數異常時能快速定位問題
    """
    warns: list[str] = []
    if not isinstance(cfg, dict):
        return ["cfg 不是 dict，請確認 turn_check_config.json / 覆寫參數。"]

    def _get_int(path: list[str], default: int) -> int:
        cur: Any = cfg
        for k in path:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        try:
            return int(cur)
        except Exception:
            return int(default)

    def _get_float(path: list[str], default: float) -> float:
        cur: Any = cfg
        for k in path:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        try:
            return float(cur)
        except Exception:
            return float(default)

    # structure
    lookback = _get_int(["structure", "lookback"], 20)
    if lookback < 10:
        warns.append(f"structure.lookback={lookback} 偏小（容易過度敏感）。")
    support_buffer = _get_float(["structure", "support_buffer"], 0.0)
    if support_buffer < 0:
        warns.append("structure.support_buffer 不可為負。")
    if support_buffer > 0.05:
        warns.append(f"structure.support_buffer={support_buffer:.3f} 偏大（可能放太寬）。")

    # momentum
    rsi_period = _get_int(["momentum", "rsi_period"], 14)
    if rsi_period < 2:
        warns.append("momentum.rsi_period 建議 >= 2。")
    div_lookback = _get_int(["momentum", "div_lookback"], 5)
    if div_lookback < 1:
        warns.append("momentum.div_lookback 建議 >= 1。")

    # volume
    compare_window = _get_int(["volume", "compare_window"], 4)
    if compare_window < 2:
        warns.append("volume.compare_window 建議 >= 2。")
    ma_window = _get_int(["volume", "ma_window"], 20)
    if ma_window < 2:
        warns.append("volume.ma_window 建議 >= 2。")
    dry_up_ratio = _get_float(["volume", "dry_up_ratio"], 0.6)
    if dry_up_ratio <= 0:
        warns.append("volume.dry_up_ratio 建議 > 0。")
    if dry_up_ratio > 1.2:
        warns.append(f"volume.dry_up_ratio={dry_up_ratio:.2f} 偏大（縮量門檻可能失去意義）。")
    top_range_window = _get_int(["volume", "top_range_window"], 5)
    top_drop_mult = _get_float(["volume", "top_drop_mult"], 1.5)
    if top_range_window and top_range_window < 2:
        warns.append("volume.top_range_window 建議 >= 2（或關閉）。")
    if top_drop_mult is not None and top_drop_mult <= 0:
        warns.append("volume.top_drop_mult 建議 > 0。")

    # chip
    trust_days = _get_int(["chip", "trust_days"], 3)
    foreign_days = _get_int(["chip", "foreign_days"], 3)
    if trust_days < 1 or foreign_days < 1:
        warns.append("chip.trust_days / chip.foreign_days 建議 >= 1。")

    # decision thresholds
    allow_score = _get_int(["decision", "bottom", "allow_score"], 3)
    watch_score = _get_int(["decision", "bottom", "watch_score"], 2)
    if watch_score > allow_score:
        warns.append(
            f"decision.bottom.watch_score({watch_score}) > allow_score({allow_score})：門檻順序反了（仍可運行，但語意不直覺）。"
        )

    block_score = _get_int(["decision", "top", "block_score"], 4)
    watch_score_top = _get_int(["decision", "top", "watch_score"], 2)
    if watch_score_top > block_score:
        warns.append(
            f"decision.top.watch_score({watch_score_top}) > block_score({block_score})：門檻順序反了（仍可運行，但語意不直覺）。"
        )

    # bias
    bias_ma_window = _get_int(["bias", "ma_window"], 20)
    if bias_ma_window < 2:
        warns.append("bias.ma_window 建議 >= 2（或關閉 bias）。")
    overheat = _get_float(["bias", "overheat_pct_top"], 8.0)
    if overheat < 0:
        warns.append("bias.overheat_pct_top 不可為負。")

    # top_shield / top_trend_filter
    top_shield_ma = _get_int(["top_shield", "ma_window"], 5)
    if top_shield_ma and top_shield_ma < 2:
        warns.append("top_shield.ma_window 建議 >= 2（或關閉 top_shield）。")
    trend_ma = _get_int(["top_trend_filter", "ma_window"], 20)
    add = _get_int(["top_trend_filter", "block_score_add"], 1)
    if trend_ma and trend_ma < 2:
        warns.append("top_trend_filter.ma_window 建議 >= 2（或關閉 top_trend_filter）。")
    if add < 0:
        warns.append("top_trend_filter.block_score_add 不可為負。")

    return warns
