from typing import Tuple, Optional

import pandas as pd

try:
    from FinMind.data import DataLoader

    FINMIND_AVAILABLE = True
except Exception:
    FINMIND_AVAILABLE = False


def get_chip_3d_net(
    stock_id: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    回傳：
    - foreign_3d_net: 外資近 3 日淨買賣（張數）
    - trust_3d_net: 投信近 3 日淨買賣（張數）
    """
    if not FINMIND_AVAILABLE:
        return None, None

    try:
        dl = DataLoader()
        df = dl.taiwan_stock_institutional_investors(stock_id=stock_id)
    except Exception:
        return None, None

    if df is None or df.empty:
        return None, None

    # 只取近 3 個交易日
    df = df.sort_values("date").tail(3)

    # FinMind 欄位名稱（注意大小寫）
    foreign = df[df["institutional_investors"] == "Foreign_Investor"]
    trust = df[df["institutional_investors"] == "Investment_Trust"]

    foreign_3d_net = foreign["buy_sell"].sum() if not foreign.empty else 0.0
    trust_3d_net = trust["buy_sell"].sum() if not trust.empty else 0.0

    return float(foreign_3d_net), float(trust_3d_net)
