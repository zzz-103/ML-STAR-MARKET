import pandas as pd
import numpy as np


def run(df: pd.DataFrame) -> pd.DataFrame:
    """情绪与量价因子组（输入为按 date/code 的 MultiIndex 日频数据）。"""
    output = pd.DataFrame(index=df.index)
    grouped = df.groupby(level="code")

    price_change = grouped["close"].diff()

    # PSY：12 日上涨天数占比（%）
    is_up = (price_change > 0).astype(float)
    output["psy_12"] = is_up.groupby(level="code").transform(
        lambda x: x.rolling(12).mean()
    ) * 100

    # VR：24 日上涨成交量 / 下跌成交量（+1 平滑避免除零）
    vol = df["volume"]
    vol_up = vol.where(price_change > 0, 0.0)
    vol_down = vol.where(price_change < 0, 0.0)
    roll_vol_up = vol_up.groupby(level="code").rolling(24).sum().droplevel(0)
    roll_vol_down = vol_down.groupby(level="code").rolling(24).sum().droplevel(0)
    output["vr_24"] = (roll_vol_up + 1.0) / (roll_vol_down + 1.0)

    # 量价相关：10 日收盘价与成交量的滚动相关系数
    output["pv_corr"] = grouped.apply(
        lambda x: x["close"].rolling(10).corr(x["volume"])
    ).reset_index(level=0, drop=True)

    return output
