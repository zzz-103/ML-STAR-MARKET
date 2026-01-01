import pandas as pd
import numpy as np


def run(df):
    output = pd.DataFrame(index=df.index)
    close = df["close"].replace(0, np.nan)
    grouped = close.groupby(level="code")

    skip = 5
    windows = [20, 60, 120, 250]
    for w in windows:
        c1 = grouped.shift(skip)
        c0 = grouped.shift(skip + w)
        output[f"momo_sl_{w}_{skip}"] = (c1 - c0) / c0

    ret_1d = grouped.pct_change(fill_method=None)
    ret_for_vol = ret_1d.groupby(level="code").shift(skip)
    vol_60 = ret_for_vol.groupby(level="code").transform(lambda x: x.rolling(60, min_periods=60).std())
    output[f"momo_sl_{60}_{skip}_voladj"] = output[f"momo_sl_{60}_{skip}"] / (vol_60 + 1e-6)
    return output

