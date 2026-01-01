import pandas as pd
import numpy as np


def run(df):
    output = pd.DataFrame(index=df.index)
    close = df["close"].replace(0, np.nan)
    high = df["high"].replace(0, np.nan)
    volume = df["volume"].replace(0, np.nan)

    g_high = high.groupby(level="code")
    g_vol = volume.groupby(level="code")

    prev_high_20 = g_high.transform(lambda x: x.rolling(20, min_periods=20).max().shift(1))
    prev_high_60 = g_high.transform(lambda x: x.rolling(60, min_periods=60).max().shift(1))
    avg_vol_20 = g_vol.transform(lambda x: x.rolling(20, min_periods=20).mean().shift(1))

    breakout_20 = (close - prev_high_20) / prev_high_20
    breakout_60 = (close - prev_high_60) / prev_high_60

    vol_ratio_20 = volume / avg_vol_20
    vol_score_20 = np.log(vol_ratio_20)

    output["breakout_20"] = breakout_20
    output["breakout_60"] = breakout_60
    output["breakout_20_vol_confirm"] = breakout_20.clip(lower=0) * vol_score_20.clip(lower=0)
    output["breakout_60_vol_confirm"] = breakout_60.clip(lower=0) * vol_score_20.clip(lower=0)
    return output

