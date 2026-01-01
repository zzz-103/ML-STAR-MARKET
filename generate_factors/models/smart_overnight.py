import pandas as pd
import numpy as np


def run(df):
    output = pd.DataFrame(index=df.index)

    required = {"open", "turnover"}
    if not required.issubset(set(df.columns)):
        return output

    open_p = pd.to_numeric(df["open"], errors="coerce")
    turnover = pd.to_numeric(df["turnover"], errors="coerce")
    turnover_safe = turnover.replace(0, np.nan)

    if "preclose" in df.columns:
        preclose = pd.to_numeric(df["preclose"], errors="coerce").replace(0, np.nan)
        ret_overnight = open_p / preclose - 1.0
    else:
        close = pd.to_numeric(df.get("close", np.nan), errors="coerce").replace(0, np.nan)
        prev_close = close.groupby(level="code").shift(1)
        ret_overnight = open_p / prev_close - 1.0

    ret_overnight = ret_overnight.replace([np.inf, -np.inf], np.nan)

    g_turnover = turnover_safe.groupby(level="code", sort=False)
    numerator = (ret_overnight * turnover_safe).groupby(level="code", sort=False).rolling(20).sum()
    denominator = g_turnover.rolling(20).sum()
    smart_overnight_20 = (numerator / denominator).reset_index(level=0, drop=True)

    lower = smart_overnight_20.quantile(0.01)
    upper = smart_overnight_20.quantile(0.99)
    if np.isfinite(lower) and np.isfinite(upper) and upper >= lower:
        smart_overnight_20 = smart_overnight_20.clip(lower, upper)

    output["smart_overnight_20"] = smart_overnight_20
    return output
