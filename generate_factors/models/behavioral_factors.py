import numpy as np
import pandas as pd


def _rolling_trend_r2(series: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    y = s.to_numpy(dtype="float64", copy=True)
    n = int(y.shape[0])
    out = np.full(n, np.nan, dtype="float64")
    w = int(window)
    if w <= 1 or n < w:
        return pd.Series(out, index=series.index)

    finite = np.isfinite(y)
    y0 = np.where(finite, y, 0.0)
    ones = np.ones(w, dtype="float64")

    sum_y = np.convolve(y0, ones, mode="valid")
    sum_y2 = np.convolve(y0 * y0, ones, mode="valid")

    x = np.arange(w, dtype="float64")
    sum_x = float(x.sum())
    sum_x2 = float((x * x).sum())
    var_x = float(sum_x2 - (sum_x * sum_x) / float(w))
    sum_xy = np.convolve(y0, x[::-1], mode="valid")

    count = np.convolve(finite.astype("float64"), ones, mode="valid")
    valid = count == float(w)

    cov_xy = sum_xy - (sum_x * sum_y) / float(w)
    var_y = sum_y2 - (sum_y * sum_y) / float(w)
    denom = var_x * var_y

    r2 = np.full_like(sum_y, np.nan, dtype="float64")
    good = valid & np.isfinite(denom) & (denom > 0.0)
    r2[good] = (cov_xy[good] * cov_xy[good]) / denom[good]
    r2 = np.clip(r2, 0.0, 1.0, out=r2, where=np.isfinite(r2))

    out[w - 1 :] = r2
    return pd.Series(out, index=series.index)


def run(df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=df.index)

    def _num_series(col: str) -> pd.Series:
        if col in df.columns:
            s = df[col]
        else:
            s = pd.Series(np.nan, index=df.index)
        return pd.to_numeric(s, errors="coerce")

    open_p = _num_series("open")
    high = _num_series("high")
    low = _num_series("low")
    close = _num_series("close")
    volume = _num_series("volume")
    turnover = _num_series("turnover")

    grouped = df.groupby(level="code", sort=False)

    if ("open" in df.columns) and ("volume" in df.columns):
        open_vol_corr_10 = grouped.apply(
            lambda x: pd.to_numeric(x["open"], errors="coerce")
            .rolling(10)
            .corr(pd.to_numeric(x["volume"], errors="coerce"))
        ).reset_index(level=0, drop=True)
        output["f_price_vol_corr_10"] = -1.0 * open_vol_corr_10
    else:
        output["f_price_vol_corr_10"] = pd.Series(np.nan, index=df.index)

    close_safe = close.replace(0.0, np.nan)
    ret_1 = close_safe.groupby(level="code").pct_change(fill_method=None)
    roc_5 = close_safe.groupby(level="code").pct_change(5, fill_method=None)
    roc_10 = close_safe.groupby(level="code").pct_change(10, fill_method=None)
    roc_20 = close_safe.groupby(level="code").pct_change(20, fill_method=None)
    output["roc_5"] = roc_5
    output["roc_10"] = roc_10
    output["roc_20"] = roc_20

    vol_5 = (
        ret_1.groupby(level="code")
        .rolling(5, min_periods=5)
        .std()
        .reset_index(level=0, drop=True)
    )
    vol_10 = (
        ret_1.groupby(level="code")
        .rolling(10, min_periods=10)
        .std()
        .reset_index(level=0, drop=True)
    )
    vol_20 = (
        ret_1.groupby(level="code")
        .rolling(20, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )
    output["vol_5"] = vol_5
    output["vol_10"] = vol_10
    output["vol_20"] = vol_20

    ma_5 = close_safe.groupby(level="code").transform(lambda x: x.rolling(5, min_periods=5).mean())
    ma_10 = close_safe.groupby(level="code").transform(lambda x: x.rolling(10, min_periods=10).mean())
    output["bias_5"] = (close_safe / ma_5) - 1.0
    output["bias_10"] = (close_safe / ma_10) - 1.0

    def _rsi_wilder(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta.clip(upper=0.0))
        avg_gain = gain.ewm(alpha=1.0 / float(period), adjust=False, min_periods=int(period)).mean()
        avg_loss = loss.ewm(alpha=1.0 / float(period), adjust=False, min_periods=int(period)).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
        return rsi

    output["rsi_6"] = close_safe.groupby(level="code", sort=False).transform(lambda x: _rsi_wilder(x, 6))

    avg_price = turnover / volume.replace(0.0, np.nan)
    avg_price_safe = avg_price.replace(0.0, np.nan)
    output["f_vwap_bias"] = (close - avg_price_safe) / avg_price_safe

    preclose = _num_series("preclose")
    if preclose.isna().all():
        preclose = close.groupby(level="code").shift(1)
    open_safe = open_p.replace(0.0, np.nan)
    preclose_safe = preclose.replace(0.0, np.nan)
    r_night = (open_safe / preclose_safe) - 1.0
    r_day = (close / open_safe) - 1.0
    output["f_intraday_reversal"] = -1.0 * r_night * r_day

    denom_hl = (high - low) + 1e-6
    output["f_candle_strength"] = (close - open_p) / denom_hl

    trend_r2_20 = (
        close_safe.groupby(level="code", sort=False)
        .apply(lambda x: _rolling_trend_r2(x, 20))
        .reset_index(level=0, drop=True)
    )
    output["f_trend_r2_20"] = trend_r2_20

    amihud_daily = ret_1.abs() / (turnover + 1e-6)
    output["f_amihud_liquidity_20"] = (
        amihud_daily.groupby(level="code")
        .rolling(20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    output["f_shadow_skew"] = ((close - low) - (high - close)) / denom_hl

    return output
