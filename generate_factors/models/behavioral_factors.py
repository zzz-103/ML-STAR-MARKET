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


def _rolling_corr_r2(x_series: pd.Series, y_series: pd.Series, window: int) -> pd.Series:
    x = pd.to_numeric(x_series, errors="coerce").to_numpy(dtype="float64", copy=True)
    y = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype="float64", copy=True)
    n = int(x.shape[0])
    out = np.full(n, np.nan, dtype="float64")
    w = int(window)
    if w <= 1 or n < w:
        return pd.Series(out, index=x_series.index)

    finite = np.isfinite(x) & np.isfinite(y)
    x0 = np.where(finite, x, 0.0)
    y0 = np.where(finite, y, 0.0)
    ones = np.ones(w, dtype="float64")

    sum_x = np.convolve(x0, ones, mode="valid")
    sum_y = np.convolve(y0, ones, mode="valid")
    sum_x2 = np.convolve(x0 * x0, ones, mode="valid")
    sum_y2 = np.convolve(y0 * y0, ones, mode="valid")
    sum_xy = np.convolve(x0 * y0, ones, mode="valid")

    count = np.convolve(finite.astype("float64"), ones, mode="valid")
    valid = count == float(w)

    cov = sum_xy - (sum_x * sum_y) / float(w)
    var_x = sum_x2 - (sum_x * sum_x) / float(w)
    var_y = sum_y2 - (sum_y * sum_y) / float(w)
    denom = var_x * var_y

    r2 = np.full_like(sum_x, np.nan, dtype="float64")
    good = valid & np.isfinite(denom) & (denom > 0.0)
    r2[good] = (cov[good] * cov[good]) / denom[good]
    r2 = np.clip(r2, 0.0, 1.0, out=r2, where=np.isfinite(r2))

    out[w - 1 :] = r2
    return pd.Series(out, index=x_series.index)


def run(df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=df.index)

    def _num_series(col: str) -> pd.Series:
        if col in df.columns:
            s = df[col]
        else:
            s = pd.Series(np.nan, index=df.index)
        return pd.to_numeric(s, errors="coerce")

    def _num_series_any(cols: list[str]) -> pd.Series:
        for c in cols:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce")
                if not s.isna().all():
                    return s
        return pd.Series(np.nan, index=df.index, dtype="float64")

    open_p = _num_series("open")
    high = _num_series("high")
    low = _num_series("low")
    close = _num_series("close")
    volume = _num_series("volume")
    turnover = _num_series("turnover")
    float_market_cap = _num_series("float_market_cap")

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
    roc_60 = close_safe.groupby(level="code").pct_change(60, fill_method=None)
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

    sp_ttm = _num_series_any(["SP_ttm", "sp_ttm"])
    sp_rank = sp_ttm.groupby(level="date", sort=False).rank(pct=True)
    momo_rank = roc_60.groupby(level="date", sort=False).rank(pct=True)
    output["f_psg_proxy"] = sp_rank + momo_rank

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
    amihud_5 = (
        amihud_daily.groupby(level="code")
        .rolling(5, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    output["f_amihud_trend"] = amihud_5 / (output["f_amihud_liquidity_20"] + 1e-6)

    output["f_shadow_skew"] = ((close - low) - (high - close)) / denom_hl

    daily_vwap = turnover / (volume + 1e-6)
    daily_vwap_safe = daily_vwap.replace(0.0, np.nan)
    smart_score = (close - daily_vwap_safe) / daily_vwap_safe
    smart_num = (
        (smart_score * volume)
        .groupby(level="code")
        .rolling(20, min_periods=20)
        .sum()
        .reset_index(level=0, drop=True)
    )
    smart_den = (
        volume.groupby(level="code")
        .rolling(20, min_periods=20)
        .sum()
        .reset_index(level=0, drop=True)
    )
    output["f_smart_money_20"] = smart_num / (smart_den + 1e-6)

    sm_rank_20 = (
        output["f_smart_money_20"]
        .groupby(level="code")
        .rolling(20, min_periods=20)
        .rank(pct=True)
        .reset_index(level=0, drop=True)
    )
    roc_rank_20 = (
        output["roc_20"]
        .groupby(level="code")
        .rolling(20, min_periods=20)
        .rank(pct=True)
        .reset_index(level=0, drop=True)
    )
    output["f_smart_money_div"] = sm_rank_20 - roc_rank_20

    vp_corr_20 = grouped.apply(
        lambda x: pd.to_numeric(x["volume"], errors="coerce")
        .diff()
        .rolling(20, min_periods=20)
        .corr(pd.to_numeric(x["close"], errors="coerce").replace(0.0, np.nan).pct_change(fill_method=None).abs())
    ).reset_index(level=0, drop=True)
    output["f_vp_rank_corr_20"] = -1.0 * vp_corr_20

    turn_rate = None
    if "turnover_rate" in df.columns:
        tr = _num_series("turnover_rate")
        if not tr.isna().all():
            turn_rate = tr
    if turn_rate is None and "Turnover" in df.columns:
        tr = _num_series("Turnover")
        if not tr.isna().all():
            turn_rate = tr
    if turn_rate is None:
        turn_rate = turnover / (float_market_cap + 1e-6)

    turn_ma5 = turn_rate.groupby(level="code").transform(lambda x: x.rolling(5, min_periods=5).mean())
    turn_ma20 = turn_rate.groupby(level="code").transform(lambda x: x.rolling(20, min_periods=20).mean())
    turn_std20 = turn_rate.groupby(level="code").transform(lambda x: x.rolling(20, min_periods=20).std())
    output["f_turn_stability_20"] = turn_std20 / (turn_ma20 + 1e-6)

    output["f_turn_acc"] = turn_ma5 / (turn_ma20 + 1e-6)

    output["f_abnormal_turn_20"] = (turn_rate - turn_ma20) / (turn_std20 + 1e-6)

    corr_absret_turn_20 = (
        pd.DataFrame({"a": ret_1.abs(), "t": turn_rate}, index=df.index)
        .groupby(level="code", sort=False)
        .apply(lambda g: g["a"].rolling(20, min_periods=20).corr(g["t"]))
        .reset_index(level=0, drop=True)
    )
    absret_std20 = (
        ret_1.abs()
        .groupby(level="code", sort=False)
        .rolling(20, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )
    output["f_resid_vol"] = absret_std20 * np.sqrt(1.0 - corr_absret_turn_20.pow(2).fillna(0.0).clip(0.0, 1.0))

    vol_60 = (
        ret_1.groupby(level="code")
        .rolling(60, min_periods=60)
        .std()
        .reset_index(level=0, drop=True)
    )

    pb = _num_series_any(["pb", "PB"])
    pb_rank = pb.groupby(level="date", sort=False).rank(pct=True)

    dates = df.index.get_level_values("date")
    market_ret_by_date = ret_1.groupby(level="date", sort=False).mean()
    market_ret_aligned = pd.Series(market_ret_by_date.reindex(dates).to_numpy(), index=df.index)
    r2_mkt_60 = (
        pd.DataFrame({"x": ret_1, "m": market_ret_aligned}, index=df.index)
        .groupby(level="code", sort=False)
        .apply(lambda g: _rolling_corr_r2(g["x"], g["m"], 60))
        .reset_index(level=0, drop=True)
    )
    idio_60 = 1.0 - r2_mkt_60.clip(lower=0.0, upper=1.0)
    idio_rank = idio_60.groupby(level="date", sort=False).rank(pct=True)
    output["f_tech_premium"] = pb_rank * idio_rank

    pe_ttm = _num_series_any(["pe_ttm", "PE_ttm", "pe", "PE"])
    ep = (1.0 / pe_ttm.replace(0.0, np.nan)).where(pe_ttm > 0.0)
    ep_rank = ep.groupby(level="date", sort=False).rank(pct=True).fillna(0.0)
    vol_rank_60 = vol_60.groupby(level="date", sort=False).rank(pct=True)
    output["f_val_vol_spread"] = ep_rank - vol_rank_60

    return output
