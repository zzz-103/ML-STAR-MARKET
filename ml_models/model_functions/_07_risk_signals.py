# 位置: 07（风控信号）| portfolio 根据指数均线/双均线生成 risk_on 信号
# 输入: merged csv 路径、指数代码、均线窗口、起止日期(YYYYMMDD)
# 输出: DataFrame(index=date; columns=close/ma[/ma_fast/ma_slow]，已 shift(1) 作为前一日信号)
# 依赖: pandas/os
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def load_stock_industry_map(json_path: str) -> dict[str, str]:
    p = str(json_path or "").strip()
    if not p or (not os.path.exists(p)):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}

    if isinstance(raw, dict):
        out: dict[str, str] = {}
        for k, v in raw.items():
            if k is None:
                continue
            code = str(k).strip()
            if not code:
                continue
            ind = str(v).strip() if v is not None else ""
            out[code] = ind if ind else "Unknown"
        return out
    return {}


def attach_industry_to_price(
    df_price: pd.DataFrame,
    stock_industry_map: dict[str, str],
    *,
    industry_col: str = "industry",
    unknown_label: str = "Unknown",
) -> pd.DataFrame:
    if not isinstance(df_price.index, pd.MultiIndex) or list(df_price.index.names)[:2] != ["date", "code"]:
        raise ValueError("df_price 需为 MultiIndex(date, code)")
    if industry_col in df_price.columns:
        return df_price
    codes = df_price.index.get_level_values("code").astype("string")
    mapper = {str(k): str(v) for k, v in (stock_industry_map or {}).items()}
    ind = codes.astype(str).map(mapper).fillna(str(unknown_label)).astype("string")
    out = df_price.copy()
    out[industry_col] = ind.to_numpy()
    return out


def compute_sector_daily_returns(
    df_price: pd.DataFrame,
    *,
    close_col: str = "close",
    industry_col: str = "industry",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if not isinstance(df_price.index, pd.MultiIndex) or list(df_price.index.names)[:2] != ["date", "code"]:
        raise ValueError("df_price 需为 MultiIndex(date, code)")
    if industry_col not in df_price.columns:
        raise ValueError(f"df_price 缺少行业列 {industry_col!r}")
    if close_col not in df_price.columns:
        raise ValueError(f"df_price 缺少价格列 {close_col!r}")

    df = df_price
    if start_date or end_date:
        s = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce") if start_date else None
        e = pd.to_datetime(end_date, format="%Y%m%d", errors="coerce") if end_date else None
        dates = df.index.get_level_values("date")
        m = pd.Series(True, index=df.index)
        if s is not None and not pd.isna(s):
            m &= dates >= s
        if e is not None and not pd.isna(e):
            m &= dates <= e
        df = df.loc[m.to_numpy(dtype=bool), :]

    close = pd.to_numeric(df[close_col], errors="coerce")
    ret = close.groupby(level="code").pct_change(fill_method=None)
    ind = df[industry_col].astype("string")
    d = df.index.get_level_values("date")
    tmp = pd.DataFrame({"date": d, "industry": ind, "ret": ret})
    tmp = tmp.dropna(subset=["ret", "industry"])
    tmp = tmp.reset_index(drop=True)
    sector_ret = tmp.groupby(["date", "industry"], sort=True)["ret"].mean().unstack("industry").sort_index()
    return sector_ret


def compute_equal_weight_index_close(
    df_price: pd.DataFrame,
    *,
    close_col: str = "close",
    start_date: str | None = None,
    end_date: str | None = None,
    base: float = 1.0,
) -> pd.Series:
    if not isinstance(df_price.index, pd.MultiIndex) or list(df_price.index.names)[:2] != ["date", "code"]:
        raise ValueError("df_price 需为 MultiIndex(date, code)")
    if close_col not in df_price.columns:
        raise ValueError(f"df_price 缺少价格列 {close_col!r}")

    df = df_price
    if start_date or end_date:
        s = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce") if start_date else None
        e = pd.to_datetime(end_date, format="%Y%m%d", errors="coerce") if end_date else None
        dates = df.index.get_level_values("date")
        m = pd.Series(True, index=df.index)
        if s is not None and not pd.isna(s):
            m &= dates >= s
        if e is not None and not pd.isna(e):
            m &= dates <= e
        df = df.loc[m.to_numpy(dtype=bool), :]

    close = pd.to_numeric(df[close_col], errors="coerce")
    close = close.where(close > 0)
    ret = close.groupby(level="code").pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    ew_ret = ret.groupby(level="date").mean().sort_index()
    ew_ret = ew_ret.fillna(0.0).astype("float64")
    idx_close = (1.0 + ew_ret).cumprod() * float(base)
    idx_close.name = "close"
    return idx_close


def load_self_equal_weight_ma_risk_signal(
    df_price: pd.DataFrame,
    ma_window: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    window = int(ma_window)
    if window <= 0:
        return None
    start_s = str(start_date).strip()
    end_s = str(end_date).strip()
    if not start_s or not end_s:
        return None

    start_dt = pd.to_datetime(start_s, format="%Y%m%d", errors="coerce")
    end_dt = pd.to_datetime(end_s, format="%Y%m%d", errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt):
        return None
    need_start = (start_dt - pd.Timedelta(days=max(60, window * 3))).strftime("%Y%m%d")

    try:
        close = compute_equal_weight_index_close(df_price, start_date=need_start, end_date=end_s)
    except Exception:
        return None
    if close is None or len(close) == 0:
        return None

    close = close.sort_index()
    ma = close.rolling(window=window, min_periods=window).mean()
    out = pd.DataFrame({"close": close, "ma": ma})
    out = out.loc[(out.index >= start_dt) & (out.index <= end_dt), :]
    if len(out) == 0:
        return None
    out = out.shift(1)
    out["risk_on"] = (out["close"] > out["ma"]).astype("float64")
    out["risk_on"] = out["risk_on"].fillna(1.0).astype("float64")
    return out


def build_sector_index(sector_ret: pd.DataFrame, *, base: float = 1.0, fillna_return: float = 0.0) -> pd.DataFrame:
    if sector_ret is None or len(sector_ret) == 0:
        return pd.DataFrame()
    r = sector_ret.copy()
    r = r.replace([np.inf, -np.inf], np.nan)
    if fillna_return is not None:
        r = r.fillna(float(fillna_return))
    idx = (1.0 + r).cumprod(axis=0) * float(base)
    return idx


def compute_sector_momentum_rank(
    sector_index: pd.DataFrame,
    *,
    window: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sector_index is None or len(sector_index) == 0:
        empty = pd.DataFrame()
        return empty, empty
    w = int(window)
    if w <= 0:
        raise ValueError("window 必须为正整数")
    mom = (sector_index / sector_index.shift(w)) - 1.0
    rank_pct = mom.rank(axis=1, pct=True, ascending=True)
    return mom, rank_pct


def compute_sector_trend_signal(
    sector_index: pd.DataFrame,
    *,
    ma_window: int = 20,
    buffer: float = 0.0,
    shift_signal: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sector_index is None or len(sector_index) == 0:
        empty = pd.DataFrame()
        return empty, empty
    w = int(ma_window)
    if w <= 0:
        raise ValueError("ma_window 必须为正整数")
    buf = float(buffer or 0.0)
    ma = sector_index.rolling(window=w, min_periods=w).mean()
    thr = ma * (1.0 + buf)
    on = (sector_index > thr).astype("float64")
    if shift_signal:
        on = on.shift(1).fillna(1.0).astype("float64")
    return ma, on


def load_index_ma_risk_signal(
    csv_path: str,
    index_code: str,
    ma_window: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    if not csv_path or not os.path.exists(csv_path):
        return None
    base = str(index_code).strip()
    if not base:
        return None
    start_s = str(start_date).strip()
    end_s = str(end_date).strip()
    if not start_s or not end_s:
        return None

    window = int(ma_window)
    if window <= 0:
        return None

    start_dt = pd.to_datetime(start_s, format="%Y%m%d", errors="coerce")
    end_dt = pd.to_datetime(end_s, format="%Y%m%d", errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt):
        return None
    need_start = (start_dt - pd.Timedelta(days=max(60, window * 3))).strftime("%Y%m%d")
    end_need = end_dt.strftime("%Y%m%d")

    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=["date", "code", "close"],
        dtype={"date": "string", "code": "string", "close": "float64"},
        chunksize=300_000,
    ):
        c = chunk["code"].astype("string")
        m_code = c.str.startswith(base, na=False)
        if not bool(m_code.any()):
            continue
        sub = chunk.loc[m_code, ["date", "code", "close"]].copy()
        d = sub["date"].astype("string")
        sub = sub.loc[d.between(need_start, end_need, inclusive="both")]
        if len(sub) == 0:
            continue
        pieces.append(sub)
    if len(pieces) == 0:
        return None

    df = pd.concat(pieces, axis=0, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    if len(df) == 0:
        return None

    df["ma"] = df["close"].rolling(window=window, min_periods=window).mean()
    df["risk_on"] = (df["close"] > df["ma"]).astype("float64")
    out = df.set_index("date")[["close", "ma", "risk_on"]]
    out = out.shift(1)
    out["risk_on"] = out["risk_on"].fillna(1.0).astype("float64")
    return out


def load_index_dual_ma_risk_signal(
    csv_path: str,
    index_code: str,
    fast_window: int,
    slow_window: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    if not csv_path or not os.path.exists(csv_path):
        return None
    base = str(index_code).strip()
    if not base:
        return None
    start_s = str(start_date).strip()
    end_s = str(end_date).strip()
    if not start_s or not end_s:
        return None

    w_fast = int(fast_window)
    w_slow = int(slow_window)
    if w_fast <= 0 or w_slow <= 0:
        return None
    if w_fast >= w_slow:
        w_fast, w_slow = min(w_fast, w_slow), max(w_fast, w_slow)

    start_dt = pd.to_datetime(start_s, format="%Y%m%d", errors="coerce")
    end_dt = pd.to_datetime(end_s, format="%Y%m%d", errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt):
        return None
    need_start = (start_dt - pd.Timedelta(days=max(120, w_slow * 3))).strftime("%Y%m%d")
    end_need = end_dt.strftime("%Y%m%d")

    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=["date", "code", "close"],
        dtype={"date": "string", "code": "string", "close": "float64"},
        chunksize=300_000,
    ):
        c = chunk["code"].astype("string")
        m_code = c.str.startswith(base, na=False)
        if not bool(m_code.any()):
            continue
        sub = chunk.loc[m_code, ["date", "code", "close"]].copy()
        d = sub["date"].astype("string")
        sub = sub.loc[d.between(need_start, end_need, inclusive="both")]
        if len(sub) == 0:
            continue
        pieces.append(sub)
    if len(pieces) == 0:
        return None

    df = pd.concat(pieces, axis=0, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    if len(df) == 0:
        return None

    df["ma_fast"] = df["close"].rolling(window=w_fast, min_periods=w_fast).mean()
    df["ma_slow"] = df["close"].rolling(window=w_slow, min_periods=w_slow).mean()
    out = df.set_index("date")[["close", "ma_fast", "ma_slow"]]
    out = out.shift(1)
    return out
