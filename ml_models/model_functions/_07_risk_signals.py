# 位置: 07（风控信号）| portfolio 根据指数均线/双均线生成 risk_on 信号
# 输入: merged csv 路径、指数代码、均线窗口、起止日期(YYYYMMDD)
# 输出: DataFrame(index=date; columns=close/ma[/ma_fast/ma_slow]，已 shift(1) 作为前一日信号)
# 依赖: pandas/os
from __future__ import annotations

import os

import pandas as pd


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
