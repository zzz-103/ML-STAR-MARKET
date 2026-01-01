# 位置: 10（评估指标）| overfit 与 main 的临时评估使用
# 输入: y/pred（按 MultiIndex: date,code 对齐的 Series）
# 输出: 按 date 聚合的 Series（Spearman IC / TopK mean）
# 依赖: numpy/pandas
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_ic(y: pd.Series, pred: pd.Series, min_n: int = 20) -> pd.Series:
    df = pd.DataFrame({"y": y, "pred": pred})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) == 0:
        return pd.Series(dtype="float64")
    if not isinstance(df.index, pd.MultiIndex) or "date" not in df.index.names:
        return pd.Series(dtype="float64")
    out: dict[pd.Timestamp, float] = {}
    for d, g in df.groupby(level="date"):
        if len(g) < int(min_n):
            continue
        ic = g["pred"].corr(g["y"], method="spearman")
        if ic is None or not np.isfinite(ic):
            continue
        out[d] = float(ic)
    return pd.Series(out).sort_index()


def daily_topk_mean(y: pd.Series, pred: pd.Series, top_k: int, min_n: int = 20) -> pd.Series:
    df = pd.DataFrame({"y": y, "pred": pred})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) == 0:
        return pd.Series(dtype="float64")
    if not isinstance(df.index, pd.MultiIndex) or "date" not in df.index.names:
        return pd.Series(dtype="float64")
    k = max(1, int(top_k))
    out: dict[pd.Timestamp, float] = {}
    for d, g in df.groupby(level="date"):
        if len(g) < max(int(min_n), k):
            continue
        sel = g.nlargest(k, columns="pred")["y"]
        if len(sel) == 0:
            continue
        m = float(sel.mean())
        if np.isfinite(m):
            out[d] = m
    return pd.Series(out).sort_index()
