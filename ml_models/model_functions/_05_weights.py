# 位置: 05（权重工具）| 训练样本权重衰减与持仓单票权重上限重分配
# 输入: 训练日期索引、参考日、衰减参数；或权重序列与 max_w
# 输出: np.ndarray|None（sample_weight）或归一化后的 pd.Series（cap 后权重）
# 依赖: numpy/pandas
from __future__ import annotations

import numpy as np
import pandas as pd


def build_sample_weights(
    mode: str,
    train_dates: pd.Index | list,
    ref_date: pd.Timestamp,
    anchor_days: int,
    half_life_days: int,
    min_weight: float,
) -> np.ndarray | None:
    """按日期衰减生成样本权重；mode=none 时返回 None"""
    mode = str(mode or "none").lower()
    if mode == "none":
        return None

    dates = pd.to_datetime(train_dates)
    ref = pd.to_datetime(ref_date)
    age_days = (ref - dates).days.astype("int64")
    eff_age = age_days - int(anchor_days)
    eff_age = np.where(eff_age > 0, eff_age, 0)

    denom = float(max(1, int(half_life_days)))
    if mode == "time_decay_linear":
        w = 1.0 - (eff_age.astype("float64") / denom)
    else:
        w = np.power(0.5, eff_age.astype("float64") / denom)

    w = np.clip(w, float(min_weight), 1.0)
    return w.astype("float64")


def apply_max_weight_cap(weights: pd.Series, max_w: float) -> pd.Series:
    """对权重做单票上限约束并重分配，返回归一化后的相对权重"""
    if len(weights) == 0:
        return weights
    max_w = float(max_w)
    if not np.isfinite(max_w) or max_w <= 0:
        return weights.astype("float64")
    w = weights.astype("float64").copy()
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        return w
    w = w / s
    if len(w) * max_w < 1 - 1e-12:
        return w
    for _ in range(10):
        over = w > (max_w + 1e-12)
        if not bool(over.any()):
            break
        excess = float((w[over] - max_w).sum())
        w[over] = max_w
        under = ~over
        if not bool(under.any()):
            break
        under_sum = float(w[under].sum())
        if not np.isfinite(under_sum) or under_sum <= 0:
            break
        w[under] += w[under] / under_sum * excess
        s2 = float(w.sum())
        if np.isfinite(s2) and s2 > 0:
            w = w / s2
    return w
