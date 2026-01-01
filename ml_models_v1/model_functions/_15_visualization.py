# 位置: 15（文本可视化）| main.py 输出评估 ASCII 报告
# 输入: dict[str, pd.Series]（例如 daily_ic/topk_mean）
# 输出: str（可直接写入日志/文件）
# 依赖: numpy/pandas
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def ascii_sparkline(values: Iterable[float], width: int = 60) -> str:
    seq = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if len(seq) == 0:
        return ""
    if width <= 0:
        width = 60
    if len(seq) > width:
        idx = np.linspace(0, len(seq) - 1, num=width).round().astype(int)
        seq = [seq[i] for i in idx]
    blocks = "▁▂▃▄▅▆▇█"
    lo = min(seq)
    hi = max(seq)
    if math.isclose(lo, hi):
        return blocks[0] * len(seq)
    out = []
    for v in seq:
        p = (v - lo) / (hi - lo)
        k = int(round(p * (len(blocks) - 1)))
        k = max(0, min(len(blocks) - 1, k))
        out.append(blocks[k])
    return "".join(out)


def summarize_series(name: str, s: pd.Series) -> str:
    s1 = pd.to_numeric(s, errors="coerce")
    s1 = s1.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s1) == 0:
        return f"{name}: empty"
    mean = float(s1.mean())
    std = float(s1.std(ddof=1)) if len(s1) > 1 else float("nan")
    ir = float(mean / std) if np.isfinite(mean) and np.isfinite(std) and std > 1e-12 else float("nan")
    pos = float((s1 > 0).mean())
    return f"{name}: n={len(s1)} mean={mean:.4f} std={std:.4f} ir={ir:.3f} pos={pos:.3f}"


def build_ascii_report(title: str, series_map: dict[str, pd.Series]) -> str:
    lines: list[str] = []
    lines.append(str(title))
    for k, s in series_map.items():
        lines.append(summarize_series(k, s))
        sp = ascii_sparkline(s.to_numpy(dtype="float64"), width=80)
        if sp:
            lines.append(sp)
    return "\n".join(lines) + "\n"
