# 位置: 02（通用解析工具）| 被特征工程/过拟合/重要性等模块调用
# 输入: 字符串参数（YYYYMMDD / CSV 列表 / 约束表达式）
# 输出: Timestamp / list[str] / dict[str,int]
# 依赖: pandas
from __future__ import annotations

import pandas as pd


def parse_yyyymmdd(value: str | None) -> pd.Timestamp | None:
    """将 YYYYMMDD 字符串解析为 Timestamp；空值返回 None。"""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return pd.to_datetime(s, format="%Y%m%d", errors="raise")


def parse_csv_list(value: str | None) -> list[str]:
    """解析逗号分隔列表字符串，返回去空白后的字符串列表。"""
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def parse_constraints(value: str | None) -> dict[str, int]:
    """解析形如 name:sign,name:sign 的单调约束字符串，返回 {name: sign}。"""
    if value is None:
        return {}
    s = str(value).strip()
    if not s:
        return {}
    out: dict[str, int] = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"constraints 格式错误，需为 name:sign，例如 ret_lag_1:1。当前: {item}")
        name, sign = item.split(":", 1)
        name = name.strip()
        sign = sign.strip()
        if not name:
            continue
        try:
            sign_i = int(sign)
        except Exception as e:
            raise ValueError(f"constraints sign 需为 -1/0/1，当前: {item}") from e
        out[name] = sign_i
    return out
