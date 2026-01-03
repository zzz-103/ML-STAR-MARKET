# 位置: 04（特征工程）| main/worker 生成最终特征列表与单调约束向量
# 输入: features(list[str])、drop/constraints 的配置字符串
# 输出: final_features(list[str])、constraints_dict(dict)、monotone_constraints(str|None)
# 依赖: /ml_models/xgb_config.py、/model_functions/_02_parsing_utils.py
from __future__ import annotations

from ml_models import xgb_config as cfg
from ml_models.model_functions._02_parsing_utils import parse_constraints, parse_csv_list


def build_drop_factors(use_default_drop_factors: bool, drop_factors_csv: str | None) -> set[str]:
    """构建因子剔除集合（默认 drop list + 命令行追加）。"""
    drop: set[str] = set()
    if bool(use_default_drop_factors):
        drop.update(cfg.DEFAULT_DROP_FACTORS)
    drop.update(parse_csv_list(drop_factors_csv))
    return drop


def build_keep_factors(use_default_keep_factors: bool, keep_factors_csv: str | None) -> set[str]:
    keep: set[str] = set()
    if bool(use_default_keep_factors):
        keep.update(getattr(cfg, "DEFAULT_KEEP_FACTORS", []))
    keep.update(parse_csv_list(keep_factors_csv))
    return keep


def apply_feature_filters(features: list[str], drop_factors: set[str], keep_factors: set[str] | None = None) -> list[str]:
    """对特征列表应用剔除规则（drop list + turnover_* 特殊规则）。"""
    out: list[str] = []
    keep = keep_factors or set()
    for f in features:
        sf = str(f)
        if keep and sf not in keep:
            continue
        if sf in drop_factors:
            continue
        if ("ret_next" in sf) or sf.endswith("_next"):
            continue
        if sf.startswith("turnover_") and sf != "turnover_bias_5":
            continue
        out.append(f)
    return out


def build_constraints_dict(use_constraints: bool, constraints_csv: str | None) -> dict[str, int]:
    """构建单调约束字典（默认约束 + 命令行覆盖/追加）。"""
    if not bool(use_constraints):
        return {}
    d = dict(cfg.DEFAULT_MONOTONE_CONSTRAINTS)
    d.update(parse_constraints(constraints_csv))
    cleaned: dict[str, int] = {}
    for k, v in d.items():
        if v is None:
            continue
        v_i = int(v)
        if v_i not in (-1, 0, 1):
            raise ValueError(f"constraints 仅支持 -1/0/1，当前: {k}={v}")
        if v_i == 0:
            continue
        cleaned[str(k)] = v_i
    return cleaned


def build_monotone_constraints(feature_names: list[str], constraints_dict: dict[str, int]) -> str | None:
    """将 {feature: sign} 映射为 XGBoost monotone_constraints 向量字符串。"""
    if not constraints_dict:
        return None
    vec = [int(constraints_dict.get(f, 0)) for f in feature_names]
    return "(" + ",".join(str(v) for v in vec) + ")"
