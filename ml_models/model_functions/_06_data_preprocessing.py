# 位置: 06（数据预处理）| main.py 构建 df_ml/df_price（避免未来函数的 shift）
# 输入: args（factor_data_path/price_data_path/label 参数/股票池/流动性阈值等）
# 输出: df_ml(MultiIndex: date,code; 含 ret_next)、df_price(MultiIndex: date,code; 含 open/turnover_prev 等)
# 依赖: /ml_models/xgb_config.py、numpy/pandas
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from ml_models import xgb_config as cfg


def _parse_yyyymmdd(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s, format="%Y%m%d", errors="raise")
    except Exception as e:
        raise ValueError(f"日期需为 YYYYMMDD，当前={value!r}") from e


def ensure_factors_index(df_factors: pd.DataFrame) -> pd.DataFrame:
    """确保因子数据为 MultiIndex(date, code)，并按索引排序。"""
    if isinstance(df_factors.index, pd.MultiIndex) and list(df_factors.index.names)[:2] == ["date", "code"]:
        return df_factors.sort_index()
    if "date" not in df_factors.columns or "code" not in df_factors.columns:
        raise ValueError("因子数据缺少 date/code 列，无法构建 MultiIndex")
    df = df_factors.copy()
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df["code"] = df["code"].astype(str)
    df = df.dropna(subset=["date", "code"])
    return df.set_index(["date", "code"]).sort_index()


def build_tradable_mask(df_price: pd.DataFrame, min_turnover: float) -> pd.Series:
    """构建可交易掩码：open>0、非涨停买入、流动性满足阈值（若字段存在）。"""
    cond_active = df_price["open"].notna() & (df_price["open"] > 0)
    if "upper_limit" in df_price.columns:
        cond_no_limit_up = df_price["open"] < df_price["upper_limit"]
    else:
        cond_no_limit_up = True
    if "turnover_prev" in df_price.columns:
        cond_liquid = df_price["turnover_prev"] > float(min_turnover)
    else:
        cond_liquid = True
    return (cond_active & cond_no_limit_up & cond_liquid).rename("tradable_t")


def prepare_dataset(args, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载因子与价格数据，构建滞后特征，并生成超额收益标签 ret_next。"""
    factor_path = str(getattr(args, "factor_data_path"))
    price_path = str(getattr(args, "price_data_path"))
    if not os.path.exists(factor_path):
        raise FileNotFoundError(f"找不到因子文件: {factor_path}")
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"找不到价格文件: {price_path}")

    logger.info("step=load_factors path=%s", factor_path)
    df_factors = ensure_factors_index(pd.read_parquet(factor_path))
    prefixes = tuple(getattr(args, "stock_pool_prefixes", cfg.DEFAULT_UNIVERSE["stock_pool_prefixes"]))
    pool_mask_f = df_factors.index.get_level_values("code").astype(str).str.startswith(prefixes)
    df_factors = df_factors.loc[pool_mask_f, :]

    end_dt = _parse_yyyymmdd(getattr(args, "end_date", None))
    train_gap = int(getattr(args, "train_gap", cfg.DEFAULT_TRAINING.get("train_gap", 6)))
    label_horizon_max = int(cfg.DEFAULT_LABEL.get("predict_days", 5))
    safe_gap_min = int(label_horizon_max + 1)
    leak_margin = int(train_gap - label_horizon_max)
    risk = int(train_gap < safe_gap_min)
    logger.info(
        "泄漏检查 gap=%d | 标签最远=+%d日 | 安全gap>=%d | 风险=%d | 最贴近标签点=target-%d (train_end+%d)",
        int(train_gap),
        int(label_horizon_max),
        int(safe_gap_min),
        int(risk),
        int(leak_margin),
        int(label_horizon_max),
    )
    if bool(risk):
        raise ValueError(f"train_gap({train_gap}) 过小：标签最远需要 t+{label_horizon_max}，为避免穿越需 train_gap>={safe_gap_min}")

    logger.info("步骤: 特征平移 shift=1")
    df_factors_shifted = df_factors.groupby(level="code").shift(1)

    logger.info("步骤: 加载价格数据 path=%s", price_path)
    df_price = pd.read_parquet(price_path).sort_index()
    logger.info("价格数据加载完毕 shape=%s industry列=%s", df_price.shape, "industry" in df_price.columns)
    pool_mask_p = df_price.index.get_level_values("code").astype(str).str.startswith(prefixes)
    df_price = df_price.loc[pool_mask_p, :]
    if end_dt is not None:
        idx = pd.IndexSlice
        df_factors_shifted = df_factors_shifted.loc[idx[:end_dt, :], :]
        df_price = df_price.loc[idx[:end_dt, :], :]

    if "turnover" in df_price.columns:
        df_price["turnover_prev"] = df_price.groupby(level="code")["turnover"].shift(1)
    if "volume" in df_price.columns:
        df_price["volume_prev"] = df_price.groupby(level="code")["volume"].shift(1)

    logger.info("步骤: 构建标签 label=%d日收益", int(label_horizon_max))
    current_open = df_price["open"].replace(0, np.nan)
    ret = (df_price.groupby(level="code")["open"].shift(-int(label_horizon_max)) - current_open) / current_open
    df_price["ret_next_raw"] = ret

    min_turnover = float(getattr(args, "min_turnover", cfg.DEFAULT_UNIVERSE["min_turnover"]))
    tradable_t = build_tradable_mask(df_price, min_turnover=min_turnover)
    bench_universe = str(getattr(args, "label_benchmark_universe", cfg.DEFAULT_LABEL["label_benchmark_universe"])).lower()
    raw_for_bench = df_price["ret_next_raw"].where(tradable_t.astype(bool)) if bench_universe == "tradable" else df_price["ret_next_raw"]

    bench_method = str(getattr(args, "label_benchmark_method", cfg.DEFAULT_LABEL["label_benchmark_method"])).lower()
    if bench_method == "mean":
        market_benchmark = raw_for_bench.groupby(level="date").transform("mean")
    else:
        market_benchmark = raw_for_bench.groupby(level="date").transform("median")
    df_price["ret_next"] = df_price["ret_next_raw"] - market_benchmark

    df_target = df_price[["ret_next"]].join(tradable_t, how="left")
    df_target = df_target.replace([np.inf, -np.inf], np.nan).dropna(subset=["ret_next"])
    df_target = df_target[df_target["tradable_t"].astype(bool)][["ret_next"]]

    logger.info("步骤: 合并特征与标签 (Inner Join)")
    df_ml = df_factors_shifted.join(df_target, how="inner")
    if bool(getattr(args, "dropna_features", False)):
        df_ml = df_ml.dropna()
    logger.info("数据集样本数=%d", int(len(df_ml)))
    if end_dt is not None and len(df_ml) > 0:
        max_date = pd.to_datetime(df_ml.index.get_level_values("date").max())
        logger.info("数据集日期范围=%s~%s (end_date=%s)", df_ml.index.get_level_values("date").min().strftime("%Y%m%d"), max_date.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"))
    return df_ml, df_price
