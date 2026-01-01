import pandas as pd
import os
import numpy as np
from pathlib import Path

# === 配置 ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FF_FILE_PATH = str(PROJECT_ROOT / "factors_data" / "ff_factors" / "ff_factors.csv")
FF_MARKETTYPE_ID = "P9714"
ROLLING_WINDOW = 60

def run(df):
    """
    计算 Fama-French 三因子的 Rolling Beta
    Beta = Cov(Stock, Factor) / Var(Factor)
    """
    # 1. 检查文件是否存在
    if not os.path.exists(FF_FILE_PATH):
        return pd.DataFrame(index=df.index)

    # 2. 读取并清洗 FF 数据
    df_ff = pd.read_csv(FF_FILE_PATH, dtype=str)
    
    # 检查字段
    required = {"MarkettypeID", "TradingDate", "RiskPremium1", "SMB1", "HML1"}
    missing = required - set(df_ff.columns)
    if missing:
        raise ValueError(f"FF 因子文件缺少字段: {sorted(missing)}")

    # 筛选 P9714 并转换格式
    df_ff = df_ff[df_ff["MarkettypeID"].astype(str) == str(FF_MARKETTYPE_ID)].copy()
    df_ff["TradingDate"] = pd.to_datetime(df_ff["TradingDate"], errors="coerce")
    
    # 提取三个因子 (使用流通市值加权: 1)
    df_ff["ff_mkt"] = pd.to_numeric(df_ff["RiskPremium1"], errors="coerce")
    df_ff["ff_smb"] = pd.to_numeric(df_ff["SMB1"], errors="coerce")
    df_ff["ff_hml"] = pd.to_numeric(df_ff["HML1"], errors="coerce")
    
    # 按日期去重并排序
    df_ff = (
        df_ff.dropna(subset=["TradingDate"])
        .set_index("TradingDate")[["ff_mkt", "ff_smb", "ff_hml"]]
        .groupby(level=0, as_index=True)
        .mean(numeric_only=True)
        .sort_index()
    )

    close_safe = df["close"].replace(0.0, np.nan)
    ret = close_safe.groupby(level="code").pct_change(fill_method=None)
    dates = df.index.get_level_values("date")

    output = pd.DataFrame(index=df.index)
    min_periods = int(ROLLING_WINDOW)

    for factor_col, out_col in (("ff_mkt", "beta_mkt"), ("ff_smb", "beta_smb"), ("ff_hml", "beta_hml")):
        factor_daily = df_ff[factor_col]
        factor_aligned = factor_daily.reindex(dates)
        factor_aligned.index = df.index

        prod = ret * factor_aligned

        mean_ret = (
            ret.groupby(level="code")
            .rolling(min_periods, min_periods=min_periods)
            .mean()
        )
        mean_ret.index = mean_ret.index.droplevel(0)

        mean_prod = (
            prod.groupby(level="code")
            .rolling(min_periods, min_periods=min_periods)
            .mean()
        )
        mean_prod.index = mean_prod.index.droplevel(0)

        mean_factor_daily = factor_daily.rolling(min_periods, min_periods=min_periods).mean()
        mean_factor_aligned = mean_factor_daily.reindex(dates)
        mean_factor_aligned.index = df.index

        mean_factor2_daily = (factor_daily * factor_daily).rolling(min_periods, min_periods=min_periods).mean()
        mean_factor2_aligned = mean_factor2_daily.reindex(dates)
        mean_factor2_aligned.index = df.index

        cov = mean_prod - mean_ret * mean_factor_aligned
        var = mean_factor2_aligned - mean_factor_aligned * mean_factor_aligned
        beta = cov / var.replace(0.0, np.nan)
        output[out_col] = beta

    return output
