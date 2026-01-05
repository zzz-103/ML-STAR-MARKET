import pandas as pd
import numpy as np

def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    研发基本面因子 (R&D Fundamental Factors)
    基于年度更新的研发数据，经过 ffill 填充后构建日频因子。
    """
    output = pd.DataFrame(index=df.index)

    # 1. 检查必要列是否存在
    required_cols = ["RDSpendSum", "RDSpendSumRatio", "RDInvestRatio", "CirculatedMarketValue"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    # 如果缺少列，为了不中断流程，可以返回空 DF 或者填充 NaN，但最好打印警告
    # 这里我们选择如果缺失核心 R&D 数据则跳过计算 (返回空列或 NaN)
    if missing_cols:
        # 如果 CirculatedMarketValue 缺失，可能无法计算 f_price_to_rd
        # 如果 R&D 列缺失，无法计算另外两个
        # 简单起见，如果缺少任何一列，对应因子就无法计算
        pass

    # 2. 提取并预处理数据
    # RD 数据是年频/季频，需要对齐到日频 (ffill)
    # 注意：generate_factors 的输入 df 已经是 MultiIndex [date, code]
    # 我们先提取这几列，并按 code 分组做 ffill
    
    # 辅助函数：安全提取并 ffill
    def get_filled_series(col_name):
        if col_name not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        s = df[col_name]
        # GroupBy code + ffill 确保非财报发布日也有数据
        # 注意：数据预处理阶段通常已经 merge 到了日线表，但如果是 merge(how='left') 可能会有 NaN
        # 用户指令：执行 groupby('code').ffill()
        s_filled = s.groupby(level='code').ffill()
        
        # 用户指令：缺失值兜底填 0 (说明从未披露过)
        s_filled = s_filled.fillna(0.0)
        return pd.to_numeric(s_filled, errors='coerce')

    rd_spend_sum = get_filled_series("RDSpendSum")
    rd_spend_sum_ratio = get_filled_series("RDSpendSumRatio")
    rd_invest_ratio = get_filled_series("RDInvestRatio")
    
    # 流通市值通常是日频的，不需要 ffill (或者假设已经是日频)
    cmv = pd.to_numeric(df.get("CirculatedMarketValue", pd.Series(np.nan, index=df.index)), errors='coerce')

    # 3. 计算因子
    
    # f_rd_intensity: 研发强度 (直接取 RDSpendSumRatio)
    # 假设 RDSpendSumRatio 单位是百分比或小数，保持原样即可
    output["f_rd_intensity"] = rd_spend_sum_ratio
    
    # f_rd_cap_ratio: 研发资本化率 (直接取 RDInvestRatio)
    output["f_rd_cap_ratio"] = rd_invest_ratio
    
    # f_price_to_rd: 市研率 = 流通市值 / (研发投入总额 + epsilon)
    # 类似于 PE, PB，越低越好 (value factor)
    output["f_price_to_rd"] = cmv / (rd_spend_sum + 1e-6)
    
    return output
