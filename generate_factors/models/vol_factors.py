import pandas as pd
import numpy as np

def run(df):
    """
    波动率策略组
    """
    output = pd.DataFrame(index=df.index)
    close_safe = df["close"].replace(0, np.nan)
    
    # 基于原始数据分组 (用于计算 vol_20)
    grouped_raw = df.groupby(level='code')
    
    # --- 因子 3: 20日历史波动率 (Vol_20) ---
    output['vol_20'] = grouped_raw['close'].transform(lambda x: x.rolling(20).std())
    
    # --- 因子 4: 真实波幅比率 (ATR_Ratio) ---
    # 1. 先算出每一天的数值 (这一步不需要分组，是列与列的加减乘除)
    output['hl_ratio'] = (df['high'] - df['low']) / close_safe
    
    # 2. 对这个新算出来的因子进行平滑 (MA5)
    # 【修正点】这里必须对 output 进行分组，因为 hl_ratio 在 output 里
    output['hl_ratio_smooth'] = output.groupby(level='code')['hl_ratio'].transform(lambda x: x.rolling(5).mean())
    
    return output
