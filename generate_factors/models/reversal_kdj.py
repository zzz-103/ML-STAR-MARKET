import pandas as pd
import numpy as np

def run(df):
    """
    反转因子组：KDJ, William R
    """
    output = pd.DataFrame(index=df.index)
    
    # 预计算 Rolling Max/Min
    # KDJ 一般由 9, 3, 3 参数构成
    window = 9
    grouped = df.groupby(level='code')
    
    high_9 = grouped['high'].transform(lambda x: x.rolling(window).max())
    low_9 = grouped['low'].transform(lambda x: x.rolling(window).min())
    close = df['close']
    
    denom_9 = high_9 - low_9
    rsv = (close - low_9) / denom_9 * 100
    rsv = rsv.mask(denom_9 == 0, 50)
    
    # --- 2. KDJ 计算 ---
    # K = 2/3 * Pre_K + 1/3 * RSV
    # 这在 pandas 里等价于 ewm(com=2)
    # 必须按 code 分组计算 ewm
    
    # 把 RSV 放入临时 DataFrame 方便分组
    temp_df = pd.DataFrame({'rsv': rsv}, index=df.index)
    
    output['kdj_k'] = temp_df.groupby(level='code')['rsv'].transform(
        lambda x: x.ewm(com=2, adjust=False).mean()
    )
    
    output['kdj_d'] = output.groupby(level='code')['kdj_k'].transform(
        lambda x: x.ewm(com=2, adjust=False).mean()
    )
    
    output['kdj_j'] = 3 * output['kdj_k'] - 2 * output['kdj_d']

    # --- 3. WR (Williams %R) ---
    # 公式: (High_n - Close) / (High_n - Low_n) * -100
    # 这里用 14 天
    wr_window = 14
    h_14 = grouped['high'].transform(lambda x: x.rolling(wr_window).max())
    l_14 = grouped['low'].transform(lambda x: x.rolling(wr_window).min())
    
    output['william_r'] = (h_14 - close) / (h_14 - l_14) * -100
    
    return output
