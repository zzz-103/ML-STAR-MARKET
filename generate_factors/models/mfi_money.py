import pandas as pd
import numpy as np

def run(df):
    """
    资金流量指标 (MFI)
    """
    output = pd.DataFrame(index=df.index)
    
    # MFI 周期
    window = 14
    
    # 1. 典型价格
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    # 2. 原始资金流 (Raw Money Flow)
    rmf = tp * df['volume']
    
    # 3. 区分正向流和负向流
    # 这里由于 tp 计算逻辑比较碎，为了代码简洁且不出错，
    # 我们直接操作 Series，利用 groupby 的 shift
    
    # 重新获取干净的 TP Series
    s_tp = (df['high'] + df['low'] + df['close']) / 3
    s_prev_tp = s_tp.groupby(level='code').shift(1)
    
    # 定义正负流
    # 如果 TP > Prev_TP，则流为 RMF，否则为 0
    pos_flow = np.where(s_tp > s_prev_tp, rmf, 0)
    # 如果 TP < Prev_TP，则流为 RMF，否则为 0
    neg_flow = np.where(s_tp < s_prev_tp, rmf, 0)
    
    # 转回 Series 方便 rolling
    s_pos = pd.Series(pos_flow, index=df.index)
    s_neg = pd.Series(neg_flow, index=df.index)
    
    # 4. 滚动求和 (14天)
    roll_pos = s_pos.groupby(level='code').transform(lambda x: x.rolling(window).sum())
    roll_neg = s_neg.groupby(level='code').transform(lambda x: x.rolling(window).sum())
    
    mfr = roll_pos / roll_neg
    mfi = 100 - (100 / (1 + mfr))
    mfi = mfi.mask((roll_neg == 0) & (roll_pos == 0), 50)
    mfi = mfi.mask((roll_neg == 0) & (roll_pos > 0), 100)
    output['mfi_14'] = mfi
    
    return output
