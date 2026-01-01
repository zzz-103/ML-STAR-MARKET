import pandas as pd
import numpy as np

def run(df):
    """
    CCI 顺势指标
    """
    output = pd.DataFrame(index=df.index)
    
    # CCI 需要用到 High, Low, Close
    # 计算典型价格 (Typical Price)
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    # 按股票分组
    # 注意：tp 是一个 Series，我们需要先把它变成 DataFrame 或者直接在 transform 里处理
    # 为了方便，我们先构造个临时 df
    temp_df = pd.DataFrame({'tp': tp}, index=df.index)
    grouped = temp_df.groupby(level='code')['tp']
    
    # 参数通常为 14
    window = 14
    
    # 1. 计算 TP 的移动平均 (MA_TP)
    ma_tp = grouped.transform(lambda x: x.rolling(window).mean())
    
    # 2. 计算平均绝对偏差 (Mean Deviation)
    # 公式: Mean(|TP - MA_TP|)
    # Pandas 的 rolling().std() 是标准差，这里我们需要手动算绝对偏差的均值
    # 技巧：先算出绝对偏差，再 rolling mean
    abs_dev = (temp_df['tp'] - ma_tp).abs()
    
    # 必须重新按 code 分组计算 rolling mean
    # 因为 abs_dev 目前只是序列，不包含分组信息
    # 我们可以利用原来的索引
    mean_dev = abs_dev.groupby(level='code').transform(lambda x: x.rolling(window).mean())
    
    # 3. 计算 CCI
    # 公式: (TP - MA_TP) / (0.015 * MeanDeviation)
    # 0.015 是常数，保证大部分数值在 -100 到 +100 之间
    output['cci_14'] = (temp_df['tp'] - ma_tp) / (0.015 * mean_dev)
    
    return output