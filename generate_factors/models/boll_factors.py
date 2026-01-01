import pandas as pd
import numpy as np

def run(df):
    """
    布林带因子组
    """
    output = pd.DataFrame(index=df.index)
    grouped = df.groupby(level='code')['close']
    
    # 标准参数：20日均线，2倍标准差
    window = 20
    k = 2
    
    # 1. 计算中轨 (MA20)
    mid = grouped.transform(lambda x: x.rolling(window).mean())
    
    # 2. 计算标准差
    std = grouped.transform(lambda x: x.rolling(window).std())
    
    # 3. 计算上轨和下轨
    upper = mid + k * std
    lower = mid - k * std
    
    # --- 因子 1: %B 指标 (价格在布林带中的相对位置) ---
    # 公式: (Close - Lower) / (Upper - Lower)
    # >1 说明突破上轨(强势)，<0 说明跌破下轨(弱势或超卖)
    # 处理分母为0的情况 (极少见，除非停牌)
    bandwidth_diff = upper - lower
    output['boll_pct_b'] = (df['close'] - lower) / bandwidth_diff.replace(0, np.nan)
    
    # --- 因子 2: BandWidth (带宽) ---
    # 公式: (Upper - Lower) / Mid
    # 用来衡量波动率的压缩与扩张。数值越小，说明盘整越久，变盘概率越大。
    output['boll_width'] = bandwidth_diff / mid
    
    return output