import pandas as pd
import numpy as np

def run(df):
    """
    情绪与量价因子组：PSY, VR
    """
    output = pd.DataFrame(index=df.index)
    grouped = df.groupby(level='code')
    
    # 计算单日涨跌
    # diff > 0 为涨， < 0 为跌
    price_change = grouped['close'].diff()
    
    # --- 1. PSY (Psychological Line) 心理线 ---
    # 过去 12 天内上涨天数的比例
    # (price_change > 0) 得到布尔值，转为 int (1或0)，然后求 rolling mean
    is_up = (price_change > 0).astype(float)
    output['psy_12'] = is_up.groupby(level='code').transform(
        lambda x: x.rolling(12).mean()
    ) * 100
    
    # --- 2. VR (Volume Ratio) 容量比率 ---
    # 公式：(上涨日成交量 + 1/2平盘日成交量) / (下跌日成交量 + 1/2平盘日成交量)
    # 这种计算在 Pandas 里如果不写循环比较麻烦，我们用简化的算法：
    # 24天内：Sum(Volume where Close Up) / Sum(Volume where Close Down)
    
    vol = df['volume']
    
    # 构造辅助列
    vol_up = vol.where(price_change > 0, 0)   # 只有上涨日的量，其他为0
    vol_down = vol.where(price_change < 0, 0) # 只有下跌日的量，其他为0
    
    roll_vol_up = vol_up.groupby(level='code').rolling(24).sum().droplevel(0)
    roll_vol_down = vol_down.groupby(level='code').rolling(24).sum().droplevel(0)
    
    # 加 1 避免除以 0
    output['vr_24'] = (roll_vol_up + 1) / (roll_vol_down + 1)
    
    # --- 3. 量价相关性 (Volume-Price Correlation) ---
    # 极其有效的因子：过去10天价格和量的相关系数
    # 缩量上涨 vs 放量下跌
    output['pv_corr'] = grouped.apply(
        lambda x: x['close'].rolling(10).corr(x['volume'])
    ).reset_index(level=0, drop=True)

    return output