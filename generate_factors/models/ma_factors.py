import pandas as pd

def run(df):
    """
    均线策略组
    输入: 清洗后的 MultiIndex DataFrame (date, code)
    输出: DataFrame, 列名为因子名称
    """
    output = pd.DataFrame(index=df.index)
    
    # 使用 groupby 加速计算
    # level='code' 表示按股票分组
    grouped_close = df.groupby(level='code')['close']
    
    # --- 因子 1: MA5 乖离率 (MA5_Bias) ---
    # 逻辑: 股价 / 5日均线 - 1
    ma5 = grouped_close.transform(lambda x: x.rolling(5).mean())
    output['ma5_bias'] = (df['close'] / ma5) - 1
    
    # --- 因子 2: 黄金交叉信号 (MA5_10_Diff) ---
    # 逻辑: MA5 - MA10 (数值越大表示短期趋势越强)
    ma10 = grouped_close.transform(lambda x: x.rolling(10).mean())
    output['ma5_10_diff'] = ma5 - ma10

    ma60 = grouped_close.transform(lambda x: x.rolling(60).mean())
    ma60_safe = ma60.where(ma60 != 0)
    output['ma60_bias'] = (df['close'] - ma60_safe) / ma60_safe
    
    return output
