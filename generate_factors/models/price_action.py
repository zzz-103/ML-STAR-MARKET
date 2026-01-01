import pandas as pd
import numpy as np

def run(df):
    """
    Price Action (量价行为) 因子组
    包含：滞后收益率序列、累计动量、K线影线形态
    """
    output = pd.DataFrame(index=df.index)
    
    # 基础数据
    close = df['close']
    open_p = df['open']
    high = df['high']
    low = df['low']
    close_safe = close.replace(0, np.nan)
    
    # 1. 每日收益率 (Daily Returns)
    # 这里的 pct_change() 是 T日 相对于 T-1日 的涨跌
    # 注意：generate_factors 是预计算，不需要 shift，
    # prepare_dataset 时会整体 shift(1) 来防止未来函数。
    ret_1d = close_safe.groupby(level='code').pct_change(fill_method=None)
    
    # --- A. 滞后收益率序列 (Sequence Features) ---
    # 让模型看到过去几天的具体走势，而不只是一个累计值
    output['ret_lag_1'] = ret_1d # 昨天(相对于交易日是前天)涨跌
    output['ret_lag_2'] = ret_1d.groupby(level='code').shift(1)
    output['ret_lag_3'] = ret_1d.groupby(level='code').shift(2)
    output['ret_lag_4'] = ret_1d.groupby(level='code').shift(3)
    output['ret_lag_5'] = ret_1d.groupby(level='code').shift(4)
    
    # --- B. 累计动量 (Momentum / ROC) ---
    # 过去 N 天的总涨幅
    output['roc_5'] = close_safe.groupby(level='code').pct_change(5, fill_method=None)
    output['roc_10'] = close_safe.groupby(level='code').pct_change(10, fill_method=None)
    output['roc_20'] = close_safe.groupby(level='code').pct_change(20, fill_method=None) # 月度动量
    
    # --- C. K线形态 (Candle Structure) ---
    # 为了消除股价绝对值影响，全部除以 Close 进行归一化
    
    # 1. 实体长度 (Body)
    # 绝对值：衡量波动幅度；符号：衡量涨跌方向
    output['candle_body'] = (close - open_p) / close_safe
    output['candle_body_abs'] = output['candle_body'].abs()
    
    # 2. 上影线 (Upper Shadow) - 压力位信号
    # 公式：(High - Max(Open, Close)) / Close
    output['shadow_upper'] = (high - pd.concat([open_p, close], axis=1).max(axis=1)) / close_safe
    
    # 3. 下影线 (Lower Shadow) - 支撑位信号
    # 公式：(Min(Open, Close) - Low) / Close
    output['shadow_lower'] = (pd.concat([open_p, close], axis=1).min(axis=1) - low) / close_safe
    
    # --- D. 振幅 (Amplitude) ---
    # (High - Low) / Close
    # 衡量当天的日内波动激烈程度
    output['candle_amplitude'] = (high - low) / close_safe

    return output
