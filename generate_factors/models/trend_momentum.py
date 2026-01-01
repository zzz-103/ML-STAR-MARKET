import pandas as pd
import numpy as np

def run(df):
    """
    趋势与动量因子组：MACD, RSI
    """
    output = pd.DataFrame(index=df.index)
    grouped = df.groupby(level='code')['close']

    # --- 1. MACD (Moving Average Convergence Divergence) ---
    # 计算快线 EMA12 和 慢线 EMA26
    # adjust=False 是为了模拟传统技术指标的递归计算方式
    ema12 = grouped.transform(lambda x: x.ewm(span=12, adjust=False, min_periods=12).mean())
    ema26 = grouped.transform(lambda x: x.ewm(span=26, adjust=False, min_periods=26).mean())
    
    # DIF (快慢线差值)
    output['macd_dif'] = ema12 - ema26
    
    # DEA (DIF的9日加权移动平均)
    # 注意：这里需要对刚才算出的 macd_dif 再进行一次 transform ewm
    # 但由于 macd_dif 已经是序列，我们需要先确保按 code 分组
    output['macd_dea'] = output.groupby(level='code')['macd_dif'].transform(
        lambda x: x.ewm(span=9, adjust=False, min_periods=9).mean()
    )
    
    # MACD Histogram (柱状图，通常作为动能强弱指标)
    output['macd_hist'] = (output['macd_dif'] - output['macd_dea']) * 2

    # --- 2. RSI (Relative Strength Index) ---
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))

        avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)
        return rsi

    output['rsi_14'] = grouped.transform(lambda x: calc_rsi(x, 14))
    
    # 为了机器学习，可以将 RSI 归一化到 0.5 附近，或者计算 RSI 的变化率
    output['rsi_rate'] = output.groupby(level='code')['rsi_14'].diff(3)

    return output
