import pandas as pd
import numpy as np

def run(df):
    """
    Turnover (换手率) 因子组
    """
    output = pd.DataFrame(index=df.index)
    
    # === 1. 基础数据准备 ===
    # 必须确保 DataFrame 是 MultiIndex [date, code] 或者包含 code 列
    # 这里的写法兼容 MultiIndex (level='code')
    
    turnover = df['turnover']
    
    # 替换 0 值，防止除法运算报错
    turnover_safe = turnover.replace(0, np.nan)
    
    # 预计算 Rolling 对象，减少重复 GroupBy 开销，提升速度
    # 针对 code 分组
    g_turnover = turnover_safe.groupby(level='code')
    
    roll_5_t = g_turnover.rolling(5)

    # ==========================================
    # Factor 1: 换手率乖离 (Turnover Bias)
    # ==========================================
    # 逻辑：当日换手率 / 5日均换手 - 1
    # 意义：
    #   - 极高值 (> 1.0)：异常放量，可能是主力抢筹或顶部出货（需结合股价位置）。
    #   - 极低值 (< -0.5)：极度缩量，往往是回调结束或底部盘整的信号。
    ma5_turnover = roll_5_t.mean().reset_index(level=0, drop=True)
    # reset_index(level=0, drop=True) 是为了对齐索引，防止出现多余的 code 层级
    output['turnover_bias_5'] = (turnover_safe / ma5_turnover) - 1

    return output
