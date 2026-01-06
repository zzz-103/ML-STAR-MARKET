import pandas as pd
import numpy as np


def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    研发相关基本面因子（将低频披露数据对齐到日频）。
    """
    output = pd.DataFrame(index=df.index)

    def get_filled_series(col_name: str) -> pd.Series:
        # R&D 类字段通常为年频/季频，按 code 向前填充到日频；从未披露则按 0 处理
        if col_name not in df.columns:
            return pd.Series(np.nan, index=df.index)

        s = df[col_name]
        s_filled = s.groupby(level="code").ffill()
        s_filled = s_filled.fillna(0.0)
        return pd.to_numeric(s_filled, errors="coerce")

    rd_spend_sum = get_filled_series("RDSpendSum")
    rd_spend_sum_ratio = get_filled_series("RDSpendSumRatio")
    rd_invest_ratio = get_filled_series("RDInvestRatio")

    cmv = pd.to_numeric(df.get("CirculatedMarketValue", pd.Series(np.nan, index=df.index)), errors="coerce")

    output["f_rd_intensity"] = rd_spend_sum_ratio
    output["f_rd_cap_ratio"] = rd_invest_ratio

    # 市研率：流通市值 / 研发投入，越低越偏“价值”
    output["f_price_to_rd"] = cmv / (rd_spend_sum + 1e-6)

    return output
