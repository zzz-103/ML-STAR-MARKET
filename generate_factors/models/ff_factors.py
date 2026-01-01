import pandas as pd
FF_FACTORS_CSV_PATH = "/Users/zhuzhuxia/Documents/SZU_w4/factors_data/ff_factors/ff_factors.csv"
FF_MARKETTYPE_ID = "P9714"


def run(df):
    df_ff = pd.read_csv(FF_FACTORS_CSV_PATH, dtype=str)
    required = {"MarkettypeID", "TradingDate", "RiskPremium1", "SMB1", "HML1"}
    missing = required - set(df_ff.columns)
    if missing:
        raise ValueError(f"FF 因子文件缺少字段: {sorted(missing)}")

    df_ff = df_ff[df_ff["MarkettypeID"].astype(str) == str(FF_MARKETTYPE_ID)].copy()
    df_ff["TradingDate"] = pd.to_datetime(df_ff["TradingDate"], errors="coerce")
    df_ff["ff_mkt"] = pd.to_numeric(df_ff["RiskPremium1"], errors="coerce")
    df_ff["ff_smb"] = pd.to_numeric(df_ff["SMB1"], errors="coerce")
    df_ff["ff_hml"] = pd.to_numeric(df_ff["HML1"], errors="coerce")

    df_ff = df_ff.dropna(subset=["TradingDate"])
    df_ff = (
        df_ff[["TradingDate", "ff_mkt", "ff_smb", "ff_hml"]]
        .groupby("TradingDate", as_index=True)
        .mean(numeric_only=True)
        .sort_index()
    )

    dates = df.index.get_level_values("date")
    aligned = df_ff.reindex(dates).fillna(0.0)
    aligned.index = df.index
    return aligned

