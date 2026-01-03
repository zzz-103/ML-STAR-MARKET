import pandas as pd
import numpy as np
import os
import glob
import importlib
import pkgutil
import argparse
import sys
import re
from pathlib import Path
from datetime import datetime

# ================= 配置区域 =================
# 1. 原始大文件路径 (你的原始合并 CSV)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_INPUT_PATH = str(PROJECT_ROOT / "pre_data" / "merged_20200101_20241231.csv")

# 2. 清洗后的缓存文件设置
CLEANED_DATA_DIR = str(PROJECT_ROOT / "pre_data")
CLEANED_FILE_NAME = "cleaned_stock_data.parquet"
CLEANED_DATA_PATH = os.path.join(CLEANED_DATA_DIR, CLEANED_FILE_NAME)
_PREFERRED_CLEANED_DATA_PATH = str(PROJECT_ROOT / "pre_data" / "cleaned_stock_data_300_688_with_idxstk_with_industry.parquet")
if os.path.exists(_PREFERRED_CLEANED_DATA_PATH):
    CLEANED_DATA_PATH = _PREFERRED_CLEANED_DATA_PATH

# 3. 因子输出根目录
OUTPUT_ROOT = str(PROJECT_ROOT / "factors_data")

# 4. 模型文件夹名称
MODELS_PKG = "models"

FUND_PRICE_PATH = str(PROJECT_ROOT / "pre_data" / "cleaned_stock_data_300_688_with_idxstk.parquet")
FUND_FACTOR_PATH = str(PROJECT_ROOT / "factors_data" / "all_factors_20200101_20241231.parquet")
FUND_OUTPUT_PATH = str(PROJECT_ROOT / "factors_data" / "all_factors_with_fundamentals.parquet")
# ===========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUMMARY_PATH = str(Path(__file__).resolve().parent / "factors_summary.txt")
QUALITY_START = "20230101"
QUALITY_END = "20241231"
MASKED_FACTORS = {"ff_mkt", "ff_hml", "ff_smb", "ff_smb_cov_60"}

def _load_keep_factors_from_ml_models() -> set[str]:
    try:
        from ml_models import xgb_config as ml_cfg

        keep = getattr(ml_cfg, "DEFAULT_KEEP_FACTORS", None)
        if keep is None:
            return set()
        return set(str(x) for x in keep)
    except Exception:
        return set()


KEEP_FACTORS = _load_keep_factors_from_ml_models()
FOCUS_STOCK_POOL_PREFIXES = ("300", "688")
QUALITY_HORIZON_DAYS = 5


def _parse_yyyymmdd(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return pd.to_datetime(s, format="%Y%m%d", errors="raise")


def _coerce_date_yyyymmdd_str(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series.astype(str), errors="coerce")
    return dt.dt.strftime("%Y%m%d")


def merge_fundamentals(price_path: str, factor_path: str, output_path: str) -> str:
    print(f"1. 加载价格/基本面数据: {price_path}")
    df_price = pd.read_parquet(price_path)
    if isinstance(df_price.index, pd.MultiIndex):
        if df_price.index.names != ["date", "code"]:
            df_price.index = df_price.index.set_names(["date", "code"])
        df_price = df_price.reset_index()
    if not {"date", "code"}.issubset(df_price.columns):
        raise ValueError("价格/基本面数据缺少 date/code")

    df_price = df_price.copy()
    df_price["date"] = _coerce_date_yyyymmdd_str(df_price["date"])
    df_price["code"] = df_price["code"].astype(str)
    df_price = df_price.dropna(subset=["date", "code"])

    print("2. 计算/提取基本面因子...")
    fund_cols = ["BP", "SP_ttm"]
    missing = [c for c in fund_cols if c not in df_price.columns]
    if missing:
        raise ValueError(f"价格/基本面数据缺少列: {missing}")

    df_fund = df_price[["date", "code"] + fund_cols].copy()
    for col in fund_cols:
        df_fund[col] = pd.to_numeric(df_fund[col], errors="coerce")
        q = df_fund.groupby("date", sort=False)[col].quantile([0.01, 0.99]).unstack(level=-1)
        q.columns = ["q01", "q99"]
        df_fund = df_fund.join(q, on="date")
        m = np.isfinite(df_fund["q01"].to_numpy()) & np.isfinite(df_fund["q99"].to_numpy()) & (df_fund["q99"] >= df_fund["q01"])
        if bool(m.any()):
            v = df_fund[col]
            df_fund.loc[m, col] = v.loc[m].clip(lower=df_fund.loc[m, "q01"], upper=df_fund.loc[m, "q99"])
        df_fund = df_fund.drop(columns=["q01", "q99"])

    print(f"3. 加载现有因子数据: {factor_path}")
    df_factors = pd.read_parquet(factor_path)
    if {"date", "code"}.issubset(df_factors.columns):
        df_factors = df_factors.copy()
    elif isinstance(df_factors.index, pd.MultiIndex) and {"date", "code"}.issubset(set(df_factors.index.names)):
        df_factors = df_factors.reset_index()
    else:
        raise ValueError("因子数据缺少 date/code")

    df_factors["date"] = _coerce_date_yyyymmdd_str(df_factors["date"])
    df_factors["code"] = df_factors["code"].astype(str)
    df_factors = df_factors.dropna(subset=["date", "code"])

    cols_to_drop = [c for c in fund_cols if c in df_factors.columns]
    if cols_to_drop:
        print(f"   删除旧的基本面列: {cols_to_drop}")
        df_factors = df_factors.drop(columns=cols_to_drop)

    print("4. 合并数据...")
    left = df_factors.set_index(["date", "code"]).sort_index()
    right = df_fund.set_index(["date", "code"]).sort_index()
    merged = left.join(right, how="left").reset_index()
    print(f"   合并后形状: {merged.shape}")
    print(f"   新增列: {fund_cols}")
    if "CFP_ttm" in merged.columns:
        print(f"   CFP_ttm 非空数量: {int(merged['CFP_ttm'].count())}")

    print(f"5. 保存到新文件: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_parquet(output_path, index=False)
    print("✅ 完成！")
    return output_path


def _safe_spearman_ic(group, factor_col, target_col, min_n=30):
    x = group[factor_col]
    y = group[target_col]
    mask = np.isfinite(x.to_numpy(dtype=np.float64)) & np.isfinite(y.to_numpy(dtype=np.float64))
    if mask.sum() < min_n:
        return np.nan
    xv = x[mask]
    yv = y[mask]
    if xv.nunique(dropna=True) < 2 or yv.nunique(dropna=True) < 2:
        return np.nan
    return xv.corr(yv, method="spearman")


def _prepare_quality_panel(
    factors_parquet_path,
    price_parquet_path,
    start_yyyymmdd,
    end_yyyymmdd,
    drop_factors=None,
    keep_factors=None,
    stock_pool_prefixes=None,
    horizon_days=QUALITY_HORIZON_DAYS,
):
    start_dt = _parse_yyyymmdd(start_yyyymmdd)
    end_dt = _parse_yyyymmdd(end_yyyymmdd)
    if start_dt is None or end_dt is None:
        raise ValueError("start/end 日期格式错误，必须为 YYYYMMDD")

    if not os.path.exists(factors_parquet_path):
        raise FileNotFoundError(f"找不到因子合并 Parquet: {factors_parquet_path}")
    if not os.path.exists(price_parquet_path):
        raise FileNotFoundError(f"找不到价格 Parquet: {price_parquet_path}")

    df_f = pd.read_parquet(factors_parquet_path)
    if not {"date", "code"}.issubset(df_f.columns):
        raise ValueError("因子 Parquet 缺少 date/code 列")
    df_f = df_f.copy()
    df_f["date"] = pd.to_datetime(df_f["date"].astype(str), format="%Y%m%d", errors="coerce")
    df_f["code"] = df_f["code"].astype(str)
    df_f = df_f.dropna(subset=["date", "code"]).set_index(["date", "code"]).sort_index()
    df_f = df_f.loc[pd.IndexSlice[start_dt:end_dt, :], :]
    if stock_pool_prefixes:
        pool_mask_f = df_f.index.get_level_values("code").astype(str).str.startswith(tuple(stock_pool_prefixes))
        df_f = df_f.loc[pool_mask_f, :]
    df_f = df_f.groupby(level="code").shift(1)

    df_p = pd.read_parquet(price_parquet_path).sort_index()
    if "close" not in df_p.columns:
        raise ValueError("价格数据缺少 close 列")
    if stock_pool_prefixes:
        pool_mask_p = df_p.index.get_level_values("code").astype(str).str.startswith(tuple(stock_pool_prefixes))
        df_p = df_p.loc[pool_mask_p, :]
    horizon_days = int(horizon_days)
    if horizon_days <= 0:
        raise ValueError("horizon_days 必须为正整数")
    future_close = df_p.groupby(level="code")["close"].shift(-horizon_days)
    current_close = df_p["close"].replace(0, np.nan)
    fwd_ret = (future_close - current_close) / current_close
    df_t = pd.DataFrame({f"ret_fwd_{horizon_days}d": fwd_ret}).replace([np.inf, -np.inf], np.nan).dropna()
    df_t = df_t.loc[pd.IndexSlice[start_dt:end_dt, :], :]

    df = df_f.join(df_t, how="inner")
    if len(df) == 0:
        raise ValueError("分析区间内无可用样本，请检查数据日期范围")

    factor_cols = [c for c in df_f.columns if c not in ("date", "code")]
    if drop_factors:
        drop_set = set(str(x) for x in drop_factors)
        factor_cols = [c for c in factor_cols if str(c) not in drop_set]
    if keep_factors:
        keep_set = set(str(x) for x in keep_factors)
        factor_cols = [c for c in factor_cols if str(c) in keep_set]
    numeric_cols = df[factor_cols].select_dtypes(include=[np.number]).columns.tolist()
    factor_cols = [c for c in factor_cols if c in set(numeric_cols)]
    return df, factor_cols


def _compute_factor_quality(df, factor_cols, target_col, *, min_n: int = 30):
    results = []
    target_col = str(target_col)
    if target_col not in df.columns:
        raise ValueError(f"目标列不存在: {target_col}")
    min_n = int(min_n)
    for col in factor_cols:
        sub = df[[col, target_col]].dropna()
        if len(sub) == 0:
            continue
        ic_by_date = (
            sub.groupby(level="date", sort=True)
            .apply(_safe_spearman_ic, factor_col=col, target_col=target_col, min_n=min_n)
            .dropna()
        )
        if len(ic_by_date) == 0:
            continue
        ic_mean = float(ic_by_date.mean())
        ic_std = float(ic_by_date.std(ddof=1)) if len(ic_by_date) > 1 else np.nan
        ir = float(ic_mean / ic_std) if (ic_std is not None and np.isfinite(ic_std) and ic_std > 0) else np.nan
        t_stat = float(ic_mean / ic_std * np.sqrt(len(ic_by_date))) if (np.isfinite(ir)) else np.nan
        pos_rate = float((ic_by_date > 0).mean())
        qs = ic_by_date.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        missing_rate = float(df[col].isna().mean())
        results.append(
            {
                "factor": col,
                "n_dates": int(len(ic_by_date)),
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ir": ir,
                "t_stat": t_stat,
                "ic_pos_rate": pos_rate,
                "ic_p05": float(qs.loc[0.05]),
                "ic_p25": float(qs.loc[0.25]),
                "ic_p50": float(qs.loc[0.5]),
                "ic_p75": float(qs.loc[0.75]),
                "ic_p95": float(qs.loc[0.95]),
                "missing_rate": missing_rate,
            }
        )

    df_res = pd.DataFrame(results)
    if df_res.empty:
        raise ValueError("未计算出任何因子的 IC，请检查因子列/数据是否为空")
    df_res["abs_ic_mean"] = df_res["ic_mean"].abs()
    df_res = df_res.sort_values(["abs_ic_mean", "ir"], ascending=[False, False]).reset_index(drop=True)
    return df_res


def _render_quality_lines(
    df_res,
    factors_parquet_path,
    price_parquet_path,
    start_yyyymmdd,
    end_yyyymmdd,
    title,
    target_expr,
    top_n=20,
    include_quantiles=True,
    include_all_factors=True,
):
    top_n = min(int(top_n), len(df_res))
    df_print = df_res.copy()
    keep = KEEP_FACTORS if isinstance(KEEP_FACTORS, set) else set()
    df_print["whitelist"] = df_print["factor"].astype(str).apply(lambda x: "✓" if x in keep else "")
    lines = []
    lines.append(title)
    lines.append(f"Date Range: {start_yyyymmdd} ~ {end_yyyymmdd}")
    lines.append(f"Target: {target_expr}")
    lines.append(f"Factors Source: {factors_parquet_path}")
    lines.append(f"Price Source: {price_parquet_path}")
    lines.append(f"Computed At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Total Factors Evaluated: {len(df_print)}")
    lines.append("")
    lines.append("Top Factors (sorted by |IC_mean|, then IR)")
    lines.append(
        df_print.head(top_n)[
            ["factor", "whitelist", "n_dates", "ic_mean", "ic_std", "ir", "t_stat", "ic_pos_rate", "missing_rate"]
        ].to_string(index=False)
    )

    if include_quantiles:
        lines.append("")
        lines.append("IC Distribution Quantiles (per factor)")
        lines.append(
            df_print[["factor", "whitelist", "ic_p05", "ic_p25", "ic_p50", "ic_p75", "ic_p95"]]
            .head(top_n)
            .to_string(index=False)
        )

    if include_all_factors:
        lines.append("")
        lines.append("All Factors")
        lines.append(
            df_print[
                ["factor", "whitelist", "n_dates", "ic_mean", "ic_std", "ir", "t_stat", "ic_pos_rate", "ic_p05", "ic_p50", "ic_p95", "missing_rate"]
            ].to_string(index=False)
        )

    return lines


def analyze_factor_quality(
    factors_parquet_path,
    price_parquet_path,
    start_yyyymmdd,
    end_yyyymmdd,
    out_path,
    drop_factors=None,
    keep_factors=None,
    stock_pool_prefixes=None,
    horizon_days=QUALITY_HORIZON_DAYS,
    min_n: int = 30,
):
    df, factor_cols = _prepare_quality_panel(
        factors_parquet_path=factors_parquet_path,
        price_parquet_path=price_parquet_path,
        start_yyyymmdd=start_yyyymmdd,
        end_yyyymmdd=end_yyyymmdd,
        drop_factors=drop_factors,
        keep_factors=keep_factors,
        stock_pool_prefixes=stock_pool_prefixes,
        horizon_days=horizon_days,
    )
    horizon_days = int(horizon_days)
    target_col = f"ret_fwd_{horizon_days}d"
    df_res = _compute_factor_quality(df, factor_cols, target_col=target_col, min_n=int(min_n))

    lines = _render_quality_lines(
        df_res=df_res,
        factors_parquet_path=factors_parquet_path,
        price_parquet_path=price_parquet_path,
        start_yyyymmdd=start_yyyymmdd,
        end_yyyymmdd=end_yyyymmdd,
        title="Factor Quality Summary",
        target_expr=f"{target_col} = close(T+{horizon_days})/close(T)-1 (per stock), factors shifted by 1 day",
        top_n=20,
        include_quantiles=True,
        include_all_factors=True,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    return out_path, df_res


def clean_raw_data(raw_path):
    """
    执行深度清洗：使用【白名单机制】剔除债、基、指数、北交所
    """
    print(f"[{datetime.now().time()}] 正在读取原始 CSV: {raw_path} ...")
    # 读取时指定 code 为 str，防止 000001 变成 1
    df = pd.read_csv(raw_path, dtype={'code': str})
    
    # 1. 日期标准化
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    
    # 2. 数值强制转换
    num_cols = ['open', 'high', 'low', 'close', 'volume']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    print(f"[{datetime.now().time()}] 正在执行白名单清洗 (剔除BSE/债/指)...")
    
    # --- 核心清洗逻辑 (Vectorized String Operations) ---
    
    # 转大写并处理空值
    df['code'] = df['code'].fillna("").astype(str).str.upper()
    
    # 拆分 Code 和 Suffix (例如 "000001.SZSE")
    # split(n=1) 确保只切分第一个点
    split_data = df['code'].str.split('.', n=1, expand=True)
    
    # 此时: 
    # split_data[0] 是数字代码 (如 '000001', '110000')
    # split_data[1] 是后缀 (如 'SZSE', 'SSE', 'BSE' 或 None)
    
    # 如果没有后缀列 (shape[1] == 1)，说明原始数据不规范，填充空字符串
    if split_data.shape[1] == 1:
        split_data[1] = ""
        
    code_num = split_data[0]
    suffix = split_data[1]

    # === 定义白名单规则 (White List) ===
    # 1. 上证主板 (60) & 科创板 (68) -> 后缀通常是 SSE 或 SH，只要数字对就行
    mask_sh = code_num.str.startswith(('60', '68'))
    
    # 2. 创业板 (30) -> 肯定是深圳
    mask_cy = code_num.str.startswith('30')
    
    # 3. 深证主板 (00) -> 必须严格检查后缀，排除上证指数(000001.SSE)
    #    逻辑: 开头是 00 且 后缀包含 'SZ'
    mask_sz_main = (code_num.str.startswith('00')) & (suffix.str.contains('SZ'))
    
    # 组合掩码: 只保留上述三类
    # 这一步会自动过滤掉:
    # - 11/12 (可转债)
    # - 8/4 (北交所)
    # - 51/15 (ETF)
    # - 000xxx.SSE (上证指数)
    final_mask = mask_sh | mask_cy | mask_sz_main
    
    # 应用过滤
    df_clean = df[final_mask].copy()
    
    # 将代码更新为纯数字格式 (去掉 .SZSE 后缀)
    df_clean['code'] = code_num[final_mask]
    
    # --- 唯一性保证 ---
    # 设置索引
    df_clean = df_clean.set_index(['date', 'code']).sort_index()
    
    # 极少数情况下，数据源可能重复 (比如同一个票同一天有两条记录)，去重
    if df_clean.index.duplicated().any():
        print(f"⚠️ 警告: 发现 {df_clean.index.duplicated().sum()} 条重复数据，正在去重...")
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]

    print(f"[{datetime.now().time()}] 清洗完成。")
    print(f"   原始行数: {len(df)}")
    print(f"   清洗后行数: {len(df_clean)} (已剔除 BSE/可转债/指数)")
    
    return df_clean

def get_data():
    """
    优先读取缓存 Parquet，不存在则清洗 CSV
    """
    if not os.path.exists(CLEANED_DATA_DIR):
        os.makedirs(CLEANED_DATA_DIR)

    # 1. 尝试读缓存
    if os.path.exists(CLEANED_DATA_PATH):
        print(f"[{datetime.now().time()}] ✅ 发现缓存文件，直接读取: {CLEANED_DATA_PATH}")
        try:
            df = pd.read_parquet(CLEANED_DATA_PATH)
            if not isinstance(df.index, pd.MultiIndex) or set(df.index.names) != {"date", "code"}:
                raise ValueError(f"缓存数据索引必须包含 date/code: {CLEANED_DATA_PATH}")
            return df.sort_index()
        except Exception as e:
            print(f"❌ 缓存损坏 ({e})，重新清洗。")

    # 2. 清洗原始数据
    df = clean_raw_data(RAW_INPUT_PATH)
    
    # 3. 保存缓存
    print(f"[{datetime.now().time()}] 正在保存缓存: {CLEANED_DATA_PATH}")
    df.to_parquet(CLEANED_DATA_PATH)
    
    return df

def normalize_factor(series):
    """
    每日截面 Z-Score 标准化
    """
    def zscore(x):
        std = x.std()
        if np.isnan(std):
            return x * 0
        if std == 0:
            return x
        return (x - x.mean()) / std
    
    return series.groupby(level='date').transform(zscore)

def parse_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        parts = []
        for v in value:
            parts.extend(parse_list(v) or [])
        return [p for p in parts if p]
    raw = str(value).strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def get_factor_output_path(factor_name, start_date, end_date):
    folder_path = os.path.join(OUTPUT_ROOT, factor_name)
    file_name = f"{factor_name}_{start_date}_{end_date}.csv"
    return folder_path, os.path.join(folder_path, file_name)


def save_factor(factor_name, factor_series, start_date, end_date, overwrite):
    """
    保存因子，格式化日期为 YYYYMMDD
    """
    folder_path, full_path = get_factor_output_path(factor_name, start_date, end_date)
    if (not overwrite) and os.path.exists(full_path):
        return False
    os.makedirs(folder_path, exist_ok=True)
    
    # 转为 DataFrame 并处理日期格式
    df_out = factor_series.dropna().reset_index()
    df_out['date'] = df_out['date'].dt.strftime('%Y%m%d')
    
    # index=False 去掉 Pandas 索引
    df_out.to_csv(full_path, index=False, header=True)
    print(f"   -> 已保存: {full_path}")
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None, help="Comma-separated model module names")
    parser.add_argument("--factors", default=None, help="Comma-separated factor names")
    parser.add_argument("--force", action="store_true", help="Recompute and overwrite existing factor files")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--list-factors", action="store_true", help="List available factors and exit")
    parser.add_argument("--diagnose", action="store_true", default=False)
    parser.add_argument("--diag-start-date", default=QUALITY_START)
    parser.add_argument("--diag-end-date", default=QUALITY_END)
    parser.add_argument("--diag-horizon-days", type=int, default=QUALITY_HORIZON_DAYS)
    parser.add_argument("--summary-only", action="store_true", default=False)
    parser.add_argument("--summary-start-date", default=QUALITY_START)
    parser.add_argument("--summary-end-date", default=QUALITY_END)
    parser.add_argument("--summary-horizon-days", type=int, default=QUALITY_HORIZON_DAYS)
    parser.add_argument("--summary-min-n", type=int, default=30)
    parser.add_argument("--summary-corr-top-k", type=int, default=30)
    parser.add_argument("--summary-out", default=SUMMARY_PATH)
    parser.add_argument("--summary-factors-parquet", default=None)
    parser.add_argument("--summary-price-parquet", default=CLEANED_DATA_PATH)
    parser.add_argument("--summary-keep-factors", default=None, help="Comma-separated factor names for quality report")
    parser.add_argument("--merge-fundamentals", action="store_true", default=False)
    parser.add_argument("--fund-price-path", default=FUND_PRICE_PATH)
    parser.add_argument("--fund-factor-path", default=FUND_FACTOR_PATH)
    parser.add_argument("--fund-output-path", default=FUND_OUTPUT_PATH)
    return parser.parse_args()


def parse_date_range_from_path(path):
    base = os.path.basename(str(path))
    m = re.search(r"(\d{8})_(\d{8})", base)
    if not m:
        return None
    return m.group(1), m.group(2)


def _find_latest_factors_parquet(output_root: str) -> str | None:
    try:
        candidates = glob.glob(os.path.join(str(output_root), "all_factors_*.parquet"))
        candidates = [p for p in candidates if os.path.isfile(p)]
        if not candidates:
            return None
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    except Exception:
        return None



def build_all_factors_parquet(output_root, start_date, end_date):
    parquet_name = f"all_factors_{start_date}_{end_date}.parquet"
    parquet_path = os.path.join(output_root, parquet_name)

    factor_dirs = []
    if os.path.isdir(output_root):
        for name in os.listdir(output_root):
            p = os.path.join(output_root, name)
            if os.path.isdir(p) and (not name.startswith(".")):
                factor_dirs.append((name, p))
    factor_dirs.sort(key=lambda x: x[0])

    merged = None
    included = 0
    for factor_name, factor_dir in factor_dirs:
        if str(factor_name) in MASKED_FACTORS:
            continue
        if KEEP_FACTORS and (str(factor_name) not in KEEP_FACTORS):
            continue
        expected = os.path.join(factor_dir, f"{factor_name}_{start_date}_{end_date}.csv")
        if os.path.exists(expected):
            csv_path = expected
        else:
            candidates = glob.glob(os.path.join(factor_dir, f"{factor_name}_*.csv"))
            parsed = []
            for p in candidates:
                rng = parse_date_range_from_path(p)
                if rng is None:
                    continue
                s, e = rng
                parsed.append((s, e, p))
            if not parsed:
                continue

            exact_end = [t for t in parsed if t[1] == end_date]
            if exact_end:
                exact_end.sort(key=lambda x: x[0])
                csv_path = exact_end[0][2]
            else:
                parsed.sort(key=lambda x: (x[1], x[0]))
                csv_path = parsed[-1][2]

        header = pd.read_csv(csv_path, nrows=0).columns.tolist()
        if "date" not in header or "code" not in header:
            continue
        value_cols = [c for c in header if c not in ("date", "code")]
        if not value_cols:
            continue
        value_col = value_cols[0]

        df_factor = pd.read_csv(
            csv_path,
            dtype={"date": str, "code": str, value_col: "float32"},
        )
        df_factor = df_factor[["date", "code", value_col]].rename(columns={value_col: factor_name})
        df_factor = df_factor.set_index(["date", "code"]).sort_index()

        if merged is None:
            merged = df_factor
        else:
            merged = merged.join(df_factor, how="outer")
        included += 1

    if merged is None:
        return None

    merged = merged.sort_index()
    merged.reset_index().to_parquet(parquet_path, index=False)
    return parquet_path, included


def _top_factor_correlations(df, factor_cols, top_k=30, min_periods=5000):
    x = df[factor_cols]
    corr = x.corr(method="pearson", min_periods=int(min_periods))
    rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iat[i, j]
            if not np.isfinite(c):
                continue
            rows.append((cols[i], cols[j], float(c), float(abs(c))))
    if not rows:
        return pd.DataFrame(columns=["factor_a", "factor_b", "corr", "abs_corr", "n_overlap"])
    rows.sort(key=lambda t: (t[3], abs(t[2])), reverse=True)
    rows = rows[: int(top_k)]
    out = []
    for a, b, c, ac in rows:
        n_overlap = int(df[[a, b]].dropna().shape[0])
        out.append({"factor_a": a, "factor_b": b, "corr": c, "abs_corr": ac, "n_overlap": n_overlap})
    return pd.DataFrame(out).sort_values(["abs_corr", "n_overlap"], ascending=[False, False]).reset_index(drop=True)


def write_quality_report(
    factors_parquet_path,
    price_parquet_path,
    quality_start,
    quality_end,
    out_path,
    yearly_ranges=None,
    corr_top_k=30,
    drop_factors=None,
    keep_factors=None,
    stock_pool_prefixes=None,
    horizon_days=QUALITY_HORIZON_DAYS,
    min_n: int = 30,
):
    df, factor_cols = _prepare_quality_panel(
        factors_parquet_path=factors_parquet_path,
        price_parquet_path=price_parquet_path,
        start_yyyymmdd=quality_start,
        end_yyyymmdd=quality_end,
        drop_factors=drop_factors,
        keep_factors=keep_factors,
        stock_pool_prefixes=stock_pool_prefixes,
        horizon_days=horizon_days,
    )
    horizon_days = int(horizon_days)
    target_col = f"ret_fwd_{horizon_days}d"
    df_res_full = _compute_factor_quality(df, factor_cols, target_col=target_col, min_n=int(min_n))
    lines = _render_quality_lines(
        df_res=df_res_full,
        factors_parquet_path=factors_parquet_path,
        price_parquet_path=price_parquet_path,
        start_yyyymmdd=quality_start,
        end_yyyymmdd=quality_end,
        title="Factor Quality Summary",
        target_expr=f"{target_col} = close(T+{horizon_days})/close(T)-1 (per stock), factors shifted by 1 day",
        top_n=20,
        include_quantiles=True,
        include_all_factors=True,
    )

    yearly_ranges = yearly_ranges or []
    for title, s, e in yearly_ranges:
        sub = df.loc[pd.IndexSlice[_parse_yyyymmdd(s) : _parse_yyyymmdd(e), :], :]
        if len(sub) == 0:
            continue
        df_res_year = _compute_factor_quality(sub, factor_cols, target_col=target_col, min_n=int(min_n))
        lines.append("")
        lines.append("")
        lines.extend(
            _render_quality_lines(
                df_res=df_res_year,
                factors_parquet_path=factors_parquet_path,
                price_parquet_path=price_parquet_path,
                start_yyyymmdd=s,
                end_yyyymmdd=e,
                title=title,
                target_expr=f"{target_col} = close(T+{horizon_days})/close(T)-1 (per stock), factors shifted by 1 day",
                top_n=20,
                include_quantiles=False,
                include_all_factors=False,
            )
        )

    corr_df = _top_factor_correlations(df, factor_cols, top_k=corr_top_k)
    lines.append("")
    lines.append("")
    lines.append(f"Factor Correlation (|corr| Top {int(corr_top_k)})")
    if corr_df.empty:
        lines.append("(empty)")
    else:
        lines.append(corr_df[["factor_a", "factor_b", "corr", "n_overlap"]].to_string(index=False))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    return out_path


def main():
    args = parse_args()

    if bool(getattr(args, "merge_fundamentals", False)):
        merge_fundamentals(
            price_path=str(getattr(args, "fund_price_path", FUND_PRICE_PATH)),
            factor_path=str(getattr(args, "fund_factor_path", FUND_FACTOR_PATH)),
            output_path=str(getattr(args, "fund_output_path", FUND_OUTPUT_PATH)),
        )
        return

    import models
    available_models = [m.name for m in pkgutil.iter_modules(models.__path__) if not m.ispkg]
    available_models.sort()

    if args.list_models:
        for m in available_models:
            print(m)
        return

    if bool(getattr(args, "diagnose", False)):
        start = str(getattr(args, "diag_start_date", QUALITY_START))
        end = str(getattr(args, "diag_end_date", QUALITY_END))
        horizon_days = int(getattr(args, "diag_horizon_days", QUALITY_HORIZON_DAYS))
        factors_parquet = os.path.join(OUTPUT_ROOT, "all_factors_20200101_20241231.parquet")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        diag_out = os.path.join(BASE_DIR, f"factors_ic_ir_{start}_{end}_h{horizon_days}_{ts}.txt")
        out_path = write_quality_report(
            factors_parquet_path=factors_parquet,
            price_parquet_path=CLEANED_DATA_PATH,
            quality_start=start,
            quality_end=end,
            out_path=diag_out,
            yearly_ranges=[
                ("Factor Quality Summary (2023 Only, Top 20)", "20230101", "20231231"),
                ("Factor Quality Summary (2024 Only, Top 20)", "20240101", "20241231"),
            ],
            corr_top_k=30,
            drop_factors=MASKED_FACTORS,
            stock_pool_prefixes=FOCUS_STOCK_POOL_PREFIXES,
            horizon_days=horizon_days,
        )
        print(f"\n[{datetime.now().time()}] ✅ 已保存逐因子 IC/IR 诊断报告: {out_path}")
        return

    if bool(getattr(args, "summary_only", False)):
        start = str(getattr(args, "summary_start_date", QUALITY_START))
        end = str(getattr(args, "summary_end_date", QUALITY_END))
        horizon_days = int(getattr(args, "summary_horizon_days", QUALITY_HORIZON_DAYS))
        min_n = int(getattr(args, "summary_min_n", 30))
        corr_top_k = int(getattr(args, "summary_corr_top_k", 30))
        out_path = str(getattr(args, "summary_out", SUMMARY_PATH))
        price_parquet = str(getattr(args, "summary_price_parquet", CLEANED_DATA_PATH))
        keep_factors = parse_list(getattr(args, "summary_keep_factors", None))
        keep_factors = keep_factors if keep_factors else None

        factors_parquet = getattr(args, "summary_factors_parquet", None)
        if factors_parquet:
            factors_parquet = str(factors_parquet)
        elif os.path.exists(FUND_OUTPUT_PATH):
            factors_parquet = str(FUND_OUTPUT_PATH)
        else:
            factors_parquet = _find_latest_factors_parquet(OUTPUT_ROOT)
        if not factors_parquet:
            raise FileNotFoundError("找不到可用于 summary 的 factors parquet")

        out_path = write_quality_report(
            factors_parquet_path=factors_parquet,
            price_parquet_path=price_parquet,
            quality_start=start,
            quality_end=end,
            out_path=out_path,
            yearly_ranges=[
                ("Factor Quality Summary (2023 Only, Top 20)", "20230101", "20231231"),
                ("Factor Quality Summary (2024 Only, Top 20)", "20240101", "20241231"),
            ],
            corr_top_k=corr_top_k,
            drop_factors=MASKED_FACTORS,
            keep_factors=keep_factors,
            stock_pool_prefixes=FOCUS_STOCK_POOL_PREFIXES,
            horizon_days=horizon_days,
            min_n=min_n,
        )
        print(f"\n[{datetime.now().time()}] ✅ 已生成因子质量概要: {out_path}")
        return

    selected_models = parse_list(args.models)
    selected_factors = parse_list(args.factors)

    if selected_models is not None and len(selected_models) == 0:
        return

    # 1. 加载数据 (自动处理清洗/缓存)
    df_base = get_data()
    
    # 获取日期范围字符串
    dates = df_base.index.get_level_values('date')
    if len(dates) == 0:
        print("❌ 错误：数据为空，请检查原始 CSV 路径或内容。")
        return
        
    s_date = dates.min().strftime('%Y%m%d')
    e_date = dates.max().strftime('%Y%m%d')

    # 2. 动态运行因子模型
    print(f"\n[{datetime.now().time()}] 开始扫描 '{MODELS_PKG}' 文件夹...")
    
    # 遍历 models 目录
    if selected_models is None:
        models_to_run = available_models
    else:
        models_to_run = [m for m in selected_models if m in set(available_models)]
        missing = [m for m in selected_models if m not in set(available_models)]
        for m in missing:
            print(f"⚠️ 找不到模型: {m}")

    if args.list_factors:
        found = set()
        for module_name in models_to_run:
            full_module_name = f"{MODELS_PKG}.{module_name}"
            module = importlib.import_module(full_module_name)
            if not hasattr(module, "run"):
                continue
            try:
                df_result = module.run(df_base)
            except Exception:
                continue
            if df_result is None or df_result.empty:
                continue
            for factor_name in df_result.columns:
                if str(factor_name) in MASKED_FACTORS:
                    continue
                if KEEP_FACTORS and str(factor_name) not in KEEP_FACTORS:
                    continue
                found.add(str(factor_name))
        for f in sorted(found):
            print(f)
        return
    models_to_run_set = set(models_to_run)
    selected_factors_set = set(selected_factors) if selected_factors is not None else None

    total_saved = 0
    total_skipped = 0
    total_failed = 0

    for _, module_name, _ in pkgutil.iter_modules(models.__path__):
        try:
            if module_name not in models_to_run_set:
                continue
            full_module_name = f"{MODELS_PKG}.{module_name}"
            module = importlib.import_module(full_module_name)
            
            if not hasattr(module, 'run'):
                continue
            
            if module_name == "ff_factors":
                print(f"\n>>> 正在加载: {module_name} (外部数据集)")
            else:
                print(f"\n>>> 正在计算: {module_name}")
            
            # 运行模型计算
            df_result = module.run(df_base)
            
            if df_result is None or df_result.empty:
                print(f"   ⚠️ 警告: {module_name} 返回为空")
                continue

            # 处理该模型下的每一个因子列
            for factor_name in df_result.columns:
                if selected_factors_set is not None and str(factor_name) not in selected_factors_set:
                    continue
                if str(factor_name) in MASKED_FACTORS:
                    continue
                if KEEP_FACTORS and str(factor_name) not in KEEP_FACTORS:
                    continue
                _, out_path = get_factor_output_path(str(factor_name), s_date, e_date)
                if (not args.force) and os.path.exists(out_path):
                    total_skipped += 1
                    continue
                # 归一化
                norm_series = normalize_factor(df_result[factor_name])
                # 保存
                ok = save_factor(str(factor_name), norm_series, s_date, e_date, overwrite=args.force)
                if ok:
                    total_saved += 1
                else:
                    total_skipped += 1
                
        except Exception as e:
            print(f"❌ 模块 {module_name} 运行失败: {e}")
            import traceback
            traceback.print_exc()
            total_failed += 1

    print(f"\n[{datetime.now().time()}] 统计: saved={total_saved}, skipped={total_skipped}, failed_models={total_failed}")

    parsed = parse_date_range_from_path(RAW_INPUT_PATH)
    if parsed is None:
        all_start, all_end = s_date, e_date
    else:
        all_start, all_end = parsed

    try:
        res = build_all_factors_parquet(OUTPUT_ROOT, all_start, all_end)
        if res is None:
            print(f"\n[{datetime.now().time()}] ⚠️ 未找到可合并的因子文件，跳过总 Parquet 生成")
        else:
            parquet_path, included = res
            print(f"\n[{datetime.now().time()}] ✅ 已生成总 Parquet: {parquet_path} (factors={included})")
            try:
                if os.path.exists(FUND_PRICE_PATH):
                    print(f"\n[{datetime.now().time()}] 开始合并基本面到: {FUND_OUTPUT_PATH}")
                    merge_fundamentals(price_path=FUND_PRICE_PATH, factor_path=parquet_path, output_path=FUND_OUTPUT_PATH)
                else:
                    print(f"\n[{datetime.now().time()}] ⚠️ 找不到 FUND_PRICE_PATH，跳过基本面合并: {FUND_PRICE_PATH}")
            except Exception as e:
                print(f"\n[{datetime.now().time()}] ⚠️ 基本面合并失败（不影响因子生成/质量分析）: {e}")
    except Exception as e:
        print(f"\n[{datetime.now().time()}] ❌ 生成总 Parquet 失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        factors_parquet = os.path.join(OUTPUT_ROOT, f"all_factors_{all_start}_{all_end}.parquet")
        print(f"\n[{datetime.now().time()}] 开始因子质量分析: {QUALITY_START} ~ {QUALITY_END}")
        out_path = write_quality_report(
            factors_parquet_path=factors_parquet,
            price_parquet_path=CLEANED_DATA_PATH,
            quality_start=QUALITY_START,
            quality_end=QUALITY_END,
            out_path=SUMMARY_PATH,
            yearly_ranges=[
                ("Factor Quality Summary (2023 Only, Top 20)", "20230101", "20231231"),
                ("Factor Quality Summary (2024 Only, Top 20)", "20240101", "20241231"),
            ],
            corr_top_k=30,
            drop_factors=MASKED_FACTORS,
            stock_pool_prefixes=FOCUS_STOCK_POOL_PREFIXES,
            horizon_days=QUALITY_HORIZON_DAYS,
        )
        print(f"[{datetime.now().time()}] ✅ 已生成因子质量概要: {out_path}")
    except Exception as e:
        print(f"\n[{datetime.now().time()}] ❌ 因子质量分析失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n[{datetime.now().time()}] ✅ 所有任务执行完毕！")

if __name__ == "__main__":
    main()
