# 位置: 13（因子诊断）| main.py --diagnose 模式调用（不训练）
# 输入: args（日期区间/步长/topk/min_n/路径等）
# 输出: 诊断报告 txt 路径；同时打印摘要表格
# 依赖: _02_parsing_utils、_04_feature_engineering、_06_data_preprocessing、xgb_config
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from ml_models import xgb_config as cfg
from ml_models.model_functions._02_parsing_utils import parse_yyyymmdd
from ml_models.model_functions._04_feature_engineering import apply_feature_filters, build_constraints_dict, build_drop_factors
from ml_models.model_functions._06_data_preprocessing import build_tradable_mask, ensure_factors_index


def diagnose_factors(args) -> str:
    start_dt = parse_yyyymmdd(getattr(args, "diag_start_date", None))
    end_dt = parse_yyyymmdd(getattr(args, "diag_end_date", None))
    if start_dt is None or end_dt is None:
        raise ValueError("diag-start-date/diag-end-date 必须为 YYYYMMDD")
    step = max(1, int(getattr(args, "diag_step", 5)))
    top_k = max(1, int(getattr(args, "diag_top_k", 30)))
    min_n = max(30, int(getattr(args, "diag_min_n", 300)))

    diag_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    diag_txt_name = (
        f"diagnose_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}"
        f"_step{step}_top{top_k}_minn{min_n}_{diag_ts}.txt"
    )
    base_dir = os.path.dirname(os.path.abspath(__file__))
    diag_txt_path = os.path.join(os.path.dirname(base_dir), diag_txt_name)
    diag_lines: list[str] = []

    def emit(msg: str = "") -> None:
        s = "" if msg is None else str(msg)
        print(s)
        diag_lines.append(s)

    emit(
        f"[{datetime.now().time()}] 诊断模式: {start_dt.strftime('%Y%m%d')}~{end_dt.strftime('%Y%m%d')} "
        f"step={step} topk={top_k} min_n={min_n}"
    )

    factor_path = str(getattr(args, "factor_data_path", cfg.DEFAULT_PATHS["factor_data_path"]))
    price_path = str(getattr(args, "price_data_path", cfg.DEFAULT_PATHS["price_data_path"]))
    if not os.path.exists(factor_path):
        raise FileNotFoundError(f"找不到因子文件: {factor_path}")
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"找不到价格文件: {price_path}")

    df_factors = ensure_factors_index(pd.read_parquet(factor_path))
    df_price = pd.read_parquet(price_path).sort_index()

    if "turnover" in df_price.columns:
        df_price["turnover_prev"] = df_price.groupby(level="code")["turnover"].shift(1)

    idx = pd.IndexSlice
    df_factors = df_factors.loc[idx[start_dt:end_dt, :], :]
    df_price = df_price.loc[idx[start_dt:end_dt, :], :]
    prefixes = tuple(getattr(args, "stock_pool_prefixes", cfg.DEFAULT_UNIVERSE["stock_pool_prefixes"]))
    pool_mask_f = df_factors.index.get_level_values("code").astype(str).str.startswith(prefixes)
    pool_mask_p = df_price.index.get_level_values("code").astype(str).str.startswith(prefixes)
    df_factors = df_factors.loc[pool_mask_f, :]
    df_price = df_price.loc[pool_mask_p, :]

    drop_factors = build_drop_factors(
        use_default_drop_factors=bool(getattr(args, "use_default_drop_factors", True)),
        drop_factors_csv=getattr(args, "drop_factors", None),
    )
    factor_cols = apply_feature_filters(list(df_factors.columns), drop_factors)
    df_factors = df_factors[factor_cols]
    factor_cols = df_factors.select_dtypes(include=[np.number]).columns.tolist()
    if len(factor_cols) == 0:
        raise ValueError("可用数值因子为空，请检查 drop list 或因子文件内容")

    df_factors_shifted = df_factors.groupby(level="code").shift(1)

    future_open = df_price.groupby(level="code")["open"].shift(-5)
    current_open = df_price["open"].replace(0, np.nan)
    ret_next_raw = (future_open - current_open) / current_open

    min_turnover = float(getattr(args, "min_turnover", cfg.DEFAULT_UNIVERSE["min_turnover"]))
    tradable = build_tradable_mask(df_price, min_turnover=min_turnover).rename("tradable")
    bench_universe = str(getattr(args, "label_benchmark_universe", cfg.DEFAULT_LABEL["label_benchmark_universe"])).lower()
    raw_for_bench = ret_next_raw.where(tradable.astype(bool)) if bench_universe == "tradable" else ret_next_raw
    bench_method = str(getattr(args, "label_benchmark_method", cfg.DEFAULT_LABEL["label_benchmark_method"])).lower()
    if bench_method == "mean":
        benchmark = raw_for_bench.groupby(level="date").transform("mean")
    else:
        benchmark = raw_for_bench.groupby(level="date").transform("median")
    alpha_5d = (ret_next_raw - benchmark).rename("alpha_5d")

    panel = df_factors_shifted.join(pd.concat([alpha_5d, tradable], axis=1), how="inner")
    if len(panel) == 0:
        raise ValueError("诊断区间内无可用样本（因子与价格/标签无法对齐）")

    all_dates = panel.index.get_level_values("date").unique().sort_values()
    dates = all_dates[::step]
    ic_rows: list[pd.Series] = []
    topk_rows: list[pd.Series] = []

    for d in tqdm(dates, desc="Diagnosing"):
        day = panel.loc[idx[d, :], :]
        day = day[day["tradable"].astype(bool)]
        if len(day) < min_n:
            continue
        y = day["alpha_5d"]
        x = day[factor_cols]
        ic = x.corrwith(y, axis=0, method="spearman")
        ic.name = d
        ic_rows.append(ic)

        topk_mean: dict[str, float] = {}
        for f in factor_cols:
            s = day[[f, "alpha_5d"]].dropna()
            if len(s) < min_n:
                topk_mean[f] = np.nan
                continue
            sel = s.nlargest(min(top_k, len(s)), columns=f)["alpha_5d"]
            topk_mean[f] = float(sel.mean()) if len(sel) > 0 else np.nan
        topk_rows.append(pd.Series(topk_mean, name=d))

    if len(ic_rows) == 0:
        raise ValueError("可用诊断日为空（可能是 min_n 太高或日期区间无可交易样本）")

    ic_df = pd.DataFrame(ic_rows).sort_index()
    topk_df = pd.DataFrame(topk_rows).sort_index()

    def summarize(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        mean = df.mean(axis=0, skipna=True)
        std = df.std(axis=0, ddof=1, skipna=True)
        ir = mean / std.replace(0.0, np.nan)
        pos = (df > 0).mean(axis=0, skipna=True)
        return pd.DataFrame({f"{prefix}_mean": mean, f"{prefix}_std": std, f"{prefix}_ir": ir, f"{prefix}_pos": pos})

    def direction_advice(ic_mean: float, ic_ir: float, top_alpha_mean: float) -> tuple[str, int]:
        if ic_mean is None or (not np.isfinite(float(ic_mean))):
            return "未知", 0
        ic_mean_f = float(ic_mean)
        ic_ir_f = float(ic_ir) if (ic_ir is not None and np.isfinite(float(ic_ir))) else np.nan
        top_f = float(top_alpha_mean) if (top_alpha_mean is not None and np.isfinite(float(top_alpha_mean))) else np.nan

        weak = (abs(ic_mean_f) < 0.01) or (np.isfinite(ic_ir_f) and abs(ic_ir_f) < 0.10)
        if weak:
            return "弱", 0
        if np.isfinite(top_f) and (ic_mean_f * top_f < 0):
            return "冲突", 0
        if ic_mean_f > 0:
            return "趋势", 1
        return "反转", -1

    def safe_year_int(dt) -> int | None:
        try:
            return int(pd.to_datetime(dt).year)
        except Exception:
            return None

    def compact_table(summary_df: pd.DataFrame, top_k_value: int, constraints_now: dict[str, int]) -> pd.DataFrame:
        s = summary_df.copy()
        alpha_col = f"top{top_k_value}_alpha_mean"
        dir_out: list[str] = []
        cons_out: list[int] = []
        for _, row in s.iterrows():
            direction, cons = direction_advice(
                row.get("ic_mean", np.nan),
                row.get("ic_ir", np.nan),
                row.get(alpha_col, np.nan),
            )
            dir_out.append(direction)
            cons_out.append(cons)
        s["方向建议"] = dir_out
        s["约束建议"] = cons_out
        s["当前约束"] = [int(constraints_now.get(str(f), 0)) for f in s.index]
        s.index.name = "factor"
        s = s.reset_index()
        keep_cols = [
            "factor",
            "ic_mean",
            "ic_ir",
            "ic_pos",
            alpha_col,
            f"top{top_k_value}_alpha_ir",
            "方向建议",
            "当前约束",
            "约束建议",
        ]
        keep_cols = [c for c in keep_cols if c in s.columns]
        return s[keep_cols]

    years = sorted({y for y in (safe_year_int(d) for d in ic_df.index) if y is not None})
    emit(f"\n诊断因子数: {len(factor_cols)} | 诊断日数: {len(ic_df)} | 年份: {years}")

    constraints_now = build_constraints_dict(
        use_constraints=bool(getattr(args, "use_constraints", True)),
        constraints_csv=getattr(args, "constraints", None),
    )
    for y in years:
        mask = ic_df.index.year == y
        ic_y = ic_df.loc[mask, :]
        topk_y = topk_df.loc[mask, :]
        if len(ic_y) == 0:
            continue
        s1 = summarize(ic_y, "ic")
        s2 = summarize(topk_y, f"top{top_k}_alpha")
        summary = s1.join(s2, how="outer")
        emit(f"\n===== Year {y} (n_days={len(ic_y)}) =====")
        compact = compact_table(summary, top_k, constraints_now)
        compact_pos = compact.sort_values(["ic_mean", "ic_ir"], ascending=[False, False]).head(10)
        compact_neg = compact.sort_values(["ic_mean", "ic_ir"], ascending=[True, True]).head(10)
        emit("\nTop 10 (IC Mean 最大)")
        emit(compact_pos.to_string(index=False, float_format=lambda v: f"{v: .5f}"))
        emit("\nBottom 10 (IC Mean 最小)")
        emit(compact_neg.to_string(index=False, float_format=lambda v: f"{v: .5f}"))

    s_all = summarize(ic_df, "ic").join(summarize(topk_df, f"top{top_k}_alpha"), how="outer")
    emit(f"\n===== All ({start_dt.strftime('%Y%m%d')}~{end_dt.strftime('%Y%m%d')}, n_days={len(ic_df)}) =====")
    compact_all = compact_table(s_all, top_k, constraints_now)
    compact_pos = compact_all.sort_values(["ic_mean", "ic_ir"], ascending=[False, False]).head(10)
    compact_neg = compact_all.sort_values(["ic_mean", "ic_ir"], ascending=[True, True]).head(10)
    emit("\nTop 10 (IC Mean 最大)")
    emit(compact_pos.to_string(index=False, float_format=lambda v: f"{v: .5f}"))
    emit("\nBottom 10 (IC Mean 最小)")
    emit(compact_neg.to_string(index=False, float_format=lambda v: f"{v: .5f}"))

    with open(diag_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(diag_lines))
        f.write("\n")
    print(f"\n✅ 已保存诊断报告: {diag_txt_path}")
    return diag_txt_path
