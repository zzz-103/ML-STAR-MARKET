# 位置: 12（因子重要性）| main.py 在最后一个预测日上拟合并输出 gain/weight/cover 等
# 输入: df_ml/all_dates/predict_dates/features/args
# 输出: (df_imp, meta) 或 None；save_factor_importance 写入 csv/meta/brief
# 依赖: _02_parsing_utils、_04_feature_engineering、xgboost/numpy/pandas
from __future__ import annotations

import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

from ml_models import xgb_config as cfg
from ml_models.model_functions._02_parsing_utils import parse_yyyymmdd
from ml_models.model_functions._04_feature_engineering import build_constraints_dict, build_monotone_constraints


def compute_factor_importance(
    df_ml: pd.DataFrame,
    all_dates: pd.Index,
    predict_dates: pd.Index,
    features: list[str],
    args,
) -> tuple[pd.DataFrame, dict] | None:
    if len(predict_dates) == 0:
        return None

    last_target_date = predict_dates[-1]
    pos = int(all_dates.get_loc(last_target_date))
    train_gap = int(getattr(args, "train_gap", cfg.DEFAULT_TRAINING["train_gap"]))
    train_window = int(getattr(args, "train_window"))
    if pos < train_window + train_gap - 1:
        return None
    train_end_date = all_dates[pos - train_gap]
    train_start_date = all_dates[pos - train_gap - train_window + 1]
    train_floor = parse_yyyymmdd(getattr(args, "train_floor_date", None))
    if train_floor is not None and train_start_date < train_floor:
        train_start_date = train_floor

    idx = pd.IndexSlice
    train_data = df_ml.loc[idx[train_start_date:train_end_date, :], :].sort_index()
    if len(train_data) < 1000:
        return None

    X_train = train_data[features]
    y_train = train_data["ret_next"]

    constraints_dict = build_constraints_dict(
        use_constraints=bool(getattr(args, "use_constraints", True)),
        constraints_csv=getattr(args, "constraints", None),
    )
    monotone_constraints = build_monotone_constraints(features, constraints_dict)

    objective = str(getattr(args, "xgb_objective", "reg:squarederror"))
    if objective.startswith("rank:"):
        train_dates = train_data.index.get_level_values("date").to_numpy()
        _, group_sizes = np.unique(train_dates, return_counts=True)
        model: xgb.XGBModel = xgb.XGBRanker(
            n_estimators=int(getattr(args, "n_estimators")),
            learning_rate=float(getattr(args, "learning_rate")),
            max_depth=int(getattr(args, "max_depth")),
            subsample=float(getattr(args, "subsample", 0.8)),
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=float(getattr(args, "reg_lambda", 1.0)),
            n_jobs=1,
            objective=objective,
            eval_metric="ndcg",
            random_state=42,
            tree_method="hist",
            monotone_constraints=monotone_constraints,
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=int(getattr(args, "n_estimators")),
            learning_rate=float(getattr(args, "learning_rate")),
            max_depth=int(getattr(args, "max_depth")),
            subsample=float(getattr(args, "subsample", 0.8)),
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=float(getattr(args, "reg_lambda", 1.0)),
            n_jobs=1,
            objective=objective,
            random_state=42,
            tree_method="hist",
            monotone_constraints=monotone_constraints,
        )
        group_sizes = None

    fit_started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    if objective.startswith("rank:"):
        model.fit(X_train, y_train, group=group_sizes)
    else:
        model.fit(X_train, y_train)
    fit_seconds = float(time.perf_counter() - t0)
    fit_finished_at = datetime.now().isoformat(timespec="seconds")

    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")
    total_gain = booster.get_score(importance_type="total_gain")

    df_imp = pd.DataFrame({"feature": features})
    df_imp["xgb_gain"] = df_imp["feature"].map(gain).fillna(0.0).astype("float64")
    df_imp["xgb_total_gain"] = df_imp["feature"].map(total_gain).fillna(0.0).astype("float64")
    df_imp["xgb_weight"] = df_imp["feature"].map(weight).fillna(0.0).astype("float64")
    df_imp["xgb_cover"] = df_imp["feature"].map(cover).fillna(0.0).astype("float64")

    missing_rate = X_train.isna().mean().astype("float64")
    df_imp["missing_rate"] = df_imp["feature"].map(missing_rate).fillna(1.0).astype("float64")

    sample_n = min(200_000, len(train_data))
    sample = train_data.sample(n=sample_n, random_state=42) if sample_n < len(train_data) else train_data
    corr = sample[features + ["ret_next"]].corr(method="spearman")["ret_next"].drop(labels=["ret_next"])
    df_imp["spearman_abs"] = df_imp["feature"].map(corr.abs()).fillna(0.0).astype("float64")

    df_imp = df_imp.sort_values(["xgb_gain", "spearman_abs"], ascending=[False, False]).reset_index(drop=True)
    df_imp["rank"] = np.arange(1, len(df_imp) + 1, dtype=np.int64)

    meta = {
        "importance_target_date": last_target_date,
        "importance_train_start": train_start_date,
        "importance_train_end": train_end_date,
        "train_rows": int(len(train_data)),
        "sample_rows": int(len(sample)),
        "feature_count": int(len(features)),
        "xgb_objective": objective,
        "importance_fit_started_at": fit_started_at,
        "importance_fit_finished_at": fit_finished_at,
        "importance_fit_seconds": fit_seconds,
    }
    return df_imp, meta


def save_factor_importance(df_imp: pd.DataFrame, meta: dict, args, logger: logging.Logger) -> tuple[str, str, str]:
    out_dir = str(getattr(args, "factors_importance_dir"))
    os.makedirs(out_dir, exist_ok=True)
    d0 = pd.to_datetime(meta["importance_train_start"]).strftime("%Y%m%d")
    d1 = pd.to_datetime(meta["importance_train_end"]).strftime("%Y%m%d")
    dt = pd.to_datetime(meta["importance_target_date"]).strftime("%Y%m%d")
    tag = (
        f"target_{dt}_train_{d0}_{d1}_tw{int(getattr(args, 'train_window'))}_"
        f"xgb{int(getattr(args, 'n_estimators'))}_d{int(getattr(args, 'max_depth'))}"
    )
    out_csv = os.path.join(out_dir, f"factor_importance_{tag}.csv")
    out_meta = os.path.join(out_dir, f"factor_importance_{tag}.meta.json")
    out_brief = os.path.join(out_dir, f"factor_importance_{tag}.brief.txt")
    df_imp.to_csv(out_csv, index=False)
    pd.Series(meta).to_json(out_meta, force_ascii=False, indent=2)

    top_n = int(min(20, len(df_imp)))
    lines: list[str] = []
    lines.append(f"target_date: {dt}")
    lines.append(f"train_window: {int(getattr(args, 'train_window'))}")
    lines.append(f"train_range: {d0} ~ {d1}")
    lines.append(f"train_rows: {int(meta.get('train_rows', 0))}")
    if meta.get("importance_fit_seconds", None) is not None:
        lines.append(f"importance_fit_seconds: {float(meta.get('importance_fit_seconds')):.3f}")
    if meta.get("importance_fit_started_at", None):
        lines.append(f"importance_fit_started_at: {meta.get('importance_fit_started_at')}")
    if meta.get("importance_fit_finished_at", None):
        lines.append(f"importance_fit_finished_at: {meta.get('importance_fit_finished_at')}")
    dropped = meta.get("dropped_factors", None)
    if isinstance(dropped, (list, tuple)):
        lines.append(f"dropped_factors_count: {len(dropped)}")
    lines.append("")
    lines.append(f"Top {top_n} factors (by xgb_gain):")
    cols = [c for c in ["rank", "feature", "xgb_gain", "spearman_abs", "missing_rate"] if c in df_imp.columns]
    lines.append(df_imp.head(top_n)[cols].to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    lines.append("")
    with open(out_brief, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    logger.info("saved_factor_importance csv=%s meta=%s brief=%s", out_csv, out_meta, out_brief)
    return out_csv, out_meta, out_brief
