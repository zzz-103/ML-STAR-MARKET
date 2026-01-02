# 位置: 14（过拟合自检）| main.py --overfit-check 调用（单点/区间）
# 输入: args；或外部传入 df_ml（面板数据，含 ret_next）
# 输出: DataFrame 结果或 None；并通过 logger 输出关键指标
# 依赖: _02_parsing_utils、_04_feature_engineering、_05_weights、_06_data_preprocessing、_08_xgb_training、_10_evaluation
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml_models import xgb_config as cfg
from ml_models.model_functions._02_parsing_utils import parse_yyyymmdd
from ml_models.model_functions._04_feature_engineering import (
    apply_feature_filters,
    build_constraints_dict,
    build_drop_factors,
    build_monotone_constraints,
)
from ml_models.model_functions._05_weights import build_sample_weights
from ml_models.model_functions._06_data_preprocessing import prepare_dataset
from ml_models.model_functions._08_xgb_training import fit_xgb_model
from ml_models.model_functions._10_evaluation import daily_ic, daily_topk_mean


@dataclass(frozen=True)
class OverfitResult:
    target_date: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    sub_train_days: int
    valid_days: int
    rmse_tr: float
    rmse_va: float
    rmse_gap: float
    ic_tr_mean: float
    ic_va_mean: float
    ic_mean_gap: float
    ic_tr_ir: float
    ic_va_ir: float
    topk_tr_mean: float
    topk_va_mean: float
    topk_mean_gap: float
    objective: str
    top_k: int


def run_overfit_check(args, logger: logging.Logger) -> pd.DataFrame | None:
    df_ml, _ = prepare_dataset(args, logger=logger)
    return run_overfit_check_on_panel(args, df_ml=df_ml, logger=logger)


def run_overfit_check_on_panel(args, df_ml: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame | None:
    all_dates = df_ml.index.get_level_values("date").unique().sort_values()
    if len(all_dates) < int(getattr(args, "train_window")) + 30:
        raise ValueError("交易日数量不足，无法进行过拟合自检")

    def rmse(a: pd.Series, b: pd.Series) -> float:
        a1 = pd.to_numeric(a, errors="coerce")
        b1 = pd.to_numeric(b, errors="coerce")
        m = (a1 - b1).astype("float64")
        m = m[np.isfinite(m)]
        if len(m) == 0:
            return float("nan")
        return float(np.sqrt(np.mean(m * m)))

    def summarize(s: pd.Series) -> dict:
        if s is None or len(s) == 0:
            return {"n": 0, "mean": float("nan"), "std": float("nan"), "ir": float("nan")}
        mean = float(s.mean())
        std = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
        ir = float(mean / std) if np.isfinite(mean) and np.isfinite(std) and std > 1e-12 else float("nan")
        return {"n": int(len(s)), "mean": mean, "std": std, "ir": ir}

    def single(target_date: pd.Timestamp) -> OverfitResult:
        if target_date not in all_dates:
            raise ValueError(f"target_date 不在数据日期范围内: {target_date.strftime('%Y%m%d')}")
        pos = int(all_dates.get_loc(target_date))
        if pos <= 0:
            raise ValueError("target_date 过早，无法构建训练窗口")

        train_gap = int(getattr(args, "train_gap", cfg.DEFAULT_TRAINING["train_gap"]))
        train_window = int(getattr(args, "train_window"))
        if pos < train_window + train_gap - 1:
            raise ValueError("target_date 过早，无法构建训练窗口")
        train_end_date = all_dates[pos - train_gap]
        train_start_date = all_dates[max(0, pos - train_gap - train_window + 1)]
        train_floor = parse_yyyymmdd(getattr(args, "train_floor_date", None))
        if train_floor is not None and train_start_date < train_floor:
            train_start_date = train_floor

        idx = pd.IndexSlice
        train_data = df_ml.loc[idx[train_start_date:train_end_date, :], :].sort_index()
        if len(train_data) < 5000:
            raise ValueError("训练样本过少，无法进行过拟合自检")

        features = [c for c in train_data.columns if c != "ret_next"]
        drop_factors = build_drop_factors(
            use_default_drop_factors=bool(getattr(args, "use_default_drop_factors", True)),
            drop_factors_csv=getattr(args, "drop_factors", None),
        )
        final_features = apply_feature_filters(features, drop_factors)
        if len(final_features) == 0:
            raise ValueError("可用特征为空")

        train_dates = train_data.index.get_level_values("date").unique().sort_values()
        valid_days = max(5, int(getattr(args, "overfit_valid_days", 20)))
        if len(train_dates) <= valid_days + 5:
            raise ValueError("训练日期过少，无法切分 train/valid")
        valid_dates = train_dates[-valid_days:]
        sub_train_dates = train_dates[: -valid_days]

        tr = train_data.loc[idx[sub_train_dates.min() : sub_train_dates.max(), :], :].sort_index()
        va = train_data.loc[idx[valid_dates.min() : valid_dates.max(), :], :].sort_index()
        if len(tr) == 0 or len(va) == 0:
            raise ValueError("train/valid 切分失败")

        X_tr = tr[final_features]
        y_tr = tr["ret_next"]
        X_va = va[final_features]
        y_va = va["ret_next"]

        objective = str(getattr(args, "xgb_objective", "reg:squarederror"))
        if objective.startswith("rank:"):
            tr_dates_u = tr.index.get_level_values("date").unique().sort_values()
            sw = build_sample_weights(
                mode=str(getattr(args, "sample_weight_mode", "none")),
                train_dates=tr_dates_u,
                ref_date=sub_train_dates.max(),
                anchor_days=int(getattr(args, "decay_anchor_days", 0)),
                half_life_days=int(getattr(args, "decay_half_life_days", 1)),
                min_weight=float(getattr(args, "decay_min_weight", 0.1)),
            )
        else:
            sw = build_sample_weights(
                mode=str(getattr(args, "sample_weight_mode", "none")),
                train_dates=tr.index.get_level_values("date"),
                ref_date=sub_train_dates.max(),
                anchor_days=int(getattr(args, "decay_anchor_days", 0)),
                half_life_days=int(getattr(args, "decay_half_life_days", 1)),
                min_weight=float(getattr(args, "decay_min_weight", 0.1)),
            )

        constraints_dict = build_constraints_dict(
            use_constraints=bool(getattr(args, "use_constraints", True)),
            constraints_csv=getattr(args, "constraints", None),
        )
        monotone_constraints = build_monotone_constraints(final_features, constraints_dict)

        model = fit_xgb_model(
            X_train=X_tr,
            y_train=y_tr,
            objective=objective,
            n_estimators=int(getattr(args, "n_estimators")),
            learning_rate=float(getattr(args, "learning_rate")),
            max_depth=int(getattr(args, "max_depth")),
            subsample=float(getattr(args, "subsample", 0.8)),
            reg_lambda=float(getattr(args, "reg_lambda", 1.0)),
            monotone_constraints=monotone_constraints,
            sample_weight=sw,
        )

        pred_tr = pd.Series(model.predict(X_tr), index=y_tr.index, dtype="float64")
        pred_va = pd.Series(model.predict(X_va), index=y_va.index, dtype="float64")

        top_k = int(getattr(args, "overfit_top_k", None) or getattr(args, "top_k", 30))
        ic_tr = daily_ic(y_tr, pred_tr)
        ic_va = daily_ic(y_va, pred_va)
        top_tr = daily_topk_mean(y_tr, pred_tr, top_k=top_k)
        top_va = daily_topk_mean(y_va, pred_va, top_k=top_k)

        s_ic_tr = summarize(ic_tr)
        s_ic_va = summarize(ic_va)
        s_top_tr = summarize(top_tr)
        s_top_va = summarize(top_va)

        rmse_tr = rmse(y_tr, pred_tr)
        rmse_va = rmse(y_va, pred_va)

        return OverfitResult(
            target_date=target_date,
            train_start=train_start_date,
            train_end=train_end_date,
            sub_train_days=int(len(sub_train_dates)),
            valid_days=int(len(valid_dates)),
            rmse_tr=float(rmse_tr),
            rmse_va=float(rmse_va),
            rmse_gap=float(rmse_va - rmse_tr) if np.isfinite(rmse_va) and np.isfinite(rmse_tr) else float("nan"),
            ic_tr_mean=float(s_ic_tr["mean"]),
            ic_va_mean=float(s_ic_va["mean"]),
            ic_mean_gap=float(s_ic_va["mean"] - s_ic_tr["mean"]),
            ic_tr_ir=float(s_ic_tr["ir"]),
            ic_va_ir=float(s_ic_va["ir"]),
            topk_tr_mean=float(s_top_tr["mean"]),
            topk_va_mean=float(s_top_va["mean"]),
            topk_mean_gap=float(s_top_va["mean"] - s_top_tr["mean"]),
            objective=str(objective),
            top_k=int(top_k),
        )

    if bool(getattr(args, "overfit_range", False)):
        train_gap = int(getattr(args, "train_gap", cfg.DEFAULT_TRAINING["train_gap"]))
        start_index = int(getattr(args, "train_window")) + train_gap - 1
        predict_dates = all_dates[start_index:]
        if getattr(args, "start_date", None):
            predict_dates = predict_dates[predict_dates >= pd.to_datetime(getattr(args, "start_date"), format="%Y%m%d")]
        if getattr(args, "end_date", None):
            predict_dates = predict_dates[predict_dates <= pd.to_datetime(getattr(args, "end_date"), format="%Y%m%d")]
        if getattr(args, "max_predict_days", None) is not None:
            predict_dates = predict_dates[: int(getattr(args, "max_predict_days"))]
        if len(predict_dates) == 0:
            raise ValueError("预测日期为空，无法进行区间过拟合自检")

        step = max(1, int(getattr(args, "overfit_range_step", 20)))
        max_points = max(1, int(getattr(args, "overfit_range_max_points", 60)))
        sample_dates = predict_dates[::step]
        if len(sample_dates) > max_points:
            sample_dates = sample_dates[:max_points]

        rows: list[dict] = []
        errors = 0
        for d in sample_dates:
            try:
                r = single(d)
                rows.append(r.__dict__)
            except Exception:
                errors += 1
                continue
        if len(rows) == 0:
            raise ValueError(f"区间过拟合自检无可用结果（errors={errors}）")

        df = pd.DataFrame(rows).sort_values("target_date")
        logger.info(
            "overfit_range range=%s~%s points=%d/%d step=%d errors=%d valid_days=%d",
            pd.to_datetime(df["target_date"].iloc[0]).strftime("%Y%m%d"),
            pd.to_datetime(df["target_date"].iloc[-1]).strftime("%Y%m%d"),
            int(len(df)),
            int(len(sample_dates)),
            int(step),
            int(errors),
            int(getattr(args, "overfit_valid_days", 20)),
        )
        return df

    target_s = getattr(args, "overfit_target_date", None) or getattr(args, "start_date", None)
    target_date = all_dates[-1] if target_s is None else pd.to_datetime(str(target_s), format="%Y%m%d", errors="raise")
    r = single(target_date)
    logger.info(
        "overfit target=%s train=%s~%s sub_train_days=%d valid_days=%d objective=%s",
        pd.to_datetime(r.target_date).strftime("%Y%m%d"),
        pd.to_datetime(r.train_start).strftime("%Y%m%d"),
        pd.to_datetime(r.train_end).strftime("%Y%m%d"),
        int(r.sub_train_days),
        int(r.valid_days),
        r.objective,
    )
    logger.info(
        "overfit train rmse=%.6f ic_mean=%.4f ic_ir=%.3f top%d_mean=%.4f",
        float(r.rmse_tr),
        float(r.ic_tr_mean),
        float(r.ic_tr_ir),
        int(r.top_k),
        float(r.topk_tr_mean),
    )
    logger.info(
        "overfit valid rmse=%.6f ic_mean=%.4f ic_ir=%.3f top%d_mean=%.4f",
        float(r.rmse_va),
        float(r.ic_va_mean),
        float(r.ic_va_ir),
        int(r.top_k),
        float(r.topk_va_mean),
    )
    logger.info(
        "overfit gap rmse=%.6f ic_mean=%.4f top%d_mean=%.4f",
        float(r.rmse_gap),
        float(r.ic_mean_gap),
        int(r.top_k),
        float(r.topk_mean_gap),
    )
    return None
