# 位置: 08（模型训练）| scoring/overfit 调用以拟合 XGBRegressor/XGBRanker
# 输入: X_train/y_train、objective、xgb 超参、monotone_constraints、sample_weight
# 输出: xgboost.XGBModel（已 fit）
# 依赖: xgboost/numpy/pandas
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


def _build_group_sizes_from_sorted_dates(dates: np.ndarray) -> np.ndarray:
    n = int(len(dates))
    if n <= 0:
        return np.asarray([], dtype=np.uint32)
    if n == 1:
        return np.asarray([1], dtype=np.uint32)
    change = dates[1:] != dates[:-1]
    cut = np.flatnonzero(change) + 1
    edges = np.concatenate([np.asarray([0], dtype=np.int64), cut.astype(np.int64), np.asarray([n], dtype=np.int64)])
    sizes = np.diff(edges).astype(np.uint32)
    return sizes


def fit_xgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    objective: str,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    reg_lambda: float,
    monotone_constraints: str | None,
    sample_weight: np.ndarray | None,
    random_state: int = 42,
) -> xgb.XGBModel:
    objective = str(objective)
    if objective.startswith("rank:"):
        if not X_train.index.is_monotonic_increasing:
            idxer = X_train.index.argsort()
            X_train = X_train.iloc[idxer]
            y_train = y_train.reindex(X_train.index)
            if sample_weight is not None:
                if isinstance(sample_weight, pd.Series):
                    sample_weight = sample_weight.reindex(X_train.index).to_numpy(dtype=np.float64)
                else:
                    sample_weight = np.asarray(sample_weight, dtype=np.float64)[idxer]
        train_dates = X_train.index.get_level_values("date").to_numpy()
        group_sizes = _build_group_sizes_from_sorted_dates(train_dates)
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if len(w) == len(y_train):
                edges = np.cumsum(np.concatenate([np.asarray([0], dtype=np.int64), group_sizes.astype(np.int64)]))
                starts = edges[:-1]
                sums = np.add.reduceat(w, starts)
                denom = group_sizes.astype(np.float64)
                denom = np.where(denom > 0, denom, 1.0)
                sample_weight = (sums / denom).astype(np.float64)
            elif len(w) != len(group_sizes):
                raise ValueError(
                    f"rank sample_weight 长度需为 n_groups({len(group_sizes)}) 或 n_rows({len(y_train)}), got {len(w)}"
                )
        model = xgb.XGBRanker(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            subsample=float(subsample),
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=float(reg_lambda),
            n_jobs=1,
            objective=objective,
            eval_metric="ndcg@20",
            random_state=int(random_state),
            tree_method="hist",
            monotone_constraints=monotone_constraints,
        )
        if sample_weight is None:
            model.fit(X_train, y_train, group=group_sizes)
        else:
            model.fit(X_train, y_train, group=group_sizes, sample_weight=sample_weight)
        return model

    model = xgb.XGBRegressor(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        subsample=float(subsample),
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=float(reg_lambda),
        n_jobs=1,
        objective=objective,
        random_state=int(random_state),
        tree_method="hist",
        monotone_constraints=monotone_constraints,
    )
    if sample_weight is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return model
