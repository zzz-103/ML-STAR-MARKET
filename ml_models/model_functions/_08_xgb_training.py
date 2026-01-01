# 位置: 08（模型训练）| scoring/overfit 调用以拟合 XGBRegressor/XGBRanker
# 输入: X_train/y_train、objective、xgb 超参、monotone_constraints、sample_weight
# 输出: xgboost.XGBModel（已 fit）
# 依赖: xgboost/numpy/pandas
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


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
        train_dates = X_train.index.get_level_values("date").to_numpy()
        _, group_sizes = np.unique(train_dates, return_counts=True)
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
            eval_metric="ndcg",
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
