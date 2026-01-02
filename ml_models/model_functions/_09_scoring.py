# 位置: 09（滚动打分 worker）| main.py 并行调用，生成每日 temp_score parquet
# 输入: target_date/train_range、df_ml/df_price、args、features、temp_dir
# 输出: str(YYYYMMDD) 或 None；副作用为写入 temp_dir/YYYYMMDD.parquet（code, score[, turnover_prev]）
# 依赖: _04_feature_engineering、_05_weights、_08_xgb_training、sklearn
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from ml_models.model_functions._04_feature_engineering import (
    apply_feature_filters,
    build_constraints_dict,
    build_drop_factors,
    build_monotone_constraints,
)
from ml_models.model_functions._05_weights import build_sample_weights
from ml_models.model_functions._08_xgb_training import fit_xgb_model


def process_single_day_score(
    target_date: pd.Timestamp,
    train_start_date: pd.Timestamp,
    train_end_date: pd.Timestamp,
    df_ml: pd.DataFrame,
    df_price: pd.DataFrame,
    args,
    features: list[str],
    temp_dir: str,
) -> Optional[str]:
    try:
        idx = pd.IndexSlice
        train_data = df_ml.loc[idx[train_start_date:train_end_date, :], :].sort_index()
        test_data = df_ml.loc[idx[target_date, :], :]
        if len(train_data) < 100 or len(test_data) == 0:
            return None

        drop_factors = build_drop_factors(
            use_default_drop_factors=bool(getattr(args, "use_default_drop_factors", True)),
            drop_factors_csv=getattr(args, "drop_factors", None),
        )
        final_features = apply_feature_filters(features, drop_factors)
        if len(final_features) == 0:
            return None

        X_train = train_data[final_features]
        y_train = train_data["ret_next"]
        X_test = test_data[final_features]
        objective = str(getattr(args, "xgb_objective", "reg:squarederror"))
        if objective.startswith("rank:"):
            train_dates_u = train_data.index.get_level_values("date").unique().sort_values()
            sample_weight = build_sample_weights(
                mode=str(getattr(args, "sample_weight_mode", "none")),
                train_dates=train_dates_u,
                ref_date=train_end_date,
                anchor_days=int(getattr(args, "decay_anchor_days", 0)),
                half_life_days=int(getattr(args, "decay_half_life_days", 1)),
                min_weight=float(getattr(args, "decay_min_weight", 0.1)),
            )
        else:
            sample_weight = build_sample_weights(
                mode=str(getattr(args, "sample_weight_mode", "none")),
                train_dates=train_data.index.get_level_values("date"),
                ref_date=train_end_date,
                anchor_days=int(getattr(args, "decay_anchor_days", 0)),
                half_life_days=int(getattr(args, "decay_half_life_days", 1)),
                min_weight=float(getattr(args, "decay_min_weight", 0.1)),
            )

        constraints_dict = build_constraints_dict(
            use_constraints=bool(getattr(args, "use_constraints", True)),
            constraints_csv=getattr(args, "constraints", None),
        )
        monotone_constraints = build_monotone_constraints(final_features, constraints_dict)

        model_xgb = fit_xgb_model(
            X_train=X_train,
            y_train=y_train,
            objective=objective,
            n_estimators=int(getattr(args, "n_estimators")),
            learning_rate=float(getattr(args, "learning_rate")),
            max_depth=int(getattr(args, "max_depth")),
            subsample=float(getattr(args, "subsample")),
            reg_lambda=float(getattr(args, "reg_lambda")),
            monotone_constraints=monotone_constraints,
            sample_weight=sample_weight,
        )
        pred_xgb = model_xgb.predict(X_test)
        final_score = np.asarray(pred_xgb, dtype=np.float64)

        if bool(getattr(args, "use_knn", False)):
            fill_values = X_train.median(axis=0, skipna=True)
            fill_values = fill_values.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")
            X_train_knn = X_train.fillna(fill_values)
            X_test_knn = X_test.fillna(fill_values)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_knn.to_numpy(dtype=np.float64))
            X_test_scaled = scaler.transform(X_test_knn.to_numpy(dtype=np.float64))
            curr_k = min(int(getattr(args, "knn_neighbors", 50)), len(X_train) - 1)
            if curr_k >= 1:
                model_knn = KNeighborsRegressor(n_neighbors=curr_k, weights="distance", n_jobs=1)
                model_knn.fit(X_train_scaled, y_train.to_numpy(dtype=np.float64))
                pred_knn = model_knn.predict(X_test_scaled)

                df_blend = pd.DataFrame(index=test_data.index)
                df_blend["xgb"] = pred_xgb
                df_blend["knn"] = pred_knn
                df_blend["r_xgb"] = df_blend["xgb"].rank(pct=True)
                df_blend["r_knn"] = df_blend["knn"].rank(pct=True)
                final_score = (
                    float(getattr(args, "blend_xgb_weight", 0.7)) * df_blend["r_xgb"]
                    + float(getattr(args, "blend_knn_weight", 0.3)) * df_blend["r_knn"]
                ).to_numpy(dtype=np.float64)

        daily_result = pd.DataFrame({"code": test_data.index.get_level_values("code"), "score": final_score})
        try:
            valid_codes = daily_result["code"].unique()
            price_today = df_price.loc[idx[target_date, valid_codes], :]
            cond_active = price_today["open"].notna() & (price_today["open"] > 0)
            if "upper_limit" in price_today.columns:
                cond_no_limit_up = price_today["open"] < price_today["upper_limit"]
            else:
                cond_no_limit_up = True
            if "turnover_prev" in price_today.columns:
                cond_liquid = price_today["turnover_prev"] > float(getattr(args, "min_turnover", 15_000_000.0))
            else:
                cond_liquid = True
            final_mask = cond_active & cond_no_limit_up & cond_liquid
            tradable_codes = price_today[final_mask].index.get_level_values("code")
            daily_result = daily_result[daily_result["code"].isin(tradable_codes)]
            if "turnover_prev" in price_today.columns:
                turnover_prev_map = price_today["turnover_prev"].reset_index(level="date", drop=True)
                daily_result = daily_result.merge(
                    turnover_prev_map.rename("turnover_prev"),
                    left_on="code",
                    right_index=True,
                    how="left",
                )
            if len(daily_result) == 0:
                return None
        except Exception:
            pass

        keep_n = max(200, int(getattr(args, "top_k", 0)), int(getattr(args, "buffer_k", 0)), int(getattr(args, "emergency_exit_rank", 0)))
        daily_result = daily_result.sort_values(by="score", ascending=False).head(keep_n)
        date_str = target_date.strftime("%Y%m%d")
        temp_file_path = os.path.join(temp_dir, f"{date_str}.parquet")
        daily_result.to_parquet(temp_file_path)
        return date_str
    except Exception as e:
        return f"Error: {str(e)}"
