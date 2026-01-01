# 位置: 总配置（默认参数中心）| 被 /ml_models/main.py 与 /model_functions/_01_cli.py 读取
# 输入: 无（仅常量/字典定义）
# 输出: DEFAULT_* 配置字典（paths/universe/label/training/model/portfolio/timing/diagnose/overfit 等）
# 依赖: 无
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: str) -> str:
    p = str(path)
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return str((PROJECT_ROOT / p).resolve(strict=False))

DEFAULT_PATHS = {
    "factor_data_path": "factors_data/all_factors_with_fundamentals.parquet",
    "price_data_path": "pre_data/cleaned_stock_data_300_688_with_idxstk.parquet",
    "risk_data_path": "pre_data/merged_20200101_20241231.csv",
    "output_dir": "ml_results",
    "sub_dir_name": "xgb_results_gem_star_momo",
    "temp_dir_name": "temp_scores_gem_star_momo",
    "factors_importance_dir": "ml_results/factors_importance",
}

DEFAULT_UNIVERSE = {
    "stock_pool_prefixes": ("300", "688"),
    "min_turnover": 15_000_000.0,
}

DEFAULT_LABEL = {
    "predict_days": 5,
    "label_benchmark_method": "median",
    "label_benchmark_universe": "tradable",
}

DEFAULT_TRAINING = {
    "train_window": 250,
    "train_gap": 6,
    "n_workers": None,
    "dropna_features": False,
    "sample_weight_mode": "time_decay_exp",
    "decay_anchor_days": 30,
    "decay_half_life_days": 60,
    "decay_min_weight": 0.1,
}

DEFAULT_MODEL = {
    "xgb_objective": "reg:squarederror",
    "n_estimators": 200,
    "learning_rate": 0.01,
    "max_depth": 4,
    "subsample": 0.7,
    "reg_lambda": 10.0,
    "use_knn": False,
    "knn_neighbors": 50,
    "blend_xgb_weight": 0.7,
    "blend_knn_weight": 0.3,
    "use_constraints": True,
}

DEFAULT_PORTFOLIO = {
    "top_k": 20,
    "buffer_k": 30,
    "rebalance_period": 5,
    "rebalance_turnover_cap": 0.90,
    "smooth_window": 5,
    "inertia_ratio": 1.02,
    "emergency_exit_rank": 50,
    "keep_missing_positions": True,
    "band_threshold": 0.001,
    "max_w": 0.06,
    "min_weight": 0.0005,
    "non_rebalance_action": "empty",
    "limit_policy": "freeze",
}

DEFAULT_TIMING = {
    "timing_method": "index_ma20",
    "timing_threshold": 0.0,
    "timing_hysteresis": 0.005,
    "timing_enter_threshold": None,
    "timing_exit_threshold": None,
    "timing_bad_exposure": 0.4,
    "risk_index_code": "399006",
    "risk_ma_window": 20,
    "risk_ma_fast_window": 5,
    "risk_ma_slow_window": 20,
    "risk_ma_buffer": 0.005,
}

DEFAULT_DIAGNOSE = {
    "diag_start_date": "20230101",
    "diag_end_date": "20241231",
    "diag_step": 5,
    "diag_top_k": 30,
    "diag_min_n": 300,
}

DEFAULT_OVERFIT = {
    "overfit_check": False,
    "overfit_target_date": None,
    "overfit_valid_days": 20,
    "overfit_top_k": None,
    "overfit_range": False,
    "overfit_range_step": 20,
    "overfit_range_max_points": 60,
    "overfit_along": False,
}


DEFAULT_QUICK_EVAL = {
    "quick_eval": False,
    "quick_eval_level": 1,
    "quick_eval_dir_name": "quick_eval",
    "quick_eval_cache": True,
    "quick_eval_risk_free": 0.0,
    "quick_eval_fee_rate": 0.0003,
    "quick_eval_slippage": 0.0,
    "quick_eval_capital": 10_000_000.0,
}

DEFAULT_DROP_FACTORS = [
    "ff_mkt",
    "ff_hml",
    "ff_smb",
    "ff_smb_cov_60",
    "beta_mkt",
    "beta_smb",
    "beta_hml",
    "kdj_k",
    "kdj_d",
    "kdj_j",
    "william_r",
    "cci_14",
    "rsi_14",
    "mfi_14",
    "psy_12",
    "vr_24",
    "macd_dea",
    "macd_dif",
    "turnover_cv_20",
    "turnover_std_10",
    "ret_lag_3",
    "ret_lag_4",
    "ret_lag_5",
    "candle_body_abs",
    "shadow_lower",
    "hl_ratio_smooth",
    "candle_amplitude",
    "turnover_cv_20",
    "turnover_std_10",
    "pe_ttm",
    "ps_ttm",
    "pb",
    "pcf_net_ttm",
    "float_market_cap",
    "smart_overnight_20"
]

DEFAULT_MONOTONE_CONSTRAINTS = {
    "EP_ttm": 0,
    "SP_ttm": 1,
    "BP": 1,
    "CFP_ttm": 0,
    "roc_5": 1,
    "roc_10": 1,
    "roc_20": 1,
    "ma60_bias": 1,
    "momo_sl_20_5": 1,
    "breakout_20_vol_confirm": 1,
    "smart_overnight_20": 1,
    "vol_20": 1,
    "pv_corr": 1,
    "pv_corr_20": 0,
    "boll_width": 1,
    "candle_body": 0,
    "shadow_upper": -1,
    "hl_ratio": -1,
    "ret_lag_1": 0,
    "ma5_bias": 0,
}
