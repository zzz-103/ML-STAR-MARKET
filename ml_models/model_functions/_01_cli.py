# 位置: 01（参数入口）| main.py 首先调用以构建 args
# 输入: argv(list[str] | None)；None 表示读取 sys.argv
# 输出: argparse.Namespace（包含所有可配置参数及默认值）
# 依赖: /ml_models/xgb_config.py
from __future__ import annotations

import argparse
import os

from ml_models import xgb_config as cfg


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器，所有默认值来自 xgb_config.py。"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--factor-data-path", default=cfg.DEFAULT_PATHS["factor_data_path"])
    parser.add_argument("--price-data-path", default=cfg.DEFAULT_PATHS["price_data_path"])
    parser.add_argument("--risk-data-path", default=cfg.DEFAULT_PATHS["risk_data_path"])
    parser.add_argument("--output-dir", default=cfg.DEFAULT_PATHS["output_dir"])
    parser.add_argument("--sub-dir-name", default=cfg.DEFAULT_PATHS["sub_dir_name"])
    parser.add_argument("--temp-dir-name", default=cfg.DEFAULT_PATHS["temp_dir_name"])
    parser.add_argument("--factors-importance-dir", default=cfg.DEFAULT_PATHS["factors_importance_dir"])

    parser.add_argument("--train-window", type=int, default=cfg.DEFAULT_TRAINING["train_window"])
    parser.add_argument("--train-gap", type=int, default=cfg.DEFAULT_TRAINING["train_gap"], help="Gap between train_end_date and target_date to avoid label leakage")
    parser.add_argument("--dropna-features", action=argparse.BooleanOptionalAction, default=cfg.DEFAULT_TRAINING["dropna_features"])
    parser.add_argument(
        "--sample-weight-mode",
        choices=["none", "time_decay_exp", "time_decay_linear"],
        default=cfg.DEFAULT_TRAINING["sample_weight_mode"],
    )
    parser.add_argument("--decay-anchor-days", type=int, default=cfg.DEFAULT_TRAINING["decay_anchor_days"])
    parser.add_argument("--decay-half-life-days", type=int, default=cfg.DEFAULT_TRAINING["decay_half_life_days"])
    parser.add_argument("--decay-min-weight", type=float, default=cfg.DEFAULT_TRAINING["decay_min_weight"])

    parser.add_argument("--n-estimators", type=int, default=cfg.DEFAULT_MODEL["n_estimators"])
    parser.add_argument("--learning-rate", type=float, default=cfg.DEFAULT_MODEL["learning_rate"])
    parser.add_argument("--max-depth", type=int, default=cfg.DEFAULT_MODEL["max_depth"])
    parser.add_argument("--subsample", type=float, default=cfg.DEFAULT_MODEL["subsample"])
    parser.add_argument("--reg-lambda", type=float, default=cfg.DEFAULT_MODEL["reg_lambda"])
    parser.add_argument("--xgb-objective", default=cfg.DEFAULT_MODEL["xgb_objective"])

    parser.add_argument("--use-knn", action=argparse.BooleanOptionalAction, default=cfg.DEFAULT_MODEL["use_knn"])
    parser.add_argument("--knn-neighbors", type=int, default=cfg.DEFAULT_MODEL["knn_neighbors"])
    parser.add_argument("--blend-xgb-weight", type=float, default=cfg.DEFAULT_MODEL["blend_xgb_weight"])
    parser.add_argument("--blend-knn-weight", type=float, default=cfg.DEFAULT_MODEL["blend_knn_weight"])

    parser.add_argument("--use-constraints", action=argparse.BooleanOptionalAction, default=cfg.DEFAULT_MODEL["use_constraints"])
    parser.add_argument("--constraints", default=None)

    parser.add_argument("--top-k", type=int, default=cfg.DEFAULT_PORTFOLIO["top_k"])
    parser.add_argument("--buffer-k", type=int, default=cfg.DEFAULT_PORTFOLIO["buffer_k"])
    parser.add_argument("--rebalance-period", type=int, default=cfg.DEFAULT_PORTFOLIO["rebalance_period"])
    parser.add_argument("--rebalance-turnover-cap", type=float, default=cfg.DEFAULT_PORTFOLIO["rebalance_turnover_cap"])
    parser.add_argument("--smooth-window", type=int, default=cfg.DEFAULT_PORTFOLIO["smooth_window"])
    parser.add_argument("--inertia-ratio", type=float, default=cfg.DEFAULT_PORTFOLIO["inertia_ratio"])
    parser.add_argument("--emergency-exit-rank", type=int, default=cfg.DEFAULT_PORTFOLIO["emergency_exit_rank"])
    parser.add_argument("--keep-missing-positions", action=argparse.BooleanOptionalAction, default=cfg.DEFAULT_PORTFOLIO["keep_missing_positions"])
    parser.add_argument("--band-threshold", type=float, default=cfg.DEFAULT_PORTFOLIO["band_threshold"])
    parser.add_argument("--max-w", type=float, default=cfg.DEFAULT_PORTFOLIO["max_w"])
    parser.add_argument("--min-weight", type=float, default=cfg.DEFAULT_PORTFOLIO["min_weight"])
    parser.add_argument("--non-rebalance-action", choices=["empty", "carry"], default=cfg.DEFAULT_PORTFOLIO["non_rebalance_action"])
    parser.add_argument("--limit-policy", choices=["freeze", "sell_only"], default=cfg.DEFAULT_PORTFOLIO["limit_policy"])

    parser.add_argument("--timing-method", choices=["index_ma20", "index_ma_dual", "score", "none"], default=cfg.DEFAULT_TIMING["timing_method"])
    parser.add_argument("--timing-threshold", type=float, default=cfg.DEFAULT_TIMING["timing_threshold"])
    parser.add_argument("--timing-bad-exposure", type=float, default=cfg.DEFAULT_TIMING["timing_bad_exposure"])
    parser.add_argument("--timing-enter-threshold", type=float, default=cfg.DEFAULT_TIMING["timing_enter_threshold"])
    parser.add_argument("--timing-exit-threshold", type=float, default=cfg.DEFAULT_TIMING["timing_exit_threshold"])
    parser.add_argument("--timing-hysteresis", type=float, default=cfg.DEFAULT_TIMING["timing_hysteresis"])
    parser.add_argument("--risk-index-code", default=cfg.DEFAULT_TIMING["risk_index_code"])
    parser.add_argument("--risk-ma-window", type=int, default=cfg.DEFAULT_TIMING["risk_ma_window"])
    parser.add_argument("--risk-ma-fast-window", type=int, default=cfg.DEFAULT_TIMING["risk_ma_fast_window"])
    parser.add_argument("--risk-ma-slow-window", type=int, default=cfg.DEFAULT_TIMING["risk_ma_slow_window"])
    parser.add_argument("--risk-ma-buffer", type=float, default=cfg.DEFAULT_TIMING["risk_ma_buffer"])

    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-predict-days", type=int, default=None)
    parser.add_argument("--train-floor-date", default=None)

    parser.add_argument("--use-default-drop-factors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-factors", default=None)
    parser.add_argument(
        "--label-benchmark-universe",
        default=cfg.DEFAULT_LABEL["label_benchmark_universe"],
        choices=["all", "tradable"],
    )

    parser.add_argument("--diagnose", action="store_true", default=False)
    parser.add_argument("--diag-start-date", default=cfg.DEFAULT_DIAGNOSE["diag_start_date"])
    parser.add_argument("--diag-end-date", default=cfg.DEFAULT_DIAGNOSE["diag_end_date"])
    parser.add_argument("--diag-step", type=int, default=cfg.DEFAULT_DIAGNOSE["diag_step"])
    parser.add_argument("--diag-top-k", type=int, default=cfg.DEFAULT_DIAGNOSE["diag_top_k"])
    parser.add_argument("--diag-min-n", type=int, default=cfg.DEFAULT_DIAGNOSE["diag_min_n"])

    parser.add_argument("--overfit-check", action="store_true", default=cfg.DEFAULT_OVERFIT["overfit_check"])
    parser.add_argument(
        "--overfit-check-only",
        action="store_true",
        default=False,
        help="Skip training and run overfit check only (optionally with quick eval)",
    )
    parser.add_argument("--overfit-target-date", default=cfg.DEFAULT_OVERFIT["overfit_target_date"])
    parser.add_argument("--overfit-valid-days", type=int, default=cfg.DEFAULT_OVERFIT["overfit_valid_days"])
    parser.add_argument("--overfit-top-k", type=int, default=cfg.DEFAULT_OVERFIT["overfit_top_k"])
    parser.add_argument("--overfit-range", action="store_true", default=cfg.DEFAULT_OVERFIT["overfit_range"])
    parser.add_argument("--overfit-range-step", type=int, default=cfg.DEFAULT_OVERFIT["overfit_range_step"])
    parser.add_argument("--overfit-range-max-points", type=int, default=cfg.DEFAULT_OVERFIT["overfit_range_max_points"])
    parser.add_argument("--overfit-along", action="store_true", default=cfg.DEFAULT_OVERFIT["overfit_along"])

    parser.add_argument("--quick-eval", action=argparse.BooleanOptionalAction, default=cfg.DEFAULT_QUICK_EVAL["quick_eval"])
    parser.add_argument("--quick-eval-only", action="store_true", default=False, help="Skip training and run quick evaluation only")
    parser.add_argument("--quick-eval-level", type=int, choices=[0, 1, 2], default=cfg.DEFAULT_QUICK_EVAL["quick_eval_level"])
    parser.add_argument("--quick-eval-dir-name", default=cfg.DEFAULT_QUICK_EVAL["quick_eval_dir_name"])
    parser.add_argument("--quick-eval-cache", action=argparse.BooleanOptionalAction, default=cfg.DEFAULT_QUICK_EVAL["quick_eval_cache"])
    parser.add_argument("--quick-eval-risk-free", type=float, default=cfg.DEFAULT_QUICK_EVAL["quick_eval_risk_free"])

    parser.add_argument("--quick-eval-fee-rate", type=float, default=cfg.DEFAULT_QUICK_EVAL["quick_eval_fee_rate"])
    parser.add_argument("--quick-eval-slippage", type=float, default=cfg.DEFAULT_QUICK_EVAL["quick_eval_slippage"])
    parser.add_argument("--quick-eval-capital", type=float, default=cfg.DEFAULT_QUICK_EVAL["quick_eval_capital"])

    default_workers = max(1, (os.cpu_count() or 1) - 2)
    parser.add_argument("--n-workers", type=int, default=default_workers)

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数；argv=None 表示直接读取 sys.argv。"""
    parser = build_arg_parser()
    return parser.parse_args(args=argv)
