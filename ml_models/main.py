# 位置: 主入口（滚动训练→打分→评估→因子重要性→生成持仓）| 对应旧入口 /ml_models/xgb_knn_runner_tech.py
# 输入: 命令行参数（args），核心数据为 df_ml(MultiIndex: date, code, 含 ret_next) 与 df_price(MultiIndex: date, code)
# 输出: temp_scores/*.parquet（每日分数），xgb_results/*.csv（每日权重），logs/*.log（运行日志），factors_importance/*（因子重要性）
# 依赖: /ml_models/xgb_config.py 与 /ml_models/model_functions/_01~_16_*.py
from __future__ import annotations


import glob
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from ml_models import xgb_config as cfg
from ml_models.model_functions._01_cli import parse_args
from ml_models.model_functions._03_logging_utils import build_logger, log_section, log_data_grid
from ml_models.model_functions._04_feature_engineering import (
    apply_feature_filters,
    build_constraints_dict,
    build_drop_factors,
    build_keep_factors,
    build_monotone_constraints,
)
from ml_models.model_functions._06_data_preprocessing import prepare_dataset
from ml_models.model_functions._09_scoring import init_scoring_worker, process_score_chunk
from ml_models.model_functions._11_portfolio import generate_positions_with_buffer
from ml_models.model_functions._12_factor_importance import compute_factor_importance, save_factor_importance
from ml_models.model_functions._13_diagnosis import diagnose_factors
from ml_models.model_functions._14_overfit import run_overfit_check, run_overfit_check_on_panel
from ml_models.model_functions._15_visualization import build_ascii_report
from ml_models.model_functions._16_run_params import format_run_params
from ml_models.model_functions._17_quick_eval import run_quick_evaluation
from ml_models.model_functions._18_factors_quick_review import write_factors_quick_review


def _attach_derived_defaults(args) -> None:
    if not hasattr(args, "stock_pool_prefixes"):
        args.stock_pool_prefixes = cfg.DEFAULT_UNIVERSE["stock_pool_prefixes"]
    if not hasattr(args, "min_turnover"):
        args.min_turnover = cfg.DEFAULT_UNIVERSE["min_turnover"]
    if not hasattr(args, "label_benchmark_method"):
        args.label_benchmark_method = cfg.DEFAULT_LABEL["label_benchmark_method"]


def _auto_enable_overfit_and_quick_eval(args) -> None:
    if getattr(args, "start_date", None) is None or getattr(args, "end_date", None) is None:
        return
    if bool(getattr(args, "diagnose", False)):
        return
    if bool(getattr(args, "overfit_check_only", False)) or bool(getattr(args, "quick_eval_only", False)):
        return
    if bool(getattr(args, "quick_eval", False)):
        return
    args.quick_eval = True


def extension_hook(stage: str, *, args, df_ml: pd.DataFrame | None, df_price: pd.DataFrame | None, logger, **context) -> None:
    _ = stage, args, df_ml, df_price, logger, context
    return


def _parse_yyyymmdd(value: str | None, *, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s, format="%Y%m%d", errors="raise")
    except Exception as e:
        raise ValueError(f"{name} 需为 YYYYMMDD，当前={value!r}") from e


def _validate_time_range(args, *, logger) -> tuple[pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]:
    start_dt = _parse_yyyymmdd(getattr(args, "start_date", None), name="start_date")
    end_dt = _parse_yyyymmdd(getattr(args, "end_date", None), name="end_date")
    floor_dt = _parse_yyyymmdd(getattr(args, "train_floor_date", None), name="train_floor_date")
    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        raise ValueError(f"start_date({start_dt:%Y%m%d}) 不能晚于 end_date({end_dt:%Y%m%d})")
    if int(getattr(args, "train_window")) <= 0:
        raise ValueError("train_window 必须为正整数")
    if int(getattr(args, "n_workers")) <= 0:
        raise ValueError("n_workers 必须为正整数")
    if int(getattr(args, "top_k")) <= 0:
        raise ValueError("top_k 必须为正整数")
    if int(getattr(args, "buffer_k")) < int(getattr(args, "top_k")):
        logger.info("buffer_k < top_k, auto_set buffer_k=%d", int(getattr(args, "top_k")))
        args.buffer_k = int(getattr(args, "top_k"))
    train_gap = int(getattr(args, "train_gap", cfg.DEFAULT_TRAINING["train_gap"]))
    if train_gap <= 0:
        raise ValueError("train_gap 必须为正整数")
    label_horizon = int(cfg.DEFAULT_LABEL.get("predict_days", 5))
    safe_gap_min = int(label_horizon + 1)
    if train_gap < safe_gap_min:
        raise ValueError(f"train_gap({train_gap}) 过小：标签最远需要 t+{label_horizon}，为避免穿越需 train_gap>={safe_gap_min}")
    return start_dt, end_dt, floor_dt


def _prepare_output_dirs(args, logger, clean_save_dir: bool = True) -> tuple[str, str, str]:
    output_dir = str(getattr(args, "output_dir"))
    save_dir = os.path.join(output_dir, str(getattr(args, "sub_dir_name")))
    temp_dir = os.path.join(output_dir, str(getattr(args, "temp_dir_name")))
    log_dir = os.path.join(output_dir, "logs")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    elif clean_save_dir:
        for fp in glob.glob(os.path.join(save_dir, "*.csv")):
            try:
                os.remove(fp)
            except Exception:
                pass

    logger.info("output save_dir=%s", save_dir)
    logger.info("output temp_dir=%s", temp_dir)
    return save_dir, temp_dir, log_dir


def _compute_temp_eval(df_ml: pd.DataFrame, temp_dir: str, top_k: int) -> dict[str, pd.Series]:
    files = sorted(glob.glob(os.path.join(temp_dir, "*.parquet")))
    ic_rows: dict[pd.Timestamp, float] = {}
    top_rows: dict[pd.Timestamp, float] = {}
    for fp in files:
        date_str = os.path.basename(fp).replace(".parquet", "")
        d = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
        if pd.isna(d):
            continue
        df_s = pd.read_parquet(fp)
        if "code" not in df_s.columns or "score" not in df_s.columns:
            continue
        df_s["code"] = df_s["code"].astype(str)
        s = df_s.dropna(subset=["code", "score"]).drop_duplicates(subset=["code"]).set_index("code")["score"]
        if len(s) < max(20, int(top_k)):
            continue
        try:
            y_day = df_ml.xs(d, level="date")["ret_next"]
        except Exception:
            continue
        y = pd.to_numeric(y_day, errors="coerce").reindex(s.index)
        # 使用concat安全对齐索引并避免不可排序警告
        df = pd.concat([
            pd.to_numeric(s, errors="coerce").rename("pred")
        ], axis=1, sort=True)
        df = df.join(y.rename("y"), how="inner").replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < max(20, int(top_k)):
            continue
        ic = df["pred"].corr(df["y"], method="spearman")
        if ic is not None and np.isfinite(float(ic)):
            ic_rows[d] = float(ic)
        sel = df.nlargest(int(top_k), columns="pred")["y"]
        if len(sel) > 0:
            m = float(sel.mean())
            if np.isfinite(m):
                top_rows[d] = m
    return {
        "daily_ic": pd.Series(ic_rows).sort_index(),
        f"top{int(top_k)}_mean_ret_next": pd.Series(top_rows).sort_index(),
    }


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv=argv)
    _attach_derived_defaults(args)
    _auto_enable_overfit_and_quick_eval(args)

    log_dir = os.path.join(str(getattr(args, "output_dir")), "logs")
    logger = build_logger(log_dir=log_dir, run_name="xgb_knn_runner")
    log_section(logger, "配置参数 (Config)")
    
    # 对参数进行分组以提高可读性
    arg_dict = vars(args)
    groups = {
        "基础路径 (Paths)": set(cfg.DEFAULT_PATHS.keys()),
        "股票池/数据 (Universe)": set(cfg.DEFAULT_UNIVERSE.keys()),
        "标签定义 (Label)": set(cfg.DEFAULT_LABEL.keys()),
        "训练设置 (Training)": set(cfg.DEFAULT_TRAINING.keys()),
        "模型参数 (Model)": set(cfg.DEFAULT_MODEL.keys()),
        "组合风控 (Portfolio)": set(cfg.DEFAULT_PORTFOLIO.keys()),
        "择时策略 (Timing)": set(cfg.DEFAULT_TIMING.keys()),
        "诊断 (Diagnose)": set(cfg.DEFAULT_DIAGNOSE.keys()),
        "过拟合自检 (Overfit)": set(cfg.DEFAULT_OVERFIT.keys()) | {"overfit_check_only"},
        "快速评估 (QuickEval)": set(cfg.DEFAULT_QUICK_EVAL.keys()) | {"quick_eval_only"},
    }
    
    # 将参数分配到组
    grouped_args = {k: {} for k in groups}
    grouped_args["其他参数 (Misc)"] = {}
    
    assigned_keys = set()
    for g_name, g_keys in groups.items():
        for k in g_keys:
            if k in arg_dict:
                grouped_args[g_name][k] = arg_dict[k]
                assigned_keys.add(k)
                
    for k, v in arg_dict.items():
        if k not in assigned_keys:
            grouped_args["其他参数 (Misc)"][k] = v

    # 打印相关组
    log_data_grid(logger, grouped_args["基础路径 (Paths)"], "基础路径")
    log_data_grid(logger, grouped_args["模型参数 (Model)"], "模型参数")
    log_data_grid(logger, grouped_args["训练设置 (Training)"], "训练设置")
    log_data_grid(logger, grouped_args["组合风控 (Portfolio)"], "组合风控")
    
    if bool(getattr(args, "diagnose", False)):
        log_data_grid(logger, grouped_args["诊断 (Diagnose)"], "诊断参数")
    if bool(getattr(args, "overfit_check", False)) or bool(getattr(args, "overfit_along", False)):
        log_data_grid(logger, grouped_args["过拟合自检 (Overfit)"], "过拟合参数")
    if bool(getattr(args, "quick_eval", False)) or bool(getattr(args, "quick_eval_only", False)):
        log_data_grid(logger, grouped_args["快速评估 (QuickEval)"], "快速评估参数")
        
    # Show Timing if enabled
    if str(getattr(args, "timing_method", "none")) != "none":
         log_data_grid(logger, grouped_args["择时策略 (Timing)"], "择时参数")

    start_dt, end_dt, floor_dt = _validate_time_range(args, logger=logger)

    if bool(getattr(args, "diagnose", False)):
        log_section(logger, "因子诊断 (Diagnose)")
        out_path = diagnose_factors(args)
        logger.info("诊断报告路径=%s", out_path)
        return

    if bool(getattr(args, "quick_eval_only", False)):
        log_section(logger, "快速评估 (QuickEvalOnly)")
        save_dir, _, _ = _prepare_output_dirs(args, logger=logger, clean_save_dir=False)

        if bool(getattr(args, "overfit_check", False)) or bool(getattr(args, "overfit_check_only", False)):
            log_section(logger, "过拟合自检 (OverfitCheckOnly)")
            run_overfit_check(args, logger=logger)

        price_path = str(getattr(args, "price_data_path"))
        logger.info("步骤: 加载价格数据 path=%s", price_path)
        if not os.path.exists(price_path):
            logger.error("错误: 找不到价格数据文件 path=%s", price_path)
            return

        df_price = pd.read_parquet(price_path).sort_index()
        prefixes = tuple(getattr(args, "stock_pool_prefixes", cfg.DEFAULT_UNIVERSE["stock_pool_prefixes"]))
        pool_mask = df_price.index.get_level_values("code").astype(str).str.startswith(prefixes)
        df_price = df_price.loc[pool_mask, :]

        try:
            run_quick_evaluation(args=args, save_dir=save_dir, df_price=df_price, logger=logger)
        except Exception as e:
            logger.info("快速评估出错: %s", str(e))
            import traceback
            logger.info(traceback.format_exc())
        return

    if bool(getattr(args, "overfit_check_only", False)):
        log_section(logger, "过拟合自检 (OverfitCheckOnly)")
        run_overfit_check(args, logger=logger)

        if bool(getattr(args, "quick_eval", False)):
            log_section(logger, "快速评估 (QuickEval)")
            save_dir, _, _ = _prepare_output_dirs(args, logger=logger, clean_save_dir=False)

            price_path = str(getattr(args, "price_data_path"))
            logger.info("步骤: 加载价格数据 path=%s", price_path)
            if not os.path.exists(price_path):
                logger.error("错误: 找不到价格数据文件 path=%s", price_path)
                return
            df_price = pd.read_parquet(price_path).sort_index()
            prefixes = tuple(getattr(args, "stock_pool_prefixes", cfg.DEFAULT_UNIVERSE["stock_pool_prefixes"]))
            pool_mask = df_price.index.get_level_values("code").astype(str).str.startswith(prefixes)
            df_price = df_price.loc[pool_mask, :]

            try:
                run_quick_evaluation(args=args, save_dir=save_dir, df_price=df_price, logger=logger)
            except Exception as e:
                logger.info("快速评估出错: %s", str(e))
                import traceback

                logger.info(traceback.format_exc())
        return

    run_overfit_along = bool(getattr(args, "overfit_check", False)) and bool(getattr(args, "overfit_along", False))
    if bool(getattr(args, "overfit_check", False)) and not run_overfit_along:
        log_section(logger, "过拟合自检 (OverfitCheckOnly)")
        run_overfit_check(args, logger=logger)
        return

    objective = str(getattr(args, "xgb_objective", "reg:squarederror"))
    model_kind = "Rank" if objective.startswith("rank:") else "Regression"
    timing_method = str(getattr(args, "timing_method", "score")).lower()
    s1, s2 = format_run_params(args, objective=objective, model_kind=model_kind, timing_method=timing_method)
    log_section(logger, "运行参数 (RunParams)")
    logger.info("%s", s1)
    logger.info("%s", s2)

    save_dir, temp_dir, _ = _prepare_output_dirs(args, logger=logger)

    log_section(logger, "数据集处理 (Dataset)")
    df_ml, df_price = prepare_dataset(args, logger=logger)
    extension_hook("after_dataset", args=args, df_ml=df_ml, df_price=df_price, logger=logger)

    all_dates = df_ml.index.get_level_values("date").unique().sort_values()
    raw_features = [c for c in df_ml.columns if c != "ret_next"]
    drop_factors = build_drop_factors(
        use_default_drop_factors=bool(getattr(args, "use_default_drop_factors", True)),
        drop_factors_csv=getattr(args, "drop_factors", None),
    )
    keep_factors = build_keep_factors(
        use_default_keep_factors=bool(getattr(args, "use_default_keep_factors", True)),
        keep_factors_csv=getattr(args, "keep_factors", None),
    )
    use_keep_filtering = bool(getattr(args, "use_default_keep_factors", True)) or bool(getattr(args, "keep_factors", None))
    if use_keep_filtering:
        keep_factors.discard("f_smart_money_div")
        keep_factors.add("f_vp_rank_corr_20")
    final_features = apply_feature_filters(raw_features, drop_factors, keep_factors)
    log_section(logger, "特征工程 (Features)")
    logger.info("最终特征数=%d 示例=%s", int(len(final_features)), final_features[:8])
    constraints_dict = build_constraints_dict(
        use_constraints=bool(getattr(args, "use_constraints", True)),
        constraints_csv=getattr(args, "constraints", None),
    )
    monotone_constraints = build_monotone_constraints(final_features, constraints_dict)

    train_window = int(getattr(args, "train_window"))
    train_gap = int(getattr(args, "train_gap", cfg.DEFAULT_TRAINING["train_gap"]))
    predict_dates = all_dates[train_window + train_gap - 1 :]
    if start_dt is not None:
        predict_dates = predict_dates[predict_dates >= start_dt]
    if end_dt is not None:
        predict_dates = predict_dates[predict_dates <= end_dt]
    if getattr(args, "max_predict_days", None) is not None:
        predict_dates = predict_dates[: int(getattr(args, "max_predict_days"))]
    if len(predict_dates) == 0:
        logger.info("预测日期为空，跳过运行")
        return
    logger.info("预测区间 %s~%s 总天数=%d", predict_dates[0].strftime("%Y%m%d"), predict_dates[-1].strftime("%Y%m%d"), int(len(predict_dates)))

    rebalance_period = int(getattr(args, "rebalance_period", cfg.DEFAULT_PORTFOLIO["rebalance_period"]))
    train_stride = max(1, int(rebalance_period))
    logger.info("打分优化: 每%d天训练一次，并复用模型预测中间交易日", int(train_stride))

    tasks: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, list[pd.Timestamp]]] = []
    all_dates_index = pd.Index(all_dates)
    n_skip = 0
    predict_list = list(predict_dates)
    for i in range(0, len(predict_list), train_stride):
        chunk = predict_list[i : i + train_stride]
        if not chunk:
            continue
        anchor_date = chunk[0]
        pos = int(all_dates_index.get_loc(anchor_date))
        if pos < train_window + train_gap - 1:
            n_skip += 1
            continue
        train_end_date = all_dates[pos - train_gap]
        train_start_date = all_dates[pos - train_gap - train_window + 1]
        if floor_dt is not None and train_start_date < floor_dt:
            train_start_date = floor_dt
        if train_start_date > train_end_date:
            n_skip += 1
            continue
        if not (train_end_date < anchor_date):
            n_skip += 1
            continue
        tasks.append((anchor_date, train_start_date, train_end_date, chunk))
    if n_skip:
        logger.info("跳过训练任务数=%d (数据不足或日期无效)", int(n_skip))
    if len(tasks) == 0:
        logger.info("训练任务列表为空，跳过运行")
        return

    log_section(logger, "模型训练与打分 (Scoring)")
    extension_hook("before_scoring", args=args, df_ml=df_ml, df_price=df_price, logger=logger, tasks=tasks, features=final_features)
    n_ok = 0
    n_none = 0
    n_err = 0
    err_samples: list[str] = []
    with ProcessPoolExecutor(
        max_workers=int(getattr(args, "n_workers")),
        initializer=init_scoring_worker,
        initargs=(df_ml, df_price, args, final_features, monotone_constraints, temp_dir),
    ) as executor:
        futures = [
            executor.submit(
                process_score_chunk,
                t[0],
                t[1],
                t[2],
                t[3],
            )
            for t in tasks
        ]
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="训练进度"):
            try:
                r = fut.result()
            except Exception as e:
                n_err += 1
                if len(err_samples) < 5:
                    err_samples.append(f"Error: {e}")
                continue
            if r is None:
                n_none += 1
                continue
            if isinstance(r, str) and r.startswith("Error:"):
                n_err += 1
                if len(err_samples) < 5:
                    err_samples.append(r)
                continue
            if isinstance(r, (list, tuple)):
                n_ok += int(len(r))
            else:
                n_ok += 1
    logger.info("打分汇总 成功天数=%d 空结果=%d 错误=%d", int(n_ok), int(n_none), int(n_err))
    for s in err_samples:
        logger.info("打分错误示例 %s", s)
    extension_hook("after_scoring", args=args, df_ml=df_ml, df_price=df_price, logger=logger, temp_dir=temp_dir)

    log_section(logger, "模型评估 (Evaluation)")
    eval_map = _compute_temp_eval(df_ml, temp_dir=temp_dir, top_k=int(getattr(args, "top_k")))
    report = build_ascii_report("临时分数评估 (基于同日ret_next)", eval_map)
    logger.info("\n%s", report.rstrip())
    try:
        out_eval = os.path.join(save_dir, "eval_report.txt")
        with open(out_eval, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info("已保存评估报告=%s", out_eval)
    except Exception:
        pass

    log_section(logger, "因子重要性 (FactorImportance)")
    importance_dates = pd.Index([t[0] for t in tasks])
    res = compute_factor_importance(df_ml, all_dates, importance_dates, final_features, args)
    df_imp: pd.DataFrame | None = None
    meta: dict | None = None
    if res is not None:
        df_imp, meta = res
        meta["dropped_factors"] = sorted(list(drop_factors))
        save_factor_importance(df_imp, meta, args, logger=logger)
    else:
        logger.info("跳过因子重要性计算")

    try:
        out_txt = write_factors_quick_review(
            df_imp,
            df_ml,
            meta=meta,
            temp_dir=temp_dir,
            top_k=int(getattr(args, "top_k", 30)),
        )
        logger.info("已刷新 quick review=%s", out_txt)
    except Exception as e:
        logger.info("quick review 生成失败: %s", e)

    log_section(logger, "组合构建 (Portfolio)")
    generate_positions_with_buffer(args, temp_dir=temp_dir, save_dir=save_dir, df_price=df_price, logger=logger)

    if run_overfit_along:
        log_section(logger, "伴随过拟合自检 (OverfitAlong)")
        if (not bool(getattr(args, "overfit_range", False))) and getattr(args, "start_date", None) and getattr(args, "end_date", None):
            args.overfit_range = True
        prior_target = getattr(args, "overfit_target_date", None)
        try:
            if (not bool(getattr(args, "overfit_range", False))) and (not prior_target):
                args.overfit_target_date = predict_dates[-1].strftime("%Y%m%d")
            run_overfit_check_on_panel(args, df_ml=df_ml, logger=logger)
        finally:
            args.overfit_target_date = prior_target

    if bool(getattr(args, "quick_eval", False)):
        log_section(logger, "快速评估 (QuickEval)")
        try:
            run_quick_evaluation(args=args, save_dir=save_dir, df_price=df_price, logger=logger)
        except Exception as e:
            logger.info("快速评估出错 %s", str(e))
    log_section(logger, "运行结束 (Done)")


if __name__ == "__main__":
    run()
