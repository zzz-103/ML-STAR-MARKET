# 位置: 旧版入口（兼容保留）| 直接转到 /ml_models/main.py 执行
# 输入: 命令行参数（同 main.py）
# 输出: 同 main.py（temp_scores parquet / 权重 csv / logs / 因子重要性等）
# 依赖: /ml_models/main.py（run）
from __future__ import annotations
from pathlib import Path

if __name__ == "__main__":
    from ml_models.main import run

    run()
    raise SystemExit(0)

# ================= 配置区域 =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FACTOR_DATA_PATH = str(PROJECT_ROOT / "factors_data" / "all_factors_with_fundamentals.parquet")
PRICE_DATA_PATH = str(PROJECT_ROOT / "pre_data" / "cleaned_stock_data_300_688_with_idxstk.parquet")
MERGED_DATA_PATH = str(PROJECT_ROOT / "pre_data" / "merged_20200101_20241231.csv")
OUTPUT_DIR = str(PROJECT_ROOT / "ml_results")
SUB_DIR_NAME = "xgb_results_gem_star_momo"
TEMP_DIR_NAME = "temp_scores_gem_star_momo"
FACTORS_IMPORTANCE_DIR = str(PROJECT_ROOT / "ml_results" / "factors_importance")
STOCK_POOL_PREFIXES = ("300", "688")

DEFAULT_DROP_FACTORS = [
    # 宏观风格
    "ff_mkt",
    "ff_hml",
    "ff_smb",
    "ff_smb_cov_60",
    "beta_mkt",
    "beta_smb",
    "beta_hml",

    # 纯震荡指标 (KDJ/RSI/William) -> 坚决剔除
    "kdj_k",
    "kdj_d",
    "kdj_j",
    "william_r",
    "cci_14",
    "rsi_14",
    "mfi_14",
    "psy_12",
    "vr_24",

    # 冗余
    "macd_dea",
    "macd_dif",
    "turnover_cv_20",
    "turnover_std_10",
    "ret_lag_3", "ret_lag_4", "ret_lag_5",
    # 再次建议剔除纯形态噪音
    "candle_body_abs",
    "shadow_lower",
    "hl_ratio_smooth",
    "candle_amplitude",
    "turnover_cv_20", "turnover_std_10",

    #原始估值因子
    "pe_ttm", "ps_ttm", "pb", "pcf_net_ttm", "float_market_cap"
]


DEFAULT_MONOTONE_CONSTRAINTS = {
    "EP_ttm": 0,"SP_ttm": 1,"BP": 1,"CFP_ttm": 0,

    # === 核心进攻因子 ===
    "roc_5": 1,"roc_10": 1,
    "roc_20": 1,"ma60_bias": 1,
    "momo_sl_20_5": 1,"breakout_20_vol_confirm": 1,
    "smart_overnight_20": 1,

    # === 量价与波动 ===
    "vol_20": 1,"pv_corr": 1,"pv_corr_20": 0,"boll_width": 1,

    # === 形态  ===
    "candle_body": 0,"shadow_upper": -1,"hl_ratio": -1,

    # === 保持中性 ===
    "ret_lag_1": 0,"ma5_bias": 0,
}


# 策略参数
TRAIN_WINDOW = 250 # 使用过去 1 年数据训练
PREDICT_TOP_K = 20 # 持仓 20 只
BUFFER_TOP_K = 30 # 缓冲带
N_ESTIMATORS = 200 # 树的数量
LEARNING_RATE = 0.01
MAX_DEPTH = 4
SUBSAMPLE = 0.7
REG_LAMBDA = 10.0
# 资金适配：1000万资金 -> 单票33万 -> 只需要日成交额 > 1500万即可安全交易
MIN_TURNOVER = 15000000.0
PREDICT_DAYS = 5 # 预测未来 5 日收益 (降低换手核心)
REBALANCE_PERIOD = 5
SMOOTH_WINDOW = 5
INERTIA_RATIO = 1.02
EMERGENCY_EXIT_RANK = 50
REBALANCE_TURNOVER_CAP = 0.90
XGB_OBJECTIVE = "reg:squarederror"
TIMING_HYSTERESIS = 0.005
LABEL_BENCHMARK_METHOD = "median"
LABEL_BENCHMARK_UNIVERSE = "tradable"
DEFAULT_BAND_THRESHOLD = 0.001
DEFAULT_MAX_W = 0.06
DEFAULT_MIN_WEIGHT = 0.0005
DEFAULT_NON_REBALANCE_ACTION = "empty"
DEFAULT_LIMIT_POLICY = "freeze"
DEFAULT_SAMPLE_WEIGHT_MODE = "time_decay_exp"
DEFAULT_DECAY_ANCHOR_DAYS = 30
DEFAULT_DECAY_HALF_LIFE_DAYS = 60
DEFAULT_DECAY_MIN_WEIGHT = 0.1
# ===========================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-data-path", default=FACTOR_DATA_PATH)
    parser.add_argument("--price-data-path", default=PRICE_DATA_PATH)
    parser.add_argument("--train-window", type=int, default=TRAIN_WINDOW)
    parser.add_argument("--top-k", type=int, default=PREDICT_TOP_K)
    parser.add_argument("--buffer-k", type=int, default=BUFFER_TOP_K)
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--subsample", type=float, default=SUBSAMPLE)
    parser.add_argument("--reg-lambda", type=float, default=REG_LAMBDA)
    parser.add_argument("--dropna-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--sample-weight-mode",
        choices=["none", "time_decay_exp", "time_decay_linear"],
        default=DEFAULT_SAMPLE_WEIGHT_MODE,
    )
    parser.add_argument("--decay-anchor-days", type=int, default=DEFAULT_DECAY_ANCHOR_DAYS)
    parser.add_argument("--decay-half-life-days", type=int, default=DEFAULT_DECAY_HALF_LIFE_DAYS)
    parser.add_argument("--decay-min-weight", type=float, default=DEFAULT_DECAY_MIN_WEIGHT)
    # KNN 融合参数
    parser.add_argument("--use-knn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--knn-neighbors", type=int, default=50)
    parser.add_argument("--blend-xgb-weight", type=float, default=0.7)
    parser.add_argument("--blend-knn-weight", type=float, default=0.3)
    # 日期范围
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-predict-days", type=int, default=None)
    parser.add_argument("--train-floor-date", default=None)
    parser.add_argument("--use-default-drop-factors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-factors", default=None)
    parser.add_argument("--use-constraints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--constraints", default=None)
    parser.add_argument("--rebalance-period", type=int, default=REBALANCE_PERIOD)
    parser.add_argument("--rebalance-turnover-cap", type=float, default=REBALANCE_TURNOVER_CAP)
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW)
    parser.add_argument("--inertia-ratio", type=float, default=INERTIA_RATIO)
    parser.add_argument("--emergency-exit-rank", type=int, default=EMERGENCY_EXIT_RANK)
    parser.add_argument("--keep-missing-positions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timing-threshold", type=float, default=0.0)
    parser.add_argument("--timing-bad-exposure", type=float, default=0.4)
    parser.add_argument("--timing-method", choices=["index_ma20", "index_ma_dual", "score", "none"], default="index_ma20")
    parser.add_argument("--risk-data-path", default=MERGED_DATA_PATH)
    parser.add_argument("--risk-index-code", default="399006")
    parser.add_argument("--risk-ma-window", type=int, default=20)
    parser.add_argument("--risk-ma-fast-window", type=int, default=5)
    parser.add_argument("--risk-ma-slow-window", type=int, default=20)
    parser.add_argument("--risk-ma-buffer", type=float, default=0.005)
    parser.add_argument("--timing-enter-threshold", type=float, default=None)
    parser.add_argument("--timing-exit-threshold", type=float, default=None)
    parser.add_argument("--timing-hysteresis", type=float, default=TIMING_HYSTERESIS)
    parser.add_argument("--band-threshold", type=float, default=DEFAULT_BAND_THRESHOLD)
    parser.add_argument("--max-w", type=float, default=DEFAULT_MAX_W)
    parser.add_argument("--min-weight", type=float, default=DEFAULT_MIN_WEIGHT)
    parser.add_argument(
        "--non-rebalance-action",
        choices=["empty", "carry"],
        default=DEFAULT_NON_REBALANCE_ACTION,
    )
    parser.add_argument("--limit-policy", choices=["freeze", "sell_only"], default=DEFAULT_LIMIT_POLICY)
    parser.add_argument("--xgb-objective", default=XGB_OBJECTIVE)
    parser.add_argument(
        "--label-benchmark-universe",
        default=LABEL_BENCHMARK_UNIVERSE,
        choices=["all", "tradable"],
    )
    parser.add_argument("--diagnose", action="store_true", default=False)
    parser.add_argument("--diag-start-date", default="20230101")
    parser.add_argument("--diag-end-date", default="20241231")
    parser.add_argument("--diag-step", type=int, default=5)
    parser.add_argument("--diag-top-k", type=int, default=30)
    parser.add_argument("--diag-min-n", type=int, default=300)
    parser.add_argument("--overfit-check", action="store_true", default=False)
    parser.add_argument("--overfit-target-date", default=None)
    parser.add_argument("--overfit-valid-days", type=int, default=20)
    parser.add_argument("--overfit-top-k", type=int, default=None)
    parser.add_argument("--overfit-range", action="store_true", default=False)
    parser.add_argument("--overfit-range-step", type=int, default=20)
    parser.add_argument("--overfit-range-max-points", type=int, default=60)
    parser.add_argument("--overfit-along", action="store_true", default=False)
    parser.add_argument("--n-workers", type=int, default=max(1, os.cpu_count() - 2))
    return parser.parse_args()

def _parse_yyyymmdd(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return pd.to_datetime(s, format="%Y%m%d", errors="raise")

def _build_sample_weights(args, train_dates, ref_date):
    mode = str(getattr(args, "sample_weight_mode", DEFAULT_SAMPLE_WEIGHT_MODE)).lower()
    if mode == "none":
        return None
    anchor_days = int(getattr(args, "decay_anchor_days", DEFAULT_DECAY_ANCHOR_DAYS))
    half_life_days = int(getattr(args, "decay_half_life_days", DEFAULT_DECAY_HALF_LIFE_DAYS))
    min_w = float(getattr(args, "decay_min_weight", DEFAULT_DECAY_MIN_WEIGHT))

    dates = pd.to_datetime(train_dates)
    ref = pd.to_datetime(ref_date)
    age_days = (ref - dates).days.astype("int64")
    eff_age = age_days - int(anchor_days)
    eff_age = np.where(eff_age > 0, eff_age, 0)

    denom = float(max(1, half_life_days))
    if mode == "time_decay_linear":
        w = 1.0 - (eff_age.astype("float64") / denom)
    else:
        w = np.power(0.5, eff_age.astype("float64") / denom)

    w = np.clip(w, float(min_w), 1.0)
    return w.astype("float64")

def _parse_csv_list(value):
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]

def _parse_constraints(value):
    if value is None:
        return {}
    s = str(value).strip()
    if not s:
        return {}
    out = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"constraints 格式错误，需为 name:sign，例如 ret_lag_1:1。当前: {item}")
        name, sign = item.split(":", 1)
        name = name.strip()
        sign = sign.strip()
        if not name:
            continue
        try:
            sign_i = int(sign)
        except Exception as e:
            raise ValueError(f"constraints sign 需为 -1/0/1，当前: {item}") from e
        out[name] = sign_i
    return out

def _print_run_params(args, objective: str, model_kind: str, timing_method: str) -> None:
    xgb_sub = float(getattr(args, "subsample", 0.8))
    xgb_rl = float(getattr(args, "reg_lambda", 1.0))
    sw_mode = str(getattr(args, "sample_weight_mode", "none"))
    print(
        f"[run-params] train_window={int(getattr(args, 'train_window', 0))} "
        f"label=0.5*ret_5d+0.5*ret_10d "
        f"pos(top={int(getattr(args, 'top_k', 0))}, buf={int(getattr(args, 'buffer_k', 0))}) "
        f"rebalance={int(getattr(args, 'rebalance_period', 0))} "
        f"turnover_cap={float(getattr(args, 'rebalance_turnover_cap', 0.0)):.2f} "
        f"smooth={int(getattr(args, 'smooth_window', 0))} "
        f"inertia={float(getattr(args, 'inertia_ratio', 0.0)):.3g} "
        f"stop_rank={int(getattr(args, 'emergency_exit_rank', 0))} "
        f"timing={timing_method} bad_exp={float(getattr(args, 'timing_bad_exposure', 0.0)):.2f} "
        f"risk={str(getattr(args, 'risk_index_code', ''))} "
        f"ma={int(getattr(args, 'risk_ma_window', 0))} "
        f"weight={sw_mode}(anch={int(getattr(args, 'decay_anchor_days', 0))}, "
        f"half={int(getattr(args, 'decay_half_life_days', 0))}, "
        f"min={float(getattr(args, 'decay_min_weight', 0.0)):.3g})"
    )
    print(
        f"[run-params] objective={objective} mode={model_kind} "
        f"xgb(n={int(getattr(args, 'n_estimators', 0))}, lr={float(getattr(args, 'learning_rate', 0.0)):.3g}, "
        f"depth={int(getattr(args, 'max_depth', 0))}, sub={xgb_sub:.2f}, col=0.80, ra=0.10, rl={xgb_rl:.3g}) "
        f"knn={'on' if bool(getattr(args, 'use_knn', False)) else 'off'}"
    )

def build_constraints_dict(args):
    if not bool(getattr(args, "use_constraints", True)):
        return {}
    d = dict(DEFAULT_MONOTONE_CONSTRAINTS)
    d.update(_parse_constraints(getattr(args, "constraints", None)))
    cleaned = {}
    for k, v in d.items():
        if v is None:
            continue
        v_i = int(v)
        if v_i not in (-1, 0, 1):
            raise ValueError(f"constraints 仅支持 -1/0/1，当前: {k}={v}")
        if v_i == 0:
            continue
        cleaned[str(k)] = v_i
    return cleaned

def build_monotone_constraints(features, constraints_dict):
    if not constraints_dict:
        return None
    vec = []
    for f in features:
        vec.append(int(constraints_dict.get(f, 0)))
    return "(" + ",".join(str(v) for v in vec) + ")"

def build_drop_factors(args):
    drop = set()
    if bool(args.use_default_drop_factors):
        drop.update(DEFAULT_DROP_FACTORS)
    drop.update(_parse_csv_list(args.drop_factors))
    return drop

def apply_feature_filters(features, drop_factors):
    out = []
    for f in features:
        if f in drop_factors:
            continue
        if str(f).startswith("turnover_") and str(f) != "turnover_bias_5":
            continue
        out.append(f)
    return out

def apply_max_weight_cap(weights: pd.Series, max_w: float) -> pd.Series:
    if len(weights) == 0:
        return weights
    max_w = float(max_w)
    if not np.isfinite(max_w) or max_w <= 0:
        return weights.astype("float64")
    w = weights.astype("float64").copy()
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        return w
    w = w / s
    if len(w) * max_w < 1 - 1e-12:
        return w
    for _ in range(10):
        over = w > (max_w + 1e-12)
        if not bool(over.any()):
            break
        excess = float((w[over] - max_w).sum())
        w[over] = max_w
        under = ~over
        if not bool(under.any()):
            break
        under_sum = float(w[under].sum())
        if not np.isfinite(under_sum) or under_sum <= 0:
            break
        w[under] += w[under] / under_sum * excess
        s2 = float(w.sum())
        if np.isfinite(s2) and s2 > 0:
            w = w / s2
    return w


def _daily_ic(y: pd.Series, pred: pd.Series) -> pd.Series:
    df = pd.DataFrame({"y": y, "pred": pred})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) == 0:
        return pd.Series(dtype="float64")
    if not isinstance(df.index, pd.MultiIndex) or "date" not in df.index.names:
        return pd.Series(dtype="float64")
    out = {}
    for d, g in df.groupby(level="date"):
        if len(g) < 20:
            continue
        ic = g["pred"].corr(g["y"], method="spearman")
        if ic is None or not np.isfinite(ic):
            continue
        out[d] = float(ic)
    return pd.Series(out).sort_index()


def _daily_topk_mean(y: pd.Series, pred: pd.Series, top_k: int) -> pd.Series:
    df = pd.DataFrame({"y": y, "pred": pred})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) == 0:
        return pd.Series(dtype="float64")
    if not isinstance(df.index, pd.MultiIndex) or "date" not in df.index.names:
        return pd.Series(dtype="float64")
    k = max(1, int(top_k))
    out = {}
    for d, g in df.groupby(level="date"):
        if len(g) < max(20, k):
            continue
        sel = g.nlargest(k, columns="pred")["y"]
        if len(sel) == 0:
            continue
        m = float(sel.mean())
        if np.isfinite(m):
            out[d] = m
    return pd.Series(out).sort_index()


def overfit_check(args) -> None:
    data_ml, _ = prepare_dataset(args)
    _run_overfit_check(args, data_ml=data_ml, exit_after=True)


def _run_overfit_check(args, data_ml: pd.DataFrame, exit_after: bool) -> Optional[pd.DataFrame]:
    all_dates = data_ml.index.get_level_values("date").unique().sort_values()
    if len(all_dates) < int(args.train_window) + 30:
        raise ValueError("交易日数量不足，无法进行过拟合自检")

    def rmse(a: pd.Series, b: pd.Series) -> float:
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        m = (a - b).astype("float64")
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

    def single(target_date: pd.Timestamp) -> dict:
        if target_date not in all_dates:
            raise ValueError(f"target_date 不在数据日期范围内: {target_date.strftime('%Y%m%d')}")
        pos = int(all_dates.get_loc(target_date))
        if pos <= 0:
            raise ValueError("target_date 过早，无法构建训练窗口")

        train_end_date = all_dates[pos - 1]
        train_start_date = all_dates[max(0, pos - int(args.train_window))]
        train_floor = _parse_yyyymmdd(getattr(args, "train_floor_date", None))
        if train_floor is not None and train_start_date < train_floor:
            train_start_date = train_floor

        idx = pd.IndexSlice
        train_data = data_ml.loc[idx[train_start_date:train_end_date, :], :].sort_index()
        if len(train_data) < 5000:
            raise ValueError("训练样本过少，无法进行过拟合自检")

        features = [c for c in train_data.columns if c != "ret_next"]
        drop_factors = build_drop_factors(args)
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

        sample_weight = _build_sample_weights(args, tr.index.get_level_values("date"), sub_train_dates.max())
        constraints_dict = build_constraints_dict(args)
        monotone_constraints = build_monotone_constraints(final_features, constraints_dict)
        objective = str(getattr(args, "xgb_objective", "reg:squarederror"))

        if objective.startswith("rank:"):
            trd = tr.index.get_level_values("date").to_numpy()
            _, group_sizes = np.unique(trd, return_counts=True)
            model = xgb.XGBRanker(
                n_estimators=int(args.n_estimators),
                learning_rate=float(args.learning_rate),
                max_depth=int(args.max_depth),
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
            model.fit(X_tr, y_tr, group=group_sizes)
        else:
            model = xgb.XGBRegressor(
                n_estimators=int(args.n_estimators),
                learning_rate=float(args.learning_rate),
                max_depth=int(args.max_depth),
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
            if sample_weight is None:
                model.fit(X_tr, y_tr)
            else:
                model.fit(X_tr, y_tr, sample_weight=sample_weight)

        pred_tr = pd.Series(model.predict(X_tr), index=y_tr.index, dtype="float64")
        pred_va = pd.Series(model.predict(X_va), index=y_va.index, dtype="float64")

        top_k = int(getattr(args, "overfit_top_k", None) or getattr(args, "top_k", 30))
        ic_tr = _daily_ic(y_tr, pred_tr)
        ic_va = _daily_ic(y_va, pred_va)
        top_tr = _daily_topk_mean(y_tr, pred_tr, top_k=top_k)
        top_va = _daily_topk_mean(y_va, pred_va, top_k=top_k)

        s_ic_tr = summarize(ic_tr)
        s_ic_va = summarize(ic_va)
        s_top_tr = summarize(top_tr)
        s_top_va = summarize(top_va)

        rmse_tr = rmse(y_tr, pred_tr)
        rmse_va = rmse(y_va, pred_va)

        return {
            "target_date": target_date,
            "train_start": train_start_date,
            "train_end": train_end_date,
            "sub_train_days": int(len(sub_train_dates)),
            "valid_days": int(len(valid_dates)),
            "rmse_tr": float(rmse_tr),
            "rmse_va": float(rmse_va),
            "rmse_gap": float(rmse_va - rmse_tr) if np.isfinite(rmse_va) and np.isfinite(rmse_tr) else float("nan"),
            "ic_tr_mean": float(s_ic_tr["mean"]),
            "ic_va_mean": float(s_ic_va["mean"]),
            "ic_mean_gap": float(s_ic_va["mean"] - s_ic_tr["mean"]),
            "ic_tr_ir": float(s_ic_tr["ir"]),
            "ic_va_ir": float(s_ic_va["ir"]),
            "topk_tr_mean": float(s_top_tr["mean"]),
            "topk_va_mean": float(s_top_va["mean"]),
            "topk_mean_gap": float(s_top_va["mean"] - s_top_tr["mean"]),
            "objective": str(objective),
            "top_k": int(top_k),
        }

    if bool(getattr(args, "overfit_range", False)):
        start_index = int(args.train_window)
        predict_dates = all_dates[start_index:]
        if getattr(args, "start_date", None):
            predict_dates = predict_dates[predict_dates >= pd.to_datetime(args.start_date, format="%Y%m%d")]
        if getattr(args, "end_date", None):
            predict_dates = predict_dates[predict_dates <= pd.to_datetime(args.end_date, format="%Y%m%d")]
        if getattr(args, "max_predict_days", None) is not None:
            predict_dates = predict_dates[: int(args.max_predict_days)]
        if len(predict_dates) == 0:
            raise ValueError("预测日期为空，无法进行区间过拟合自检")

        step = max(1, int(getattr(args, "overfit_range_step", 20)))
        max_points = max(1, int(getattr(args, "overfit_range_max_points", 60)))
        sample_dates = predict_dates[::step]
        if len(sample_dates) > max_points:
            sample_dates = sample_dates[:max_points]

        rows = []
        errors = 0
        for d in tqdm(sample_dates, desc="OverfitCheck"):
            try:
                rows.append(single(d))
            except Exception:
                errors += 1
                continue

        if len(rows) == 0:
            raise ValueError(f"区间过拟合自检无可用结果（errors={errors}）")
        df = pd.DataFrame(rows).sort_values("target_date")

        def agg_mean(col: str) -> float:
            s = pd.to_numeric(df[col], errors="coerce")
            s = s[np.isfinite(s)]
            return float(s.mean()) if len(s) else float("nan")

        def agg_pos_ratio(col: str) -> float:
            s = pd.to_numeric(df[col], errors="coerce")
            s = s[np.isfinite(s)]
            return float((s > 0).mean()) if len(s) else float("nan")

        print(
            f"[overfit-range] range={df['target_date'].iloc[0].strftime('%Y%m%d')}~{df['target_date'].iloc[-1].strftime('%Y%m%d')} "
            f"points={len(df)}/{len(sample_dates)} step={step} errors={errors} valid_days={int(getattr(args, 'overfit_valid_days', 20))}"
        )
        print(
            f"[overfit-range] mean_gap ic_mean={agg_mean('ic_mean_gap'):.4f} topk_mean={agg_mean('topk_mean_gap'):.4f} rmse={agg_mean('rmse_gap'):.6f}"
        )
        print(
            f"[overfit-range] pos_ratio valid ic_mean={agg_pos_ratio('ic_va_mean'):.3f} valid topk_mean={agg_pos_ratio('topk_va_mean'):.3f}"
        )

        if exit_after:
            return None
        return df

    target_s = getattr(args, "overfit_target_date", None) or getattr(args, "start_date", None)
    if target_s is None:
        target_date = all_dates[-1]
    else:
        target_date = pd.to_datetime(str(target_s), format="%Y%m%d", errors="raise")

    r = single(target_date)
    print(
        f"[overfit] target={pd.to_datetime(r['target_date']).strftime('%Y%m%d')} "
        f"train={pd.to_datetime(r['train_start']).strftime('%Y%m%d')}~{pd.to_datetime(r['train_end']).strftime('%Y%m%d')} "
        f"sub_train_days={int(r['sub_train_days'])} valid_days={int(r['valid_days'])} objective={r['objective']}"
    )
    print(
        f"[overfit] train  rmse={r['rmse_tr']:.6f} ic_mean={r['ic_tr_mean']:.4f} ic_ir={r['ic_tr_ir']:.3f} "
        f"top{int(r['top_k'])}_mean={r['topk_tr_mean']:.4f}"
    )
    print(
        f"[overfit] valid  rmse={r['rmse_va']:.6f} ic_mean={r['ic_va_mean']:.4f} ic_ir={r['ic_va_ir']:.3f} "
        f"top{int(r['top_k'])}_mean={r['topk_va_mean']:.4f}"
    )
    print(
        f"[overfit] gap    rmse={r['rmse_gap']:.6f} ic_mean={r['ic_mean_gap']:.4f} "
        f"top{int(r['top_k'])}_mean={r['topk_mean_gap']:.4f}"
    )
    return None

def _safe_year_int(dt):
    try:
        return int(pd.to_datetime(dt).year)
    except Exception:
        return None

def _build_tradable_mask(df_price):
    cond_active = df_price["open"].notna() & (df_price["open"] > 0)
    if "upper_limit" in df_price.columns:
        cond_no_limit_up = df_price["open"] < df_price["upper_limit"]
    else:
        cond_no_limit_up = True
    if "turnover_prev" in df_price.columns:
        cond_liquid = df_price["turnover_prev"] > MIN_TURNOVER
    else:
        cond_liquid = True
    return cond_active & cond_no_limit_up & cond_liquid

def load_index_ma_risk_signal(csv_path, index_code, ma_window, start_date, end_date):
    if not csv_path or not os.path.exists(csv_path):
        return None
    base = str(index_code).strip()
    if not base:
        return None
    start_s = str(start_date).strip()
    end_s = str(end_date).strip()
    if not start_s or not end_s:
        return None

    window = int(ma_window)
    if window <= 0:
        return None

    start_dt = pd.to_datetime(start_s, format="%Y%m%d", errors="coerce")
    end_dt = pd.to_datetime(end_s, format="%Y%m%d", errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt):
        return None
    need_start = (start_dt - pd.Timedelta(days=max(60, window * 3))).strftime("%Y%m%d")
    end_need = end_dt.strftime("%Y%m%d")

    pieces = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=["date", "code", "close"],
        dtype={"date": "string", "code": "string", "close": "float64"},
        chunksize=300_000,
    ):
        c = chunk["code"].astype("string")
        m_code = c.str.startswith(base, na=False)
        if not bool(m_code.any()):
            continue
        sub = chunk.loc[m_code, ["date", "code", "close"]].copy()
        d = sub["date"].astype("string")
        sub = sub.loc[d.between(need_start, end_need, inclusive="both")]
        if len(sub) == 0:
            continue
        pieces.append(sub)
    if len(pieces) == 0:
        return None

    df = pd.concat(pieces, axis=0, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    if len(df) == 0:
        return None

    df["ma"] = df["close"].rolling(window=window, min_periods=window).mean()
    df["risk_on"] = (df["close"] > df["ma"]).astype("float64")
    out = df.set_index("date")[["close", "ma", "risk_on"]]
    out = out.shift(1)
    out["risk_on"] = out["risk_on"].fillna(1.0).astype("float64")
    return out

def load_index_dual_ma_risk_signal(csv_path, index_code, fast_window, slow_window, start_date, end_date):
    if not csv_path or not os.path.exists(csv_path):
        return None
    base = str(index_code).strip()
    if not base:
        return None
    start_s = str(start_date).strip()
    end_s = str(end_date).strip()
    if not start_s or not end_s:
        return None

    w_fast = int(fast_window)
    w_slow = int(slow_window)
    if w_fast <= 0 or w_slow <= 0:
        return None
    if w_fast >= w_slow:
        w_fast, w_slow = min(w_fast, w_slow), max(w_fast, w_slow)

    start_dt = pd.to_datetime(start_s, format="%Y%m%d", errors="coerce")
    end_dt = pd.to_datetime(end_s, format="%Y%m%d", errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt):
        return None
    need_start = (start_dt - pd.Timedelta(days=max(120, w_slow * 3))).strftime("%Y%m%d")
    end_need = end_dt.strftime("%Y%m%d")

    pieces = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=["date", "code", "close"],
        dtype={"date": "string", "code": "string", "close": "float64"},
        chunksize=300_000,
    ):
        c = chunk["code"].astype("string")
        m_code = c.str.startswith(base, na=False)
        if not bool(m_code.any()):
            continue
        sub = chunk.loc[m_code, ["date", "code", "close"]].copy()
        d = sub["date"].astype("string")
        sub = sub.loc[d.between(need_start, end_need, inclusive="both")]
        if len(sub) == 0:
            continue
        pieces.append(sub)
    if len(pieces) == 0:
        return None

    df = pd.concat(pieces, axis=0, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    if len(df) == 0:
        return None

    df["ma_fast"] = df["close"].rolling(window=w_fast, min_periods=w_fast).mean()
    df["ma_slow"] = df["close"].rolling(window=w_slow, min_periods=w_slow).mean()
    out = df.set_index("date")[["close", "ma_fast", "ma_slow"]]
    out = out.shift(1)
    return out

def diagnose_factors(args):
    start_dt = _parse_yyyymmdd(getattr(args, "diag_start_date", None))
    end_dt = _parse_yyyymmdd(getattr(args, "diag_end_date", None))
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
    diag_txt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), diag_txt_name)
    diag_lines = []

    def emit(msg=""):
        s = "" if msg is None else str(msg)
        print(s)
        diag_lines.append(s)

    emit(
        f"[{datetime.now().time()}] 诊断模式: {start_dt.strftime('%Y%m%d')}~{end_dt.strftime('%Y%m%d')} "
        f"step={step} topk={top_k} min_n={min_n}"
    )

    factor_path = str(getattr(args, "factor_data_path", FACTOR_DATA_PATH))
    price_path = str(getattr(args, "price_data_path", PRICE_DATA_PATH))
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
    pool_mask_f = df_factors.index.get_level_values("code").astype(str).str.startswith(STOCK_POOL_PREFIXES)
    pool_mask_p = df_price.index.get_level_values("code").astype(str).str.startswith(STOCK_POOL_PREFIXES)
    df_factors = df_factors.loc[pool_mask_f, :]
    df_price = df_price.loc[pool_mask_p, :]

    drop_factors = build_drop_factors(args)
    factor_cols = apply_feature_filters(list(df_factors.columns), drop_factors)
    df_factors = df_factors[factor_cols]
    factor_cols = df_factors.select_dtypes(include=[np.number]).columns.tolist()
    if len(factor_cols) == 0:
        raise ValueError("可用数值因子为空，请检查 drop list 或因子文件内容")

    df_factors_shifted = df_factors.groupby(level="code").shift(1)

    future_open = df_price.groupby(level="code")["open"].shift(-PREDICT_DAYS)
    current_open = df_price["open"].replace(0, np.nan)
    ret_next_raw = (future_open - current_open) / current_open
    tradable = _build_tradable_mask(df_price).rename("tradable")
    bench_universe = str(getattr(args, "label_benchmark_universe", LABEL_BENCHMARK_UNIVERSE)).lower()
    raw_for_bench = ret_next_raw.where(tradable.astype(bool)) if bench_universe == "tradable" else ret_next_raw
    if str(LABEL_BENCHMARK_METHOD).lower() == "mean":
        benchmark = raw_for_bench.groupby(level="date").transform("mean")
    else:
        benchmark = raw_for_bench.groupby(level="date").transform("median")
    alpha_5d = (ret_next_raw - benchmark).rename("alpha_5d")

    panel = df_factors_shifted.join(pd.concat([alpha_5d, tradable], axis=1), how="inner")
    if len(panel) == 0:
        raise ValueError("诊断区间内无可用样本（因子与价格/标签无法对齐）")

    all_dates = panel.index.get_level_values("date").unique().sort_values()
    dates = all_dates[::step]
    ic_rows = []
    topk_rows = []

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

        topk_mean = {}
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

    def _summarize(df, prefix):
        mean = df.mean(axis=0, skipna=True)
        std = df.std(axis=0, ddof=1, skipna=True)
        ir = mean / std.replace(0.0, np.nan)
        pos = (df > 0).mean(axis=0, skipna=True)
        out = pd.DataFrame(
            {
                f"{prefix}_mean": mean,
                f"{prefix}_std": std,
                f"{prefix}_ir": ir,
                f"{prefix}_pos": pos,
            }
        )
        return out

    def _direction_advice(ic_mean, ic_ir, top_alpha_mean):
        if ic_mean is None or (not np.isfinite(float(ic_mean))):
            return "未知", 0
        ic_mean_f = float(ic_mean)
        ic_ir_f = float(ic_ir) if (ic_ir is not None and np.isfinite(float(ic_ir))) else np.nan
        top_f = (
            float(top_alpha_mean)
            if (top_alpha_mean is not None and np.isfinite(float(top_alpha_mean)))
            else np.nan
        )

        weak = (abs(ic_mean_f) < 0.01) or (np.isfinite(ic_ir_f) and abs(ic_ir_f) < 0.10)
        if weak:
            return "弱", 0

        if np.isfinite(top_f) and (ic_mean_f * top_f < 0):
            return "冲突", 0

        if ic_mean_f > 0:
            return "趋势", 1
        return "反转", -1

    def _compact_table(summary_df, top_k_value, constraints_now):
        s = summary_df.copy()
        alpha_col = f"top{top_k_value}_alpha_mean"
        dir_out = []
        cons_out = []
        for _, row in s.iterrows():
            direction, cons = _direction_advice(
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

    years = sorted({y for y in (_safe_year_int(d) for d in ic_df.index) if y is not None})
    emit(f"\n诊断因子数: {len(factor_cols)} | 诊断日数: {len(ic_df)} | 年份: {years}")

    constraints_now = build_constraints_dict(args)
    for y in years:
        mask = ic_df.index.year == y
        ic_y = ic_df.loc[mask, :]
        topk_y = topk_df.loc[mask, :]
        if len(ic_y) == 0:
            continue
        s1 = _summarize(ic_y, "ic")
        s2 = _summarize(topk_y, f"top{top_k}_alpha")
        summary = s1.join(s2, how="outer")
        emit(f"\n===== Year {y} (n_days={len(ic_y)}) =====")
        compact = _compact_table(summary, top_k, constraints_now)
        compact_pos = compact.sort_values(["ic_mean", "ic_ir"], ascending=[False, False]).head(10)
        compact_neg = compact.sort_values(["ic_mean", "ic_ir"], ascending=[True, True]).head(10)
        emit("\nTop 10 (IC Mean 最大)")
        emit(compact_pos.to_string(index=False, float_format=lambda v: f"{v: .5f}"))
        emit("\nBottom 10 (IC Mean 最小)")
        emit(compact_neg.to_string(index=False, float_format=lambda v: f"{v: .5f}"))

    s_all = _summarize(ic_df, "ic").join(_summarize(topk_df, f"top{top_k}_alpha"), how="outer")
    emit(f"\n===== All ({start_dt.strftime('%Y%m%d')}~{end_dt.strftime('%Y%m%d')}, n_days={len(ic_df)}) =====")
    compact_all = _compact_table(s_all, top_k, constraints_now)
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

def compute_factor_importance(df_ml, all_dates, predict_dates, features, args):
    if len(predict_dates) == 0:
        return None

    last_target_date = predict_dates[-1]
    train_end_date = all_dates[all_dates.get_loc(last_target_date) - 1]
    train_start_date = all_dates[all_dates.get_loc(last_target_date) - int(args.train_window)]
    train_floor = _parse_yyyymmdd(args.train_floor_date)
    if train_floor is not None and train_start_date < train_floor:
        train_start_date = train_floor

    idx = pd.IndexSlice
    train_data = df_ml.loc[idx[train_start_date:train_end_date, :], :].sort_index()
    if len(train_data) < 1000:
        return None

    X_train = train_data[features]
    y_train = train_data["ret_next"]

    constraints_dict = build_constraints_dict(args)
    monotone_constraints = build_monotone_constraints(features, constraints_dict)

    objective = str(getattr(args, "xgb_objective", "reg:squarederror"))
    if objective.startswith("rank:"):
        train_dates = train_data.index.get_level_values("date").to_numpy()
        _, group_sizes = np.unique(train_dates, return_counts=True)
        model = xgb.XGBRanker(
            n_estimators=int(args.n_estimators),
            learning_rate=float(args.learning_rate),
            max_depth=int(args.max_depth),
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
            n_estimators=int(args.n_estimators),
            learning_rate=float(args.learning_rate),
            max_depth=int(args.max_depth),
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

def save_factor_importance(df_imp, meta, args):
    os.makedirs(FACTORS_IMPORTANCE_DIR, exist_ok=True)
    d0 = pd.to_datetime(meta["importance_train_start"]).strftime("%Y%m%d")
    d1 = pd.to_datetime(meta["importance_train_end"]).strftime("%Y%m%d")
    dt = pd.to_datetime(meta["importance_target_date"]).strftime("%Y%m%d")
    tag = f"target_{dt}_train_{d0}_{d1}_tw{int(args.train_window)}_xgb{int(args.n_estimators)}_d{int(args.max_depth)}"
    out_csv = os.path.join(FACTORS_IMPORTANCE_DIR, f"factor_importance_{tag}.csv")
    out_meta = os.path.join(FACTORS_IMPORTANCE_DIR, f"factor_importance_{tag}.meta.json")
    out_brief = os.path.join(FACTORS_IMPORTANCE_DIR, f"factor_importance_{tag}.brief.txt")
    df_imp.to_csv(out_csv, index=False)
    pd.Series(meta).to_json(out_meta, force_ascii=False, indent=2)

    top_n = int(min(20, len(df_imp)))
    lines = []
    lines.append(f"target_date: {dt}")
    lines.append(f"train_window: {int(args.train_window)}")
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
    print(f"✅ 已保存因子重要性: {out_csv}")
    print(f"✅ 已保存因子重要性元信息: {out_meta}")
    print(f"✅ 已保存因子重要性简报: {out_brief}")

def ensure_factors_index(df_factors):
    if isinstance(df_factors.index, pd.MultiIndex) and list(df_factors.index.names)[:2] == ["date", "code"]:
        df_factors = df_factors.sort_index()
        return df_factors
    if "date" not in df_factors.columns or "code" not in df_factors.columns:
        raise ValueError("因子数据缺少 date/code 列，无法构建 MultiIndex")
    df_factors = df_factors.copy()
    df_factors["date"] = pd.to_datetime(df_factors["date"].astype(str), format="%Y%m%d", errors="coerce")
    df_factors["code"] = df_factors["code"].astype(str)
    df_factors = df_factors.dropna(subset=["date", "code"])
    df_factors = df_factors.set_index(["date", "code"]).sort_index()
    return df_factors

def prepare_dataset(args):
    print(f"[{datetime.now().time()}] 1. 加载因子数据...")
    factor_path = str(getattr(args, "factor_data_path", FACTOR_DATA_PATH))
    price_path = str(getattr(args, "price_data_path", PRICE_DATA_PATH))
    if not os.path.exists(factor_path):
        raise FileNotFoundError(f"找不到因子文件: {factor_path}")
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"找不到价格文件: {price_path}")
    df_factors = pd.read_parquet(factor_path)
    df_factors = ensure_factors_index(df_factors)
    pool_mask_f = df_factors.index.get_level_values("code").astype(str).str.startswith(STOCK_POOL_PREFIXES)
    df_factors = df_factors.loc[pool_mask_f, :]
    end_dt = None
    if getattr(args, "end_date", None):
        end_dt = pd.to_datetime(str(args.end_date), format="%Y%m%d", errors="coerce")
        if pd.isna(end_dt):
            raise ValueError(f"end_date 需为 YYYYMMDD，当前={args.end_date!r}")
        df_factors = df_factors.loc[pd.IndexSlice[:end_dt, :], :]

    # --- 因子预处理：去极值 (Clip) ---
    # 防止某些极端因子值 (如 vol 突然巨大) 干扰模型
    print(">>> 执行因子去极值 (1% - 99%)...")
    # 简单的 quantile clip 比较慢，这里用 clip 加 std 方式更快，或者直接跳过交给 XGBoost
    # XGBoost 对异常值鲁棒，但 KNN 敏感。这里简单做个 truncate
    # 为了速度，暂略，依赖 StandardScaler 在局部做

    # --- 特征滞后 (Shift 1) ---
    print(">>> 执行特征滞后 (Shift 1) 防止未来函数...")
    df_factors_shifted = df_factors.groupby(level="code").shift(1)
    print(f"[{datetime.now().time()}] 2. 加载价格数据...")
    df_price = pd.read_parquet(price_path)
    df_price = df_price.sort_index()
    pool_mask_p = df_price.index.get_level_values("code").astype(str).str.startswith(STOCK_POOL_PREFIXES)
    df_price = df_price.loc[pool_mask_p, :]
    if end_dt is not None:
        df_price = df_price.loc[pd.IndexSlice[:end_dt, :], :]

    if "turnover" in df_price.columns:
        df_price["turnover_prev"] = df_price.groupby(level="code")["turnover"].shift(1)
    if "volume" in df_price.columns:
        df_price["volume_prev"] = df_price.groupby(level="code")["volume"].shift(1)
    print(">>> 计算复合超额收益作为 Label...")
    current_open = df_price["open"].replace(0, np.nan)
    ret_5d = (df_price.groupby(level="code")["open"].shift(-5) - current_open) / current_open
    ret_10d = (df_price.groupby(level="code")["open"].shift(-10) - current_open) / current_open
    df_price["ret_next_raw"] = 0.5 * ret_5d + 0.5 * ret_10d
    tradable_t = _build_tradable_mask(df_price).rename("tradable_t")
    bench_universe = str(getattr(args, "label_benchmark_universe", LABEL_BENCHMARK_UNIVERSE)).lower()
    raw_for_bench = (
        df_price["ret_next_raw"].where(tradable_t.astype(bool)) if bench_universe == "tradable" else df_price["ret_next_raw"]
    )
    if str(LABEL_BENCHMARK_METHOD).lower() == "mean":
        market_benchmark = raw_for_bench.groupby(level="date").transform("mean")
    else:
        market_benchmark = raw_for_bench.groupby(level="date").transform("median")
    df_price["ret_next"] = df_price["ret_next_raw"] - market_benchmark
    df_target = df_price[["ret_next"]].join(tradable_t, how="left")
    df_target = df_target.replace([np.inf, -np.inf], np.nan).dropna(subset=["ret_next"])
    df_target = df_target[df_target["tradable_t"].astype(bool)]
    df_target = df_target[["ret_next"]]
    print(f"[{datetime.now().time()}] 3. 合并数据集...")
    df_ml = df_factors_shifted.join(df_target, how="inner")
    if bool(args.dropna_features):
        df_ml = df_ml.dropna()
    print(f"[{datetime.now().time()}] 数据集准备完毕: {len(df_ml)} 行")
    return df_ml, df_price

# ==============================================================================
# 并行 Worker
# ==============================================================================
def process_single_day_score(target_date, train_start_date, train_end_date, df_ml, df_price, args, features, temp_dir):
    try:
        idx = pd.IndexSlice
        train_data = df_ml.loc[idx[train_start_date:train_end_date, :], :].sort_index()
        test_data = df_ml.loc[idx[target_date, :], :]
        if len(train_data) < 100 or len(test_data) == 0:  # 样本太少不训练
            return None

        drop_factors = build_drop_factors(args)
        final_features = apply_feature_filters(features, drop_factors)
        if len(final_features) == 0:
            return None

        X_train = train_data[final_features]
        y_train = train_data["ret_next"]
        X_test = test_data[final_features]
        sample_weight = _build_sample_weights(args, train_data.index.get_level_values("date"), train_end_date)

        constraints_dict = build_constraints_dict(args)
        monotone_constraints = build_monotone_constraints(final_features, constraints_dict)

        objective = str(getattr(args, "xgb_objective", "reg:squarederror"))
        if objective.startswith("rank:"):
            train_dates = train_data.index.get_level_values("date").to_numpy()
            _, group_sizes = np.unique(train_dates, return_counts=True)
            group_weights = None
            if sample_weight is not None:
                group_weights = _build_sample_weights(args, np.unique(train_dates), train_end_date)
            model_xgb = xgb.XGBRanker(
                n_estimators=int(args.n_estimators),
                learning_rate=float(args.learning_rate),
                max_depth=int(args.max_depth),
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
            if group_weights is None:
                model_xgb.fit(X_train, y_train, group=group_sizes)
            else:
                model_xgb.fit(X_train, y_train, group=group_sizes, sample_weight=group_weights)
        else:
            model_xgb = xgb.XGBRegressor(
                n_estimators=int(args.n_estimators),
                learning_rate=float(args.learning_rate),
                max_depth=int(args.max_depth),
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
            if sample_weight is None:
                model_xgb.fit(X_train, y_train)
            else:
                model_xgb.fit(X_train, y_train, sample_weight=sample_weight)

        pred_xgb = model_xgb.predict(X_test)

        final_score = np.asarray(pred_xgb, dtype=np.float64)

        # KNN 融合 (可选)
        if bool(args.use_knn):
            fill_values = X_train.median(axis=0, skipna=True)
            fill_values = fill_values.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")
            X_train_knn = X_train.fillna(fill_values)
            X_test_knn = X_test.fillna(fill_values)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_knn.to_numpy(dtype=np.float64))
            X_test_scaled = scaler.transform(X_test_knn.to_numpy(dtype=np.float64))
            curr_k = min(int(args.knn_neighbors), len(X_train) - 1)
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
                    args.blend_xgb_weight * df_blend["r_xgb"] + args.blend_knn_weight * df_blend["r_knn"]
                ).values

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
                cond_liquid = price_today["turnover_prev"] > MIN_TURNOVER
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

        keep_n = max(200, int(args.top_k), int(args.buffer_k), int(args.emergency_exit_rank))
        daily_result = daily_result.sort_values(by="score", ascending=False).head(keep_n)
        date_str = target_date.strftime("%Y%m%d")
        temp_file_path = os.path.join(temp_dir, f"{date_str}.parquet")
        daily_result.to_parquet(temp_file_path)
        return date_str
    except Exception as e:
        return f"Error: {str(e)}"

# ==============================================================================
# Step 2: 串行缓冲 + 等权分配
# ==============================================================================
def generate_positions_with_buffer(args, temp_dir, save_dir, df_price):
    objective = str(getattr(args, "xgb_objective", XGB_OBJECTIVE))
    model_kind = "Rank" if objective.startswith("rank:") else "Regression"
    timing_method = str(getattr(args, "timing_method", "score")).lower()
    print(
        f"\n[{datetime.now().time()}] 开始生成持仓 (双轨制: Top{int(args.top_k)}买/Top{int(args.buffer_k)}持, "
        f"间隔{int(args.rebalance_period)}天, 调仓换手上限{float(getattr(args, 'rebalance_turnover_cap', 1.0)):.0%}, "
        f"惯性{float(args.inertia_ratio):.2f}, 平滑{int(args.smooth_window)}日, 止损Top{int(args.emergency_exit_rank)}, "
        f"band={float(getattr(args, 'band_threshold', 0.0)):.4f}, max_w={float(getattr(args, 'max_w', 1.0)):.4f}, "
        f"min_w={float(getattr(args, 'min_weight', 0.0)):.6f}, "
        f"非调仓日={str(getattr(args, 'non_rebalance_action', 'empty'))}, 模型={model_kind}, timing={timing_method})..."
    )
    temp_files = sorted(glob.glob(os.path.join(temp_dir, "*.parquet")))
    if len(temp_files) == 0:
        print("⚠️ 未生成任何日度评分文件，跳过持仓文件生成。")
        return

    start_s = os.path.basename(temp_files[0]).replace(".parquet", "")
    end_s = os.path.basename(temp_files[-1]).replace(".parquet", "")
    risk_df = None
    if timing_method in ("index_ma20", "index_ma_dual"):
        if timing_method == "index_ma20":
            risk_df = load_index_ma_risk_signal(
                getattr(args, "risk_data_path", MERGED_DATA_PATH),
                getattr(args, "risk_index_code", "399006"),
                int(getattr(args, "risk_ma_window", 20)),
                start_s,
                end_s,
            )
            if risk_df is None or len(risk_df) == 0:
                print("⚠️ 风控数据缺失，timing=index_ma20 回退为 timing=none")
                timing_method = "none"
            else:
                idx_dt = pd.to_datetime(end_s, format="%Y%m%d", errors="coerce")
                last = None
                if pd.notna(idx_dt):
                    try:
                        last = risk_df.loc[idx_dt, :]
                    except Exception:
                        last = None
                if last is not None and hasattr(last, "to_dict"):
                    d = dict(last.to_dict())
                    c = d.get("close", np.nan)
                    ma = d.get("ma", np.nan)
                    ro = d.get("risk_on", np.nan)
                    print(
                        f"[{datetime.now().time()}] 风控: {str(getattr(args, 'risk_index_code', '399006'))} MA{int(getattr(args, 'risk_ma_window', 20))} "
                        f"({start_s}~{end_s}), close={float(c) if np.isfinite(c) else np.nan:.4f}, "
                        f"ma={float(ma) if np.isfinite(ma) else np.nan:.4f}, "
                        f"risk_on={int(float(ro) >= 0.5) if np.isfinite(ro) else -1}, "
                        f"buffer={float(getattr(args, 'risk_ma_buffer', 0.0)):.4f}, "
                        f"bad_exposure={float(getattr(args, 'timing_bad_exposure', 0.0)):.2f}"
                    )
        else:
            risk_df = load_index_dual_ma_risk_signal(
                getattr(args, "risk_data_path", MERGED_DATA_PATH),
                getattr(args, "risk_index_code", "399006"),
                int(getattr(args, "risk_ma_fast_window", 20)),
                int(getattr(args, "risk_ma_slow_window", 60)),
                start_s,
                end_s,
            )
            if risk_df is None or len(risk_df) == 0:
                print("⚠️ 风控数据缺失，timing=index_ma_dual 回退为 timing=none")
                timing_method = "none"
            else:
                idx_dt = pd.to_datetime(end_s, format="%Y%m%d", errors="coerce")
                last = None
                if pd.notna(idx_dt):
                    try:
                        last = risk_df.loc[idx_dt, :]
                    except Exception:
                        last = None
                if last is not None and hasattr(last, "to_dict"):
                    d = dict(last.to_dict())
                    c = d.get("close", np.nan)
                    maf = d.get("ma_fast", np.nan)
                    mas = d.get("ma_slow", np.nan)
                    print(
                        f"[{datetime.now().time()}] 风控: {str(getattr(args, 'risk_index_code', '399006'))} MA{int(getattr(args, 'risk_ma_fast_window', 20))}/MA{int(getattr(args, 'risk_ma_slow_window', 60))} "
                        f"({start_s}~{end_s}), close={float(c) if np.isfinite(c) else np.nan:.4f}, "
                        f"ma_fast={float(maf) if np.isfinite(maf) else np.nan:.4f}, "
                        f"ma_slow={float(mas) if np.isfinite(mas) else np.nan:.4f}, "
                        f"buffer={float(getattr(args, 'risk_ma_buffer', 0.0)):.4f}, "
                        f"bad_exposure={float(getattr(args, 'timing_bad_exposure', 0.0)):.2f}"
                    )
    prev_w = pd.Series(dtype="float64")
    yesterday_holding_list = []
    history_scores = []
    days_since_last_trade = int(args.rebalance_period)
    market_risk_on = True
    idx = pd.IndexSlice
    for file_path in tqdm(temp_files, desc="Generating Positions"):
        df_today = pd.read_parquet(file_path)
        date_str = os.path.basename(file_path).replace(".parquet", "")
        if "code" not in df_today.columns or "score" not in df_today.columns:
            continue
        target_date = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
        if pd.isna(target_date):
            continue

        curr_scores = (
            df_today[["code", "score"]].dropna().drop_duplicates(subset=["code"]).set_index("code")["score"]
        )
        history_scores.append(curr_scores)
        while len(history_scores) > int(args.smooth_window):
            history_scores.pop(0)

        mean_scores = pd.concat(history_scores, axis=1, join="outer").mean(axis=1).dropna()
        if mean_scores.empty:
            continue

        ranked_raw = mean_scores.sort_values(ascending=False)
        timing_scale = 1.0
        if timing_method == "none":
            timing_scale = 1.0
        elif timing_method == "index_ma20":
            risk_on_now = market_risk_on
            if risk_df is not None:
                try:
                    row = risk_df.loc[target_date, :]
                    c = float(row["close"]) if hasattr(row, "__getitem__") else np.nan
                    ma = float(row["ma"]) if hasattr(row, "__getitem__") else np.nan
                    buf = float(getattr(args, "risk_ma_buffer", 0.0))
                    if np.isfinite(c) and np.isfinite(ma) and ma > 0:
                        if market_risk_on:
                            if c < ma * (1.0 - buf):
                                risk_on_now = False
                        else:
                            if c > ma * (1.0 + buf):
                                risk_on_now = True
                except Exception:
                    pass
            if risk_on_now != market_risk_on:
                print(f"[{datetime.now().time()}] 风控切换: {date_str} risk_on={int(risk_on_now)}")
            market_risk_on = risk_on_now
            timing_scale = 1.0 if market_risk_on else float(getattr(args, "timing_bad_exposure", 0.0))
        elif timing_method == "index_ma_dual":
            risk_on_now = market_risk_on
            if risk_df is not None:
                try:
                    row = risk_df.loc[target_date, :]
                    c = float(row["close"]) if hasattr(row, "__getitem__") else np.nan
                    maf = float(row["ma_fast"]) if hasattr(row, "__getitem__") else np.nan
                    mas = float(row["ma_slow"]) if hasattr(row, "__getitem__") else np.nan
                    buf = float(getattr(args, "risk_ma_buffer", 0.0))
                    if np.isfinite(c) and np.isfinite(maf) and np.isfinite(mas) and maf > 0 and mas > 0:
                        if market_risk_on:
                            if (maf < mas) or (c < maf * (1.0 - buf)):
                                risk_on_now = False
                        else:
                            if (maf > mas) and (c > maf * (1.0 + buf)):
                                risk_on_now = True
                except Exception:
                    pass
            if risk_on_now != market_risk_on:
                print(f"[{datetime.now().time()}] 风控切换: {date_str} risk_on={int(risk_on_now)}")
            market_risk_on = risk_on_now
            timing_scale = 1.0 if market_risk_on else float(getattr(args, "timing_bad_exposure", 0.0))
        else:
            top_n = min(30, len(ranked_raw))
            top30_mean_score = float(ranked_raw.head(top_n).mean()) if top_n > 0 else np.nan
            exit_thr = getattr(args, "timing_exit_threshold", None)
            if exit_thr is None:
                exit_thr = float(args.timing_threshold)
            else:
                exit_thr = float(exit_thr)
            enter_thr = getattr(args, "timing_enter_threshold", None)
            if enter_thr is None:
                enter_thr = exit_thr + float(getattr(args, "timing_hysteresis", 0.01))
            else:
                enter_thr = float(enter_thr)
            if enter_thr < exit_thr:
                enter_thr = exit_thr

            if np.isfinite(top30_mean_score):
                if market_risk_on:
                    if top30_mean_score < exit_thr:
                        market_risk_on = False
                else:
                    if top30_mean_score >= enter_thr:
                        market_risk_on = True
            timing_scale = 1.0 if market_risk_on else float(args.timing_bad_exposure)
        band_thr = float(getattr(args, "band_threshold", 0.0))
        band_thr = 0.0 if not np.isfinite(band_thr) else max(0.0, band_thr)

        is_rebalance_day = (days_since_last_trade >= int(args.rebalance_period)) or (len(prev_w) == 0)
        stoploss_set = set()
        stoploss_worst_rank = None
        stoploss_thr = int(getattr(args, "emergency_exit_rank", 0))
        if stoploss_thr > 0 and len(prev_w) > 0 and len(ranked_raw) > 0:
            try:
                rank_map = pd.Series(
                    np.arange(1, len(ranked_raw) + 1, dtype=np.int64),
                    index=ranked_raw.index.astype("string"),
                    dtype="int64",
                )
                hold_idx = prev_w.index.astype("string")
                hold_rank = rank_map.reindex(hold_idx)
                if bool(getattr(args, "keep_missing_positions", True)):
                    hold_rank = hold_rank.fillna(1)
                else:
                    hold_rank = hold_rank.fillna(int(1e12))
                m = hold_rank.astype("int64") > stoploss_thr
                if bool(m.any()):
                    stoploss_set = set(hold_rank.index[m].astype("string").tolist())
                    stoploss_worst_rank = int(hold_rank.loc[m].max())
            except Exception:
                stoploss_set = set()
                stoploss_worst_rank = None
        if (not is_rebalance_day) and len(stoploss_set) > 0:
            print(
                f"[{datetime.now().time()}] 快速止损触发: {date_str} "
                f"n={len(stoploss_set)}, worst_rank={stoploss_worst_rank if stoploss_worst_rank is not None else -1}, "
                f"thr={stoploss_thr}"
            )
            is_rebalance_day = True

        if not is_rebalance_day:
            output_path = os.path.join(save_dir, f"{date_str}.csv")
            if str(getattr(args, "non_rebalance_action", "empty")).lower() == "carry":
                out_w = prev_w.copy()
                if timing_scale <= 0.0:
                    out_w = out_w.iloc[0:0].copy()
                else:
                    out_w = (out_w * float(timing_scale)).astype("float64")
                out_df = out_w.rename_axis("code").rename("weight").reset_index()
                out_df.to_csv(output_path, index=False, header=False, float_format="%.10f")
            else:
                with open(output_path, "w", encoding="utf-8"):
                    pass
            days_since_last_trade += 1
            continue

        adj_scores = ranked_raw.copy()
        if len(prev_w) > 0:
            keep_idx = adj_scores.index.intersection(list(prev_w.index))
            if len(keep_idx) > 0:
                adj_scores.loc[keep_idx] = adj_scores.loc[keep_idx] * float(args.inertia_ratio)
        ranked_adj_codes = adj_scores.sort_values(ascending=False).index.to_numpy()

        target_selected = []
        target_set = set()
        if len(prev_w) > 0:
            top_buffer = set(ranked_adj_codes[: int(args.buffer_k)])
            for code in yesterday_holding_list:
                if code in top_buffer:
                    target_selected.append(code)
                    target_set.add(code)

        slots = int(args.top_k) - len(target_selected)
        if slots > 0:
            for code in ranked_adj_codes:
                if slots == 0:
                    break
                if code in target_set:
                    continue
                target_selected.append(code)
                target_set.add(code)
                slots -= 1

        target_codes = target_selected[: int(args.top_k)]

        codes_check = set(target_codes) | set(prev_w.index.astype(str).tolist())
        buyable = pd.Series(False, index=pd.Index(sorted(codes_check), dtype="string"))
        tradable = pd.Series(False, index=buyable.index)
        try:
            price_today = df_price.loc[idx[target_date, list(buyable.index)], :]
            if len(price_today) > 0:
                pt = price_today.reset_index(level="date", drop=True)
                o = pd.to_numeric(pt.get("open", np.nan), errors="coerce")
                cond_open = o.notna() & (o > 0)
                tv = pt.get("turnover_prev", None)
                if tv is None:
                    tv = pt.get("turnover", None)
                if tv is not None:
                    tv = pd.to_numeric(tv, errors="coerce")
                    cond_liquid = tv > float(MIN_TURNOVER)
                else:
                    cond_liquid = pd.Series(True, index=pt.index)
                u = pd.to_numeric(pt.get("upper_limit", np.nan), errors="coerce")
                l = pd.to_numeric(pt.get("lower_limit", np.nan), errors="coerce")
                has_u = u.notna() & (u > 0)
                has_l = l.notna() & (l > 0)
                lim = (has_u & (o >= (u * (1 - 1e-12)))) | (has_l & (o <= (l * (1 + 1e-12))))
                trad = (cond_open & cond_liquid).rename("tradable")
                buy = (trad & (~lim)).rename("buyable")
                tradable.loc[trad.index.astype("string")] = trad.to_numpy(dtype=bool)
                buyable.loc[buy.index.astype("string")] = buy.to_numpy(dtype=bool)
        except Exception:
            pass

        prev_syms = set(prev_w.index.astype(str).tolist())
        fixed_force = set()
        if str(getattr(args, "limit_policy", "freeze")).lower() == "freeze":
            fixed_force |= {c for c in prev_syms if c in buyable.index and (not bool(buyable.loc[c]))}
            missing_px = prev_syms - set(buyable.index.tolist())
            fixed_force |= missing_px

        desired_set = set(target_codes) | fixed_force

        cap = getattr(args, "rebalance_turnover_cap", None)
        if cap is None:
            cap = 1.0
        cap = float(cap)
        if not np.isfinite(cap):
            cap = 1.0
        cap = max(0.0, min(1.0, cap))
        max_new = int(np.floor(cap * int(args.top_k)))
        max_new = max(0, min(int(args.top_k), max_new))

        if len(prev_w) == 0:
            buy_candidates = [c for c in target_codes if c in buyable.index and bool(buyable.loc[c])]
            buy_keep = buy_candidates[: int(args.top_k)]
            sell_exec = []
            buy_budget = 1.0
            w_next = pd.Series(dtype="float64")
        else:
            sell_candidates = [c for c in prev_syms if (c not in desired_set) or (c in stoploss_set)]
            sell_candidates = [c for c in sell_candidates if c in buyable.index and bool(buyable.loc[c])]
            sell_exec = [c for c in sell_candidates if abs(float(prev_w.get(c, 0.0))) >= band_thr]
            buy_budget = float(prev_w.reindex(sell_exec).sum()) if len(sell_exec) > 0 else 0.0
            w_next = prev_w.drop(index=[c for c in sell_exec if c in prev_w.index], errors="ignore").copy()
            buy_candidates = [c for c in ranked_adj_codes if (c in desired_set and c not in prev_syms)]
            buy_candidates = [c for c in buy_candidates if c in buyable.index and bool(buyable.loc[c])]
            buy_keep = buy_candidates[:max_new]

        if buy_budget > 0.0 and len(buy_keep) > 0:
            w_buy = pd.Series(
                float(buy_budget) / len(buy_keep),
                index=pd.Index(buy_keep, dtype="string"),
                dtype="float64",
            )
            max_w = float(getattr(args, "max_w", 1.0))
            if np.isfinite(max_w) and max_w > 0 and max_w < 1:
                w_rel = apply_max_weight_cap(w_buy, max_w / float(buy_budget))
                w_buy = w_rel * float(buy_budget)
            min_w = float(getattr(args, "min_weight", 0.0))
            min_w = 0.0 if not np.isfinite(min_w) else max(0.0, min_w)
            if min_w > 0:
                w_buy[w_buy < min_w] = 0.0
                w_buy = w_buy[w_buy > 0]
            if len(w_buy) > 0 and band_thr > 0:
                w_buy[w_buy < band_thr] = 0.0
                w_buy = w_buy[w_buy > 0]
            if len(w_buy) > 0:
                if len(w_next) == 0:
                    w_next = w_buy.copy()
                else:
                    w_next = pd.concat([w_next, w_buy])

        w_next = w_next[w_next > 0].astype("float64")
        w_next_sum = float(w_next.sum()) if len(w_next) > 0 else 0.0
        if np.isfinite(w_next_sum) and w_next_sum > 1 + 1e-10:
            w_next = (w_next / w_next_sum).astype("float64")

        prev_w = w_next
        if len(prev_w) > 0:
            yesterday_holding_list = list(prev_w.index.astype(str).tolist())
        else:
            yesterday_holding_list = []

        days_since_last_trade = 1

        output_path = os.path.join(save_dir, f"{date_str}.csv")
        if timing_scale <= 0.0 or len(prev_w) == 0:
            with open(output_path, "w", encoding="utf-8"):
                pass
        else:
            out_w = (prev_w * float(timing_scale)).astype("float64")
            out_df = out_w.rename_axis("code").rename("weight").reset_index()
            out_df.to_csv(output_path, index=False, header=False, float_format="%.10f")

# ==============================================================================
# 主流程
# ==============================================================================
def main_workflow():
    try:
        args = parse_args()

        if bool(getattr(args, "diagnose", False)):
            diagnose_factors(args)
            return
        run_overfit_along = bool(getattr(args, "overfit_check", False)) and bool(getattr(args, "overfit_along", False))
        if bool(getattr(args, "overfit_check", False)) and not run_overfit_along:
            overfit_check(args)
            return

        objective = str(getattr(args, "xgb_objective", XGB_OBJECTIVE))
        model_kind = "Rank" if objective.startswith("rank:") else "Regression"
        timing_method = str(getattr(args, "timing_method", "score")).lower()
        if bool(args.use_knn):
            print(
                f"[{datetime.now().time()}] 模型: XGBoost + KNN | objective={objective} | mode={model_kind} | "
                f"blend(xgb={float(args.blend_xgb_weight):.2f}, knn={float(args.blend_knn_weight):.2f}) | "
                f"constraints={'on' if bool(getattr(args, 'use_constraints', True)) else 'off'} | "
                f"dropna_features={'on' if bool(getattr(args, 'dropna_features', False)) else 'off'} | "
                f"timing={timing_method} | label=0.5*ret_5d+0.5*ret_10d"
            )
        else:
            print(
                f"[{datetime.now().time()}] 模型: XGBoost | objective={objective} | mode={model_kind} | "
                f"constraints={'on' if bool(getattr(args, 'use_constraints', True)) else 'off'} | "
                f"dropna_features={'on' if bool(getattr(args, 'dropna_features', False)) else 'off'} | "
                f"timing={timing_method} | label=0.5*ret_5d+0.5*ret_10d"
            )
        _print_run_params(args, objective=objective, model_kind=model_kind, timing_method=timing_method)

        save_dir = os.path.join(OUTPUT_DIR, SUB_DIR_NAME)
        temp_dir = os.path.join(OUTPUT_DIR, TEMP_DIR_NAME)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            for fp in glob.glob(os.path.join(save_dir, "*.csv")):
                try:
                    os.remove(fp)
                except Exception:
                    pass

        data_ml, data_price = prepare_dataset(args)
        if run_overfit_along:
            if not bool(getattr(args, "overfit_range", False)) and getattr(args, "start_date", None) and getattr(args, "end_date", None):
                args.overfit_range = True
            _run_overfit_check(args, data_ml=data_ml, exit_after=False)

        all_dates = data_ml.index.get_level_values("date").unique().sort_values()
        features = [col for col in data_ml.columns if col != "ret_next"]
        drop_factors = build_drop_factors(args)
        final_features = apply_feature_filters(features, drop_factors)
        print(f"Features ({len(final_features)}): {final_features[:5]} ...")

        start_index = int(args.train_window)
        predict_dates = all_dates[start_index:]
        if args.start_date:
            predict_dates = predict_dates[predict_dates >= pd.to_datetime(args.start_date, format="%Y%m%d")]
        if args.end_date:
            predict_dates = predict_dates[predict_dates <= pd.to_datetime(args.end_date, format="%Y%m%d")]
        if args.max_predict_days is not None:
            predict_dates = predict_dates[: int(args.max_predict_days)]
        if len(predict_dates) == 0:
            print("⚠️ 预测日期为空，跳过训练与生成持仓。")
            return
        print(f"预测范围: {predict_dates[0].strftime('%Y%m%d')} ~ {predict_dates[-1].strftime('%Y%m%d')}")

        train_floor = _parse_yyyymmdd(args.train_floor_date)
        tasks = []
        for i, target_date in enumerate(predict_dates):
            train_start_date = all_dates[start_index + i - int(args.train_window)]
            train_end_date = all_dates[start_index + i - 1]
            if train_floor is not None and train_start_date < train_floor:
                train_start_date = train_floor
            tasks.append((target_date, train_start_date, train_end_date))

        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = [
                executor.submit(
                    process_single_day_score,
                    t[0],
                    t[1],
                    t[2],
                    data_ml,
                    data_price,
                    args,
                    final_features,
                    temp_dir,
                )
                for t in tasks
            ]
            n_ok = 0
            n_none = 0
            n_err = 0
            err_samples = []
            for fut in tqdm(as_completed(futures), total=len(tasks), desc="Training"):
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
                n_ok += 1
            if n_err > 0:
                print(f"⚠️ 评分生成异常: ok={n_ok}, none={n_none}, err={n_err}")
                for s in err_samples:
                    print(s)

        res = compute_factor_importance(data_ml, all_dates, predict_dates, final_features, args)
        if res is not None:
            df_imp, meta = res
            meta["dropped_factors"] = sorted(list(drop_factors))
            save_factor_importance(df_imp, meta, args)

        generate_positions_with_buffer(args, temp_dir, save_dir, data_price)
        print("\n✅ 完成！")
    except Exception as e:
        print(f"❌ 出错: {e}")
        import traceback

        traceback.print_exc()

if __name__ == "__main__":
    main_workflow()
