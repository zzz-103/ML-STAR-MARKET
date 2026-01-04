# 位置: 11（组合构建）| 将每日分数转为逐日权重 CSV（TopK/缓冲/惯性/调仓/择时）
# 输入: args、temp_dir(parquet 分数)、save_dir(csv 输出)、df_price（用于可交易/涨跌停/流动性）
# 输出: save_dir/YYYYMMDD.csv（code, weight，无表头）；并记录日志
# 依赖: _05_weights、_07_risk_signals、numpy/pandas
from __future__ import annotations

import glob
import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from ml_models.model_functions._05_weights import apply_max_weight_cap
from ml_models.model_functions._07_risk_signals import (
    attach_industry_to_price,
    build_sector_index,
    compute_sector_daily_returns,
    compute_sector_momentum_rank,
    load_index_dual_ma_risk_signal,
    load_index_ma_risk_signal,
    load_self_equal_weight_ma_risk_signal,
    load_stock_industry_map,
)


def generate_positions_with_buffer(args, temp_dir: str, save_dir: str, df_price: pd.DataFrame, logger: logging.Logger) -> None:
    objective = str(getattr(args, "xgb_objective", "reg:squarederror"))
    model_kind = "Rank" if objective.startswith("rank:") else "Regression"
    timing_method = str(getattr(args, "timing_method", "score")).lower()
    keep_cash_idle = bool(getattr(args, "keep_cash_idle", False))
    logger.info(
        "step=generate_positions top_k=%d buffer_k=%d rebalance=%d turnover_cap=%.3f keep_cash_idle=%d inertia=%.4f smooth=%d stop_rank=%d band=%.6f max_w=%.6f min_w=%.6f non_rebalance=%s model=%s timing=%s",
        int(getattr(args, "top_k")),
        int(getattr(args, "buffer_k")),
        int(getattr(args, "rebalance_period")),
        float(getattr(args, "rebalance_turnover_cap", 1.0)),
        int(keep_cash_idle),
        float(getattr(args, "inertia_ratio", 1.0)),
        int(getattr(args, "smooth_window", 1)),
        int(getattr(args, "emergency_exit_rank", 0)),
        float(getattr(args, "band_threshold", 0.0)),
        float(getattr(args, "max_w", 1.0)),
        float(getattr(args, "min_weight", 0.0)),
        str(getattr(args, "non_rebalance_action", "empty")),
        model_kind,
        timing_method,
    )

    temp_files = sorted(glob.glob(os.path.join(temp_dir, "*.parquet")))
    if len(temp_files) == 0:
        logger.info("no_temp_scores skip_positions=1")
        return

    start_s = os.path.basename(temp_files[0]).replace(".parquet", "")
    end_s = os.path.basename(temp_files[-1]).replace(".parquet", "")

    risk_df = None
    risk_df_300 = None
    risk_df_688 = None
    if timing_method in ("index_ma20", "index_ma_dual", "self_eq_ma20", "split_index_ma20"):
        if timing_method == "index_ma20":
            risk_df = load_index_ma_risk_signal(
                str(getattr(args, "risk_data_path")),
                str(getattr(args, "risk_index_code")),
                int(getattr(args, "risk_ma_window")),
                start_s,
                end_s,
            )
            if risk_df is None or len(risk_df) == 0:
                logger.info("risk_data_missing timing_fallback=none method=index_ma20")
                timing_method = "none"
        elif timing_method == "index_ma_dual":
            risk_df = load_index_dual_ma_risk_signal(
                str(getattr(args, "risk_data_path")),
                str(getattr(args, "risk_index_code")),
                int(getattr(args, "risk_ma_fast_window")),
                int(getattr(args, "risk_ma_slow_window")),
                start_s,
                end_s,
            )
            if risk_df is None or len(risk_df) == 0:
                logger.info("risk_data_missing timing_fallback=none method=index_ma_dual")
                timing_method = "none"
        elif timing_method == "self_eq_ma20":
            risk_df = load_self_equal_weight_ma_risk_signal(
                df_price,
                int(getattr(args, "risk_ma_window")),
                start_s,
                end_s,
            )
            if risk_df is None or len(risk_df) == 0:
                logger.info("risk_data_missing timing_fallback=none method=self_eq_ma20")
                timing_method = "none"
        else:
            risk_df_300 = load_index_ma_risk_signal(
                str(getattr(args, "risk_data_path")),
                str(getattr(args, "risk_index_code_300", "399006")),
                int(getattr(args, "risk_ma_window")),
                start_s,
                end_s,
            )
            risk_df_688 = load_index_ma_risk_signal(
                str(getattr(args, "risk_data_path")),
                str(getattr(args, "risk_index_code_688", "000688")),
                int(getattr(args, "risk_ma_window")),
                start_s,
                end_s,
            )
            if (risk_df_300 is None or len(risk_df_300) == 0) and (risk_df_688 is None or len(risk_df_688) == 0):
                logger.info("risk_data_missing timing_fallback=none method=split_index_ma20")
                timing_method = "none"

    bull_df = None
    if bool(getattr(args, "industry_bull_enable", False)):
        bull_df = load_index_ma_risk_signal(
            str(getattr(args, "risk_data_path")),
            str(getattr(args, "risk_index_code")),
            int(getattr(args, "industry_bull_ma_window", 60)),
            start_s,
            end_s,
        )

    industry_enabled = bool(getattr(args, "industry_enable", False))
    code_to_industry: dict[str, str] = {}
    industry_rank_pct_df: pd.DataFrame | None = None
    industry_risk_off_df: pd.DataFrame | None = None
    industry_fast_reversal_df: pd.DataFrame | None = None
    if industry_enabled:
        industry_map_path = str(getattr(args, "industry_map_path", "")).strip()
        code_to_industry = load_stock_industry_map(industry_map_path)
        try:
            df_price = attach_industry_to_price(df_price, code_to_industry)
            mom_window = int(getattr(args, "industry_mom_window", 20))
            ma_window = int(getattr(args, "industry_ma_window", 20))
            riskoff_buf = float(getattr(args, "industry_ma_riskoff_buffer", 0.01))

            start_dt = pd.to_datetime(start_s, format="%Y%m%d", errors="coerce")
            end_dt = pd.to_datetime(end_s, format="%Y%m%d", errors="coerce")
            if (not pd.isna(start_dt)) and (not pd.isna(end_dt)) and mom_window > 0 and ma_window > 0:
                need_days = max(120, int(mom_window * 4), int(ma_window * 4))
                need_start_dt = (start_dt - pd.Timedelta(days=need_days)).strftime("%Y%m%d")
                sector_ret = compute_sector_daily_returns(df_price, start_date=need_start_dt, end_date=end_s)
                sector_idx = build_sector_index(sector_ret)
                _, rank_pct = compute_sector_momentum_rank(sector_idx, window=mom_window)
                industry_rank_pct_df = rank_pct.shift(1)
                ma = sector_idx.rolling(window=ma_window, min_periods=ma_window).mean()
                risk_off = (sector_idx < (ma * (1.0 - riskoff_buf))).astype("float64")
                industry_risk_off_df = risk_off.shift(1).fillna(0.0).astype("float64")
                if bool(getattr(args, "industry_fast_reversal_enable", False)):
                    turnover_col = None
                    for c in ("turnover_prev", "turnover", "amount", "volume", "vol"):
                        if c in df_price.columns:
                            turnover_col = c
                            break
                    if turnover_col is not None:
                        ret_thr = float(getattr(args, "industry_fast_reversal_ret_threshold", 0.03))
                        vol_window = int(getattr(args, "industry_fast_reversal_vol_window", 20))
                        vol_mult = float(getattr(args, "industry_fast_reversal_vol_mult", 1.5))
                        vol_window = 0 if vol_window is None else int(vol_window)
                        if vol_window > 1 and np.isfinite(ret_thr) and np.isfinite(vol_mult) and vol_mult > 0:
                            s_tv = pd.to_datetime(need_start_dt, format="%Y%m%d", errors="coerce")
                            e_tv = pd.to_datetime(end_s, format="%Y%m%d", errors="coerce")
                            df_tv = df_price
                            if (not pd.isna(s_tv)) and (not pd.isna(e_tv)):
                                dates = df_tv.index.get_level_values("date")
                                m = (dates >= s_tv) & (dates <= e_tv)
                                df_tv = df_tv.loc[m.to_numpy(dtype=bool), :]
                            tv = pd.to_numeric(df_tv[turnover_col], errors="coerce")
                            ind = df_tv["industry"].astype("string")
                            d = df_tv.index.get_level_values("date")
                            tmp = pd.DataFrame({"date": d, "industry": ind, "tv": tv}).dropna(subset=["tv", "industry"])
                            sector_tv = tmp.groupby(["date", "industry"], sort=True)["tv"].mean().unstack("industry").sort_index()
                            tv_ma = sector_tv.rolling(window=vol_window, min_periods=vol_window).mean()
                            vol_ratio = sector_tv / tv_ma
                            under_ma = sector_idx < ma
                            fast = (under_ma & (sector_ret > ret_thr) & (vol_ratio > vol_mult)).astype("float64")
                            industry_fast_reversal_df = fast.shift(1).fillna(0.0).astype("float64")
            else:
                industry_enabled = False
        except Exception:
            industry_enabled = False

    prev_w = pd.Series(dtype="float64")
    yesterday_holding_list: list[str] = []
    history_scores: list[pd.Series] = []
    days_since_last_trade = int(getattr(args, "rebalance_period"))
    market_risk_on = True
    market_risk_on_300 = True
    market_risk_on_688 = True
    industry_riskoff_state: dict[str, bool] = {}
    idx = pd.IndexSlice

    for file_path in tqdm(temp_files, desc="Generating Positions"):
        df_today = pd.read_parquet(file_path)
        date_str = os.path.basename(file_path).replace(".parquet", "")
        if "code" not in df_today.columns or "score" not in df_today.columns:
            continue
        target_date = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
        if pd.isna(target_date):
            continue

        curr_scores = df_today[["code", "score"]].dropna().drop_duplicates(subset=["code"]).set_index("code")["score"]
        history_scores.append(curr_scores)
        while len(history_scores) > int(getattr(args, "smooth_window")):
            history_scores.pop(0)

        mean_scores = pd.concat(history_scores, axis=1, join="outer").mean(axis=1).dropna()
        if mean_scores.empty:
            continue

        ranked_raw = mean_scores.sort_values(ascending=False)
        timing_scale = 1.0
        timing_scale_by_code: pd.Series | None = None
        if timing_method == "none":
            timing_scale = 1.0
        elif timing_method in ("index_ma20", "self_eq_ma20"):
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
                logger.info("risk_switch date=%s risk_on=%d", date_str, int(risk_on_now))
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
                logger.info("risk_switch date=%s risk_on=%d", date_str, int(risk_on_now))
            market_risk_on = risk_on_now
            timing_scale = 1.0 if market_risk_on else float(getattr(args, "timing_bad_exposure", 0.0))
        elif timing_method == "split_index_ma20":
            buf = float(getattr(args, "risk_ma_buffer", 0.0))
            risk_on_300 = market_risk_on_300
            if risk_df_300 is not None:
                try:
                    row = risk_df_300.loc[target_date, :]
                    c = float(row["close"]) if hasattr(row, "__getitem__") else np.nan
                    ma = float(row["ma"]) if hasattr(row, "__getitem__") else np.nan
                    if np.isfinite(c) and np.isfinite(ma) and ma > 0:
                        if market_risk_on_300:
                            if c < ma * (1.0 - buf):
                                risk_on_300 = False
                        else:
                            if c > ma * (1.0 + buf):
                                risk_on_300 = True
                except Exception:
                    pass
            if risk_on_300 != market_risk_on_300:
                logger.info("risk_switch_300 date=%s risk_on=%d", date_str, int(risk_on_300))
            market_risk_on_300 = risk_on_300

            risk_on_688 = market_risk_on_688
            if risk_df_688 is not None:
                try:
                    row = risk_df_688.loc[target_date, :]
                    c = float(row["close"]) if hasattr(row, "__getitem__") else np.nan
                    ma = float(row["ma"]) if hasattr(row, "__getitem__") else np.nan
                    if np.isfinite(c) and np.isfinite(ma) and ma > 0:
                        if market_risk_on_688:
                            if c < ma * (1.0 - buf):
                                risk_on_688 = False
                        else:
                            if c > ma * (1.0 + buf):
                                risk_on_688 = True
                except Exception:
                    pass
            if risk_on_688 != market_risk_on_688:
                logger.info("risk_switch_688 date=%s risk_on=%d", date_str, int(risk_on_688))
            market_risk_on_688 = risk_on_688

            bad = float(getattr(args, "timing_bad_exposure", 0.0))
            scale_300 = 1.0 if market_risk_on_300 else bad
            scale_688 = 1.0 if market_risk_on_688 else bad
            codes = ranked_raw.index.astype("string")
            is_300 = codes.str.startswith("300", na=False)
            is_688 = codes.str.startswith("688", na=False)
            scale = pd.Series(1.0, index=codes, dtype="float64")
            if bool(is_300.any()):
                scale.loc[is_300] = float(scale_300)
            if bool(is_688.any()):
                scale.loc[is_688] = float(scale_688)
            timing_scale_by_code = scale
        else:
            top_n = min(30, len(ranked_raw))
            top30_mean_score = float(ranked_raw.head(top_n).mean()) if top_n > 0 else np.nan
            exit_thr = getattr(args, "timing_exit_threshold", None)
            exit_thr = float(getattr(args, "timing_threshold")) if exit_thr is None else float(exit_thr)
            enter_thr = getattr(args, "timing_enter_threshold", None)
            enter_thr = exit_thr + float(getattr(args, "timing_hysteresis", 0.01)) if enter_thr is None else float(enter_thr)
            if enter_thr < exit_thr:
                enter_thr = exit_thr

            if np.isfinite(top30_mean_score):
                if market_risk_on:
                    if top30_mean_score < exit_thr:
                        market_risk_on = False
                else:
                    if top30_mean_score >= enter_thr:
                        market_risk_on = True
            timing_scale = 1.0 if market_risk_on else float(getattr(args, "timing_bad_exposure"))

        def _apply_timing(w: pd.Series) -> pd.Series:
            if w is None or len(w) == 0:
                return pd.Series(dtype="float64")
            if timing_scale_by_code is not None:
                s = timing_scale_by_code.reindex(w.index.astype("string")).fillna(1.0).astype("float64")
                out = (w.astype("float64") * s).astype("float64")
                out = out[out > 0.0]
                return out
            if timing_scale <= 0.0:
                return w.iloc[0:0].copy()
            return (w * float(timing_scale)).astype("float64")

        band_thr = float(getattr(args, "band_threshold", 0.0))
        band_thr = 0.0 if not np.isfinite(band_thr) else max(0.0, band_thr)

        is_rebalance_day = (days_since_last_trade >= int(getattr(args, "rebalance_period"))) or (len(prev_w) == 0)
        stoploss_set: set[str] = set()
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
            logger.info(
                "stoploss_trigger date=%s n=%d worst_rank=%d thr=%d",
                date_str,
                int(len(stoploss_set)),
                int(stoploss_worst_rank if stoploss_worst_rank is not None else -1),
                int(stoploss_thr),
            )
            is_rebalance_day = True

        if not is_rebalance_day:
            output_path = os.path.join(save_dir, f"{date_str}.csv")
            if str(getattr(args, "non_rebalance_action", "empty")).lower() == "carry":
                out_w = _apply_timing(prev_w.copy())
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
                adj_scores.loc[keep_idx] = adj_scores.loc[keep_idx] * float(getattr(args, "inertia_ratio"))
        ranked_adj_codes = adj_scores.sort_values(ascending=False).index.to_numpy()

        prev_syms = set(prev_w.index.astype(str).tolist())
        industry_rank_row = None
        industry_risk_off_row = None
        industry_fast_reversal_row = None
        if industry_enabled and industry_rank_pct_df is not None and target_date in industry_rank_pct_df.index:
            try:
                industry_rank_row = industry_rank_pct_df.loc[target_date, :]
            except Exception:
                industry_rank_row = None
        if industry_enabled and industry_risk_off_df is not None and target_date in industry_risk_off_df.index:
            try:
                industry_risk_off_row = industry_risk_off_df.loc[target_date, :]
            except Exception:
                industry_risk_off_row = None
        if industry_enabled and industry_fast_reversal_df is not None and target_date in industry_fast_reversal_df.index:
            try:
                industry_fast_reversal_row = industry_fast_reversal_df.loc[target_date, :]
            except Exception:
                industry_fast_reversal_row = None
        if industry_enabled and industry_risk_off_row is not None:
            try:
                for ind, v in industry_risk_off_row.items():
                    now_off = bool(float(v) >= 0.5) if v is not None else False
                    industry_riskoff_state[str(ind)] = bool(now_off)
            except Exception:
                pass

        strong_pct = float(getattr(args, "industry_rank_strong_pct", 0.2))
        weak_pct = float(getattr(args, "industry_rank_weak_pct", 0.2))
        strong_thr = 1.0 - max(0.0, min(1.0, strong_pct))
        weak_thr = max(0.0, min(1.0, weak_pct))
        strong_max = int(getattr(args, "industry_strong_max_count", 8))
        neutral_max = int(getattr(args, "industry_neutral_max_count", 5))
        weak_max = int(getattr(args, "industry_weak_max_count", 2))
        unknown_max = int(getattr(args, "industry_unknown_max_count", 2))
        min_industries = int(getattr(args, "industry_min_industries", 4))
        riskoff_policy = str(getattr(args, "industry_riskoff_policy", "ban_new")).lower()
        bull_market = False
        if bull_df is not None and target_date in bull_df.index:
            try:
                bull_market = bool(float(bull_df.loc[target_date, "risk_on"]) >= 0.5)
            except Exception:
                bull_market = False

        def _industry_of(code: str) -> str:
            if not industry_enabled:
                return "Unknown"
            c = str(code)
            ind = code_to_industry.get(c, "Unknown")
            ind = str(ind).strip()
            return ind if ind else "Unknown"

        def _is_risk_off(industry: str) -> bool:
            if industry_risk_off_row is None:
                return False
            try:
                v = industry_risk_off_row.get(industry, 0.0)
                return bool(float(v) >= 0.5)
            except Exception:
                return False

        def _has_fast_reversal(industry: str) -> bool:
            if industry_fast_reversal_row is None:
                return False
            try:
                v = industry_fast_reversal_row.get(industry, 0.0)
                return bool(float(v) >= 0.5)
            except Exception:
                return False

        def _quota(industry: str) -> int:
            if not industry_enabled:
                return int(getattr(args, "top_k"))
            if industry == "Unknown":
                return max(0, int(unknown_max))
            if industry_rank_row is None:
                return max(0, int(neutral_max))
            try:
                rp = float(industry_rank_row.get(industry, np.nan))
            except Exception:
                rp = np.nan
            if np.isfinite(rp):
                if rp >= strong_thr:
                    return max(0, int(strong_max))
                if rp <= weak_thr:
                    return max(0, int(weak_max))
            return max(0, int(neutral_max))

        target_selected: list[str] = []
        target_set: set[str] = set()
        industry_count: dict[str, int] = {}
        fast_reversal_new_candidates: set[str] = set()
        if len(prev_w) > 0:
            top_buffer = set(ranked_adj_codes[: int(getattr(args, "buffer_k"))])
            for code in yesterday_holding_list:
                if code in top_buffer:
                    target_selected.append(code)
                    target_set.add(code)
                    ind = _industry_of(code)
                    industry_count[ind] = int(industry_count.get(ind, 0)) + 1

        slots = int(getattr(args, "top_k")) - len(target_selected)
        if slots > 0:
            for code in ranked_adj_codes:
                if slots == 0:
                    break
                if code in target_set:
                    continue
                ind = _industry_of(code)
                is_new = str(code) not in prev_syms
                if industry_enabled and _is_risk_off(ind):
                    if riskoff_policy == "ban_all":
                        continue
                    if riskoff_policy == "ban_new" and is_new:
                        if not _has_fast_reversal(ind):
                            continue
                        fast_reversal_new_candidates.add(str(code))
                q = _quota(ind)
                if industry_enabled and int(industry_count.get(ind, 0)) >= int(q):
                    continue
                target_selected.append(code)
                target_set.add(code)
                industry_count[ind] = int(industry_count.get(ind, 0)) + 1
                slots -= 1

        target_codes = target_selected[: int(getattr(args, "top_k"))]

        if industry_enabled and min_industries > 0:
            ind_set = set(_industry_of(c) for c in target_codes)
            if len(ind_set) < min_industries:
                need = int(min_industries - len(ind_set))
                rank_pos = {str(c): int(i) for i, c in enumerate(ranked_adj_codes)}
                best_code_for_ind: dict[str, str] = {}
                for c in ranked_adj_codes:
                    c = str(c)
                    if c in target_set:
                        continue
                    ind = _industry_of(c)
                    if ind in ind_set or ind in best_code_for_ind:
                        continue
                    is_new = c not in prev_syms
                    if _is_risk_off(ind):
                        if riskoff_policy == "ban_all":
                            continue
                        if riskoff_policy == "ban_new" and is_new:
                            if not _has_fast_reversal(ind):
                                continue
                            fast_reversal_new_candidates.add(str(c))
                    if int(industry_count.get(ind, 0)) >= int(_quota(ind)):
                        continue
                    best_code_for_ind[ind] = c
                    if len(best_code_for_ind) >= need:
                        break

                if len(best_code_for_ind) > 0:
                    removable = sorted(
                        [str(c) for c in target_codes],
                        key=lambda c: ((c in prev_syms), -int(rank_pos.get(str(c), 1_000_000))),
                    )
                    for ind, add_code in list(best_code_for_ind.items()):
                        if len(ind_set) >= min_industries or len(removable) == 0:
                            break
                        if add_code in target_set:
                            continue
                        if int(industry_count.get(ind, 0)) >= int(_quota(ind)):
                            continue
                        drop_code = None
                        while len(removable) > 0:
                            cand = removable.pop(0)
                            if cand in target_set:
                                drop_code = cand
                                break
                        if drop_code is None:
                            break
                        drop_ind = _industry_of(drop_code)
                        try:
                            target_selected.remove(drop_code)
                        except ValueError:
                            continue
                        target_set.discard(drop_code)
                        industry_count[drop_ind] = max(0, int(industry_count.get(drop_ind, 0)) - 1)
                        target_selected.append(add_code)
                        target_set.add(add_code)
                        industry_count[ind] = int(industry_count.get(ind, 0)) + 1
                        ind_set.add(ind)

                target_codes = target_selected[: int(getattr(args, "top_k"))]

        codes_check = set(target_codes) | prev_syms
        buyable = pd.Series(False, index=pd.Index(sorted(codes_check), dtype="string"))
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
                    cond_liquid = tv > float(getattr(args, "min_turnover", 15_000_000.0))
                else:
                    cond_liquid = pd.Series(True, index=pt.index)
                u = pd.to_numeric(pt.get("upper_limit", np.nan), errors="coerce")
                l = pd.to_numeric(pt.get("lower_limit", np.nan), errors="coerce")
                has_u = u.notna() & (u > 0)
                has_l = l.notna() & (l > 0)
                lim = (has_u & (o >= (u * (1 - 1e-12)))) | (has_l & (o <= (l * (1 + 1e-12))))
                trad = (cond_open & cond_liquid).rename("tradable")
                buy = (trad & (~lim)).rename("buyable")
                buyable.loc[buy.index.astype("string")] = buy.to_numpy(dtype=bool)
        except Exception:
            pass

        fixed_force: set[str] = set()
        if str(getattr(args, "limit_policy", "freeze")).lower() == "freeze":
            fixed_force |= {c for c in prev_syms if c in buyable.index and (not bool(buyable.loc[c]))}
            missing_px = prev_syms - set(buyable.index.tolist())
            fixed_force |= missing_px

        desired_set = set(target_codes) | fixed_force

        cap = getattr(args, "rebalance_turnover_cap", None)
        cap = 1.0 if cap is None or (not np.isfinite(float(cap))) else float(cap)
        cap = max(0.0, min(1.0, cap))
        max_new = int(np.floor(cap * int(getattr(args, "top_k"))))
        max_new = max(0, min(int(getattr(args, "top_k")), max_new))

        if len(prev_w) == 0:
            buy_candidates = [c for c in target_codes if c in buyable.index and bool(buyable.loc[c])]
            buy_keep = buy_candidates[: int(getattr(args, "top_k"))]
            sell_exec: list[str] = []
            buy_budget = 1.0
            w_next = pd.Series(dtype="float64")
        else:
            sell_candidates = [c for c in prev_syms if (c not in desired_set) or (c in stoploss_set)]
            sell_candidates = [c for c in sell_candidates if c in buyable.index and bool(buyable.loc[c])]
            sell_exec = [c for c in sell_candidates if abs(float(prev_w.get(c, 0.0))) >= band_thr]
            w_next = prev_w.drop(index=[c for c in sell_exec if c in prev_w.index], errors="ignore").copy()
            if keep_cash_idle:
                buy_budget = float(prev_w.reindex(sell_exec).sum()) if len(sell_exec) > 0 else 0.0
            else:
                buy_budget = 1.0 - float(w_next.sum()) if len(w_next) > 0 else 1.0
            if (not np.isfinite(buy_budget)) or buy_budget < 0.0:
                buy_budget = 0.0
            buy_candidates = [c for c in ranked_adj_codes if (c in desired_set and c not in prev_syms)]
            buy_candidates = [c for c in buy_candidates if c in buyable.index and bool(buyable.loc[c])]
            buy_keep = buy_candidates[:max_new]

        if buy_budget > 0.0 and len(buy_keep) > 0:
            top_k_i = int(getattr(args, "top_k"))
            rank_pos_map = {str(c): int(i) + 1 for i, c in enumerate(ranked_adj_codes)}
            buy_keep_idx = pd.Index([str(c) for c in buy_keep], dtype="string")
            pos = np.asarray([rank_pos_map.get(str(c), 1_000_000) for c in buy_keep_idx], dtype=np.int64)
            mult = np.ones_like(pos, dtype=np.float64)
            peak_l = 4
            peak_r = min(7, top_k_i)
            mid_l = 2
            mid_r = min(10, top_k_i)
            mult[pos == 1] = 0.6
            if mid_r >= mid_l:
                mult[(pos >= mid_l) & (pos <= mid_r)] = 1.25
            if peak_r >= peak_l:
                mult[(pos >= peak_l) & (pos <= peak_r)] = 1.6
            tail1_l = mid_r + 1
            tail1_r = min(15, top_k_i)
            if tail1_r >= tail1_l:
                mult[(pos >= tail1_l) & (pos <= tail1_r)] = 0.85
            mult[pos >= max(16, tail1_l)] = 0.6
            s = float(np.nansum(mult))
            if (not np.isfinite(s)) or s <= 0:
                mult = np.ones_like(mult, dtype=np.float64)
                s = float(mult.sum())
            w_buy = pd.Series((mult / s) * float(buy_budget), index=buy_keep_idx, dtype="float64")
            observe_scale = float(getattr(args, "industry_fast_reversal_observe_scale", 1.0))
            observe_scale = 1.0 if not np.isfinite(observe_scale) else max(0.0, observe_scale)
            if observe_scale < 1.0 and len(fast_reversal_new_candidates) > 0 and len(w_buy) > 0:
                m = w_buy.index.isin(list(fast_reversal_new_candidates))
                if bool(m.any()):
                    w_buy.loc[m] = (w_buy.loc[m] * observe_scale).astype("float64")
            max_w = float(getattr(args, "max_w", 1.0))
            if np.isfinite(max_w) and 0 < max_w < 1:
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
                w_next = w_buy.copy() if len(w_next) == 0 else pd.concat([w_next, w_buy])

        w_next = w_next[w_next > 0].astype("float64")
        if industry_enabled and len(w_next) > 0:
            riskoff_scale = float(getattr(args, "industry_riskoff_weight_scale", 1.0))
            riskoff_scale = 1.0 if not np.isfinite(riskoff_scale) else max(0.0, riskoff_scale)
            if riskoff_scale < 1.0 and industry_risk_off_row is not None:
                ind_by_code = pd.Series({_c: _industry_of(_c) for _c in w_next.index.astype(str)}, dtype="string")
                riskoff_inds = sorted({str(i) for i in ind_by_code.to_list() if _is_risk_off(str(i))})
                entering = [ind for ind in riskoff_inds if (not bool(industry_riskoff_state.get(str(ind), False)))]
                if len(riskoff_inds) > 0:
                    for ind in entering:
                        m = ind_by_code == ind
                        w_before = float(w_next.loc[m].sum())
                        if w_before <= 0:
                            continue
                        w_next.loc[m] = (w_next.loc[m] * riskoff_scale).astype("float64")
                        w_after = float(w_next.loc[m].sum())
                        logger.info(
                            "industry_riskoff_scale date=%s industry=%s scale=%.4f w=%.6f->%.6f",
                            date_str,
                            str(ind),
                            float(riskoff_scale),
                            float(w_before),
                            float(w_after),
                        )
                    for ind in riskoff_inds:
                        industry_riskoff_state[str(ind)] = True
                for ind in ind_by_code.unique().tolist():
                    if str(ind) not in riskoff_inds:
                        industry_riskoff_state[str(ind)] = False

            max_ind_w = float(getattr(args, "industry_max_weight", 1.0))
            max_ind_w = 1.0 if not np.isfinite(max_ind_w) else max(0.0, max_ind_w)
            cap_today = float(max_ind_w)
            if bull_market:
                bull_cap = float(getattr(args, "industry_bull_max_weight", cap_today))
                bull_cap = cap_today if (not np.isfinite(bull_cap)) else max(0.0, bull_cap)
                if bull_cap > cap_today:
                    cap_today = float(bull_cap)
            if 0 < cap_today < 1:
                ind_by_code = pd.Series({_c: _industry_of(_c) for _c in w_next.index.astype(str)}, dtype="string")
                df_w = pd.DataFrame(
                    {"w": w_next.to_numpy(dtype="float64"), "industry": ind_by_code.astype(str).to_numpy(dtype=object)},
                    index=w_next.index,
                )
                sector_w = df_w.groupby("industry", sort=False)["w"].sum()
                hit = sector_w[sector_w > float(cap_today) + 1e-12].sort_values(ascending=False)
                for ind, w_before in hit.items():
                    w_before = float(w_before)
                    if w_before <= 0:
                        continue
                    scale = float(cap_today) / w_before
                    m = df_w["industry"] == ind
                    w_next.loc[m.to_numpy(dtype=bool)] = (w_next.loc[m.to_numpy(dtype=bool)] * scale).astype("float64")
                    w_after = float(w_next.loc[m.to_numpy(dtype=bool)].sum())
                    logger.info(
                        "industry_weight_cap date=%s industry=%s cap=%.4f scale=%.6f w=%.6f->%.6f",
                        date_str,
                        str(ind),
                        float(cap_today),
                        float(scale),
                        float(w_before),
                        float(w_after),
                    )

        w_next_sum = float(w_next.sum()) if len(w_next) > 0 else 0.0
        if np.isfinite(w_next_sum) and w_next_sum > 1 + 1e-10:
            w_next = (w_next / w_next_sum).astype("float64")

        prev_w = w_next
        yesterday_holding_list = list(prev_w.index.astype(str).tolist()) if len(prev_w) > 0 else []
        days_since_last_trade = 1

        output_path = os.path.join(save_dir, f"{date_str}.csv")
        out_w = _apply_timing(prev_w)
        if len(out_w) == 0:
            with open(output_path, "w", encoding="utf-8"):
                pass
        else:
            out_df = out_w.rename_axis("code").rename("weight").reset_index()
            out_df.to_csv(output_path, index=False, header=False, float_format="%.10f")
