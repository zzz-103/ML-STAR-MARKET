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
from ml_models.model_functions._07_risk_signals import load_index_dual_ma_risk_signal, load_index_ma_risk_signal


def generate_positions_with_buffer(args, temp_dir: str, save_dir: str, df_price: pd.DataFrame, logger: logging.Logger) -> None:
    objective = str(getattr(args, "xgb_objective", "reg:squarederror"))
    model_kind = "Rank" if objective.startswith("rank:") else "Regression"
    timing_method = str(getattr(args, "timing_method", "score")).lower()
    logger.info(
        "step=generate_positions top_k=%d buffer_k=%d rebalance=%d turnover_cap=%.3f inertia=%.4f smooth=%d stop_rank=%d band=%.6f max_w=%.6f min_w=%.6f non_rebalance=%s model=%s timing=%s",
        int(getattr(args, "top_k")),
        int(getattr(args, "buffer_k")),
        int(getattr(args, "rebalance_period")),
        float(getattr(args, "rebalance_turnover_cap", 1.0)),
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
    if timing_method in ("index_ma20", "index_ma_dual"):
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
        else:
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

    prev_w = pd.Series(dtype="float64")
    yesterday_holding_list: list[str] = []
    history_scores: list[pd.Series] = []
    days_since_last_trade = int(getattr(args, "rebalance_period"))
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

        curr_scores = df_today[["code", "score"]].dropna().drop_duplicates(subset=["code"]).set_index("code")["score"]
        history_scores.append(curr_scores)
        while len(history_scores) > int(getattr(args, "smooth_window")):
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
                adj_scores.loc[keep_idx] = adj_scores.loc[keep_idx] * float(getattr(args, "inertia_ratio"))
        ranked_adj_codes = adj_scores.sort_values(ascending=False).index.to_numpy()

        target_selected: list[str] = []
        target_set: set[str] = set()
        if len(prev_w) > 0:
            top_buffer = set(ranked_adj_codes[: int(getattr(args, "buffer_k"))])
            for code in yesterday_holding_list:
                if code in top_buffer:
                    target_selected.append(code)
                    target_set.add(code)

        slots = int(getattr(args, "top_k")) - len(target_selected)
        if slots > 0:
            for code in ranked_adj_codes:
                if slots == 0:
                    break
                if code in target_set:
                    continue
                target_selected.append(code)
                target_set.add(code)
                slots -= 1

        target_codes = target_selected[: int(getattr(args, "top_k"))]

        codes_check = set(target_codes) | set(prev_w.index.astype(str).tolist())
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

        prev_syms = set(prev_w.index.astype(str).tolist())
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
        w_next_sum = float(w_next.sum()) if len(w_next) > 0 else 0.0
        if np.isfinite(w_next_sum) and w_next_sum > 1 + 1e-10:
            w_next = (w_next / w_next_sum).astype("float64")

        prev_w = w_next
        yesterday_holding_list = list(prev_w.index.astype(str).tolist()) if len(prev_w) > 0 else []
        days_since_last_trade = 1

        output_path = os.path.join(save_dir, f"{date_str}.csv")
        if timing_scale <= 0.0 or len(prev_w) == 0:
            with open(output_path, "w", encoding="utf-8"):
                pass
        else:
            out_w = (prev_w * float(timing_scale)).astype("float64")
            out_df = out_w.rename_axis("code").rename("weight").reset_index()
            out_df.to_csv(output_path, index=False, header=False, float_format="%.10f")
