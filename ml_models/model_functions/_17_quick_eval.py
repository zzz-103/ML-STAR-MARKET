from __future__ import annotations

import base64
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class QuickEvalPaths:
    out_dir: Path
    cache_parquet: Path
    cache_meta_json: Path
    report_html: Path
    fig_winrate_png: Path
    fig_equity_png: Path
    fig_excess_hist_png: Path
    fig_heatmap_png: Path
    fig_relative_equity_png: Path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _ansi(color: str) -> str:
    if color == "green":
        return "\x1b[32m"
    if color == "red":
        return "\x1b[31m"
    if color == "yellow":
        return "\x1b[33m"
    if color == "bold":
        return "\x1b[1m"
    if color == "reset":
        return "\x1b[0m"
    return "\x1b[0m"


def _colorize(value: float | None, *, higher_better: bool, good: float, bad: float, fmt: str) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or not math.isfinite(value))):
        return "NaN"
    s = fmt.format(value)
    if higher_better:
        if value >= good:
            return f"{_ansi('green')}{s}{_ansi('reset')}"
        if value <= bad:
            return f"{_ansi('red')}{s}{_ansi('reset')}"
        return f"{_ansi('yellow')}{s}{_ansi('reset')}"
    if value <= good:
        return f"{_ansi('green')}{s}{_ansi('reset')}"
    if value >= bad:
        return f"{_ansi('red')}{s}{_ansi('reset')}"
    return f"{_ansi('yellow')}{s}{_ansi('reset')}"


def _read_weight_csv(path: Path) -> pd.Series:
    if (not path.exists()) or path.stat().st_size == 0:
        return pd.Series(dtype="float64")
    df = pd.read_csv(path, header=None, names=["code", "weight"], dtype={"code": "string"}, engine="c")
    if df.empty:
        return pd.Series(dtype="float64")
    w = pd.to_numeric(df["weight"], errors="coerce").astype(float)
    s = pd.Series(w.to_numpy(), index=df["code"].astype("string"), dtype="float64")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s > 0]
    if len(s) == 0:
        return pd.Series(dtype="float64")
    tot = float(s.sum())
    if np.isfinite(tot) and tot > 1.0 + 1e-10:
        s = (s / tot).astype("float64")
    return s


def _list_weight_files(weights_dir: Path) -> list[tuple[pd.Timestamp, Path]]:
    items: list[tuple[pd.Timestamp, Path]] = []
    if not weights_dir.exists():
        return items
    for fp in sorted(weights_dir.glob("*.csv")):
        stem = fp.stem.strip()
        try:
            d = pd.to_datetime(stem, format="%Y%m%d", errors="raise")
        except Exception:
            continue
        items.append((d, fp))
    items.sort(key=lambda x: x[0])
    return items


def _load_index_close_series(csv_path: str, index_code: str, start_date: str, end_date: str) -> pd.Series:
    if (not csv_path) or (not os.path.exists(csv_path)):
        return pd.Series(dtype="float64")
    base = str(index_code).strip()
    if not base:
        return pd.Series(dtype="float64")
    start_s = str(start_date).strip()
    end_s = str(end_date).strip()
    if not start_s or not end_s:
        return pd.Series(dtype="float64")

    start_dt = pd.to_datetime(start_s, format="%Y%m%d", errors="coerce")
    end_dt = pd.to_datetime(end_s, format="%Y%m%d", errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt):
        return pd.Series(dtype="float64")
    need_start = (start_dt - pd.Timedelta(days=90)).strftime("%Y%m%d")
    end_need = end_dt.strftime("%Y%m%d")

    pieces: list[pd.DataFrame] = []
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
        sub = chunk.loc[m_code, ["date", "close"]].copy()
        d = sub["date"].astype("string")
        sub = sub.loc[d.between(need_start, end_need, inclusive="both")]
        if len(sub) == 0:
            continue
        pieces.append(sub)
    if len(pieces) == 0:
        return pd.Series(dtype="float64")

    df = pd.concat(pieces, axis=0, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    if len(df) == 0:
        return pd.Series(dtype="float64")
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    s = pd.Series(close.to_numpy(), index=df["date"], dtype="float64")
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def _compute_equity_and_metrics(daily_ret: pd.Series, *, risk_free_annual: float) -> tuple[pd.Series, dict[str, float]]:
    r = pd.to_numeric(daily_ret, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    equity = (1.0 + r).cumprod()
    n = int(r.shape[0])
    if n == 0:
        return equity, {
            "total_return": float("nan"),
            "annualized_return": float("nan"),
            "max_drawdown": float("nan"),
            "annualized_volatility": float("nan"),
            "annualized_sharpe": float("nan"),
        }
    total_return = float(equity.iloc[-1] - 1.0)
    annualized_return = float(equity.iloc[-1] ** (TRADING_DAYS_PER_YEAR / n) - 1.0) if n > 0 else float("nan")
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else float("nan")
    daily_vol = float(r.std(ddof=1)) if n >= 2 else 0.0
    annualized_volatility = float(daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR))
    rf_daily = float(risk_free_annual) / TRADING_DAYS_PER_YEAR
    excess = r - rf_daily
    excess_mean = float(excess.mean()) if n >= 1 else 0.0
    annualized_sharpe = float((excess_mean / daily_vol) * math.sqrt(TRADING_DAYS_PER_YEAR)) if daily_vol > 0 else float("nan")
    return equity, {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "annualized_volatility": annualized_volatility,
        "annualized_sharpe": annualized_sharpe,
    }


def _monthly_return_table(daily_ret: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"ret": daily_ret}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["month", "return"]).set_index("month")
    df["month"] = df.index.to_period("M").astype(str)
    mret = df.groupby("month")["ret"].apply(lambda x: float((1.0 + x).prod() - 1.0))
    out = mret.rename("return").to_frame()
    return out


def _compute_winrate_tables(daily_ret: pd.Series) -> dict[str, pd.DataFrame]:
    df = pd.DataFrame({"ret": daily_ret}).dropna()
    if df.empty:
        empty = pd.DataFrame(columns=["win_rate", "n_days"])
        return {"monthly": empty, "quarter_rolling_3m": empty, "half_year": empty, "annual": empty}

    df["win"] = (df["ret"] > 0).astype(int)
    df["month"] = df.index.to_period("M")

    monthly = df.groupby("month").agg(win_rate=("win", "mean"), n_days=("win", "size"))
    monthly.index = monthly.index.astype(str)

    months = monthly.index.to_series()
    month_period = pd.PeriodIndex(months, freq="M")
    uniq_months = pd.Series(month_period.unique()).sort_values()

    rolling_rows: list[dict[str, object]] = []
    for i in range(len(uniq_months)):
        if i < 2:
            continue
        end_m = uniq_months.iloc[i]
        window = [uniq_months.iloc[i - 2], uniq_months.iloc[i - 1], end_m]
        mset = set(window)
        sub = df[df["month"].isin(mset)]
        if sub.empty:
            continue
        rolling_rows.append(
            {
                "window_end_month": str(end_m),
                "win_rate": float(sub["win"].mean()),
                "n_days": int(sub.shape[0]),
            }
        )
    quarter_roll = pd.DataFrame(rolling_rows).set_index("window_end_month") if rolling_rows else pd.DataFrame(columns=["win_rate", "n_days"])

    half = ((df.index.month - 1) // 6 + 1).astype(int)
    half_key = df.index.year.astype(str) + "H" + half.astype(str)
    half_year = df.groupby(half_key).agg(win_rate=("win", "mean"), n_days=("win", "size"))

    annual = df.groupby(df.index.year).agg(win_rate=("win", "mean"), n_days=("win", "size"))
    annual.index = annual.index.astype(str)

    return {"monthly": monthly, "quarter_rolling_3m": quarter_roll, "half_year": half_year, "annual": annual}


def _plot_winrate_tables(tables: dict[str, pd.DataFrame], out_path: Path) -> None:
    plt.figure(figsize=(14, 10), dpi=120)

    def _bar(ax, df: pd.DataFrame, title: str, max_points: int) -> None:
        if df is None or df.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return
        x = df.index.astype(str).to_list()
        y = df["win_rate"].astype(float).to_numpy()
        if len(x) > max_points:
            x = x[-max_points:]
            y = y[-max_points:]
        
        # Plot bars
        ax.bar(np.arange(len(x)), y, color="#4C72B0", alpha=0.8)
        
        # Add 50% Win Rate Line
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="50% WinRate")
        
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x, rotation=45, ha="right", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax1 = plt.subplot(2, 2, 1)
    _bar(ax1, tables.get("monthly"), "Monthly Win Rate", 36)
    ax2 = plt.subplot(2, 2, 2)
    _bar(ax2, tables.get("quarter_rolling_3m"), "Rolling 3-Month Win Rate (by days)", 36)
    ax3 = plt.subplot(2, 2, 3)
    _bar(ax3, tables.get("half_year"), "Half-Year Win Rate", 20)
    ax4 = plt.subplot(2, 2, 4)
    _bar(ax4, tables.get("annual"), "Annual Win Rate", 20)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_equity_curve(dates: pd.DatetimeIndex, equity: pd.Series, bench_equity: pd.Series | None, out_path: Path) -> None:
    plt.figure(figsize=(14, 6), dpi=120)
    
    x = dates
    y = equity.to_numpy()
    
    # Plot Strategy Line
    plt.plot(x, y, label="Strategy", linewidth=1.5, color="#1f77b4", zorder=5)
    
    # Fill areas: Green for Profit (>= 1.0), Red for Loss (< 1.0)
    plt.fill_between(x, y, 1.0, where=(y >= 1.0), interpolate=True, color="green", alpha=0.15, label="Profit Area")
    plt.fill_between(x, y, 1.0, where=(y < 1.0), interpolate=True, color="red", alpha=0.15, label="Loss Area")
    
    # Baseline
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Plot Benchmark
    if bench_equity is not None and len(bench_equity) == len(equity):
        plt.plot(x, bench_equity.to_numpy(), label="Benchmark Index", linewidth=1.5, color="#888888", alpha=0.8, linestyle="-.")

    plt.title("Equity Curve (Normalized to 1.0)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_excess_hist(excess_daily: pd.Series, out_path: Path) -> None:
    x = pd.to_numeric(excess_daily, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    plt.figure(figsize=(10, 5), dpi=120)
    if len(x) == 0:
        plt.title("Excess Return Histogram (no data)")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return
    plt.hist(x.to_numpy(), bins=60, color="#55A868", alpha=0.8)
    plt.axvline(0.0, color="#333", linewidth=1)
    plt.title("Daily Excess Return Histogram (Strategy - Benchmark)")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_monthly_heatmap(monthly_return: pd.DataFrame, out_path: Path) -> None:
    if monthly_return.empty:
        plt.figure(figsize=(10, 4), dpi=120)
        plt.title("Monthly Return Heatmap (no data)")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return
    m = monthly_return.copy()
    m.index = pd.PeriodIndex(m.index.astype(str), freq="M")
    df = m.reset_index()
    df["year"] = df["month"].dt.year
    df["mon"] = df["month"].dt.month
    pivot = df.pivot(index="year", columns="mon", values="return").sort_index()
    data = pivot.to_numpy()

    plt.figure(figsize=(14, max(3.5, 0.6 * len(pivot.index))), dpi=120)
    im = plt.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-0.15, vmax=0.15)
    plt.colorbar(im, fraction=0.02, pad=0.02, label="Monthly Return")
    plt.yticks(np.arange(len(pivot.index)), pivot.index.astype(str))
    plt.xticks(np.arange(12), [str(i) for i in range(1, 13)])
    plt.title("Monthly Return Heatmap")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_relative_equity(dates: pd.DatetimeIndex, equity: pd.Series, bench_equity: pd.Series | None, out_path: Path) -> None:
    plt.figure(figsize=(14, 6), dpi=120)
    
    if bench_equity is None or len(bench_equity) != len(equity):
        plt.text(0.5, 0.5, "Benchmark Equity Data Missing or Length Mismatch", ha="center", va="center")
        plt.title("Relative Equity Curve (Strategy / Benchmark)")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return

    eq = pd.to_numeric(equity, errors="coerce").astype("float64")
    beq = pd.to_numeric(bench_equity, errors="coerce").astype("float64")
    beq = beq.replace(0.0, np.nan)
    rel_s = (eq / beq).replace([np.inf, -np.inf], np.nan)
    if len(rel_s) > 0 and (not np.isfinite(float(rel_s.iloc[0]))):
        rel_s.iloc[0] = 1.0
    rel_s = rel_s.ffill()
    
    x = dates
    y = rel_s.to_numpy(dtype="float64")
    
    # Plot Relative Line
    plt.plot(x, y, label="Relative Strength (Strategy/Benchmark)", linewidth=1.5, color="#d62728", zorder=5)
    
    # Baseline at 1.0
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Baseline (1.0)")
    
    # Fill areas: Green for Outperform (> 1.0), Red for Underperform (< 1.0)
    plt.fill_between(x, y, 1.0, where=(y >= 1.0), interpolate=True, color="green", alpha=0.15, label="Outperform Area")
    plt.fill_between(x, y, 1.0, where=(y < 1.0), interpolate=True, color="red", alpha=0.15, label="Underperform Area")
    
    plt.title("Relative Equity Curve (Strategy vs Benchmark)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _encode_img_base64(path: Path) -> str:
    b = path.read_bytes()
    ext = path.suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else f"image/{ext}"
    return f"data:{mime};base64," + base64.b64encode(b).decode("ascii")


def _build_paths(save_dir: str, dir_name: str) -> QuickEvalPaths:
    base = Path(save_dir) / str(dir_name)
    return QuickEvalPaths(
        out_dir=base,
        cache_parquet=base / "daily_cache.parquet",
        cache_meta_json=base / "daily_cache.meta.json",
        report_html=base / "quick_eval_report.html",
        fig_winrate_png=base / "winrate.png",
        fig_equity_png=base / "equity.png",
        fig_excess_hist_png=base / "excess_hist.png",
        fig_heatmap_png=base / "monthly_heatmap.png",
        fig_relative_equity_png=base / "relative_equity.png",
    )


def _load_cache(paths: QuickEvalPaths) -> tuple[pd.DataFrame | None, dict | None]:
    if not paths.cache_parquet.exists() or not paths.cache_meta_json.exists():
        return None, None
    try:
        df = pd.read_parquet(paths.cache_parquet)
        meta = json.loads(paths.cache_meta_json.read_text(encoding="utf-8"))
        return df, meta
    except Exception:
        return None, None


def _save_cache(paths: QuickEvalPaths, df: pd.DataFrame, meta: dict) -> None:
    paths.cache_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.cache_parquet, index=False)
    paths.cache_meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")



def _compute_daily_from_weights(
    *,
    weights_dir: Path,
    df_price: pd.DataFrame,
    risk_data_path: str,
    risk_index_code: str,
    bench_method: str,
    enable_cache: bool,
    paths: QuickEvalPaths,
    risk_free_annual: float,
    fee_rate: float,
    slippage: float = 0.0,  # Added slippage parameter
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """
    Compute daily returns from weights and price data.
    
    Parameters:
    - slippage: Estimated impact cost per trade (one-way). 
                Positive value = cost (reduces return). 
                Negative value = benefit (increases return).
                Default 0.0.
    """
    items = _list_weight_files(weights_dir)
    if len(items) == 0:
        return pd.DataFrame(columns=["date", "port_ret", "bench_ret", "excess_ret"]), {}, {}

    dates = [d for d, _ in items]
    start_s = min(dates).strftime("%Y%m%d")
    end_s = max(dates).strftime("%Y%m%d")

    cache_df, cache_meta = (None, None)
    if enable_cache:
        cache_df, cache_meta = _load_cache(paths)

    meta = {
        "weights_dir": str(weights_dir),
        "price_data_path": str(getattr(df_price, "_source_path", "")),
        "risk_data_path": str(risk_data_path),
        "risk_index_code": str(risk_index_code),
        "bench_method": str(bench_method),
        "risk_free_annual": float(risk_free_annual),
        "fee_rate": float(fee_rate),
        "slippage": float(slippage),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "risk_data_mtime": float(Path(risk_data_path).stat().st_mtime) if risk_data_path and os.path.exists(risk_data_path) else None,
    }

    can_reuse = False
    if enable_cache and cache_df is not None and cache_meta is not None:

        keys = ["weights_dir", "risk_data_path", "risk_index_code", "bench_method", "fee_rate", "slippage"]
        can_reuse = all(str(cache_meta.get(k)) == str(meta.get(k)) for k in keys)

    cached_map: dict[pd.Timestamp, dict[str, float]] = {}
    if can_reuse and cache_df is not None and len(cache_df) > 0:
        try:
            cdf = cache_df.copy()
            cdf["date"] = pd.to_datetime(cdf["date"])
            for row in cdf.to_dict(orient="records"):
                d = pd.to_datetime(row["date"])
                cached_map[d] = row
        except Exception:
            cached_map = {}


    open_s = pd.to_numeric(df_price.get("open", np.nan), errors="coerce").astype(float)
    
    # Prepare Adjusted Return Series (Open-to-NextOpen with PreClose Adjustment)
    # R = (Close_t / Open_t) * (Open_{t+1} / PreClose_{t+1}) - 1
    p = df_price.copy()
    if "preclose" in p.columns:
        p_cols = ["open", "close", "preclose"]
    else:
        p_cols = ["open", "close"]
    
    for c in p_cols:
        p[c] = pd.to_numeric(p.get(c, np.nan), errors="coerce").astype(float)
    
    p["open_next"] = p.groupby(level="code")["open"].shift(-1)
    
    if "preclose" in p.columns:
        p["preclose_next"] = p.groupby(level="code")["preclose"].shift(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_intraday = p["close"] / p["open"]
            r_overnight = p["open_next"] / p["preclose_next"]
            rr_full = r_intraday * r_overnight - 1.0
    else:
        # Fallback to simple Open-to-NextOpen
        with np.errstate(divide="ignore", invalid="ignore"):
            rr_full = p["open_next"] / p["open"] - 1.0
            
    rr_full = rr_full.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    idx = pd.IndexSlice
    if str(bench_method).lower() in ("self_eq", "self_equal_weight", "self_equal"):
        close = pd.to_numeric(df_price.get("close", np.nan), errors="coerce").astype(float)
        close = close.where(close > 0)
        ret = close.groupby(level="code").pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        bench_ret = ret.groupby(level="date").mean().sort_index().fillna(0.0).astype("float64")
        start_dt = pd.to_datetime(start_s, format="%Y%m%d", errors="coerce")
        end_dt = pd.to_datetime(end_s, format="%Y%m%d", errors="coerce")
        if not pd.isna(start_dt) and not pd.isna(end_dt):
            bench_ret = bench_ret.loc[(bench_ret.index >= start_dt) & (bench_ret.index <= end_dt)]
    else:
        bench_close = _load_index_close_series(risk_data_path, risk_index_code, start_s, end_s)
        bench_ret = bench_close.pct_change().astype("float64")

    rows: list[dict[str, object]] = []
    last_w = pd.Series(dtype="float64")
    for d, fp in items:
        mtime = float(fp.stat().st_mtime) if fp.exists() else None
        size = int(fp.stat().st_size) if fp.exists() else 0

        cached = cached_map.get(d)
        if cached is not None and float(cached.get("w_mtime", -1)) == float(mtime) and int(cached.get("w_size", -1)) == int(size):
            rows.append(
                {
                    "date": d,
                    "port_ret": float(cached.get("port_ret", np.nan)),
                    "port_ret_gross": float(cached.get("port_ret_gross", np.nan)),
                    "turnover": float(cached.get("turnover", 0.0)),
                    "fee_cost": float(cached.get("fee_cost", 0.0)),
                    "rebalanced": int(cached.get("rebalanced", 0)),
                    "passive_sell_cnt": int(cached.get("passive_sell_cnt", 0)),
                    "reduce_cnt": int(cached.get("reduce_cnt", 0)),
                    "bench_ret": float(cached.get("bench_ret", np.nan)),
                    "excess_ret": float(cached.get("excess_ret", np.nan)),
                    "w_mtime": mtime,
                    "w_size": size,
                }
            )
            continue

        prev_w = last_w
        w_today = _read_weight_csv(fp)
        rebalanced = 1 if len(w_today) > 0 else 0
        if rebalanced:
            last_w = w_today
        w = last_w

        turnover = 0.0

        fee_cost = 0.0
        slippage_cost = 0.0
        passive_sell_cnt = 0
        reduce_cnt = 0
        if rebalanced and len(prev_w) > 0:
            all_codes = prev_w.index.union(w_today.index).astype("string")
            prev_vec = prev_w.reindex(all_codes).fillna(0.0).astype(float)
            new_vec = w_today.reindex(all_codes).fillna(0.0).astype(float)
            with np.errstate(invalid="ignore"):
                turnover = 0.5 * float((new_vec - prev_vec).abs().sum())
            sold_out = (prev_vec > 0) & (new_vec <= 0)
            reduced = (prev_vec > 0) & (new_vec > 0) & (new_vec < prev_vec)
            passive_sell_cnt = int(sold_out.sum())
            reduce_cnt = int(reduced.sum())
            fee_cost = float(turnover * (2.0 * float(fee_rate)))
            # Slippage on turnover (buy + sell amount) = turnover * 2 * slippage
            slippage_cost = float(turnover * (2.0 * float(slippage)))

        elif rebalanced and len(prev_w) == 0:
             # First day buy turnover = 1.0 (0 -> 1)
             # Fee is only on Buy (1.0), not Buy+Sell. 
             # However, to be conservative or match some systems that charge round-trip on entry for PnL estimation?
             # User's data: 2023 Fee ~2.3%. 
             # If I use 2.0x, I get ~2.5%. If 1.0x, I might get ~2.3%.
             turnover = 1.0
             fee_cost = float(turnover * (float(fee_rate))) # 1.0x for initial entry
             slippage_cost = float(turnover * float(slippage)) # 1.0x for initial entry

        if len(w) == 0:
            port_r = 0.0
        else:
            codes = w.index.astype("string").tolist()
            try:
                m = pd.MultiIndex.from_product([[d], codes], names=["date", "code"])
                rr = rr_full.reindex(m).to_numpy(dtype=float)
                rr = np.where(np.isfinite(rr), rr, 0.0)
                ww = w.reindex(pd.Index(codes, dtype="string")).to_numpy(dtype=float)
                ww = np.where(np.isfinite(ww), ww, 0.0)
                port_r = float(np.dot(ww, rr))
            except Exception:
                port_r = 0.0

        port_r_net = float(port_r - fee_cost - slippage_cost)
        b = float(bench_ret.loc[d]) if d in bench_ret.index else float("nan")
        ex = float(port_r_net - b) if np.isfinite(b) else float("nan")
        rows.append(
            {
                "date": d,
                "port_ret": float(port_r_net),
                "port_ret_gross": float(port_r),
                "turnover": float(turnover),
                "fee_cost": float(fee_cost),
                "rebalanced": int(rebalanced),
                "passive_sell_cnt": int(passive_sell_cnt),
                "reduce_cnt": int(reduce_cnt),
                "bench_ret": b,
                "excess_ret": ex,
                "w_mtime": mtime,
                "w_size": size,
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return pd.DataFrame(columns=["date", "port_ret", "bench_ret", "excess_ret"]), {}, {}

    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out = out.replace([np.inf, -np.inf], np.nan)

    valid = out.dropna(subset=["port_ret"])
    daily_ret = pd.Series(pd.to_numeric(valid["port_ret"], errors="coerce").to_numpy(), index=pd.to_datetime(valid["date"]))
    equity, metrics = _compute_equity_and_metrics(daily_ret, risk_free_annual=risk_free_annual)

    bench_valid = out.dropna(subset=["bench_ret"])
    bench_ret_s = pd.Series(pd.to_numeric(bench_valid["bench_ret"], errors="coerce").to_numpy(), index=pd.to_datetime(bench_valid["date"]))
    bench_eq, bench_metrics = _compute_equity_and_metrics(bench_ret_s.fillna(0.0), risk_free_annual=risk_free_annual)

    if enable_cache:
        _save_cache(paths, out, meta)
    return out, metrics, bench_metrics


def _render_cli_tables(
    df_daily: pd.DataFrame,
    metrics: dict[str, float],
    bench_metrics: dict[str, float],
    *,
    level: int,
    initial_capital: float,
    fee_rate: float,
) -> None:
    if df_daily.empty:
        print("QuickEval: empty daily series, skip.")
        return

    d0 = pd.to_datetime(df_daily["date"].min()).strftime("%Y%m%d")
    d1 = pd.to_datetime(df_daily["date"].max()).strftime("%Y%m%d")
    daily = pd.Series(df_daily["port_ret"].to_numpy(dtype=float), index=pd.to_datetime(df_daily["date"]))
    bench = pd.Series(df_daily["bench_ret"].to_numpy(dtype=float), index=pd.to_datetime(df_daily["date"]))
    ex = pd.Series(df_daily["excess_ret"].to_numpy(dtype=float), index=pd.to_datetime(df_daily["date"]))

    equity, _ = _compute_equity_and_metrics(daily, risk_free_annual=0.0)
    end_capital = float(initial_capital * float(equity.iloc[-1])) if len(equity) else float("nan")
    if "fee_cost" in df_daily.columns:
        fee_frac = pd.Series(df_daily["fee_cost"].to_numpy(dtype=float), index=pd.to_datetime(df_daily["date"])).fillna(0.0)
        cap_before = initial_capital * equity.shift(1).fillna(1.0)
        total_fee = float((fee_frac * cap_before).sum())
    else:
        total_fee = float("nan")

    win = (daily > 0).astype(int)
    overall_win = float(win.mean()) if len(win) else float("nan")

    out_day = float((ex > 0).mean()) if ex.notna().any() else float("nan")
    mret = _monthly_return_table(daily)
    mret_b = _monthly_return_table(bench.fillna(0.0))
    if (not mret.empty) and (not mret_b.empty):
        mb = mret_b.reindex(mret.index)["return"]
        out_month = float((mret["return"] > mb).mean()) if len(mb.dropna()) else float("nan")
    else:
        out_month = float("nan")

    sharpe = float(metrics.get("annualized_sharpe", float("nan")))
    ann = float(metrics.get("annualized_return", float("nan")))
    mdd = float(metrics.get("max_drawdown", float("nan")))

    b_sharpe = float(bench_metrics.get("annualized_sharpe", float("nan")))
    b_ann = float(bench_metrics.get("annualized_return", float("nan")))
    b_mdd = float(bench_metrics.get("max_drawdown", float("nan")))

    print(f"{_ansi('bold')}QuickEval {d0}~{d1}{_ansi('reset')}")
    print(
        "  "
        f"AnnRet={_colorize(ann, higher_better=True, good=0.15, bad=0.00, fmt='{:.2%}')}"
        f"  MDD={_colorize(abs(mdd), higher_better=False, good=0.20, bad=0.35, fmt='{:.2%}')}"
        f"  Sharpe={_colorize(sharpe, higher_better=True, good=1.00, bad=0.00, fmt='{:.3f}')}"
        f"  Win={_colorize(overall_win, higher_better=True, good=0.55, bad=0.50, fmt='{:.2%}')}"
    )
    print(
        "  "
        f"OutDay={_colorize(out_day, higher_better=True, good=0.55, bad=0.50, fmt='{:.2%}')}"
        f"  OutMonth={_colorize(out_month, higher_better=True, good=0.55, bad=0.50, fmt='{:.2%}')}"
    )
    print(f"  Benchmark: AnnRet={b_ann:.2%}  MDD={b_mdd:.2%}  Sharpe={b_sharpe:.3f}")
    if initial_capital > 0 and np.isfinite(end_capital):
        pnl = float(end_capital - initial_capital)
        print(f"  Capital: start={initial_capital:,.0f}  end={end_capital:,.0f}  pnl={pnl:,.0f}")
    if np.isfinite(fee_rate) and fee_rate > 0:
        if np.isfinite(total_fee):
            print(f"  Fee: rate(one-side)={fee_rate:.6f}  total_fee≈{total_fee:,.0f}")
        else:
            print(f"  Fee: rate(one-side)={fee_rate:.6f}")
    if "turnover" in df_daily.columns and df_daily["turnover"].notna().any():
        avg_turnover = float(pd.to_numeric(df_daily["turnover"], errors="coerce").fillna(0.0).mean())
        rebal_days = int(pd.to_numeric(df_daily.get("rebalanced", 0), errors="coerce").fillna(0).sum())
        print(f"  Turnover: avg={avg_turnover:.3f}  rebalanced_days={rebal_days}")
    if "passive_sell_cnt" in df_daily.columns and df_daily["passive_sell_cnt"].notna().any():
        ps = int(pd.to_numeric(df_daily["passive_sell_cnt"], errors="coerce").fillna(0).sum())
        rd = int(pd.to_numeric(df_daily.get("reduce_cnt", 0), errors="coerce").fillna(0).sum())
        print(f"  PassiveSell: sold_out_total={ps}  reduced_total={rd}")

    if level <= 0:
        return

    tables = _compute_winrate_tables(daily)
    show = {}
    for k, df in tables.items():
        if df is None or df.empty:
            continue
        show[k] = df.copy()
        show[k]["win_rate"] = (show[k]["win_rate"] * 100).round(2)
    if show:
        print("\nWinRate Tables (win_rate in %)")
        for k in ["monthly", "quarter_rolling_3m", "half_year", "annual"]:
            if k not in show:
                continue
            df = show[k]
            if len(df) > 24 and level < 2:
                df = df.tail(24)
            print(f"\n[{k}]")
            print(df.to_string())

    if level >= 2:
        if ex.notna().any():
            q = ex.dropna().quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_frame("excess_ret")
            print("\nExcess Return Quantiles")
            print((q * 100).round(3).to_string())


def _render_html_report(
    paths: QuickEvalPaths,
    *,
    df_daily: pd.DataFrame,
    metrics: dict[str, float],
    bench_metrics: dict[str, float],
    win_tables: dict[str, pd.DataFrame],
    monthly_ret: pd.DataFrame,
    monthly_ret_bench: pd.DataFrame,
    initial_capital: float,
    fee_rate: float,
    source_info: dict[str, str],
) -> None:
    d0 = pd.to_datetime(df_daily["date"].min()).strftime("%Y%m%d") if not df_daily.empty else ""
    d1 = pd.to_datetime(df_daily["date"].max()).strftime("%Y%m%d") if not df_daily.empty else ""

    def _badge(v: float | None, *, higher_better: bool, good: float, bad: float, fmt: str) -> str:
        if v is None or (isinstance(v, float) and (math.isnan(v) or not math.isfinite(v))):
            return "<span class='na'>NaN</span>"
        s = fmt.format(v)
        cls = "mid"
        if higher_better:
            cls = "good" if v >= good else ("bad" if v <= bad else "mid")
        else:
            cls = "good" if v <= good else ("bad" if v >= bad else "mid")
        return f"<span class='{cls}'>{s}</span>"

    daily = pd.Series(df_daily["port_ret"].to_numpy(dtype=float), index=pd.to_datetime(df_daily["date"]))
    bench = pd.Series(df_daily["bench_ret"].to_numpy(dtype=float), index=pd.to_datetime(df_daily["date"]))
    ex = pd.Series(df_daily["excess_ret"].to_numpy(dtype=float), index=pd.to_datetime(df_daily["date"]))
    equity, _ = _compute_equity_and_metrics(daily, risk_free_annual=0.0)
    end_capital = float(initial_capital * float(equity.iloc[-1])) if len(equity) else float("nan")
    if "fee_cost" in df_daily.columns:
        fee_frac = pd.Series(df_daily["fee_cost"].to_numpy(dtype=float), index=pd.to_datetime(df_daily["date"])).fillna(0.0)
        cap_before = initial_capital * equity.shift(1).fillna(1.0)
        total_fee = float((fee_frac * cap_before).sum())
    else:
        total_fee = float("nan")
    overall_win = float((daily > 0).mean()) if len(daily) else float("nan")
    out_day = float((ex > 0).mean()) if ex.notna().any() else float("nan")

    if (not monthly_ret.empty) and (not monthly_ret_bench.empty):
        mb = monthly_ret_bench.reindex(monthly_ret.index)["return"]
        out_month = float((monthly_ret["return"] > mb).mean()) if len(mb.dropna()) else float("nan")
    else:
        out_month = float("nan")

    fig_winrate = _encode_img_base64(paths.fig_winrate_png) if paths.fig_winrate_png.exists() else ""
    fig_equity = _encode_img_base64(paths.fig_equity_png) if paths.fig_equity_png.exists() else ""
    fig_hist = _encode_img_base64(paths.fig_excess_hist_png) if paths.fig_excess_hist_png.exists() else ""
    fig_heat = _encode_img_base64(paths.fig_heatmap_png) if paths.fig_heatmap_png.exists() else ""
    fig_rel = _encode_img_base64(paths.fig_relative_equity_png) if paths.fig_relative_equity_png.exists() else ""

    # Metrics Table
    mt = []
    mt.append("<div class='table-responsive'><table class='metrics-table'>")
    mt.append("<thead><tr><th>指标 (Metric)</th><th>策略 (Strategy)</th><th>基准 (Benchmark)</th></tr></thead><tbody>")
    
    def _row(label, val_s, val_b=None, is_header=False):
        return f"<tr><td>{label}</td><td>{val_s}</td><td>{val_b if val_b else '-'}</td></tr>"

    mt.append(_row(
        "年化收益率 (Annualized Return)",
        _badge(metrics.get('annualized_return'), higher_better=True, good=0.15, bad=0.00, fmt='{:.2%}'),
        _badge(bench_metrics.get('annualized_return'), higher_better=True, good=0.15, bad=0.00, fmt='{:.2%}')
    ))
    mt.append(_row(
        "最大回撤 (Max Drawdown)",
        _badge(abs(metrics.get('max_drawdown', float('nan'))), higher_better=False, good=0.20, bad=0.35, fmt='{:.2%}'),
        _badge(abs(bench_metrics.get('max_drawdown', float('nan'))), higher_better=False, good=0.20, bad=0.35, fmt='{:.2%}')
    ))
    mt.append(_row(
        "年化波动率 (Annualized Volatility)",
        _badge(metrics.get('annualized_volatility'), higher_better=False, good=0.20, bad=0.35, fmt='{:.2%}'),
        _badge(bench_metrics.get('annualized_volatility'), higher_better=False, good=0.20, bad=0.35, fmt='{:.2%}')
    ))
    mt.append(_row(
        "年化夏普比率 (Sharpe Ratio)",
        _badge(metrics.get('annualized_sharpe'), higher_better=True, good=1.0, bad=0.0, fmt='{:.3f}'),
        _badge(bench_metrics.get('annualized_sharpe'), higher_better=True, good=1.0, bad=0.0, fmt='{:.3f}')
    ))
    mt.append(_row(
        "日胜率 (Daily Win Rate)",
        _badge(overall_win, higher_better=True, good=0.55, bad=0.50, fmt='{:.2%}'),
        None
    ))
    mt.append(_row(
        "跑赢概率 (Outperform Rate 日/月)",
        f"{_badge(out_day, higher_better=True, good=0.55, bad=0.50, fmt='{:.2%}')} / {_badge(out_month, higher_better=True, good=0.55, bad=0.50, fmt='{:.2%}')}",
        None
    ))

    if initial_capital > 0:
        mt.append(_row("期初资金 (Initial Capital)", f"{initial_capital:,.0f}"))
        if np.isfinite(end_capital):
            cls_pnl = "good" if end_capital >= initial_capital else "bad"
            mt.append(_row("期末资金 (End Capital)", f"<span class='{cls_pnl}'>{end_capital:,.0f}</span>"))


    if np.isfinite(fee_rate) and fee_rate > 0:
        mt.append(_row("单边费率 (Fee Rate)", f"{fee_rate:.6f}"))
        if np.isfinite(total_fee):
            mt.append(_row("预估总手续费 (Total Fee)", f"{total_fee:,.0f}"))
            
    # Optional: Display slippage info if non-zero
    # Since we don't track total slippage in df_daily (yet), we just skip detail or add if we want.
    # But user didn't ask for explicit slippage display in table, just optimization.
    # We can add a row if we want.

    if "turnover" in df_daily.columns and df_daily["turnover"].notna().any():
        avg_turnover = float(pd.to_numeric(df_daily["turnover"], errors="coerce").fillna(0.0).mean())
        mt.append(_row("平均换手率 (Avg Turnover)", f"{avg_turnover:.3f}"))

    if "passive_sell_cnt" in df_daily.columns and df_daily["passive_sell_cnt"].notna().any():
        ps = int(pd.to_numeric(df_daily["passive_sell_cnt"], errors="coerce").fillna(0).sum())
        rd = int(pd.to_numeric(df_daily.get('reduce_cnt', 0), errors='coerce').fillna(0).sum())
        mt.append(_row("被动卖出 (清仓/减仓)", f"{ps} / {rd}"))
    
    mt.append("</tbody></table></div>")
    metrics_html = "\n".join(mt)

    def _df_html(df: pd.DataFrame, title: str, tail: int | None = None) -> str:
        if df is None or df.empty:
            return f"<div class='card'><h3>{title}</h3><div class='muted'>无数据 (No data)</div></div>"
        dd = df.copy()
        if tail is not None and len(dd) > tail:
            dd = dd.tail(tail)
        dd = dd.copy()
        
        # Colorize win_rate: < 50% red, >= 50% green
        def _color_win(x):
            color = "#cf222e" if x < 0.5 else "#1a7f37"
            return f"<span style='color: {color}; font-weight: bold;'>{x:.2%}</span>"
        
        if "win_rate" in dd.columns:
            dd["win_rate"] = dd["win_rate"].apply(_color_win)
            
        return f"<div class='card'><h3>{title}</h3>" + dd.to_html(classes="tbl", escape=False, border=0) + "</div>"

    win_html_list = [
        _df_html(win_tables.get("monthly"), "月度胜率 (Monthly, latest 24)", tail=24),
        _df_html(win_tables.get("quarter_rolling_3m"), "滚动季度胜率 (Rolling 3M, latest 24)", tail=24),
        _df_html(win_tables.get("half_year"), "半年度胜率 (Half-Year)", tail=None),
        _df_html(win_tables.get("annual"), "年度胜率 (Annual)", tail=None),
    ]
    win_html = "<div class='grid-2'>" + "\n".join(win_html_list) + "</div>"

    month_cmp = ""
    if (not monthly_ret.empty) and (not monthly_ret_bench.empty):
        cmp_df = monthly_ret.join(monthly_ret_bench.rename(columns={"return": "bench_return"}), how="left")
        cmp_df["excess"] = cmp_df["return"] - cmp_df["bench_return"]
        
        # Colorize excess return
        def _color_ret(x):
            if pd.isna(x): return "-"
            color = "#cf222e" if x < 0 else "#1a7f37"
            return f"<span style='color: {color}; font-weight: bold;'>{x:.2%}</span>"
            
        view = cmp_df.tail(24).copy()
        view["return"] = view["return"].apply(lambda x: f"{x:.2%}")
        view["bench_return"] = view["bench_return"].apply(lambda x: f"{x:.2%}")
        view["excess"] = view["excess"].apply(_color_ret)
        
        html_tbl = view.to_html(classes="tbl", escape=False, border=0)
        month_cmp = f"<div class='card full-width'><h3>月度收益 vs 基准 (Monthly Return vs Benchmark, latest 24)</h3>{html_tbl}</div>"
    else:
        month_cmp = "<div class='card full-width'><h3>月度收益 vs 基准</h3><div class='muted'>无数据</div></div>"

    passive_html = ""
    if "passive_sell_cnt" in df_daily.columns:
        ps_df = df_daily[["date", "passive_sell_cnt", "reduce_cnt", "rebalanced"]].copy()
        ps_df["date"] = pd.to_datetime(ps_df["date"])
        ps_df = ps_df.sort_values("date")
        ps_df["month"] = ps_df["date"].dt.to_period("M").astype(str)
        m = (
            ps_df.groupby("month")
            .agg(
                sold_out=("passive_sell_cnt", "sum"),
                reduced=("reduce_cnt", "sum"),
                rebalanced_days=("rebalanced", "sum"),
            )
            .tail(24)
        )
        
        # Colorize non-zero sold_out
        def _color_int(x):
            if x > 0:
                return f"<span style='color: #cf222e; font-weight: 600;'>{x}</span>"
            return str(x)
            
        m["sold_out"] = m["sold_out"].apply(_color_int)
        # reduced is neutral/mixed, maybe keep black or mild color
        
        passive_html = f"<div class='card full-width'><h3>被动卖出统计 (Passive Sell Counts, latest 24)</h3>{m.to_html(classes='tbl', escape=False, border=0)}</div>"

    source_html = ""
    if source_info:
        rows = []
        for k, v in source_info.items():
            rows.append(f"<div><strong>{k}:</strong> <span class='source-path'>{v}</span></div>")
        source_html = f"<div class='source-info'>{''.join(rows)}</div>"

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>QuickEval Report</title>
  <style>
    :root {{
        --primary-color: #0969da;
        --bg-color: #f6f8fa;
        --card-bg: #ffffff;
        --text-main: #24292f;
        --text-muted: #57606a;
        --border-color: #d0d7de;
        --success-color: #1a7f37;
        --danger-color: #cf222e;
        --warning-color: #9a6700;
        --table-header-bg: #f6f8fa;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: var(--bg-color);
        color: var(--text-main);
        margin: 0;
        padding: 20px;
        line-height: 1.5;
    }}
    .container {{
        max-width: 1200px;
        margin: 0 auto;
        background: var(--card-bg);
        padding: 40px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }}
    h1 {{
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 16px;
        font-size: 28px;
        margin-bottom: 24px;
        margin-top: 0;
        color: var(--text-main);
    }}
    h2 {{
        font-size: 22px;
        margin-top: 32px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-color);
        color: var(--text-main);
    }}
    h3 {{
        font-size: 16px;
        margin-top: 0;
        margin-bottom: 12px;
        color: var(--text-main);
        font-weight: 600;
    }}
    .meta-info {{
        color: var(--text-muted);
        font-size: 14px;
        margin-bottom: 20px;
        background: #f3f4f6;
        padding: 10px 14px;
        border-radius: 6px;
    }}
    .source-info {{
        color: var(--text-muted);
        font-size: 13px;
        margin-bottom: 30px;
        background: #fff;
        border: 1px solid #eee;
        padding: 10px 14px;
        border-radius: 6px;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        word-break: break-all;
    }}
    .source-path {{
        color: #0969da;
    }}
    
    /* Metrics Table */
    .metrics-table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
    }}
    .metrics-table th, .metrics-table td {{
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid var(--border-color);
    }}
    .metrics-table th {{
        background-color: var(--table-header-bg);
        font-weight: 600;
        color: var(--text-muted);
        width: 40%;
    }}
    .metrics-table td {{
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 14px;
        width: 30%;
    }}
    .metrics-table tr:last-child td {{
        border-bottom: none;
    }}

    /* Data Tables */
    .tbl {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}
    .tbl th, .tbl td {{
        padding: 8px 12px;
        border: 1px solid var(--border-color);
        text-align: right;
    }}
    .tbl th {{
        background-color: var(--table-header-bg);
        font-weight: 600;
        text-align: center;
    }}
    .tbl tr:nth-child(even) {{
        background-color: #fcfcfc;
    }}
    .tbl tr:hover {{
        background-color: #f8f9fa;
    }}
    
    /* Colors */
    .good {{ color: var(--success-color); font-weight: 600; }}
    .bad {{ color: var(--danger-color); font-weight: 600; }}
    .mid {{ color: var(--warning-color); font-weight: 600; }}
    .na {{ color: #888; }}
    
    /* Layout */
    .grid-2 {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-bottom: 20px;
    }}
    .card {{
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 16px;
        background: #fff;
    }}
    .full-width {{
        grid-column: 1 / -1;
        margin-bottom: 20px;
    }}
    
    /* Images */
    .img-container {{
        display: flex;
        flex-direction: column;
        gap: 24px;
    }}
    .chart-card {{
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 16px;
        background: #fff;
        text-align: center;
    }}
    .chart-card img {{
        max-width: 100%;
        height: auto;
        border-radius: 4px;
    }}
    
    /* Responsive */
    @media (max-width: 768px) {{
        .container {{ padding: 20px; }}
        .grid-2 {{ grid-template-columns: 1fr; }}
        .metrics-table th, .metrics-table td {{ display: block; width: auto; }}
        .metrics-table th {{ background: transparent; padding-bottom: 4px; color: var(--text-muted); border-bottom: none; }}
        .metrics-table td {{ padding-top: 0; border-bottom: 1px solid var(--border-color); }}
    }}
    
    .notes {{
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid var(--border-color);
        color: var(--text-muted);
        font-size: 13px;
    }}
  </style>
</head>
<body>
<div class="container">
  <h1>快速评估报告 (QuickEval Report)</h1>
  <div class="meta-info">
    <strong>回测区间 (Period):</strong> {d0} ~ {d1} &nbsp;|&nbsp; 
    <strong>生成时间 (Generated):</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
  </div>
  
  {source_html}

  <h2>核心指标摘要 (Summary Metrics)</h2>
  {metrics_html}

  <h2>图表分析 (Charts)</h2>
  <div class="img-container">
    <div class="chart-card">
        <h3>资金曲线 (Equity Curve)</h3>
        <img src="{fig_equity}"/>
    </div>
    <div class="chart-card">
        <h3>相对净值曲线 (Relative Equity vs Benchmark)</h3>
        <img src="{fig_rel}"/>
    </div>
    <div class="chart-card">
        <h3>胜率分布 (Win Rate)</h3>
        <img src="{fig_winrate}"/>
    </div>
    <div class="chart-card">
        <h3>超额收益直方图 (Excess Return Histogram)</h3>
        <img src="{fig_hist}"/>
    </div>
    <div class="chart-card">
        <h3>月度收益热力图 (Monthly Return Heatmap)</h3>
        <img src="{fig_heat}"/>
    </div>
  </div>

  <h2>详细数据表 (Detailed Tables)</h2>
  {win_html}
  {month_cmp}
  {passive_html}

  <div class="notes">
    <strong>说明:</strong> 基准日收益：当 bench_method=self_eq 时，采用 price_data_path 中全样本等权 close-to-close 平均涨跌幅；否则采用 risk_data_path 指定指数的收盘价涨跌幅。策略收益采用 price_data_path 数据的 open-to-next-open 收益，并应用当日权重。净收益已扣除基于调仓权重的双边交易手续费。
  </div>
</div>
</body>
</html>
"""
    paths.report_html.write_text(html.strip(), encoding="utf-8")


def run_quick_evaluation(*, args, save_dir: str, df_price: pd.DataFrame, logger) -> str | None:
    weights_dir = Path(save_dir)
    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        output_dir = str(weights_dir.parent)
    paths = _build_paths(
        str(Path(str(output_dir)) / str(getattr(args, "quick_eval_dir_name", "quick_eval")) / str(weights_dir.name)),
        "",
    )
    _ensure_dir(paths.out_dir)

    level = int(getattr(args, "quick_eval_level", 1))

    enable_cache = bool(getattr(args, "quick_eval_cache", True))
    risk_free = float(getattr(args, "quick_eval_risk_free", 0.0))
    fee_rate = float(getattr(args, "quick_eval_fee_rate", getattr(args, "quick_eval_fee", 0.0)))
    slippage = float(getattr(args, "quick_eval_slippage", 0.0))
    initial_capital = float(getattr(args, "quick_eval_capital", 10_000_000.0))

    risk_data_path = str(getattr(args, "risk_data_path"))
    risk_index_code = str(getattr(args, "risk_index_code"))
    timing_method = str(getattr(args, "timing_method", "index_ma20"))
    bench_method = "self_eq" if (timing_method == "self_eq_ma20" or str(risk_index_code).lower() in ("self_eq", "self_equal_weight", "self_equal")) else "index_close"

    df_daily, metrics, bench_metrics = _compute_daily_from_weights(
        weights_dir=weights_dir,
        df_price=df_price,
        risk_data_path=risk_data_path,
        risk_index_code=risk_index_code,
        bench_method=bench_method,
        enable_cache=enable_cache,
        paths=paths,
        risk_free_annual=risk_free,
        fee_rate=fee_rate,
        slippage=slippage,
    )

    if df_daily.empty:
        logger.info("quick_eval_empty skip=1")
        return None

    df_daily = df_daily.dropna(subset=["date", "port_ret"]).copy()
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.sort_values("date").reset_index(drop=True)

    daily = pd.Series(df_daily["port_ret"].to_numpy(dtype=float), index=pd.to_datetime(df_daily["date"]))
    bench = pd.Series(df_daily["bench_ret"].to_numpy(dtype=float), index=pd.to_datetime(df_daily["date"])).fillna(0.0)
    ex = pd.Series(df_daily["excess_ret"].to_numpy(dtype=float), index=pd.to_datetime(df_daily["date"]))

    equity, _ = _compute_equity_and_metrics(daily, risk_free_annual=risk_free)
    bench_eq, _ = _compute_equity_and_metrics(bench, risk_free_annual=risk_free)

    win_tables = _compute_winrate_tables(daily)
    monthly_ret = _monthly_return_table(daily)
    monthly_ret_bench = _monthly_return_table(bench)

    _plot_winrate_tables(win_tables, paths.fig_winrate_png)
    _plot_equity_curve(daily.index, equity, bench_eq.reindex(equity.index).ffill(), paths.fig_equity_png)
    _plot_relative_equity(daily.index, equity, bench_eq.reindex(equity.index).ffill(), paths.fig_relative_equity_png)
    _plot_excess_hist(ex, paths.fig_excess_hist_png)
    _plot_monthly_heatmap(monthly_ret, paths.fig_heatmap_png)

    source_info = {
        "权重目录 (Weights Dir)": str(weights_dir),
        "价格数据 (Price Data)": str(getattr(df_price, "_source_path", "InMemory")),
        "风控基准 (Risk Data)": f"{risk_data_path} (Index: {risk_index_code}, BenchMethod: {bench_method})",
    }

    _render_html_report(
        paths,
        df_daily=df_daily,
        metrics=metrics,
        bench_metrics=bench_metrics,
        win_tables=win_tables,
        monthly_ret=monthly_ret,
        monthly_ret_bench=monthly_ret_bench,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        source_info=source_info,
    )

    _render_cli_tables(df_daily, metrics, bench_metrics, level=level, initial_capital=initial_capital, fee_rate=fee_rate)

    logger.info("quick_eval_saved html=%s", str(paths.report_html))
    logger.info("quick_eval_saved cache=%s", str(paths.cache_parquet))
    return str(paths.report_html)
