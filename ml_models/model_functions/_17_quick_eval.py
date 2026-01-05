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
    fig_dual_equity_png: Path
    fig_rank_perf_png: Path


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


def _col_as_series(df_price: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df_price.columns:
        s = df_price[col]
        if isinstance(s, pd.Series):
            return s
    return pd.Series(default, index=df_price.index, name=col)


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


def _get_exec_price_series(df_price: pd.DataFrame, *, price_method: str) -> pd.Series:
    method = str(price_method).lower().strip()
    if method == "vwap":
        if "vwap" in df_price.columns:
            raw = _col_as_series(df_price, "vwap")
        elif ("turnover" in df_price.columns) and ("volume" in df_price.columns):
            vol = pd.to_numeric(_col_as_series(df_price, "volume"), errors="coerce").astype(float)
            amt = pd.to_numeric(_col_as_series(df_price, "turnover"), errors="coerce").astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                raw = amt / vol
        elif ("amount" in df_price.columns) and ("volume" in df_price.columns):
            vol = pd.to_numeric(_col_as_series(df_price, "volume"), errors="coerce").astype(float)
            amt = pd.to_numeric(_col_as_series(df_price, "amount"), errors="coerce").astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                raw = amt / vol
        else:
            raw = _col_as_series(df_price, "open")
    else:
        raw = _col_as_series(df_price, "open")

    s = pd.to_numeric(raw, errors="coerce").astype(float)
    s = s.where(s > 0)
    return s


def _get_limit_series(df_price: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    upper = pd.to_numeric(_col_as_series(df_price, "upper_limit"), errors="coerce").astype(float)
    lower = pd.to_numeric(_col_as_series(df_price, "lower_limit"), errors="coerce").astype(float)
    upper = upper.where(upper > 0)
    lower = lower.where(lower > 0)
    return upper, lower


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


def _compute_self_eq_ret(df_price: pd.DataFrame, start_s: str, end_s: str) -> pd.Series:
    close = pd.to_numeric(_col_as_series(df_price, "close"), errors="coerce").astype(float)
    close = close.where(close > 0)
    ret = close.groupby(level="code").pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    bench_ret = ret.groupby(level="date").mean().sort_index().fillna(0.0).astype("float64")
    start_dt = pd.to_datetime(start_s, format="%Y%m%d", errors="coerce")
    end_dt = pd.to_datetime(end_s, format="%Y%m%d", errors="coerce")
    if not pd.isna(start_dt) and not pd.isna(end_dt):
        bench_ret = bench_ret.loc[(bench_ret.index >= start_dt) & (bench_ret.index <= end_dt)]
    return bench_ret


def _plot_dual_equity_curve(
    dates: pd.DatetimeIndex,
    equity: pd.Series,
    bench1_equity: pd.Series,
    bench1_name: str,
    bench2_equity: pd.Series,
    bench2_name: str,
    out_path: Path,
    win_rates: dict[str, str] = {},
) -> None:
    plt.figure(figsize=(14, 10), dpi=120)
    
    # Subplot 1: Equity Curves
    ax1 = plt.subplot(2, 1, 1)
    x = dates
    y = equity.to_numpy()
    ax1.plot(x, y, label="Strategy", linewidth=2.0, color="#1f77b4", zorder=5)
    
    if len(bench1_equity) == len(equity):
        ax1.plot(x, bench1_equity.to_numpy(), label=f"{bench1_name}", linewidth=1.5, color="#ff7f0e", linestyle="--", alpha=0.9)
    if len(bench2_equity) == len(equity):
        ax1.plot(x, bench2_equity.to_numpy(), label=f"{bench2_name}", linewidth=1.5, color="#2ca02c", linestyle="-.", alpha=0.9)
        
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_title("Dual Benchmark Equity Comparison")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left")
    
    # Add win rate text
    if win_rates:
        info_text = "\n".join([f"{k}: {v}" for k, v in win_rates.items()])
        # Place text in bottom right of the plot area
        ax1.text(0.98, 0.02, info_text, transform=ax1.transAxes, fontsize=10, 
                 verticalalignment='bottom', horizontalalignment='right', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Subplot 2: Relative Strength
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    
    if len(bench1_equity) == len(equity):
        rel1 = equity / bench1_equity
        ax2.plot(x, rel1.to_numpy(), label=f"Relative to {bench1_name}", linewidth=1.5, color="#ff7f0e")
        
    if len(bench2_equity) == len(equity):
        rel2 = equity / bench2_equity
        ax2.plot(x, rel2.to_numpy(), label=f"Relative to {bench2_name}", linewidth=1.5, color="#2ca02c")
        
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_title("Relative Strength (Strategy / Benchmark)")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper left")
    
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
        fig_dual_equity_png=base / "dual_equity.png",
        fig_rank_perf_png=base / "rank_performance.png",
    )


def _compute_rank_performance(
    *,
    weights_dir: Path,
    df_price: pd.DataFrame,
    top_n: int,
    risk_free_annual: float,
    price_method: str,
) -> tuple[pd.DataFrame, pd.Series]:
    top_n = int(top_n)
    if top_n <= 0:
        return pd.DataFrame(), pd.Series(dtype="float64")

    items = _list_weight_files(weights_dir)
    if len(items) == 0:
        return pd.DataFrame(), pd.Series(dtype="float64")

    px = _get_exec_price_series(df_price, price_method=price_method)
    px_next = px.groupby(level="code").shift(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rr_full = px_next / px - 1.0
    rr_full = rr_full.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    rows: list[dict[str, object]] = []
    ic_rows: dict[pd.Timestamp, float] = {}
    last_w = pd.Series(dtype="float64")

    for d, fp in items:
        w_today = _read_weight_csv(fp)
        if len(w_today) > 0:
            last_w = w_today
        w = last_w
        if len(w) == 0:
            continue

        w_sorted = w.sort_values(ascending=False)
        codes = w_sorted.index.astype("string").tolist()[:top_n]
        if len(codes) == 0:
            continue

        m = pd.MultiIndex.from_product([[d], codes], names=["date", "code"])
        rr = rr_full.reindex(m).to_numpy(dtype=float)
        rr = np.where(np.isfinite(rr), rr, 0.0)
        ww = w_sorted.reindex(pd.Index(codes, dtype="string")).to_numpy(dtype=float)
        ww = np.where(np.isfinite(ww), ww, 0.0)

        if len(rr) >= 3:
            rank_num = np.arange(1, len(rr) + 1, dtype="float64")
            ic = pd.Series(-rank_num).corr(pd.Series(rr), method="spearman")
            if ic is not None and np.isfinite(float(ic)):
                ic_rows[pd.to_datetime(d)] = float(ic)

        for i, code in enumerate(codes):
            rows.append(
                {
                    "date": pd.to_datetime(d),
                    "rank": int(i + 1),
                    "code": str(code),
                    "weight": float(ww[i]) if i < len(ww) else float("nan"),
                    "ret": float(rr[i]) if i < len(rr) else float("nan"),
                }
            )

    if len(rows) == 0:
        return pd.DataFrame(), pd.Series(ic_rows).sort_index()

    df = pd.DataFrame(rows).dropna(subset=["date", "rank", "ret"])
    if df.empty:
        return pd.DataFrame(), pd.Series(ic_rows).sort_index()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["rank", "date"]).reset_index(drop=True)

    metrics_rows: list[dict[str, object]] = []
    for r in range(1, int(top_n) + 1):
        s = df.loc[df["rank"] == r, ["date", "ret"]].dropna().drop_duplicates(subset=["date"]).set_index("date")["ret"].sort_index()
        if len(s) == 0:
            continue
        equity, met = _compute_equity_and_metrics(s, risk_free_annual=risk_free_annual)
        win_rate = float((s > 0).mean()) if len(s) else float("nan")
        mean_ret = float(s.mean()) if len(s) else float("nan")
        std_ret = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
        metrics_rows.append(
            {
                "rank": int(r),
                "n_days": int(len(s)),
                "mean_ret": mean_ret,
                "std_ret": std_ret,
                "win_rate": win_rate,
                "annualized_return": float(met.get("annualized_return", float("nan"))),
                "annualized_volatility": float(met.get("annualized_volatility", float("nan"))),
                "annualized_sharpe": float(met.get("annualized_sharpe", float("nan"))),
                "max_drawdown": float(met.get("max_drawdown", float("nan"))),
                "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else float("nan"),
            }
        )

    df_rank = pd.DataFrame(metrics_rows)
    if df_rank.empty:
        return df_rank, pd.Series(ic_rows).sort_index()

    df_rank = df_rank.sort_values("rank").reset_index(drop=True)
    return df_rank, pd.Series(ic_rows).sort_index()


def _plot_rank_performance(df_rank: pd.DataFrame, out_path: Path) -> None:
    if df_rank is None or df_rank.empty:
        plt.figure(figsize=(12, 4), dpi=120)
        plt.title("Rank Performance (no data)")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return

    d = df_rank.copy()
    d["rank"] = pd.to_numeric(d["rank"], errors="coerce").astype(int)
    d = d.sort_values("rank")
    x = d["rank"].to_numpy(dtype=int)
    ann = pd.to_numeric(d["annualized_return"], errors="coerce").to_numpy(dtype=float)
    shp = pd.to_numeric(d["annualized_sharpe"], errors="coerce").to_numpy(dtype=float)
    win = pd.to_numeric(d["win_rate"], errors="coerce").to_numpy(dtype=float)

    best_idx = int(np.nanargmax(shp)) if np.isfinite(np.nanmax(shp)) else 0
    best_rank = int(x[best_idx]) if len(x) else 1

    plt.figure(figsize=(14, 10), dpi=120)

    ax1 = plt.subplot(2, 1, 1)
    ax1.bar(x, ann, color="#4C72B0", alpha=0.85)
    ax1.axvline(best_rank, color="#cf222e", linestyle="--", linewidth=1.5, alpha=0.9)
    ax1.set_title("Single-Rank Strategy Annualized Return (per rank)")
    ax1.set_xlabel("Predicted Rank (1 = best)")
    ax1.set_ylabel("Annualized Return")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax2 = plt.subplot(2, 1, 2)
    ax2.bar(x, shp, color="#55A868", alpha=0.85, label="Sharpe")
    ax2.axvline(best_rank, color="#cf222e", linestyle="--", linewidth=1.5, alpha=0.9, label=f"Best Sharpe Rank={best_rank}")
    ax2.set_title("Single-Rank Strategy Sharpe & Win Rate (per rank)")
    ax2.set_xlabel("Predicted Rank (1 = best)")
    ax2.set_ylabel("Sharpe")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right")

    ax3 = ax2.twinx()
    ax3.plot(x, win, color="#8172B2", linewidth=1.8, marker="o", markersize=4, alpha=0.9, label="WinRate")
    ax3.set_ylabel("Win Rate")
    ax3.set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


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
    price_method: str = "open",
    exec_model: str = "limit_aware",
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
        "price_method": str(price_method),
        "exec_model": str(exec_model),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "risk_data_mtime": float(Path(risk_data_path).stat().st_mtime) if risk_data_path and os.path.exists(risk_data_path) else None,
    }

    can_reuse = False
    if enable_cache and cache_df is not None and cache_meta is not None:

        keys = ["weights_dir", "risk_data_path", "risk_index_code", "bench_method", "fee_rate", "slippage", "price_method", "exec_model"]
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

    px = _get_exec_price_series(df_price, price_method=price_method)
    px_next = px.groupby(level="code").shift(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rr_full = px_next / px - 1.0
    rr_full = rr_full.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    upper_s, lower_s = _get_limit_series(df_price)

    idx = pd.IndexSlice
    if str(bench_method).lower() in ("self_eq", "self_equal_weight", "self_equal"):
        close = pd.to_numeric(_col_as_series(df_price, "close"), errors="coerce").astype(float)
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
    last_target_w = pd.Series(dtype="float64")
    hold_w = pd.Series(dtype="float64")
    cash_w = 1.0
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

        prev_target = last_target_w
        w_today = _read_weight_csv(fp)
        rebalanced = 1 if len(w_today) > 0 else 0
        if rebalanced:
            last_target_w = w_today
        target_w = last_target_w

        exec_mode = str(exec_model).lower().strip()
        if exec_mode not in ("naive", "limit_aware"):
            exec_mode = "limit_aware"

        turnover = 0.0
        fee_cost = 0.0
        slippage_cost = 0.0
        passive_sell_cnt = 0
        reduce_cnt = 0

        if rebalanced and len(prev_target) > 0:
            all_codes = prev_target.index.union(w_today.index).astype("string")
            prev_vec = prev_target.reindex(all_codes).fillna(0.0).astype(float)
            new_vec = w_today.reindex(all_codes).fillna(0.0).astype(float)
            sold_out = (prev_vec > 0) & (new_vec <= 0)
            reduced = (prev_vec > 0) & (new_vec > 0) & (new_vec < prev_vec)
            passive_sell_cnt = int(sold_out.sum())
            reduce_cnt = int(reduced.sum())

        if exec_mode == "naive":
            if rebalanced and len(prev_target) > 0:
                all_codes = prev_target.index.union(w_today.index).astype("string")
                prev_vec = prev_target.reindex(all_codes).fillna(0.0).astype(float)
                new_vec = w_today.reindex(all_codes).fillna(0.0).astype(float)
                with np.errstate(invalid="ignore"):
                    turnover = 0.5 * float((new_vec - prev_vec).abs().sum())
                fee_cost = float(turnover * (2.0 * float(fee_rate)))
                slippage_cost = float(turnover * (2.0 * float(slippage)))
            elif rebalanced and len(prev_target) == 0 and len(w_today) > 0:
                turnover = float(w_today.sum())
                fee_cost = float(turnover * float(fee_rate))
                slippage_cost = float(turnover * float(slippage))

            if len(target_w) == 0:
                port_r = 0.0
            else:
                codes = target_w.index.astype("string").tolist()
                try:
                    m = pd.MultiIndex.from_product([[d], codes], names=["date", "code"])
                    rr = rr_full.reindex(m).to_numpy(dtype=float)
                    rr = np.where(np.isfinite(rr), rr, 0.0)
                    ww = target_w.reindex(pd.Index(codes, dtype="string")).to_numpy(dtype=float)
                    ww = np.where(np.isfinite(ww), ww, 0.0)
                    port_r = float(np.dot(ww, rr))
                except Exception:
                    port_r = 0.0
        else:
            if rebalanced and len(target_w) == 0:
                hold_w = pd.Series(dtype="float64")
                cash_w = 1.0
            elif rebalanced:
                hold_codes = hold_w.index.astype("string") if len(hold_w) else pd.Index([], dtype="string")
                tgt_codes = target_w.index.astype("string") if len(target_w) else pd.Index([], dtype="string")
                all_codes = hold_codes.union(tgt_codes).astype("string")

                prev_hold = hold_w.reindex(all_codes).fillna(0.0).astype(float)
                desired = target_w.reindex(all_codes).fillna(0.0).astype(float)

                m = pd.MultiIndex.from_product([[d], all_codes], names=["date", "code"])
                px_d = px.reindex(m)
                up_d = upper_s.reindex(m)
                lo_d = lower_s.reindex(m)

                ok_base = px_d.notna() & (px_d > 0)
                buy_ok = ok_base & (up_d.isna() | (px_d < (up_d * (1.0 - 1e-12))))
                sell_ok = ok_base & (lo_d.isna() | (px_d > (lo_d * (1.0 + 1e-12))))
                buy_ok_s = pd.Series(buy_ok.to_numpy(dtype=bool), index=all_codes)
                sell_ok_s = pd.Series(sell_ok.to_numpy(dtype=bool), index=all_codes)

                cash_now = float(cash_w)
                pos_after = prev_hold.copy()

                need_sell = (pos_after - desired).clip(lower=0.0)
                can_sell = (need_sell > 0) & sell_ok_s
                if bool(can_sell.any()):
                    sell_amt = float(need_sell.loc[can_sell].sum())
                    pos_after.loc[can_sell] = desired.loc[can_sell]
                    cash_now += sell_amt

                need_buy = (desired - pos_after).clip(lower=0.0)
                can_buy = (need_buy > 0) & buy_ok_s
                if bool(can_buy.any()) and cash_now > 1e-12:
                    need_total = float(need_buy.loc[can_buy].sum())
                    if need_total > 1e-12:
                        fill_ratio = float(min(1.0, cash_now / need_total))
                        buy_inc = need_buy.loc[can_buy] * fill_ratio
                        pos_after.loc[can_buy] = pos_after.loc[can_buy] + buy_inc
                        cash_now -= float(buy_inc.sum())

                cash_w = float(max(0.0, min(1.0, cash_now)))
                hold_w = pos_after.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                hold_w = hold_w[hold_w > 0].astype("float64")

                delta = (hold_w.reindex(all_codes).fillna(0.0) - prev_hold).astype(float)
                buy_amt = float(delta[delta > 0].sum()) if len(delta) else 0.0
                sell_amt = float((-delta[delta < 0]).sum()) if len(delta) else 0.0
                turnover = 0.5 * float(buy_amt + sell_amt)
                fee_cost = float((buy_amt + sell_amt) * float(fee_rate))
                slippage_cost = float((buy_amt + sell_amt) * float(slippage))

            if len(hold_w) == 0:
                port_r = 0.0
            else:
                codes = hold_w.index.astype("string").tolist()
                try:
                    m = pd.MultiIndex.from_product([[d], codes], names=["date", "code"])
                    rr = rr_full.reindex(m).to_numpy(dtype=float)
                    rr = np.where(np.isfinite(rr), rr, 0.0)
                    ww = hold_w.reindex(pd.Index(codes, dtype="string")).to_numpy(dtype=float)
                    ww = np.where(np.isfinite(ww), ww, 0.0)
                    port_r = float(np.dot(ww, rr))
                except Exception:
                    port_r = 0.0

        port_r_net = float(float(port_r) - float(fee_cost) - float(slippage_cost))
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
    rank_perf_html: str = "",
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
    fig_dual = _encode_img_base64(paths.fig_dual_equity_png) if paths.fig_dual_equity_png.exists() else ""
    fig_rank = _encode_img_base64(paths.fig_rank_perf_png) if paths.fig_rank_perf_png.exists() else ""
    dual_chart_html = ""
    if fig_dual:
        dual_chart_html = (
            "<div class=\"chart-card\">"
            "<h3>双基准资金曲线 (Dual Benchmark Equity)</h3>"
            f"<img src=\"{fig_dual}\"/>"
            "</div>"
        )
    rank_chart_html = ""
    if fig_rank:
        rank_chart_html = (
            "<div class=\"chart-card\">"
            "<h3>预测排名表现 (Rank Performance)</h3>"
            f"<img src=\"{fig_rank}\"/>"
            "</div>"
        )

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
    {dual_chart_html}
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
    {rank_chart_html}
  </div>

  <h2>详细数据表 (Detailed Tables)</h2>
  {win_html}
  {month_cmp}
  {passive_html}
  {rank_perf_html}

  <div class="notes">
    <strong>说明:</strong> 主基准日收益固定采用 risk_data_path 中 399006 指数的 close-to-close 涨跌幅；双基准图中额外展示 self_eq（price_data 的全样本等权 close-to-close）用于衡量与股票池的差距。策略收益采用 price_data 的执行价到下一日执行价收益（由 quick_eval_price 控制），并应用当日权重；若 quick_eval_exec_model=limit_aware，则会按涨跌停/价格缺失对调仓进行简化撮合，可能产生现金仓位。净收益已扣除基于调仓权重估算的交易成本（手续费与滑点）。
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
    price_method = str(getattr(args, "quick_eval_price", "open"))
    exec_model = str(getattr(args, "quick_eval_exec_model", "limit_aware"))

    risk_data_path = str(getattr(args, "risk_data_path"))
    primary_bench_code = "399006"
    pool_bench_code = "self_eq"
    bench_method = "index_close"

    df_daily, metrics, bench_metrics = _compute_daily_from_weights(
        weights_dir=weights_dir,
        df_price=df_price,
        risk_data_path=risk_data_path,
        risk_index_code=primary_bench_code,
        bench_method=bench_method,
        enable_cache=enable_cache,
        paths=paths,
        risk_free_annual=risk_free,
        fee_rate=fee_rate,
        slippage=slippage,
        price_method=price_method,
        exec_model=exec_model,
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

    # --- Dual Benchmark Comparison (399006 & self_eq) ---
    d_min = daily.index.min().strftime("%Y%m%d")
    d_max = daily.index.max().strftime("%Y%m%d")

    bench_399006_ret = bench.reindex(daily.index).fillna(0.0)
    bench_399006_eq = bench_eq.reindex(daily.index).ffill()

    bench_self_ret_raw = _compute_self_eq_ret(df_price, d_min, d_max)
    bench_self_ret = bench_self_ret_raw.reindex(daily.index).fillna(0.0)
    bench_self_eq, _ = _compute_equity_and_metrics(bench_self_ret, risk_free_annual=risk_free)

    win_rates_info = {}
    
    # vs 399006
    daily_win_399006 = float((daily > bench_399006_ret).mean())
    m_daily = _monthly_return_table(daily)["return"]
    m_399006 = _monthly_return_table(bench_399006_ret)["return"]
    if not m_daily.empty and not m_399006.empty:
        m_win_399006 = float((m_daily > m_399006.reindex(m_daily.index).fillna(0.0)).mean())
        win_rates_info["WinRate vs 399006"] = f"Daily={daily_win_399006:.1%}, Monthly={m_win_399006:.1%}"
    else:
        win_rates_info["WinRate vs 399006"] = f"Daily={daily_win_399006:.1%}"

    # vs self_eq
    daily_win_self = float((daily > bench_self_ret).mean())
    m_self = _monthly_return_table(bench_self_ret)["return"]
    m_daily_s = _monthly_return_table(daily)["return"]
    if not m_daily_s.empty and not m_self.empty:
        m_win_self = float((m_daily_s > m_self.reindex(m_daily_s.index).fillna(0.0)).mean())
        win_rates_info["WinRate vs self_eq"] = f"Daily={daily_win_self:.1%}, Monthly={m_win_self:.1%}"
    else:
        win_rates_info["WinRate vs self_eq"] = f"Daily={daily_win_self:.1%}"

    # 4. Plot
    _plot_dual_equity_curve(
        dates=daily.index,
        equity=equity,
        bench1_equity=bench_399006_eq,
        bench1_name=primary_bench_code,
        bench2_equity=bench_self_eq,
        bench2_name=pool_bench_code,
        out_path=paths.fig_dual_equity_png,
        win_rates=win_rates_info
    )

    win_tables = _compute_winrate_tables(daily)
    monthly_ret = _monthly_return_table(daily)
    monthly_ret_bench = _monthly_return_table(bench)

    _plot_winrate_tables(win_tables, paths.fig_winrate_png)
    _plot_equity_curve(daily.index, equity, bench_eq.reindex(equity.index).ffill(), paths.fig_equity_png)
    _plot_relative_equity(daily.index, equity, bench_eq.reindex(equity.index).ffill(), paths.fig_relative_equity_png)
    _plot_excess_hist(ex, paths.fig_excess_hist_png)
    _plot_monthly_heatmap(monthly_ret, paths.fig_heatmap_png)

    rank_top_n = int(getattr(args, "quick_eval_rank_top_n", 20))
    df_rank, rank_ic_s = _compute_rank_performance(
        weights_dir=weights_dir,
        df_price=df_price,
        top_n=rank_top_n,
        risk_free_annual=risk_free,
        price_method=price_method,
    )
    if df_rank is not None and not df_rank.empty:
        _plot_rank_performance(df_rank, paths.fig_rank_perf_png)

    rank_perf_html = ""
    if df_rank is not None and not df_rank.empty:
        view = df_rank.copy()
        view["rank"] = pd.to_numeric(view["rank"], errors="coerce").astype(int)
        view = view.sort_values("rank")
        for c in ["mean_ret", "std_ret", "annualized_return", "annualized_volatility", "max_drawdown", "total_return"]:
            if c in view.columns:
                view[c] = pd.to_numeric(view[c], errors="coerce")
        if "annualized_sharpe" in view.columns:
            view["annualized_sharpe"] = pd.to_numeric(view["annualized_sharpe"], errors="coerce")
        if "win_rate" in view.columns:
            view["win_rate"] = pd.to_numeric(view["win_rate"], errors="coerce")

        cols = [
            "rank",
            "n_days",
            "mean_ret",
            "win_rate",
            "annualized_return",
            "annualized_sharpe",
            "max_drawdown",
            "total_return",
        ]
        cols = [c for c in cols if c in view.columns]
        show = view[cols].copy()
        if "mean_ret" in show.columns:
            show["mean_ret"] = show["mean_ret"].map(lambda x: f"{x:.4%}" if np.isfinite(float(x)) else "NaN")
        if "win_rate" in show.columns:
            show["win_rate"] = show["win_rate"].map(lambda x: f"{x:.2%}" if np.isfinite(float(x)) else "NaN")
        if "annualized_return" in show.columns:
            show["annualized_return"] = show["annualized_return"].map(lambda x: f"{x:.2%}" if np.isfinite(float(x)) else "NaN")
        if "annualized_sharpe" in show.columns:
            show["annualized_sharpe"] = show["annualized_sharpe"].map(lambda x: f"{x:.3f}" if np.isfinite(float(x)) else "NaN")
        if "max_drawdown" in show.columns:
            show["max_drawdown"] = show["max_drawdown"].map(lambda x: f"{x:.2%}" if np.isfinite(float(x)) else "NaN")
        if "total_return" in show.columns:
            show["total_return"] = show["total_return"].map(lambda x: f"{x:.2%}" if np.isfinite(float(x)) else "NaN")

        ic_mean = float(rank_ic_s.mean()) if rank_ic_s is not None and len(rank_ic_s) else float("nan")
        ic_std = float(rank_ic_s.std(ddof=1)) if rank_ic_s is not None and len(rank_ic_s) > 1 else float("nan")
        ic_ir = float(ic_mean / ic_std) if np.isfinite(ic_mean) and np.isfinite(ic_std) and ic_std > 1e-12 else float("nan")
        ic_pos = float((rank_ic_s > 0).mean()) if rank_ic_s is not None and len(rank_ic_s) else float("nan")
        ic_n = int(len(rank_ic_s)) if rank_ic_s is not None else 0

        best_by_sharpe = None
        if "annualized_sharpe" in df_rank.columns:
            tmp = pd.to_numeric(df_rank["annualized_sharpe"], errors="coerce")
            if tmp.notna().any():
                best_by_sharpe = int(df_rank.loc[tmp.idxmax(), "rank"])

        best_by_ann = None
        if "annualized_return" in df_rank.columns:
            tmp = pd.to_numeric(df_rank["annualized_return"], errors="coerce")
            if tmp.notna().any():
                best_by_ann = int(df_rank.loc[tmp.idxmax(), "rank"])

        summary = (
            f"<div class='muted'>rank_ic(Spearman, pred=-rank vs ret): mean={ic_mean:.4f} ir={ic_ir:.3f} pos_rate={ic_pos:.1%} n_days={ic_n}"
            + (f" | best_rank_by_sharpe={best_by_sharpe}" if best_by_sharpe is not None else "")
            + (f" | best_rank_by_annret={best_by_ann}" if best_by_ann is not None else "")
            + "</div>"
        )

        rank_perf_html = (
            "<div class='card full-width'>"
            "<h3>预测排名-盈利关系 (Rank → Return)</h3>"
            + summary
            + show.to_html(classes="tbl", escape=False, border=0, index=False)
            + "</div>"
        )

    source_info = {
        "权重目录 (Weights Dir)": str(weights_dir),
        "价格数据 (Price Data)": str(getattr(df_price, "_source_path", "InMemory")),
        "风控基准 (Risk Data)": f"{risk_data_path} (PrimaryIndex: {primary_bench_code}, PoolBench: {pool_bench_code}, BenchMethod: {bench_method})",
        "执行价格 (Exec Price)": str(price_method),
        "撮合模型 (Exec Model)": str(exec_model),
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
        rank_perf_html=rank_perf_html,
    )

    _render_cli_tables(df_daily, metrics, bench_metrics, level=level, initial_capital=initial_capital, fee_rate=fee_rate)

    logger.info("quick_eval_saved html=%s", str(paths.report_html))
    logger.info("quick_eval_saved cache=%s", str(paths.cache_parquet))
    return str(paths.report_html)
