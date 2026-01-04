from __future__ import annotations

from datetime import datetime
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

def _parse_yyyymmdd(value: str | int | None) -> pd.Timestamp | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    if pd.isna(dt):
        return None
    return dt


def _summarize_series(s: pd.Series) -> tuple[float, float, int]:
    if s is None or len(s) == 0:
        return float("nan"), float("nan"), 0
    s1 = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s1) == 0:
        return float("nan"), float("nan"), 0
    mean = float(s1.mean())
    std = float(s1.std(ddof=1)) if len(s1) > 1 else float("nan")
    ir = float(mean / std) if np.isfinite(mean) and np.isfinite(std) and std > 1e-12 else float("nan")
    return mean, ir, int(len(s1))


def _year_stats(s: pd.Series, year: int) -> tuple[float, float, int]:
    if s is None or len(s) == 0:
        return float("nan"), float("nan"), 0
    idx = pd.to_datetime(s.index, errors="coerce")
    mask = idx.year == int(year)
    return _summarize_series(s.loc[mask])


def _filter_df_by_date_range(
    df: pd.DataFrame,
    *,
    start_dt: pd.Timestamp | None,
    end_dt: pd.Timestamp | None,
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if not isinstance(df.index, pd.MultiIndex) or "date" not in df.index.names:
        return pd.DataFrame()
    d = pd.to_datetime(df.index.get_level_values("date"), errors="coerce")
    m = pd.Series(True, index=df.index)
    if start_dt is not None:
        m &= d >= start_dt
    if end_dt is not None:
        m &= d <= end_dt
    out = df.loc[m.to_numpy(dtype=bool), :]
    return out


def _daily_ic_table_from_panel(
    df_ml: pd.DataFrame,
    *,
    features: list[str],
    y_col: str,
    start_dt: pd.Timestamp | None,
    end_dt: pd.Timestamp | None,
    min_n: int,
) -> pd.DataFrame:
    if df_ml is None or len(df_ml) == 0:
        return pd.DataFrame()
    if y_col not in df_ml.columns:
        return pd.DataFrame()
    features = [f for f in features if f in df_ml.columns and f != y_col]
    if len(features) == 0:
        return pd.DataFrame()

    panel = df_ml[[y_col] + features]
    panel = _filter_df_by_date_range(panel, start_dt=start_dt, end_dt=end_dt)
    if len(panel) == 0:
        return pd.DataFrame(columns=features, dtype="float64")

    rows: list[pd.DataFrame] = []
    for d, g in panel.groupby(level="date", sort=True):
        gg = g.replace([np.inf, -np.inf], np.nan)
        if gg[y_col].notna().sum() < int(min_n):
            continue
        ranked = gg.rank(axis=0, method="average", na_option="keep")
        ic = ranked[features].corrwith(ranked[y_col], method="pearson")
        mask_y = gg[y_col].notna()
        cnt = gg[features].notna().mul(mask_y, axis=0).sum(axis=0)
        ic = ic.where(cnt >= int(min_n))
        rows.append(pd.DataFrame([ic.to_numpy(dtype=float)], index=[d], columns=features))

    if len(rows) == 0:
        return pd.DataFrame(columns=features, dtype="float64")
    out = pd.concat(rows, axis=0)
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.sort_index()
    return out


def _summarize_ic_table(
    ic_df: pd.DataFrame,
    *,
    years: tuple[int, int],
    weak_thr: float = 0.01,
) -> pd.DataFrame:
    if ic_df is None or ic_df.empty:
        return pd.DataFrame()
    y1, y2 = int(years[0]), int(years[1])

    rows: list[dict] = []
    for f in ic_df.columns.astype(str).tolist():
        s_all = pd.to_numeric(ic_df[f], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        m_all, ir_all, n_all = _summarize_series(s_all)
        m1, ir1, n1 = _year_stats(s_all, y1)
        m2, ir2, n2 = _year_stats(s_all, y2)
        pos_rate = float((s_all > 0).mean()) if len(s_all) else float("nan")

        direction = "弱"
        if np.isfinite(m_all) and abs(float(m_all)) >= float(weak_thr):
            signs: list[int] = []
            for mm in (m1, m2):
                if np.isfinite(mm) and abs(float(mm)) >= float(weak_thr):
                    signs.append(1 if float(mm) > 0 else -1)
            if len(signs) >= 2 and any(sg != signs[0] for sg in signs[1:]):
                direction = "冲突"
            else:
                direction = "趋势" if float(m_all) > 0 else "反转"

        rows.append(
            {
                "factor": f,
                "ic_mean": m_all,
                "ir": ir_all,
                "n_days": n_all,
                f"{y1}ic": m1,
                f"{y1}ir": ir1,
                f"{y1}n": n1,
                f"{y2}ic": m2,
                f"{y2}ir": ir2,
                f"{y2}n": n2,
                "ic_pos_rate": pos_rate,
                "direction": direction,
            }
        )

    out = pd.DataFrame(rows)
    out["abs_ic_mean"] = pd.to_numeric(out["ic_mean"], errors="coerce").abs()
    out = out.sort_values(["abs_ic_mean", "ir"], ascending=[False, False]).reset_index(drop=True)
    out = out.drop(columns=["abs_ic_mean"])
    return out


def _compute_model_daily_from_temp_dir(
    df_ml: pd.DataFrame,
    *,
    temp_dir: str | None,
    y_col: str,
    start_dt: pd.Timestamp | None,
    end_dt: pd.Timestamp | None,
    min_n: int,
    top_k: int,
) -> tuple[pd.Series, pd.Series]:
    if df_ml is None or len(df_ml) == 0:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")
    if not temp_dir or (not os.path.exists(str(temp_dir))):
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")

    files = sorted(glob.glob(os.path.join(str(temp_dir), "*.parquet")))
    ic_rows: dict[pd.Timestamp, float] = {}
    top_rows: dict[pd.Timestamp, float] = {}

    for fp in files:
        date_str = os.path.basename(fp).replace(".parquet", "")
        d = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
        if pd.isna(d):
            continue
        if start_dt is not None and d < start_dt:
            continue
        if end_dt is not None and d > end_dt:
            continue

        df_s = pd.read_parquet(fp)
        if "code" not in df_s.columns or "score" not in df_s.columns:
            continue
        df_s["code"] = df_s["code"].astype(str)
        s = df_s.dropna(subset=["code", "score"]).drop_duplicates(subset=["code"]).set_index("code")["score"]
        if len(s) < max(int(min_n), int(top_k), 20):
            continue
        try:
            y_day = df_ml.xs(d, level="date")[y_col]
        except Exception:
            continue
        y = pd.to_numeric(y_day, errors="coerce").reindex(s.index)

        df = pd.concat(
            [
                pd.to_numeric(s, errors="coerce").rename("pred"),
            ],
            axis=1,
            sort=True,
        ).replace([np.inf, -np.inf], np.nan)
        df = df.join(y.rename("y"), how="inner").dropna()
        if len(df) < max(int(min_n), int(top_k), 20):
            continue

        ranked = df.rank(axis=0, method="average", na_option="keep")
        ic = ranked["pred"].corr(ranked["y"], method="pearson")
        if ic is not None and np.isfinite(float(ic)):
            ic_rows[d] = float(ic)

        sel = df.nlargest(int(top_k), columns="pred")["y"]
        if len(sel) > 0:
            m = float(sel.mean())
            if np.isfinite(m):
                top_rows[d] = m

    return pd.Series(ic_rows).sort_index(), pd.Series(top_rows).sort_index()


def generate_factors_quick_review(
    df_imp: pd.DataFrame,
    df_ml: pd.DataFrame,
    *,
    years: tuple[int, int] = (2023, 2024),
    min_n: int = 300,
    review_start: str | int | None = "20230101",
    review_end: str | int | None = "20241231",
) -> pd.DataFrame:
    if df_ml is None or len(df_ml) == 0:
        return pd.DataFrame()
    if "ret_next" not in df_ml.columns:
        return pd.DataFrame()

    start_dt = _parse_yyyymmdd(review_start)
    end_dt = _parse_yyyymmdd(review_end)

    numeric_cols = df_ml.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c != "ret_next"]
    ic_df = _daily_ic_table_from_panel(
        df_ml,
        features=features,
        y_col="ret_next",
        start_dt=start_dt,
        end_dt=end_dt,
        min_n=int(min_n),
    )
    stats = _summarize_ic_table(ic_df, years=years)
    if stats.empty:
        return pd.DataFrame()

    sub = _filter_df_by_date_range(df_ml[features], start_dt=start_dt, end_dt=end_dt)
    if len(sub) > 0:
        mr = sub.isna().mean().rename("missing_rate").reset_index()
        mr.columns = ["factor", "missing_rate"]
        stats = stats.merge(mr, on="factor", how="left")
    else:
        stats["missing_rate"] = float("nan")

    gain_map: dict[str, float] = {}
    if df_imp is not None and len(df_imp) > 0 and "feature" in df_imp.columns and "xgb_gain" in df_imp.columns:
        tmp = df_imp[["feature", "xgb_gain"]].copy()
        tmp["feature"] = tmp["feature"].astype(str)
        tmp["xgb_gain"] = pd.to_numeric(tmp["xgb_gain"], errors="coerce")
        gain_map = dict(zip(tmp["feature"].to_list(), tmp["xgb_gain"].to_list()))
    stats["xgbgain"] = stats["factor"].map(lambda x: float(gain_map.get(str(x), float("nan"))))
    return stats


def write_factors_quick_review(
    df_imp: pd.DataFrame | None,
    df_ml: pd.DataFrame | None,
    *,
    meta: dict | None = None,
    output_path: str | None = None,
    years: tuple[int, int] = (2023, 2024),
    min_n: int = 300,
    model_min_n: int = 20,
    review_start: str | int | None = "20230101",
    review_end: str | int | None = "20241231",
    temp_dir: str | None = None,
    top_k: int = 30,
    show_top_n: int = 20,
) -> str:
    out_path = Path(output_path) if output_path else (Path(__file__).resolve().parent.parent / "factors_quick_review.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _ = show_top_n

    df_stats = generate_factors_quick_review(
        df_imp if df_imp is not None else pd.DataFrame(),
        df_ml if df_ml is not None else pd.DataFrame(),
        years=years,
        min_n=min_n,
        review_start=review_start,
        review_end=review_end,
    )

    lines: list[str] = []
    lines.append(f"updated_at: {datetime.now().isoformat(timespec='seconds')}")

    rs = _parse_yyyymmdd(review_start)
    re = _parse_yyyymmdd(review_end)
    rs_s = rs.strftime("%Y%m%d") if rs is not None else ""
    re_s = re.strftime("%Y%m%d") if re is not None else ""
    if rs_s and re_s:
        lines.append(f"review_range: {rs_s} ~ {re_s}")

    lines.append(
        f"target: ret_next (df_ml label), ic=Spearman | single_factor_min_n={int(min_n)} | multi_factor_min_n={int(model_min_n)}"
    )

    lines.append("")
    y1, y2 = int(years[0]), int(years[1])

    model_ic_s, _ = _compute_model_daily_from_temp_dir(
        df_ml if df_ml is not None else pd.DataFrame(),
        temp_dir=temp_dir,
        y_col="ret_next",
        start_dt=rs,
        end_dt=re,
        min_n=int(model_min_n),
        top_k=int(top_k),
    )
    if len(model_ic_s) > 0:
        m1, ir1, n1 = _year_stats(model_ic_s, y1)
        m2, ir2, n2 = _year_stats(model_ic_s, y2)
        lines.append(
            f"multi_factor(model_score)_ic: {y1} mean={m1: .6f} ir={ir1: .3f} n_days={n1} | {y2} mean={m2: .6f} ir={ir2: .3f} n_days={n2}"
        )
    else:
        lines.append(
            f"multi_factor(model_score)_ic: {y1} mean=nan ir=nan n_days=0 | {y2} mean=nan ir=nan n_days=0"
        )

    if meta is not None:
        rank_meta = meta.get("rank_perf") if isinstance(meta, dict) else None
        if rank_meta is None and isinstance(meta, dict):
            rank_meta = meta.get("rank_performance")
        if rank_meta is None and isinstance(meta, dict):
            rank_meta = meta.get("quick_eval_rank_perf")
        if rank_meta is not None:
            lines.append("")
            lines.append("rank_performance:")
            if isinstance(rank_meta, dict):
                for k in sorted(rank_meta.keys(), key=lambda x: str(x)):
                    lines.append(f"  {k}: {rank_meta.get(k)}")
            else:
                lines.append(f"  {rank_meta}")

    lines.append("")
    if df_stats is None or len(df_stats) == 0:
        lines.append("factors: (empty)")
        lines.append("(empty)")
    else:
        cols = [
            "factor",
            "ic_mean",
            "ir",
            "n_days",
            "xgbgain",
            "missing_rate",
            "ic_pos_rate",
            "direction",
        ]
        cols = [c for c in cols if c in df_stats.columns]

        df_view = df_stats.copy()
        df_view["_abs_ic_mean"] = pd.to_numeric(df_view["ic_mean"], errors="coerce").abs()
        if "xgbgain" in df_view.columns:
            df_view["_xgbgain_sort"] = pd.to_numeric(df_view["xgbgain"], errors="coerce")
        else:
            df_view["_xgbgain_sort"] = float("nan")
        df_view = df_view.sort_values(["_abs_ic_mean", "_xgbgain_sort"], ascending=[False, False]).drop(
            columns=["_abs_ic_mean", "_xgbgain_sort"],
            errors="ignore",
        )

        lines.append(f"factors (all {len(df_view)}), sorted by |ic_mean| then xgbgain:")
        lines.append(df_view[cols].to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out_path)
