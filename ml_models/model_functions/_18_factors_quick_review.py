from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml_models.model_functions._10_evaluation import daily_ic


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


def generate_factors_quick_review(
    df_imp: pd.DataFrame,
    df_ml: pd.DataFrame,
    *,
    model_daily_ic: pd.Series | None = None,
    years: tuple[int, int] = (2023, 2024),
    top_n: int = 20,
    min_n: int = 300,
) -> pd.DataFrame:
    if df_imp is None or len(df_imp) == 0:
        return pd.DataFrame()
    if "feature" not in df_imp.columns:
        return pd.DataFrame()
    if "xgb_gain" not in df_imp.columns:
        return pd.DataFrame()
    if df_ml is None or len(df_ml) == 0:
        return pd.DataFrame()
    if "ret_next" not in df_ml.columns:
        return pd.DataFrame()

    y_all = pd.to_numeric(df_ml["ret_next"], errors="coerce")
    dates = pd.to_datetime(df_ml.index.get_level_values("date"), errors="coerce")

    y1, y2 = int(years[0]), int(years[1])

    def factor_stats(feature: str, year: int) -> tuple[float, float]:
        if feature not in df_ml.columns:
            return float("nan"), float("nan")
        mask = dates.year == int(year)
        if not bool(mask.any()):
            return float("nan"), float("nan")
        y = y_all.loc[mask]
        pred = pd.to_numeric(df_ml.loc[mask, feature], errors="coerce")
        ic_s = daily_ic(y, pred, min_n=int(min_n))
        m, ir, _n = _summarize_series(ic_s)
        return m, ir

    m1, mir1, _mn1 = _year_stats(model_daily_ic, y1) if model_daily_ic is not None else (float("nan"), float("nan"), 0)
    m2, mir2, _mn2 = _year_stats(model_daily_ic, y2) if model_daily_ic is not None else (float("nan"), float("nan"), 0)

    top = df_imp.sort_values(["xgb_gain"], ascending=[False]).head(int(top_n)).copy()
    top["xgbgain"] = pd.to_numeric(top["xgb_gain"], errors="coerce").astype("float64")

    ic_y1: list[float] = []
    ic_y2: list[float] = []
    ir_y1: list[float] = []
    ir_y2: list[float] = []
    for f in top["feature"].astype(str).to_list():
        ic1, ir1 = factor_stats(f, y1)
        ic2, ir2 = factor_stats(f, y2)
        ic_y1.append(ic1)
        ic_y2.append(ic2)
        ir_y1.append(ir1)
        ir_y2.append(ir2)

    out = pd.DataFrame(
        {
            "feature": top["feature"].astype(str).to_numpy(),
            "xgbgain": top["xgbgain"].to_numpy(dtype=float),
            f"{y1}ic": np.asarray(ic_y1, dtype=float),
            f"{y2}ic": np.asarray(ic_y2, dtype=float),
            f"{y1}ir": np.asarray(ir_y1, dtype=float),
            f"{y2}ir": np.asarray(ir_y2, dtype=float),
            f"model_{y1}ic": float(m1),
            f"model_{y2}ic": float(m2),
            f"model_{y1}ir": float(mir1),
            f"model_{y2}ir": float(mir2),
        }
    )
    return out


def write_factors_quick_review(
    df_imp: pd.DataFrame | None,
    df_ml: pd.DataFrame | None,
    *,
    model_daily_ic: pd.Series | None = None,
    meta: dict | None = None,
    output_path: str | None = None,
    years: tuple[int, int] = (2023, 2024),
    top_n: int = 20,
    min_n: int = 300,
) -> str:
    out_path = Path(output_path) if output_path else (Path(__file__).resolve().parent.parent / "factors_quick_review.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = (
        generate_factors_quick_review(
            df_imp if df_imp is not None else pd.DataFrame(),
            df_ml if df_ml is not None else pd.DataFrame(),
            model_daily_ic=model_daily_ic,
            years=years,
            top_n=top_n,
            min_n=min_n,
        )
        if (df_imp is not None and df_ml is not None)
        else pd.DataFrame()
    )

    lines: list[str] = []
    lines.append(f"updated_at: {datetime.now().isoformat(timespec='seconds')}")

    if isinstance(meta, dict) and len(meta) > 0:
        ts = meta.get("importance_target_date", None)
        tr0 = meta.get("importance_train_start", None)
        tr1 = meta.get("importance_train_end", None)
        if ts is not None:
            lines.append(f"importance_target_date: {pd.to_datetime(ts).strftime('%Y%m%d')}")
        if tr0 is not None and tr1 is not None:
            lines.append(
                f"importance_train_range: {pd.to_datetime(tr0).strftime('%Y%m%d')} ~ {pd.to_datetime(tr1).strftime('%Y%m%d')}"
            )
        if meta.get("train_rows", None) is not None:
            lines.append(f"importance_train_rows: {int(meta.get('train_rows'))}")
        if meta.get("importance_fit_seconds", None) is not None:
            lines.append(f"importance_fit_seconds: {float(meta.get('importance_fit_seconds')):.3f}")

    lines.append("")
    y1, y2 = int(years[0]), int(years[1])

    if model_daily_ic is not None and len(model_daily_ic) > 0:
        m1, ir1, n1 = _year_stats(model_daily_ic, y1)
        m2, ir2, n2 = _year_stats(model_daily_ic, y2)
        lines.append(
            f"model_ic: {y1} mean={m1: .6f} ir={ir1: .3f} n_days={n1} | {y2} mean={m2: .6f} ir={ir2: .3f} n_days={n2}"
        )
    else:
        lines.append(f"model_ic: {y1} mean=nan ir=nan n_days=0 | {y2} mean=nan ir=nan n_days=0")

    lines.append("")
    lines.append(f"Top {int(min(top_n, len(df_out)))} factors (by xgbgain):")
    if len(df_out) == 0:
        lines.append("(empty)")
    else:
        cols = [
            "feature",
            "xgbgain",
            f"{y1}ic",
            f"{y2}ic",
            f"{y1}ir",
            f"{y2}ir",
            f"model_{y1}ic",
            f"model_{y2}ic",
            f"model_{y1}ir",
            f"model_{y2}ir",
        ]
        cols = [c for c in cols if c in df_out.columns]
        lines.append(df_out[cols].to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out_path)

