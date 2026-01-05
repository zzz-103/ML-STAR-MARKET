import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def get_industry_map_cache(*, cache_path: str, refresh: bool) -> dict[str, str]:
    if (not refresh) and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): str(v) for k, v in data.items()}

    try:
        import akshare as ak
    except Exception as e:
        raise RuntimeError("未安装 akshare，无法拉取行业映射表；请先安装后再运行") from e

    industry_list_df = ak.stock_board_industry_name_em()
    if industry_list_df is None or industry_list_df.empty:
        return {}

    if "板块名称" not in industry_list_df.columns:
        raise ValueError("ak.stock_board_industry_name_em 返回缺少列: 板块名称")

    industry_names = industry_list_df["板块名称"].astype(str).tolist()
    stock_to_industry: dict[str, str] = {}

    for industry in tqdm(industry_names, desc="fetch_industry_cons"):
        try:
            cons_df = ak.stock_board_industry_cons_em(symbol=industry)
            if cons_df is None or cons_df.empty:
                continue
            if "代码" not in cons_df.columns:
                continue
            codes = cons_df["代码"].astype(str).tolist()
            for code in codes:
                c = str(code).strip()
                if c:
                    stock_to_industry[c] = str(industry)
            time.sleep(0.05)
        except Exception:
            continue

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(stock_to_industry, f, ensure_ascii=False)

    return stock_to_industry


def _extract_code_series(df: pd.DataFrame) -> pd.Series:
    if "code" in df.columns:
        return df["code"]
    if isinstance(df.index, pd.MultiIndex) and ("code" in set(df.index.names)):
        return pd.Series(df.index.get_level_values("code"), index=df.index)
    if df.index.name == "code":
        return pd.Series(df.index, index=df.index)
    raise ValueError("数据缺少 code 列或 code 索引")


def _sanitize_code_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.(SZ|SH|BJ)$", "", regex=True)
    out = out.str.replace(r"^(SZ|SH|BJ)", "", regex=True)
    return out


def add_industry_column(
    *,
    input_path: str,
    output_path: str,
    mapping: dict[str, str],
    missing_value: str,
) -> tuple[int, int]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    suffix = str(Path(input_path).suffix).lower()
    if suffix == ".csv":
        df = pd.read_csv(input_path, dtype={"code": str})
        had_multiindex = False
        index_names: list[str] | None = None
    else:
        df = pd.read_parquet(input_path)
        had_multiindex = isinstance(df.index, pd.MultiIndex)
        index_names = list(df.index.names) if had_multiindex else None

    code_raw = _extract_code_series(df)
    code = _sanitize_code_series(code_raw)

    industry = code.map(mapping)
    industry = industry.fillna(str(missing_value))

    if "industry" in df.columns:
        df = df.copy()
        df["industry"] = industry.to_numpy()
    else:
        df = df.assign(industry=industry.to_numpy())

    missing_count = int((industry == str(missing_value)).sum())
    total_count = int(len(industry))

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    out_suffix = str(Path(output_path).suffix).lower()
    tmp_path = str(output_path) + ".tmp"
    if out_suffix == ".csv":
        df.to_csv(tmp_path, index=False)
    else:
        if had_multiindex and index_names:
            df = df.reset_index().set_index(index_names).sort_index()
        df.to_parquet(tmp_path)

    os.replace(tmp_path, output_path)

    return total_count, missing_count


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default=str(project_root / "pre_data" / "cleaned_stock_data_300_688_with_idxstk.parquet"),
    )
    parser.add_argument(
        "--output-path",
        default=str(project_root / "pre_data" / "cleaned_stock_data_300_688_with_idxstk_with_industry.parquet"),
    )
    parser.add_argument(
        "--cache-path",
        default=str(Path(__file__).resolve().parent / "industry_map.json"),
    )
    parser.add_argument("--refresh-cache", action="store_true", default=False)
    parser.add_argument("--missing-value", default="Unknown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mapping = get_industry_map_cache(cache_path=str(args.cache_path), refresh=bool(args.refresh_cache))
    if not mapping:
        raise RuntimeError("行业映射表为空，请检查 akshare/网络，或缓存文件是否正确")

    total, missing = add_industry_column(
        input_path=str(args.input_path),
        output_path=str(args.output_path),
        mapping=mapping,
        missing_value=str(args.missing_value),
    )

    print(f"已写入: {args.output_path}")
    print(f"样本行数: {total} | 未匹配行数: {missing} | 未匹配占比: {missing / max(1, total):.2%}")


if __name__ == "__main__":
    main()
