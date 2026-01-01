import argparse
import csv
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class Columns:
    names: Tuple[str, ...]
    index: dict

    @staticmethod
    def from_header(header: Sequence[str]) -> "Columns":
        seen = set()
        normalized: List[str] = []
        for col in header:
            col = (col or "").strip()
            if not col or col in seen:
                continue
            normalized.append(col)
            seen.add(col)

        names: List[str] = ["date", "code"]
        names.extend([c for c in normalized if c not in ("date", "code")])
        final_names = tuple(names)
        return Columns(names=final_names, index={c: i for i, c in enumerate(final_names)})


def iter_dates(start_yyyymmdd: str, end_yyyymmdd: str) -> Iterable[str]:
    start = datetime.strptime(start_yyyymmdd, "%Y%m%d").date()
    end = datetime.strptime(end_yyyymmdd, "%Y%m%d").date()
    if end < start:
        raise ValueError(f"end_date < start_date: {end_yyyymmdd} < {start_yyyymmdd}")
    current: date = start
    while current <= end:
        yield current.strftime("%Y%m%d")
        current += timedelta(days=1)


def detect_code_from_path(csv_path: str) -> str:
    return Path(csv_path).stem


def read_csv_rows(csv_path: str, day: str, columns: Columns) -> List[List[str]]:
    code = detect_code_from_path(csv_path)
    out_rows: List[List[str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_out = [""] * len(columns.names)
            row_out[columns.index["date"]] = day
            row_out[columns.index["code"]] = code
            for k, v in row.items():
                if k is None:
                    continue
                k = k.strip()
                if not k or k in ("date", "code"):
                    continue
                idx = columns.index.get(k)
                if idx is not None:
                    row_out[idx] = "" if v is None else str(v)
            out_rows.append(row_out)
    return out_rows


def find_first_csv(input_root: str, start_date: str, end_date: str) -> Optional[str]:
    for day in iter_dates(start_date, end_date):
        day_dir = os.path.join(input_root, day)
        if not os.path.isdir(day_dir):
            continue
        matches = glob.glob(os.path.join(day_dir, "*.csv"))
        if matches:
            matches.sort()
            return matches[0]
    return None


def load_header(csv_path: str) -> List[str]:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    return [h.strip() for h in header if (h or "").strip()]


def merge_stockdata(
    input_root: str,
    output_csv_path: str,
    start_date: str,
    end_date: str,
    max_workers: int,
) -> Tuple[int, int]:
    first_csv = find_first_csv(input_root, start_date, end_date)
    if not first_csv:
        raise FileNotFoundError(f"no csv found under {input_root} within {start_date}-{end_date}")

    columns = Columns.from_header(load_header(first_csv))
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    days_with_data = 0
    total_rows = 0

    with open(output_csv_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(list(columns.names))

        for day in iter_dates(start_date, end_date):
            day_dir = os.path.join(input_root, day)
            if not os.path.isdir(day_dir):
                continue
            csv_paths = glob.glob(os.path.join(day_dir, "*.csv"))
            if not csv_paths:
                continue
            csv_paths.sort()
            days_with_data += 1

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for rows in ex.map(lambda p: read_csv_rows(p, day, columns), csv_paths):
                    if not rows:
                        continue
                    writer.writerows(rows)
                    total_rows += len(rows)

    return days_with_data, total_rows


def csv_to_parquet(
    csv_path: str,
    parquet_path: str,
    compression: str,
) -> int:
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    if os.path.exists(parquet_path):
        os.remove(parquet_path)

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    header = [h.strip() for h in header if (h or "").strip()]
    if not header:
        raise ValueError(f"empty csv header: {csv_path}")

    column_types = {name: (pa.string() if name in ("date", "code") else pa.float64()) for name in header}

    reader = pacsv.open_csv(
        csv_path,
        read_options=pacsv.ReadOptions(use_threads=True),
        convert_options=pacsv.ConvertOptions(column_types=column_types),
    )

    total_rows = 0
    writer: Optional[pq.ParquetWriter] = None
    try:
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            if batch is None:
                break
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, batch.schema, compression=compression)
            writer.write_batch(batch)
            total_rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()

    return total_rows


def _parse_prefixes(value: Optional[str]) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    parts: List[str] = []
    for item in str(value).split(","):
        s = item.strip()
        if s:
            parts.append(s)
    return tuple(parts)


def _norm_csv_field_name(name: Optional[str]) -> str:
    s = "" if name is None else str(name)
    s = s.replace("\ufeff", "").strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    return s


def _year_from_yyyy_mm_dd(value: str) -> Optional[str]:
    s = "" if value is None else str(value).strip()
    if len(s) >= 4 and s[:4].isdigit():
        return s[:4]
    return None


def validate_idx_idxstk(
    csv_path: str,
    prefixes: Tuple[str, ...],
    code_col: str = "Stkcd",
    date_col: str = "Idxstk01",
    sample_codes: int = 20,
    verbose: bool = True,
) -> Dict:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"csv not found: {csv_path}")
    if not prefixes:
        raise ValueError("prefixes is empty")

    total_rows = 0
    codes_all = set()
    codes_by_prefix = {p: set() for p in prefixes}
    counts_all_by_date = {}
    counts_by_prefix_by_date = {p: {} for p in prefixes}
    min_date = None
    max_date = None

    def consume_row(code: str, dt: str) -> None:
        nonlocal total_rows, min_date, max_date
        total_rows += 1
        code = str(code).strip().strip('"').zfill(6)
        if not code:
            return
        codes_all.add(code)

        dt = "" if dt is None else str(dt).strip().strip('"')
        if dt:
            if min_date is None or dt < min_date:
                min_date = dt
            if max_date is None or dt > max_date:
                max_date = dt
            counts_all_by_date[dt] = counts_all_by_date.get(dt, 0) + 1

        for p in prefixes:
            if code.startswith(p):
                codes_by_prefix[p].add(code)
                if dt:
                    d = counts_by_prefix_by_date[p]
                    d[dt] = d.get(dt, 0) + 1

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader0 = csv.reader(f)
        first = next(reader0, None)
        if first is None:
            raise ValueError(f"empty csv: {csv_path}")

        normalized_first = [_norm_csv_field_name(x) for x in first]
        looks_like_header = (code_col in normalized_first) or (_norm_csv_field_name(code_col) in normalized_first)
        if looks_like_header:
            f.seek(0)
            reader = csv.DictReader(f)
            header = list(reader.fieldnames or [])
            header_map = {_norm_csv_field_name(h): h for h in header}
            code_key = header_map.get(_norm_csv_field_name(code_col))
            date_key = header_map.get(_norm_csv_field_name(date_col))
            if code_key is None:
                raise ValueError(f"code_col not found: {code_col}, header={header}")
            if date_key is None:
                raise ValueError(f"date_col not found: {date_col}, header={header}")

            for row in reader:
                code_raw = row.get(code_key)
                dt_raw = row.get(date_key)
                if code_raw is None:
                    continue
                consume_row(code=str(code_raw), dt="" if dt_raw is None else str(dt_raw))
        else:
            code_idx = 0
            date_idx = 1
            if len(first) > max(code_idx, date_idx):
                consume_row(code=str(first[code_idx]), dt=str(first[date_idx]))
            for row in reader0:
                if not row or len(row) <= max(code_idx, date_idx):
                    continue
                consume_row(code=str(row[code_idx]), dt=str(row[date_idx]))

    unique_dates = sorted(counts_all_by_date.keys())
    stats = {
        "csv_path": csv_path,
        "total_rows": total_rows,
        "unique_codes": len(codes_all),
        "unique_dates": len(unique_dates),
        "min_date": min_date,
        "max_date": max_date,
        "prefixes": {},
        "all_counts_by_date": dict(counts_all_by_date),
        "prefix_counts_by_date": {p: dict(counts_by_prefix_by_date[p]) for p in prefixes},
        "all_daily_counts": [counts_all_by_date[d] for d in unique_dates] if unique_dates else [],
        "prefix_daily_counts": {
            p: [counts_by_prefix_by_date[p].get(d, 0) for d in unique_dates] if unique_dates else []
            for p in prefixes
        },
        "prefix_codes": {p: sorted(list(codes_by_prefix[p])) for p in prefixes},
    }

    for p in prefixes:
        counts = stats["prefix_daily_counts"][p]
        stats["prefixes"][p] = {
            "unique_codes": len(codes_by_prefix[p]),
            "sample_codes": stats["prefix_codes"][p][: int(max(1, sample_codes))],
            "daily_count_min": min(counts) if counts else 0,
            "daily_count_median": float(median(counts)) if counts else 0.0,
            "daily_count_max": max(counts) if counts else 0,
        }

    if verbose:
        print(f"input_csv={csv_path}")
        print(f"columns={code_col},{date_col}")
        print(f"total_rows={stats['total_rows']}")
        print(f"unique_codes={stats['unique_codes']}")
        print(f"unique_dates={stats['unique_dates']}")
        print(f"date_range={stats['min_date']}~{stats['max_date']}")
        for p in prefixes:
            s = stats["prefixes"][p]
            print(f"prefix={p} unique_codes={s['unique_codes']}")
            if s["unique_codes"] > 0:
                print(f"prefix={p} sample_codes={','.join(s['sample_codes'])}")
            print(
                f"prefix={p} daily_count min={s['daily_count_min']} median={s['daily_count_median']} max={s['daily_count_max']}"
            )
        if stats["all_daily_counts"]:
            all_counts = stats["all_daily_counts"]
            print(f"all_prefixes daily_count min={min(all_counts)} median={median(all_counts)} max={max(all_counts)}")

    return stats


def _expand_csv_inputs(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        s = "" if v is None else str(v).strip()
        if not s:
            continue
        matches = glob.glob(s)
        if matches:
            matches.sort()
            out.extend(matches)
        else:
            out.append(s)
    uniq: List[str] = []
    seen = set()
    for p in out:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        uniq.append(ap)
    return uniq


def validate_idx_idxstk_multi(
    csv_paths: Sequence[str],
    prefixes: Tuple[str, ...],
    code_col: str = "Stkcd",
    date_col: str = "Idxstk01",
    sample_codes: int = 20,
) -> None:
    paths = _expand_csv_inputs(csv_paths)
    if not paths:
        raise ValueError("no csv paths provided")

    per_file: List[Dict] = []
    combined_codes_by_prefix = {p: set() for p in prefixes}
    combined_counts_by_prefix_by_date = {p: {} for p in prefixes}
    combined_min_date = None
    combined_max_date = None

    for pth in paths:
        st = validate_idx_idxstk(
            csv_path=pth,
            prefixes=prefixes,
            code_col=code_col,
            date_col=date_col,
            sample_codes=sample_codes,
            verbose=False,
        )
        per_file.append(st)
        if st["min_date"] is not None:
            if combined_min_date is None or st["min_date"] < combined_min_date:
                combined_min_date = st["min_date"]
        if st["max_date"] is not None:
            if combined_max_date is None or st["max_date"] > combined_max_date:
                combined_max_date = st["max_date"]

        for pref in prefixes:
            for c in st["prefix_codes"][pref]:
                combined_codes_by_prefix[pref].add(c)
            for dt, cnt in st["prefix_counts_by_date"][pref].items():
                d = combined_counts_by_prefix_by_date[pref]
                d[dt] = d.get(dt, 0) + int(cnt)

    print(f"files={len(paths)}")
    for st in per_file:
        base = os.path.basename(st["csv_path"])
        print(f"[file] {base} rows={st['total_rows']} dates={st['unique_dates']} range={st['min_date']}~{st['max_date']}")
        for pref in prefixes:
            s = st["prefixes"][pref]
            print(
                f"[file] {base} prefix={pref} codes={s['unique_codes']} daily(min/med/max)={s['daily_count_min']}/{s['daily_count_median']}/{s['daily_count_max']}"
            )

    print(f"[combined] date_range={combined_min_date}~{combined_max_date}")
    for pref in prefixes:
        codes = sorted(list(combined_codes_by_prefix[pref]))
        dmap = combined_counts_by_prefix_by_date[pref]
        dates_sorted = sorted(dmap.keys())
        counts = [dmap[d] for d in dates_sorted] if dates_sorted else []
        print(f"[combined] prefix={pref} unique_codes={len(codes)}")
        if codes:
            print(f"[combined] prefix={pref} sample_codes={','.join(codes[:20])}")
        if counts:
            print(f"[combined] prefix={pref} daily_count min={min(counts)} median={median(counts)} max={max(counts)}")
            by_year: Dict[str, List[int]] = {}
            for d, c in zip(dates_sorted, counts):
                y = _year_from_yyyy_mm_dd(d)
                if y is None:
                    continue
                by_year.setdefault(y, []).append(int(c))
            for y in sorted(by_year.keys()):
                arr = by_year[y]
                print(f"[combined] prefix={pref} year={y} daily_count min={min(arr)} median={median(arr)} max={max(arr)}")
        else:
            print(f"[combined] prefix={pref} daily_count min=0 median=0 max=0")


def _normalize_code(value: Optional[str]) -> str:
    s = "" if value is None else str(value).strip().strip('"')
    if "." in s:
        s = s.split(".", 1)[0].strip()
    s = s.replace("\ufeff", "").strip()
    if s.isdigit():
        return s.zfill(6)
    return s


def _parse_yyyymmdd(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    ts = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _parse_yyyy_mm_dd(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    s = str(value).strip().strip('"')
    if not s:
        return None
    ts = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _parse_date_any(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    s = str(value).strip().strip('"')
    if not s:
        return None
    if s.isdigit() and len(s) == 8:
        return _parse_yyyymmdd(s)
    if "-" in s and len(s) >= 10:
        ts = pd.to_datetime(s[:10], format="%Y-%m-%d", errors="coerce")
        if pd.isna(ts):
            return None
        return ts
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _expand_paths(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        s = "" if v is None else str(v).strip()
        if not s:
            continue
        m = glob.glob(s)
        if m:
            m.sort()
            out.extend(m)
        else:
            out.append(s)
    uniq: List[str] = []
    seen = set()
    for p in out:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        uniq.append(ap)
    return uniq


def load_idx_idxstk_as_df(
    csv_paths: Sequence[str],
    prefixes: Tuple[str, ...],
    code_col: str = "Stkcd",
    date_col: str = "Idxstk01",
) -> pd.DataFrame:
    paths = _expand_paths(csv_paths)
    if not paths:
        raise ValueError("idxstk csv paths is empty")

    frames: List[pd.DataFrame] = []
    fixed_cols = ["Stkcd", "Idxstk01", "Idxstk02", "Idxstk03", "Idxstk06", "Idxstk07"]

    for pth in paths:
        with open(pth, "r", newline="", encoding="utf-8") as f:
            reader0 = csv.reader(f)
            first = next(reader0, None)
        if first is None:
            continue
        normalized_first = [_norm_csv_field_name(x) for x in first]
        looks_like_header = (code_col in normalized_first) or (_norm_csv_field_name(code_col) in normalized_first)

        if looks_like_header:
            df = pd.read_csv(pth, dtype=str)
            df = df.rename(columns={c: _norm_csv_field_name(c) for c in df.columns})
        else:
            df = pd.read_csv(pth, header=None, names=fixed_cols, dtype=str)

        if code_col != "Stkcd" and code_col in df.columns:
            df = df.rename(columns={code_col: "Stkcd"})
        if date_col != "Idxstk01" and date_col in df.columns:
            df = df.rename(columns={date_col: "Idxstk01"})

        keep_cols = [c for c in fixed_cols if c in df.columns]
        df = df[keep_cols].copy()
        df["code"] = df["Stkcd"].map(_normalize_code)
        df["date"] = df["Idxstk01"].map(_parse_date_any)

        mask = df["code"].astype(str).str.startswith(prefixes)
        df = df.loc[mask, :].copy()

        for c in ("Idxstk02", "Idxstk03", "Idxstk06", "Idxstk07"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.drop(columns=[c for c in ("Stkcd", "Idxstk01") if c in df.columns], errors="ignore")
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["date", "code", "Idxstk02", "Idxstk03", "Idxstk06", "Idxstk07"])

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.dropna(subset=["date", "code"])
    out = out.drop_duplicates(subset=["date", "code"], keep="last")
    return out


def load_stk_mkt_dalyr_as_df(
    csv_paths: Sequence[str],
    prefixes: Tuple[str, ...],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    code_col: str = "Symbol",
    date_col: str = "TradingDate",
) -> pd.DataFrame:
    paths = _expand_paths(csv_paths)
    if not paths:
        return pd.DataFrame(columns=["date", "code", "Turnover", "CirculatedMarketValue"])

    keep_cols = [date_col, code_col, "Turnover", "CirculatedMarketValue"]
    keep_norm = {_norm_csv_field_name(c) for c in keep_cols}

    start_dt = _parse_date_any(start_date) if start_date else None
    end_dt = _parse_date_any(end_date) if end_date else None

    frames: List[pd.DataFrame] = []

    def _usecols(c: str) -> bool:
        return _norm_csv_field_name(c) in keep_norm

    for pth in paths:
        with open(pth, "r", newline="", encoding="utf-8") as f:
            reader0 = csv.reader(f)
            first = next(reader0, None)
        if first is None:
            continue
        normalized_first = [_norm_csv_field_name(x) for x in first]
        looks_like_header = (date_col in normalized_first) or (_norm_csv_field_name(date_col) in normalized_first)
        if not looks_like_header:
            continue

        for chunk in pd.read_csv(pth, dtype=str, usecols=_usecols, chunksize=500_000):
            chunk = chunk.rename(columns={c: _norm_csv_field_name(c) for c in chunk.columns})

            if code_col != "Symbol" and code_col in chunk.columns:
                chunk = chunk.rename(columns={code_col: "Symbol"})
            if date_col != "TradingDate" and date_col in chunk.columns:
                chunk = chunk.rename(columns={date_col: "TradingDate"})

            if "Symbol" not in chunk.columns or "TradingDate" not in chunk.columns:
                continue

            chunk["code"] = chunk["Symbol"].map(_normalize_code)
            chunk["date"] = chunk["TradingDate"].map(_parse_date_any)
            chunk = chunk.dropna(subset=["date", "code"])
            chunk = chunk.loc[chunk["code"].astype(str).str.startswith(prefixes), :].copy()
            if start_dt is not None:
                chunk = chunk.loc[chunk["date"] >= start_dt, :].copy()
            if end_dt is not None:
                chunk = chunk.loc[chunk["date"] <= end_dt, :].copy()
            if len(chunk) == 0:
                continue

            for c in ("Turnover", "CirculatedMarketValue"):
                if c in chunk.columns:
                    chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

            frames.append(chunk[["date", "code", "Turnover", "CirculatedMarketValue"]])

    if not frames:
        return pd.DataFrame(columns=["date", "code", "Turnover", "CirculatedMarketValue"])

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.dropna(subset=["date", "code"])
    out = out.drop_duplicates(subset=["date", "code"], keep="last")
    return out


def load_fi_t10_as_df(
    csv_paths: Sequence[str],
    prefixes: Tuple[str, ...],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    code_col: str = "Stkcd",
    date_col: str = "Accper",
) -> pd.DataFrame:
    paths = _expand_paths(csv_paths)
    if not paths:
        return pd.DataFrame(columns=["date", "code", "PETTM", "PSTTM", "PCFTTM", "PB"])

    alias_map = {
        "F100103C": "PETTM",
        "F100203C": "PSTTM",
        "F100303C": "PCFTTM",
        "F100401A": "PB",
    }
    keep_cols = [
        code_col,
        date_col,
        "PETTM",
        "PSTTM",
        "PCFTTM",
        "PB",
        *list(alias_map.keys()),
    ]
    keep_norm = {_norm_csv_field_name(c) for c in keep_cols}

    start_dt = _parse_date_any(start_date) if start_date else None
    end_dt = _parse_date_any(end_date) if end_date else None

    frames: List[pd.DataFrame] = []

    def _usecols(c: str) -> bool:
        return _norm_csv_field_name(c) in keep_norm

    for pth in paths:
        with open(pth, "r", newline="", encoding="utf-8") as f:
            reader0 = csv.reader(f)
            first = next(reader0, None)
        if first is None:
            continue
        normalized_first = [_norm_csv_field_name(x) for x in first]
        looks_like_header = (date_col in normalized_first) or (_norm_csv_field_name(date_col) in normalized_first)
        if not looks_like_header:
            continue

        for chunk in pd.read_csv(pth, dtype=str, usecols=_usecols, chunksize=500_000):
            chunk = chunk.rename(columns={c: _norm_csv_field_name(c) for c in chunk.columns})
            chunk = chunk.rename(columns=alias_map)

            if code_col != "Stkcd" and code_col in chunk.columns:
                chunk = chunk.rename(columns={code_col: "Stkcd"})
            if date_col != "Accper" and date_col in chunk.columns:
                chunk = chunk.rename(columns={date_col: "Accper"})

            if "Stkcd" not in chunk.columns or "Accper" not in chunk.columns:
                continue

            chunk["code"] = chunk["Stkcd"].map(_normalize_code)
            chunk["date"] = chunk["Accper"].map(_parse_date_any)
            chunk = chunk.dropna(subset=["date", "code"])
            chunk = chunk.loc[chunk["code"].astype(str).str.startswith(prefixes), :].copy()
            if start_dt is not None:
                chunk = chunk.loc[chunk["date"] >= start_dt, :].copy()
            if end_dt is not None:
                chunk = chunk.loc[chunk["date"] <= end_dt, :].copy()
            if len(chunk) == 0:
                continue

            for c in ("PETTM", "PSTTM", "PCFTTM", "PB"):
                if c in chunk.columns:
                    chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

            for c in ("PETTM", "PSTTM", "PCFTTM", "PB"):
                if c not in chunk.columns:
                    chunk[c] = pd.NA

            frames.append(chunk[["date", "code", "PETTM", "PSTTM", "PCFTTM", "PB"]])

    if not frames:
        return pd.DataFrame(columns=["date", "code", "PETTM", "PSTTM", "PCFTTM", "PB"])

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.dropna(subset=["date", "code"])
    out = out.drop_duplicates(subset=["date", "code"], keep="last")
    return out


def load_shuangchuang_as_df(
    csv_paths: Sequence[str],
    prefixes: Tuple[str, ...],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    paths = _expand_paths(csv_paths)
    frames: List[pd.DataFrame] = []
    extra_cols: List[str] = []
    extra_seen = set()

    start_dt = _parse_date_any(start_date) if start_date else None
    end_dt = _parse_date_any(end_date) if end_date else None

    for pth in paths:
        if not os.path.exists(pth):
            continue
        df = pd.read_csv(pth, dtype=str)
        df = df.rename(columns={c: _norm_csv_field_name(c) for c in df.columns})
        if "date" not in df.columns or "instrument" not in df.columns:
            continue

        df["code"] = df["instrument"].map(_normalize_code)
        df["date"] = df["date"].map(_parse_date_any)
        df = df.dropna(subset=["date", "code"])
        df = df.loc[df["code"].astype(str).str.startswith(prefixes), :].copy()
        if start_dt is not None:
            df = df.loc[df["date"] >= start_dt, :].copy()
        if end_dt is not None:
            df = df.loc[df["date"] <= end_dt, :].copy()
        if len(df) == 0:
            continue

        cols = [c for c in df.columns if c not in ("date", "code", "instrument")]
        for c in cols:
            if c not in extra_seen:
                extra_seen.add(c)
                extra_cols.append(c)
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df[["date", "code"] + cols].copy()
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["date", "code"])

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.dropna(subset=["date", "code"])
    out = out.drop_duplicates(subset=["date", "code"], keep="last")
    return out[["date", "code"] + [c for c in extra_cols if c in out.columns]]


def load_base_merged_as_df(
    csv_path: str,
    prefixes: Tuple[str, ...],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"base csv not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    norm_to_raw = {_norm_csv_field_name(c).strip().lower(): c for c in df.columns}
    date_candidates = ("date", "trddt", "trade_date", "trading_date")
    code_candidates = ("code", "stkcd", "stock_code", "ts_code", "ticker", "symbol")
    date_col_raw = next((norm_to_raw.get(c) for c in date_candidates if norm_to_raw.get(c) is not None), None)
    code_col_raw = next((norm_to_raw.get(c) for c in code_candidates if norm_to_raw.get(c) is not None), None)
    if date_col_raw is None or code_col_raw is None:
        raise ValueError(f"base csv must include date and code columns: {csv_path}")
    if date_col_raw != "date":
        df = df.rename(columns={date_col_raw: "date"})
    if code_col_raw != "code":
        df = df.rename(columns={code_col_raw: "code"})

    df["code"] = df["code"].map(_normalize_code)
    df["date"] = df["date"].map(_parse_date_any)
    df = df.dropna(subset=["date", "code"])
    df = df[df["code"].astype(str).str.startswith(prefixes)]

    start_dt = _parse_date_any(start_date) if start_date else None
    end_dt = _parse_date_any(end_date) if end_date else None
    if start_dt is not None:
        df = df[df["date"] >= start_dt]
    if end_dt is not None:
        df = df[df["date"] <= end_dt]

    for c in df.columns:
        if c in ("date", "code"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def merge_base_with_idxstk_to_parquet(
    base_csv_path: str,
    idxstk_csv_paths: Sequence[str],
    valuation_csv_paths: Sequence[str],
    shuangchuang_csv_paths: Sequence[str],
    output_parquet_path: str,
    prefixes: Tuple[str, ...],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    compression: str = "snappy",
) -> None:
    base_df = load_base_merged_as_df(
        csv_path=base_csv_path,
        prefixes=prefixes,
        start_date=start_date,
        end_date=end_date,
    )
    val_df = load_stk_mkt_dalyr_as_df(
        csv_paths=valuation_csv_paths,
        prefixes=prefixes,
        start_date=start_date,
        end_date=end_date,
    )
    sc_df = load_shuangchuang_as_df(
        csv_paths=shuangchuang_csv_paths,
        prefixes=prefixes,
        start_date=start_date,
        end_date=end_date,
    )

    base_df = base_df.drop_duplicates(subset=["date", "code"], keep="last")
    val_df = val_df.drop_duplicates(subset=["date", "code"], keep="last")
    sc_df = sc_df.drop_duplicates(subset=["date", "code"], keep="last")

    base_idx = base_df.set_index(["date", "code"]).sort_index()
    val_idx = val_df.set_index(["date", "code"]).sort_index()
    sc_idx = sc_df.set_index(["date", "code"]).sort_index()

    merged = base_idx.join(val_idx, how="left").join(sc_idx, how="left")
    base_cols = [
        "preclose",
        "open",
        "high",
        "low",
        "close",
        "last",
        "upper_limit",
        "lower_limit",
        "volume",
        "turnover",
        "open_interest",
    ]
    keep_cols = [c for c in base_cols if c in merged.columns]
    for c in ("CirculatedMarketValue", "Turnover"):
        if c in merged.columns:
            keep_cols.append(c)
    for c in sc_idx.columns:
        if c in merged.columns and c not in keep_cols:
            keep_cols.append(c)
    merged = merged[keep_cols]

    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    if os.path.exists(output_parquet_path):
        os.remove(output_parquet_path)
    engine = None
    try:
        import pyarrow  # noqa: F401

        engine = "pyarrow"
    except ModuleNotFoundError:
        try:
            import fastparquet  # noqa: F401

            engine = "fastparquet"
        except ModuleNotFoundError as e:
            raise ImportError(
                "parquet support requires 'pyarrow' or 'fastparquet' to be installed"
            ) from e

    merged.to_parquet(output_parquet_path, compression=compression, engine=engine)

    print(f"base_csv_path={base_csv_path}")
    print(f"valuation_files={len(_expand_paths(valuation_csv_paths))}")
    print(f"output_parquet_path={output_parquet_path}")
    print(f"rows={len(merged)}")
    print(f"date_range={merged.index.get_level_values('date').min()}~{merged.index.get_level_values('date').max()}")


def merge_shuangchuang_into_existing_parquet(
    parquet_path: str,
    shuangchuang_csv_paths: Sequence[str],
    output_parquet_path: str,
    prefixes: Tuple[str, ...],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    compression: str = "snappy",
) -> None:
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if isinstance(df.index, pd.MultiIndex):
        if df.index.names != ["date", "code"]:
            df.index = df.index.set_names(["date", "code"])
        base_idx = df.sort_index()
    else:
        if "date" not in df.columns or "code" not in df.columns:
            raise ValueError(f"parquet must include MultiIndex(date,code) or columns date/code: {parquet_path}")
        df["date"] = df["date"].map(_parse_date_any)
        df["code"] = df["code"].map(_normalize_code)
        base_idx = df.set_index(["date", "code"]).sort_index()

    sc_df = load_shuangchuang_as_df(
        csv_paths=shuangchuang_csv_paths,
        prefixes=prefixes,
        start_date=start_date,
        end_date=end_date,
    )
    if len(sc_df):
        sc_idx = sc_df.set_index(["date", "code"]).sort_index()
        merged = base_idx.join(sc_idx, how="left")
    else:
        merged = base_idx

    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    if os.path.exists(output_parquet_path):
        os.remove(output_parquet_path)
    engine = None
    try:
        import pyarrow  # noqa: F401

        engine = "pyarrow"
    except ModuleNotFoundError:
        try:
            import fastparquet  # noqa: F401

            engine = "fastparquet"
        except ModuleNotFoundError as e:
            raise ImportError("parquet support requires 'pyarrow' or 'fastparquet' to be installed") from e

    merged.to_parquet(output_parquet_path, compression=compression, engine=engine)
    print(f"parquet_path={parquet_path}")
    print(f"shuangchuang_files={len(_expand_paths(shuangchuang_csv_paths))}")
    print(f"output_parquet_path={output_parquet_path}")
    print(f"rows={len(merged)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="/Users/zhuzhuxia/Documents/SZU_w4/StockData_Backup",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/zhuzhuxia/Documents/SZU_w4/pre_data",
    )
    parser.add_argument("--start-date", default="20200101")
    parser.add_argument("--end-date", default="20241231")
    parser.add_argument("--max-workers", type=int, default=min(32, (os.cpu_count() or 4) * 2))
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--write-parquet", action="store_true")
    parser.add_argument("--parquet-name", default=None)
    parser.add_argument("--parquet-compression", default="snappy")
    parser.add_argument("--convert-csv", default=None)
    parser.add_argument("--validate-idx-idxstk", default=None)
    parser.add_argument("--validate-idx-idxstk-multi", nargs="*", default=None)
    parser.add_argument("--idx-idxstk-code-col", default="Stkcd")
    parser.add_argument("--idx-idxstk-date-col", default="Idxstk01")
    parser.add_argument("--stock-pool-prefixes", default="300,688")
    parser.add_argument("--merge-idxstk-to-parquet", action="store_true", default=False)
    parser.add_argument("--merge-shuangchuang-into-parquet", action="store_true", default=False)
    parser.add_argument(
        "--base-csv-path",
        default="/Users/zhuzhuxia/Documents/SZU_w4/pre_data/merged_20200101_20241231.csv",
    )
    parser.add_argument(
        "--idxstk-csvs",
        nargs="*",
        default=[
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/2020/IDX_Idxstk.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/2021-2022/IDX_Idxstk.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/2021-2022/IDX_Idxstk1.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/2021-2022/IDX_Idxstk2.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/2023-2024/IDX_Idxstk.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/2023-2024/IDX_Idxstk1.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/2023-2024/IDX_Idxstk2.csv",
        ],
    )
    parser.add_argument(
        "--valuation-csvs",
        nargs="*",
        default=[
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/pe2020/STK_MKT_DALYR.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/pe2021-2022/STK_MKT_DALYR.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/pe2021-2022/STK_MKT_DALYR1.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/pe2021-2022/STK_MKT_DALYR2.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/pe2023-2024/STK_MKT_DALYR.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/pe2023-2024/STK_MKT_DALYR1.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/pe2023-2024/STK_MKT_DALYR2.csv",
        ],
    )
    parser.add_argument(
        "--shuangchuang-csvs",
        nargs="*",
        default=[
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/shuangchuang_2020_H1.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/shuangchuang_2020_H2.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/shuangchuang_2021_H1.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/shuangchuang_2021_H2.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/shuangchuang_2022_H1.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/shuangchuang_2022_H2.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/shuangchuang_2023_H1.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/shuangchuang_2023_H2.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/shuangchuang_2024_H1.csv",
            "/Users/zhuzhuxia/Documents/SZU_w4/pre_data/shuangchuang_2024_H2.csv",
        ],
    )
    parser.add_argument(
        "--existing-parquet-path",
        default="/Users/zhuzhuxia/Documents/SZU_w4/pre_data/cleaned_stock_data_300_688_with_idxstk.parquet",
    )
    parser.add_argument(
        "--output-parquet-path",
        default="/Users/zhuzhuxia/Documents/SZU_w4/pre_data/cleaned_stock_data_300_688_with_idxstk.parquet",
    )
    parser.add_argument("--merge-start-date", default=None)
    parser.add_argument("--merge-end-date", default=None)
    parser.add_argument("--parquet-compression-merge", default="snappy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.convert_csv:
        csv_path = args.convert_csv
        parquet_name = args.parquet_name or (Path(csv_path).stem + ".parquet")
        parquet_path = os.path.join(args.output_dir, parquet_name)
        started = datetime.now()
        parquet_rows = csv_to_parquet(
            csv_path=csv_path,
            parquet_path=parquet_path,
            compression=args.parquet_compression,
        )
        elapsed = datetime.now() - started
        print(f"input_csv_path={csv_path}")
        print(f"output_parquet_path={parquet_path}")
        print(f"total_rows={parquet_rows}")
        print(f"elapsed_seconds={elapsed.total_seconds():.3f}")
        return

    if args.validate_idx_idxstk:
        validate_idx_idxstk(
            csv_path=str(args.validate_idx_idxstk),
            prefixes=_parse_prefixes(args.stock_pool_prefixes),
            code_col=str(args.idx_idxstk_code_col),
            date_col=str(args.idx_idxstk_date_col),
        )
        return

    if args.validate_idx_idxstk_multi:
        validate_idx_idxstk_multi(
            csv_paths=list(args.validate_idx_idxstk_multi),
            prefixes=_parse_prefixes(args.stock_pool_prefixes),
            code_col=str(args.idx_idxstk_code_col),
            date_col=str(args.idx_idxstk_date_col),
        )
        return

    if bool(args.merge_idxstk_to_parquet):
        merge_base_with_idxstk_to_parquet(
            base_csv_path=str(args.base_csv_path),
            idxstk_csv_paths=list(args.idxstk_csvs or []),
            valuation_csv_paths=list(args.valuation_csvs or []),
            shuangchuang_csv_paths=list(args.shuangchuang_csvs or []),
            output_parquet_path=str(args.output_parquet_path),
            prefixes=_parse_prefixes(args.stock_pool_prefixes),
            start_date=str(args.merge_start_date) if args.merge_start_date else None,
            end_date=str(args.merge_end_date) if args.merge_end_date else None,
            compression=str(args.parquet_compression_merge),
        )
        return

    if bool(args.merge_shuangchuang_into_parquet):
        merge_shuangchuang_into_existing_parquet(
            parquet_path=str(args.existing_parquet_path),
            shuangchuang_csv_paths=list(args.shuangchuang_csvs or []),
            output_parquet_path=str(args.output_parquet_path),
            prefixes=_parse_prefixes(args.stock_pool_prefixes),
            start_date=str(args.merge_start_date) if args.merge_start_date else None,
            end_date=str(args.merge_end_date) if args.merge_end_date else None,
            compression=str(args.parquet_compression_merge),
        )
        return

    output_name = args.output_name or f"merged_{args.start_date}_{args.end_date}.csv"
    output_csv_path = os.path.join(args.output_dir, output_name)

    started = datetime.now()
    days, rows = merge_stockdata(
        input_root=args.input_root,
        output_csv_path=output_csv_path,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=max(1, int(args.max_workers)),
    )
    elapsed = datetime.now() - started

    print(f"output_csv_path={output_csv_path}")
    print(f"days_with_data={days}")
    print(f"total_rows={rows}")
    print(f"elapsed_seconds={elapsed.total_seconds():.3f}")

    if args.write_parquet:
        parquet_name = args.parquet_name or (Path(output_csv_path).stem + ".parquet")
        parquet_path = os.path.join(args.output_dir, parquet_name)
        started = datetime.now()
        parquet_rows = csv_to_parquet(
            csv_path=output_csv_path,
            parquet_path=parquet_path,
            compression=args.parquet_compression,
        )
        elapsed = datetime.now() - started
        print(f"output_parquet_path={parquet_path}")
        print(f"parquet_total_rows={parquet_rows}")
        print(f"parquet_elapsed_seconds={elapsed.total_seconds():.3f}")


if __name__ == "__main__":
    main()
