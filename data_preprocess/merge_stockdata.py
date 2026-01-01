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
