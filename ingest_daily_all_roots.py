# ingest_daily_all_roots.py
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import databento as db
import polars as pl
from dateutil.relativedelta import relativedelta
from tqdm import tqdm


# ----------------------------
# Config defaults
# ----------------------------
DATASET_DEFAULT = "GLBX.MDP3"
SCHEMAS_DAILY = ["ohlcv-1d", "statistics"]

RAW_ROOT = Path("/data/lake/raw/databento")
CURATED_ROOT = Path("/data/lake/curated")
NORMALIZED_ROOT = Path("/data/lake/curated_normalized")
STATE_ROOT = Path("/data/lake/state/catalog_daily")

PROGRESS_PATH = Path("/data/lake/state/ingest_daily_progress.json")


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_roots(path: Path) -> list[str]:
    roots = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        roots.append(s)
    return roots


def month_ranges(start: date, end: date):
    """
    Yield month windows clipped to [start, end).
    """
    cur = date(start.year, start.month, 1)
    while cur < end:
        nxt = cur + relativedelta(months=1)
        m0 = max(cur, start)
        m1 = min(nxt, end)
        if m0 < m1:
            yield m0, m1
        cur = nxt


def overlaps(a0: date, a1: date, b0: date, b1: date) -> bool:
    # overlap if ranges intersect (treat end as exclusive-ish)
    return a0 < b1 and b0 < a1


def chunked(lst: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def load_progress(path: Path, dataset: str) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if data.get("dataset") != dataset:
                return {"dataset": dataset, "schemas_done": {}, "normalized_done": {}}
            data.setdefault("schemas_done", {})
            data.setdefault("normalized_done", {})
            return data
        except Exception:
            return {"dataset": dataset, "schemas_done": {}, "normalized_done": {}}
    return {"dataset": dataset, "schemas_done": {}, "normalized_done": {}}


def save_progress_atomic(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def is_schema_done(progress: dict, root: str, month: str, schema: str) -> bool:
    return schema in progress.get("schemas_done", {}).get(root, {}).get(month, [])


def mark_schema_done(progress: dict, root: str, month: str, schema: str) -> None:
    progress.setdefault("schemas_done", {}).setdefault(root, {}).setdefault(month, [])
    if schema not in progress["schemas_done"][root][month]:
        progress["schemas_done"][root][month].append(schema)


def is_normalized_done(progress: dict, root: str, month: str) -> bool:
    return bool(progress.get("normalized_done", {}).get(root, {}).get(month, False))


def mark_normalized_done(progress: dict, root: str, month: str) -> None:
    progress.setdefault("normalized_done", {}).setdefault(root, {})
    progress["normalized_done"][root][month] = True


def raw_outdir(dataset: str, schema: str, root: str, d: date) -> Path:
    # /data/lake/raw/databento/GLBX.MDP3/<schema>/root=ES/year=2015/month=01
    return RAW_ROOT / dataset / schema / f"root={root}" / f"year={d.year:04d}" / f"month={d.month:02d}"


def curated_outdir(schema: str, root: str, d: date) -> Path:
    table = f"futures_{schema.replace('-', '_')}"
    return CURATED_ROOT / table / f"root={root}" / f"year={d.year:04d}" / f"month={d.month:02d}"


def normalized_outfile(table: str, root: str, d: date) -> Path:
    return (
        NORMALIZED_ROOT
        / table
        / f"root={root}"
        / f"year={d.year:04d}"
        / f"month={d.month:02d}"
        / "part-0000.parquet"
    )


# ----------------------------
# Symbology / contract discovery
# ----------------------------
@dataclass
class ActiveUniverse:
    root: str
    month_start: date
    month_end: date
    contracts_only: list[str]  # raw symbols (e.g., ESH5)
    raw_symbology_response: dict  # audit/debug


def resolve_active_contracts_for_month(
    client: db.Historical,
    dataset: str,
    root: str,
    month_start: date,
    month_end: date,
    write_state: bool = True,
) -> ActiveUniverse:
    """
    Uses parent symbology (ROOT.FUT) -> instrument_id to get child symbols + d0/d1 validity,
    then filters:
      - outright contracts only (no '-')
      - active in [month_start, month_end)
    """
    parent = f"{root}.FUT"
    resp = client.symbology.resolve(
        dataset=dataset,
        symbols=[parent],
        stype_in="parent",
        stype_out="instrument_id",
        start_date=month_start,
        end_date=month_end,
    )

    if not isinstance(resp, dict) or "result" not in resp or not isinstance(resp["result"], dict):
        raise RuntimeError(
            f"Unexpected symbology response for {root} {month_start}: {str(resp)[:2000]}"
        )

    result = resp["result"]  # dict: child_symbol -> [{d0,d1,s}, ...]

    active_contracts: list[str] = []
    for child_symbol, rows in result.items():
        if not isinstance(child_symbol, str):
            continue
        if "-" in child_symbol:
            continue  # drop spreads for daily truth
        if not isinstance(rows, list) or not rows:
            continue
        r0 = rows[0]
        if not isinstance(r0, dict):
            continue
        d0 = r0.get("d0")
        d1 = r0.get("d1")
        if not d0 or not d1:
            continue
        try:
            s0 = date.fromisoformat(d0)
            s1 = date.fromisoformat(d1)
        except Exception:
            continue

        if overlaps(s0, s1, month_start, month_end):
            active_contracts.append(child_symbol)

    active_contracts = sorted(set(active_contracts))

    uni = ActiveUniverse(
        root=root,
        month_start=month_start,
        month_end=month_end,
        contracts_only=active_contracts,
        raw_symbology_response=resp,
    )

    if write_state:
        out = {
            "root": root,
            "parent": parent,
            "month_start": month_start.isoformat(),
            "month_end": month_end.isoformat(),
            "contracts_only": active_contracts,
            "raw_response_parent_to_instrument_id": resp,
        }
        ensure_dir(STATE_ROOT / f"root={root}")
        state_path = STATE_ROOT / f"root={root}" / f"{month_start.year:04d}-{month_start.month:02d}.json"
        state_path.write_text(json.dumps(out, indent=2))

    return uni


# ----------------------------
# IO: raw + curated
# ----------------------------
def save_dbn(store, path: Path) -> None:
    ensure_dir(path.parent)
    if hasattr(store, "to_file"):
        store.to_file(str(path))
    elif hasattr(store, "to_bytes"):
        path.write_bytes(store.to_bytes())
    else:
        raise TypeError("DBNStore missing to_file/to_bytes; check databento version.")


def transcode_dbn_to_parquet(dbn_path: Path, parquet_path: Path) -> None:
    ensure_dir(parquet_path.parent)
    store = db.DBNStore.from_file(str(dbn_path))
    store.to_parquet(str(parquet_path))


# ----------------------------
# Normalize: daily truth tables
# ----------------------------
def normalize_daily_month(root: str, month_start: date) -> None:
    """
    Reads curated parquet (ohlcv-1d and statistics) for root/year/month,
    writes normalized parquet files:
      - daily_bars_clean
      - daily_stats_pivoted
      - daily_joined
    """
    bars_dir = curated_outdir("ohlcv-1d", root, month_start)
    stats_dir = curated_outdir("statistics", root, month_start)
    bars_files = list(bars_dir.glob("*.parquet"))
    stats_files = list(stats_dir.glob("*.parquet"))

    if not bars_files:
        print(f"[normalize] No bars parquet for {root} {month_start:%Y-%m}, skipping")
        return
    if not stats_files:
        print(f"[normalize] No stats parquet for {root} {month_start:%Y-%m}, skipping")
        return

    bars_glob = str(bars_dir / "*.parquet")
    stats_glob = str(stats_dir / "*.parquet")

    bars = (
        pl.scan_parquet(bars_glob)
        .sort(["instrument_id", "ts_event"])
        .with_columns(pl.col("ts_event").dt.date().alias("trade_date_utc"))
        .unique(subset=["instrument_id", "ts_event"], keep="last")
        .select(
            [
                "instrument_id",
                "symbol",
                "trade_date_utc",
                "ts_event",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )
        .collect()
    )

    # stat_type codes: settlement=3, cleared_volume=6, open_interest=9
    stats = (
        pl.scan_parquet(stats_glob)
        .filter(pl.col("stat_type").is_in([3, 6, 9]))
        .with_columns(pl.col("ts_event").dt.date().alias("trade_date_utc"))
        .sort(["instrument_id", "trade_date_utc", "stat_type", "ts_event"])
        .unique(subset=["instrument_id", "trade_date_utc", "stat_type"], keep="last")
        .with_columns(
            [
                pl.when(pl.col("stat_type") == 3)
                .then(pl.col("price"))
                .otherwise(None)
                .alias("settlement_price"),
                pl.when(pl.col("stat_type") == 9)
                .then(pl.col("quantity"))
                .otherwise(None)
                .alias("open_interest"),
                pl.when(pl.col("stat_type") == 6)
                .then(pl.col("quantity"))
                .otherwise(None)
                .alias("cleared_volume"),
            ]
        )
        .group_by(["instrument_id", "symbol", "trade_date_utc"])
        .agg(
            [
                pl.max("settlement_price").alias("settlement_price"),
                pl.max("open_interest").alias("open_interest"),
                pl.max("cleared_volume").alias("cleared_volume"),
            ]
        )
        .collect()
    )

    joined = bars.join(stats, on=["instrument_id", "symbol", "trade_date_utc"], how="left")

    bars_out = normalized_outfile("daily_bars_clean", root, month_start)
    stats_out = normalized_outfile("daily_stats_pivoted", root, month_start)
    join_out = normalized_outfile("daily_joined", root, month_start)

    ensure_dir(bars_out.parent)
    ensure_dir(stats_out.parent)
    ensure_dir(join_out.parent)

    bars.write_parquet(bars_out, compression="zstd")
    stats.write_parquet(stats_out, compression="zstd")
    joined.write_parquet(join_out, compression="zstd")

    print(
        f"[normalize] {root} {month_start:%Y-%m}: bars={bars.height} stats={stats.height} joined={joined.height}"
    )


# ----------------------------
# Ingest: one month for one root
# ----------------------------
def ingest_month_for_root(
    client: db.Historical,
    dataset: str,
    root: str,
    m_start: date,
    m_end: date,
    batch_size: int,
    transcode: bool,
    normalize: bool,
    progress: dict,
    force_window: bool,
) -> None:
    month = _month_key(m_start)

    # If both schemas done and normalized done, skip immediately (unless forcing window).
    if (not force_window) and all(
        is_schema_done(progress, root, month, s) for s in SCHEMAS_DAILY
    ) and (not normalize or is_normalized_done(progress, root, month)):
        print(f"[skip] {root} {month}: already complete (progress)")
        return

    # Universe discovery is cheap and also produces audit state files.
    try:
        uni = resolve_active_contracts_for_month(client, dataset, root, m_start, m_end, write_state=True)
    except Exception as e:
        msg = str(e)
        # Databento returns 422 symbology_invalid_request when smart parent doesn't exist for that era
        if "symbology_invalid_request" in msg and f"Could not resolve smart symbols: {root}.FUT" in msg:
            print(f"[skip] {root} {_month_key(m_start)}: parent symbol not available in this window ({root}.FUT)")
            return
        raise

    if not uni.contracts_only:
        print(f"[skip] {root} {month}: no active contracts")
        return

    for schema in SCHEMAS_DAILY:
        # IMPORTANT: schema-level progress skip must also be disabled when forcing a window
        if (not force_window) and is_schema_done(progress, root, month, schema):
            print(f"[skip] {root} {month} {schema}: progress says done")
            continue

        batches = list(chunked(uni.contracts_only, batch_size))
        for bi, batch in enumerate(tqdm(batches, desc=f"{root} {month} {schema}"), start=1):
            dbn_dir = raw_outdir(dataset, schema, root, m_start)
            ensure_dir(dbn_dir)
            dbn_name = f"{schema}__{root}__{m_start.isoformat()}__{m_end.isoformat()}__b{bi}_n{len(batch)}.dbn"
            dbn_path = dbn_dir / dbn_name

            # RAW skip (filesystem watermark)
            if dbn_path.exists() and dbn_path.stat().st_size > 0:
                pass
            else:
                store = client.timeseries.get_range(
                    dataset=dataset,
                    schema=schema,
                    symbols=batch,
                    stype_in="raw_symbol",
                    start=m_start,
                    end=m_end,
                )
                save_dbn(store, dbn_path)

            # Curated transcode skip
            if transcode:
                pq_dir = curated_outdir(schema, root, m_start)
                ensure_dir(pq_dir)
                pq_path = pq_dir / dbn_name.replace(".dbn", ".parquet")

                if pq_path.exists() and pq_path.stat().st_size > 0:
                    pass
                else:
                    transcode_dbn_to_parquet(dbn_path, pq_path)

        # Mark schema done if we completed all batches without raising
        mark_schema_done(progress, root, month, schema)
        save_progress_atomic(PROGRESS_PATH, progress)
        print(f"[done] {root} {month} {schema}")

    # Normalization step
    if normalize:
        # IMPORTANT: normalization progress/output skips must be disabled when forcing a window
        if (not force_window) and is_normalized_done(progress, root, month):
            print(f"[skip] {root} {month} normalize: progress says done")
        else:
            join_out = normalized_outfile("daily_joined", root, m_start)
            if (not force_window) and join_out.exists() and join_out.stat().st_size > 0:
                mark_normalized_done(progress, root, month)
                save_progress_atomic(PROGRESS_PATH, progress)
                print(f"[skip] {root} {month} normalize: output exists")
            else:
                normalize_daily_month(root, m_start)
                mark_normalized_done(progress, root, month)
                save_progress_atomic(PROGRESS_PATH, progress)
                print(f"[done] {root} {month} normalize")


# ----------------------------
# Main driver
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Ingest daily futures truth layer for all roots (ohlcv-1d + statistics) with resume + idempotency."
    )
    ap.add_argument("--roots-file", type=str, default="roots.txt", help="Path to roots.txt (one root per line).")
    ap.add_argument("--dataset", type=str, default=DATASET_DEFAULT, help="Databento dataset, e.g. GLBX.MDP3")
    ap.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD (exclusive-ish)")
    ap.add_argument("--batch-size", type=int, default=200, help="Symbols per request batch")
    ap.add_argument("--no-transcode", action="store_true", help="Do not transcode DBN -> parquet")
    ap.add_argument("--no-normalize", action="store_true", help="Do not build normalized daily tables")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if ANY month fails (default: soft-fail per root/month).")
    ap.add_argument(
        "--force-window",
        action="store_true",
        help="Ignore progress-based skips and ingest exactly the requested start/end window (still uses filesystem watermarks).",
    )
    args = ap.parse_args()

    roots_path = Path(args.roots_file).expanduser().resolve()
    roots = read_roots(roots_path)
    if not roots:
        raise RuntimeError(f"No roots found in {roots_path}")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end <= start:
        raise ValueError("end must be > start")

    ensure_dir(STATE_ROOT)
    ensure_dir(PROGRESS_PATH.parent)

    transcode = not args.no_transcode
    normalize = not args.no_normalize

    # ---- Guard: ensure Databento key is present early (prevents confusing Prefect failures) ----
    # databento client reads DATABENTO_API_KEY from env; we fail fast with a helpful message.
    if not (getattr(db, "__version__", None) is None):
        pass
    # The Historical() constructor throws "invalid API key, was None" if missing.
    # We'll pre-check env to make the error message obvious.
    import os
    if not os.environ.get("DATABENTO_API_KEY"):
        print("[FATAL] DATABENTO_API_KEY is not set in the environment.", file=sys.stderr)
        print("        If running under Prefect, set it as a Prefect Secret/Environment Variable for the worker job.", file=sys.stderr)
        sys.exit(3)

    progress = load_progress(PROGRESS_PATH, args.dataset)
    client = db.Historical()

    print(
        f"roots={len(roots)} dataset={args.dataset} start={start} end={end} "
        f"transcode={transcode} normalize={normalize} force_window={bool(args.force_window)}"
    )
    print(f"progress_file={PROGRESS_PATH}")

    # --- Run report (JSON) ---
    run_report = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "dataset": args.dataset,
        "start": args.start,
        "end": args.end,
        "roots_file": str(roots_path),
        "transcode": transcode,
        "normalize": normalize,
        "strict": bool(args.strict),
        "force_window": bool(args.force_window),
        "months_ok": 0,
        "months_error": 0,
        "errors": [],  # list of dicts: {root, month, error_type, error}
    }

    # Iterate month by month to keep files manageable
    for root in roots:
        print(f"\n=== ROOT {root} ===")
        for m_start, m_end in month_ranges(start, end):
            month = _month_key(m_start)
            print(f"[month] {root} {month} ({m_start} -> {m_end})")
            try:
                ingest_month_for_root(
                    client=client,
                    dataset=args.dataset,
                    root=root,
                    m_start=m_start,
                    m_end=m_end,
                    batch_size=args.batch_size,
                    transcode=transcode,
                    normalize=normalize,
                    progress=progress,
                    force_window=bool(args.force_window),
                )
                run_report["months_ok"] += 1
            except Exception as e:
                save_progress_atomic(PROGRESS_PATH, progress)
                run_report["months_error"] += 1
                run_report["errors"].append(
                    {
                        "root": root,
                        "month": month,
                        "error_type": type(e).__name__,
                        "error": str(e),
                    }
                )
                print(f"[ERROR] {root} {month}: {type(e).__name__}: {e}")
                if args.strict:
                    # write report then fail immediately
                    out = Path("/data/lake/state") / f"ingest_daily_run_{args.start}_to_{args.end}.json"
                    out.write_text(json.dumps(run_report, indent=2))
                    print(f"\nWrote run report: {out}")
                    print("Strict mode: failing because at least one month failed.")
                    sys.exit(1)

            # Save after each month for maximum restartability
            save_progress_atomic(PROGRESS_PATH, progress)

    # Write the run report
    out = Path("/data/lake/state") / f"ingest_daily_run_{args.start}_to_{args.end}.json"
    out.write_text(json.dumps(run_report, indent=2))
    print(f"\nWrote run report: {out}")
    print(f"Summary: months_ok={run_report['months_ok']} months_error={run_report['months_error']}")

    # Exit code policy
    if run_report["months_ok"] == 0:
        print("No successful months ingested. Failing.")
        sys.exit(2)

    print("\nDone.")
    sys.exit(0)


if __name__ == "__main__":
    main()