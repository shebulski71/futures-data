#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

import databento as db
from dateutil.relativedelta import relativedelta
from tqdm import tqdm


# ----------------------------
# Config defaults
# ----------------------------
DATASET_DEFAULT = "GLBX.MDP3"
SCHEMAS_1M = ["ohlcv-1m"]

RAW_ROOT = Path("/data/lake/raw/databento")
STATE_ROOT = Path("/data/lake/state/catalog_1m")
PROGRESS_PATH = Path("/data/lake/state/ingest_1m_progress.json")


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_roots(path: Path) -> list[str]:
    roots: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        roots.append(s)
    return roots


def month_ranges(start: date, end: date):
    cur = date(start.year, start.month, 1)
    while cur < end:
        nxt = cur + relativedelta(months=1)
        m0 = max(cur, start)
        m1 = min(nxt, end)
        if m0 < m1:
            yield m0, m1
        cur = nxt


def overlaps(a0: date, a1: date, b0: date, b1: date) -> bool:
    return a0 < b1 and b0 < a1


def chunked(lst: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def load_progress(path: Path, dataset: str) -> dict:
    empty = {"dataset": dataset, "schemas_done": {}}
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if data.get("dataset") != dataset:
                return empty
            data.setdefault("schemas_done", {})
            return data
        except Exception:
            return empty
    return empty


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


def raw_outdir(dataset: str, schema: str, root: str, d: date) -> Path:
    return RAW_ROOT / dataset / schema / f"root={root}" / f"year={d.year:04d}" / f"month={d.month:02d}"


# ----------------------------
# Symbology / contract discovery
# ----------------------------
@dataclass
class ActiveUniverse:
    root: str
    month_start: date
    month_end: date
    contracts_only: list[str]
    raw_symbology_response: dict


def resolve_active_contracts_for_month(
    client: db.Historical,
    dataset: str,
    root: str,
    month_start: date,
    month_end: date,
    write_state: bool = True,
) -> ActiveUniverse:
    """
    Resolve ROOT.FUT into active outright child raw symbols for the month window.
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
        raise RuntimeError(f"Unexpected symbology response for {root} {month_start}: {str(resp)[:2000]}")

    result = resp["result"]

    active_contracts: list[str] = []
    for child_symbol, rows in result.items():
        if not isinstance(child_symbol, str):
            continue
        if "-" in child_symbol:
            continue  # skip spreads
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
# IO
# ----------------------------
def save_dbn(store, path: Path) -> None:
    ensure_dir(path.parent)
    if hasattr(store, "to_file"):
        store.to_file(str(path))
    elif hasattr(store, "to_bytes"):
        path.write_bytes(store.to_bytes())
    else:
        raise TypeError("DBNStore missing to_file/to_bytes; check databento version.")


def _safe_skip_for_missing_parent_symbol(msg: str, root: str) -> bool:
    return "symbology_invalid_request" in msg and f"Could not resolve smart symbols: {root}.FUT" in msg


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
    progress: dict,
    force_window: bool,
) -> None:
    month = _month_key(m_start)

    if (not force_window) and all(is_schema_done(progress, root, month, s) for s in SCHEMAS_1M):
        print(f"[skip] {root} {month}: already complete (progress)")
        return

    try:
        uni = resolve_active_contracts_for_month(client, dataset, root, m_start, m_end, write_state=True)
    except Exception as e:
        msg = str(e)
        if _safe_skip_for_missing_parent_symbol(msg, root):
            print(f"[skip] {root} {month}: parent symbol not available in this window ({root}.FUT)")
            return
        raise

    if not uni.contracts_only:
        print(f"[skip] {root} {month}: no active contracts")
        return

    for schema in SCHEMAS_1M:
        if (not force_window) and is_schema_done(progress, root, month, schema):
            print(f"[skip] {root} {month} {schema}: progress says done")
            continue

        batches = list(chunked(uni.contracts_only, batch_size))
        for bi, batch in enumerate(tqdm(batches, desc=f"{root} {month} {schema}"), start=1):
            dbn_dir = raw_outdir(dataset, schema, root, m_start)
            ensure_dir(dbn_dir)

            dbn_name = f"{schema}__{root}__{m_start.isoformat()}__{m_end.isoformat()}__b{bi}_n{len(batch)}.dbn"
            dbn_path = dbn_dir / dbn_name

            if dbn_path.exists() and dbn_path.stat().st_size > 0:
                continue

            store = client.timeseries.get_range(
                dataset=dataset,
                schema=schema,
                symbols=batch,
                stype_in="raw_symbol",
                start=m_start,
                end=m_end,
            )
            save_dbn(store, dbn_path)

        mark_schema_done(progress, root, month, schema)
        save_progress_atomic(PROGRESS_PATH, progress)
        print(f"[done] {root} {month} {schema}")


# ----------------------------
# Main driver
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ingest 1-minute futures raw layer for all roots (ohlcv-1m) with resume + idempotency."
    )
    ap.add_argument("--roots-file", type=str, default="roots.txt", help="Path to roots.txt (one root per line).")
    ap.add_argument("--dataset", type=str, default=DATASET_DEFAULT, help="Databento dataset, e.g. GLBX.MDP3")
    ap.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD (exclusive-ish)")
    ap.add_argument("--batch-size", type=int, default=100, help="Symbols per request batch")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if ANY month fails.")
    ap.add_argument(
        "--force-window",
        action="store_true",
        help="Ignore progress-based skips and ingest exactly the requested start/end window (still uses filesystem watermarks).",
    )
    args = ap.parse_args()

    roots_path = Path(args.roots_file).expanduser().resolve()
    if not roots_path.exists():
        raise RuntimeError(f"roots file not found: {roots_path}")

    roots = read_roots(roots_path)
    if not roots:
        raise RuntimeError(f"No roots found in {roots_path}")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end <= start:
        raise ValueError("end must be > start")

    ensure_dir(STATE_ROOT)
    ensure_dir(PROGRESS_PATH.parent)

    progress = load_progress(PROGRESS_PATH, args.dataset)
    client = db.Historical()

    print(
        f"roots={len(roots)} dataset={args.dataset} start={start} end={end} "
        f"schema=ohlcv-1m force_window={args.force_window}"
    )
    print(f"progress_file={PROGRESS_PATH}")

    run_report = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "dataset": args.dataset,
        "schema": "ohlcv-1m",
        "start": args.start,
        "end": args.end,
        "roots_file": str(roots_path),
        "strict": bool(args.strict),
        "force_window": bool(args.force_window),
        "months_ok": 0,
        "months_error": 0,
        "errors": [],
    }

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
                    progress=progress,
                    force_window=args.force_window,
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

            save_progress_atomic(PROGRESS_PATH, progress)

    out = Path("/data/lake/state") / f"ingest_1m_run_{args.start}_to_{args.end}.json"
    out.write_text(json.dumps(run_report, indent=2))
    print(f"\nWrote run report: {out}")
    print(f"Summary: months_ok={run_report['months_ok']} months_error={run_report['months_error']}")

    if args.strict and run_report["months_error"] > 0:
        print("Strict mode: failing because at least one month failed.")
        sys.exit(1)

    if run_report["months_ok"] == 0:
        print("No successful months ingested. Failing.")
        sys.exit(2)

    print("\nDone.")
    sys.exit(0)


if __name__ == "__main__":
    main()