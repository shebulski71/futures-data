from __future__ import annotations

import os
import json
from datetime import date
from pathlib import Path
from typing import Iterable
import databento as db
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from datetime import datetime

DATASET = "GLBX.MDP3"
SCHEMAS = ["ohlcv-1d", "statistics"]

RAW_ROOT = Path("/data/lake/raw/databento/GLBX.MDP3")
CATALOG_DIR = Path("/data/lake/state/catalog")

BATCH_SIZE = 200  # safe default; with ES you won't hit it




def parse_d(s: str) -> date:
    return date.fromisoformat(s)

def overlaps(a0: date, a1: date, b0: date, b1: date) -> bool:
    # treat d1/end as exclusive-like; overlap if ranges intersect
    return a0 < b1 and b0 < a1

def load_active_contract_symbols(root: str, start: date, end: date) -> list[str]:
    p = latest_catalog_path(root)
    data = json.loads(p.read_text())

    resp = data.get("raw_response_parent_to_instrument_id", {})
    result = resp.get("result", {}) if isinstance(resp, dict) else {}

    active = []
    for sym, rows in result.items():
        if "-" in sym:
            continue  # contracts only
        if not isinstance(rows, list) or not rows:
            continue
        r0 = rows[0]
        if not isinstance(r0, dict):
            continue
        d0 = r0.get("d0")
        d1 = r0.get("d1")
        if not d0 or not d1:
            continue
        s0, s1 = parse_d(d0), parse_d(d1)
        if overlaps(s0, s1, start, end):
            active.append(sym)

    active = sorted(set(active))
    if not active:
        raise RuntimeError(f"No active contract symbols for root={root} in {start}..{end} using catalog {p}")
    return active

def month_ranges(start: date, end: date):
    cur = date(start.year, start.month, 1)
    while cur < end:
        nxt = cur + relativedelta(months=1)
        yield cur, min(nxt, end)
        cur = nxt

def chunked(lst: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def outdir(schema: str, root: str, d: date) -> Path:
    return RAW_ROOT / schema / f"root={root}" / f"year={d.year:04d}" / f"month={d.month:02d}"

def latest_catalog_path(root: str) -> Path:
    files = sorted(CATALOG_DIR.glob(f"{root}_catalog_*.json"))
    if not files:
        raise FileNotFoundError(f"No catalog found for root={root} in {CATALOG_DIR}. Run discover_contracts.py first.")
    return files[-1]

def load_contract_symbols(root: str) -> list[str]:
    p = latest_catalog_path(root)
    data = json.loads(p.read_text())
    m = data.get("child_symbol_to_instrument_id_contracts", {})
    symbols = sorted(m.keys())
    if not symbols:
        raise RuntimeError(f"Catalog {p} contains zero contract symbols.")
    return symbols

def save_dbn(store, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(store, "to_file"):
        store.to_file(str(path))
    elif hasattr(store, "to_bytes"):
        path.write_bytes(store.to_bytes())
    else:
        raise TypeError("DBNStore missing to_file/to_bytes; check databento version.")

def main(root: str, start: date, end: date):
    client = db.Historical()
    symbols = load_active_contract_symbols(root, start, end)

    print(f"root={root} contracts={len(symbols)} start={start} end={end}")
    print("symbols:", symbols)

    for m_start, m_end in month_ranges(start, end):
        for schema in SCHEMAS:
            batches = list(chunked(symbols, BATCH_SIZE))
            for batch in tqdm(batches, desc=f"{root} {schema} {m_start}"):
                store = client.timeseries.get_range(
                    dataset=DATASET,
                    schema=schema,
                    symbols=batch,
                    stype_in="raw_symbol",
                    start=m_start,
                    end=m_end,
                )
                fn = f"{schema}__{root}__{m_start.isoformat()}__{m_end.isoformat()}__n{len(batch)}.dbn"
                save_dbn(store, outdir(schema, root, m_start) / fn)

if __name__ == "__main__":
    root = os.environ.get("ROOT", "ES")
    start = date.fromisoformat(os.environ.get("START", "2015-01-01"))
    end = date.fromisoformat(os.environ.get("END", "2015-01-05"))
    main(root, start, end)