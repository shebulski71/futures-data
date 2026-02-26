from __future__ import annotations

import os
import json
from datetime import date
from pathlib import Path
import databento as db

DATASET = "GLBX.MDP3"

def _extract_ids(mapping, parent: str) -> list[int]:
    ids: list[int] = []

    if isinstance(mapping, dict) and parent in mapping:
        entries = mapping[parent]
        if isinstance(entries, list):
            for e in entries:
                if isinstance(e, dict):
                    v = e.get("instrument_id")
                    if v is not None:
                        ids.append(int(v))
                else:
                    # sometimes entries are already IDs
                    try:
                        ids.append(int(e))
                    except Exception:
                        pass
    elif isinstance(mapping, list):
        for x in mapping:
            try:
                ids.append(int(x))
            except Exception:
                pass

    return sorted(set(ids))

def _extract_symbols(mapping) -> dict[int, str]:
    out: dict[int, str] = {}
    # expected: { "<id>": [{"raw_symbol": "ESH4", ...}, ...] } OR { "<id>": "ESH4" } depending on version
    if isinstance(mapping, dict):
        for k, v in mapping.items():
            try:
                iid = int(k)
            except Exception:
                continue

            if isinstance(v, list) and v:
                # take the first mapping entry
                e = v[0]
                if isinstance(e, dict):
                    sym = e.get("raw_symbol") or e.get("symbol")
                    if sym:
                        out[iid] = str(sym)
            elif isinstance(v, dict):
                sym = v.get("raw_symbol") or v.get("symbol")
                if sym:
                    out[iid] = str(sym)
            elif isinstance(v, str):
                out[iid] = v
    return out

def discover(root: str, start: date, end: date, out_dir: Path) -> dict:
    client = db.Historical()

    parent = f"{root}.FUT"

    # 1) parent -> instrument_id (SUPPORTED for GLBX.MDP3)
    mapping_ids = client.symbology.resolve(
        dataset=DATASET,
        symbols=[parent],
        stype_in="parent",
        stype_out="instrument_id",
        start_date=start,
        end_date=end,
    )
    instrument_ids = _extract_ids(mapping_ids, parent)

    if not instrument_ids:
        raise RuntimeError(f"No instrument_ids discovered for {parent} in {start}..{end}. "
                           f"Try widening the date window (e.g. 2014-01-01..2016-01-01).")

    # 2) instrument_id -> raw_symbol (SUPPORTED)
    mapping_syms = client.symbology.resolve(
        dataset=DATASET,
        symbols=[str(i) for i in instrument_ids],
        stype_in="instrument_id",
        stype_out="raw_symbol",
        start_date=start,
        end_date=end,
    )
    id_to_symbol = _extract_symbols(mapping_syms)

    # Keep both, because instrument_id is the best stable key
    result = {
        "root": root,
        "parent": parent,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "instrument_ids": instrument_ids,
        "id_to_raw_symbol": id_to_symbol,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{root}_catalog_{start.isoformat()}_{end.isoformat()}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote catalog: ids={len(instrument_ids)} symbols_mapped={len(id_to_symbol)} -> {out_path}")
    return result

if __name__ == "__main__":
    root = os.environ.get("ROOT", "ES")
    start = date.fromisoformat(os.environ.get("START", "2015-01-01"))
    end = date.fromisoformat(os.environ.get("END", "2015-01-05"))

    out_dir = Path("/data/lake/state/catalog")
    discover(root, start, end, out_dir)