from __future__ import annotations

import os
import json
from datetime import date
from pathlib import Path
import databento as db

DATASET = "GLBX.MDP3"

def _get_result_block(resp: dict) -> dict:
    # Databento resolve responses are shaped like: {"result": {...}, "symbols": [...], ...}
    if isinstance(resp, dict) and "result" in resp and isinstance(resp["result"], dict):
        return resp["result"]
    return {}

def discover(root: str, start: date, end: date, out_dir: Path) -> dict:
    client = db.Historical()

    parent = f"{root}.FUT"

    # 1) parent -> instrument_id (supported for GLBX.MDP3)
    resp_ids = client.symbology.resolve(
        dataset=DATASET,
        symbols=[parent],
        stype_in="parent",
        stype_out="instrument_id",
        start_date=start,
        end_date=end,
    )

    result_ids = _get_result_block(resp_ids)
    rows = result_ids.get(parent, []) if isinstance(result_ids, dict) else []

    instrument_ids: list[int] = []
    for r in rows:
        # each row looks like {"d0": "...", "d1": "...", "s": "3403"}
        if isinstance(r, dict) and "s" in r:
            try:
                instrument_ids.append(int(r["s"]))
            except Exception:
                pass

    instrument_ids = sorted(set(instrument_ids))
    if not instrument_ids:
        # dump response for debugging if still empty
        raise RuntimeError(
            f"No instrument_ids discovered for {parent} in {start}..{end}.\n"
            f"Raw response was:\n{json.dumps(resp_ids, indent=2)[:4000]}"
        )

    # 2) instrument_id -> raw_symbol (supported)
    resp_syms = client.symbology.resolve(
        dataset=DATASET,
        symbols=[str(i) for i in instrument_ids],
        stype_in="instrument_id",
        stype_out="raw_symbol",
        start_date=start,
        end_date=end,
    )

    result_syms = _get_result_block(resp_syms)
    id_to_raw: dict[int, str] = {}

    # result_syms maps each input instrument_id (as string) to list of {..,"s":"ESH4"}
    for iid_str, mappings in result_syms.items():
        try:
            iid = int(iid_str)
        except Exception:
            continue
        if isinstance(mappings, list) and mappings:
            m0 = mappings[0]
            if isinstance(m0, dict) and "s" in m0:
                id_to_raw[iid] = str(m0["s"])

    out = {
        "root": root,
        "parent": parent,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "instrument_ids": instrument_ids,
        "id_to_raw_symbol": id_to_raw,
        "symbology_resolve_parent_to_id": resp_ids,   # keep for audit/debug
        "symbology_resolve_id_to_raw": resp_syms,     # keep for audit/debug
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{root}_catalog_{start.isoformat()}_{end.isoformat()}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote catalog: ids={len(instrument_ids)} raw_symbols={len(id_to_raw)} -> {out_path}")
    return out

if __name__ == "__main__":
    root = os.environ.get("ROOT", "ES")
    start = date.fromisoformat(os.environ.get("START", "2014-01-01"))
    end = date.fromisoformat(os.environ.get("END", "2016-01-01"))

    out_dir = Path("/data/lake/state/catalog")
    discover(root, start, end, out_dir)