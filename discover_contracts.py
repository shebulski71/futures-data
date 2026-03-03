from __future__ import annotations

import os
import json
from datetime import date
from pathlib import Path
import databento as db

DATASET = "GLBX.MDP3"

def discover(root: str, start: date, end: date, out_dir: Path) -> dict:
    client = db.Historical()
    parent = f"{root}.FUT"

    resp = client.symbology.resolve(
        dataset=DATASET,
        symbols=[parent],
        stype_in="parent",
        stype_out="instrument_id",
        start_date=start,
        end_date=end,
    )

    if not isinstance(resp, dict) or "result" not in resp or not isinstance(resp["result"], dict):
        raise RuntimeError(f"Unexpected response shape:\n{json.dumps(resp, indent=2)[:4000]}")

    result = resp["result"]

    # Build two maps:
    # 1) child_symbol -> instrument_id
    # 2) instrument_id -> child_symbol (not 1:1 historically, but good enough for labeling)
    child_to_id: dict[str, int] = {}
    instrument_ids: set[int] = set()

    for child_symbol, rows in result.items():
        if not isinstance(rows, list) or not rows:
            continue
        r0 = rows[0]
        if isinstance(r0, dict) and "s" in r0:
            try:
                iid = int(r0["s"])
            except Exception:
                continue
            child_to_id[str(child_symbol)] = iid
            instrument_ids.add(iid)

    # Optional: keep only outright futures contracts (no spreads)
    contracts_only = {sym: iid for sym, iid in child_to_id.items() if "-" not in sym}

    out = {
        "root": root,
        "parent": parent,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "count_all_children": len(child_to_id),
        "count_contracts_only": len(contracts_only),
        "child_symbol_to_instrument_id_all": child_to_id,
        "child_symbol_to_instrument_id_contracts": contracts_only,
        "raw_response_parent_to_instrument_id": resp,  # keep for audit/debug
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{root}_catalog_{start.isoformat()}_{end.isoformat()}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(
        f"Wrote catalog for {root}: "
        f"children={len(child_to_id)} contracts_only={len(contracts_only)} -> {out_path}"
    )
    return out

if __name__ == "__main__":
    root = os.environ.get("ROOT", "ES")
    start = date.fromisoformat(os.environ.get("START", "2014-01-01"))
    end = date.fromisoformat(os.environ.get("END", "2016-01-01"))
    out_dir = Path("/data/lake/state/catalog")
    discover(root, start, end, out_dir)