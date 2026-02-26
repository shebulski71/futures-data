from __future__ import annotations

import os
import json
from datetime import date
from pathlib import Path
import databento as db

DATASET = "GLBX.MDP3"

def discover(root: str, start: date, end: date, out_dir: Path) -> list[str]:
    client = db.Historical()

    parent = f"{root}.FUT"
    # Your client requires stype_out (we saw that error), so include it:
    mapping = client.symbology.resolve(
        dataset=DATASET,
        symbols=[parent],
        stype_in="parent",
        stype_out="raw_symbol",
        start_date=start,
        end_date=end,
    )

    # Mapping shape can vary by databento version; handle common cases:
    symbols: list[str] = []
    if isinstance(mapping, dict) and parent in mapping:
        entries = mapping[parent]
        if isinstance(entries, list):
            for e in entries:
                if isinstance(e, dict):
                    s = e.get("raw_symbol") or e.get("symbol")
                    if s:
                        symbols.append(str(s))
                else:
                    symbols.append(str(e))
    elif isinstance(mapping, list):
        symbols = [str(x) for x in mapping]
    else:
        # last-resort: try to stringify
        symbols = [str(mapping)]

    symbols = sorted(set(symbols))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{root}_contracts_{start.isoformat()}_{end.isoformat()}.json"
    out_path.write_text(json.dumps({"root": root, "start": start.isoformat(), "end": end.isoformat(), "symbols": symbols}, indent=2))
    print(f"Wrote {len(symbols)} symbols to {out_path}")
    return symbols

if __name__ == "__main__":
    root = os.environ.get("ROOT", "ES")
    start = date.fromisoformat(os.environ.get("START", "2015-01-01"))
    end = date.fromisoformat(os.environ.get("END", "2015-01-05"))

    out_dir = Path("/data/lake/state/contracts")
    discover(root, start, end, out_dir)