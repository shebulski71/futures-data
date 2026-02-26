from __future__ import annotations

from pathlib import Path
import databento as db
from tqdm import tqdm

RAW_ROOT = Path("/data/lake/raw/databento/GLBX.MDP3")
CURATED_ROOT = Path("/data/lake/curated")

def curated_dir(schema: str, root: str, year: str, month: str) -> Path:
    table = f"futures_{schema.replace('-', '_')}"
    return CURATED_ROOT / table / f"root={root}" / f"year={year}" / f"month={month}"

def main():
    dbn_files = list(RAW_ROOT.rglob("*.dbn"))
    print(f"Found {len(dbn_files)} DBN files")

    for p in tqdm(dbn_files, desc="Transcoding"):
        # parse pieces from path:
        # .../<schema>/root=ES/year=2015/month=01/file.dbn
        parts = p.parts
        schema = parts[parts.index("GLBX.MDP3")+1]
        root = parts[parts.index(schema)+1].split("=", 1)[1]
        year = parts[parts.index(schema)+2].split("=", 1)[1]
        month = parts[parts.index(schema)+3].split("=", 1)[1]

        out_dir = curated_dir(schema, root, year, month)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / p.name.replace(".dbn", ".parquet")

        store = db.DBNStore.from_file(str(p))
        store.to_parquet(str(out_path))

if __name__ == "__main__":
    main()