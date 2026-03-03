from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from tqdm import tqdm


def parse_month_from_filename(p: Path) -> date:
    # filename like 2015-01.json
    stem = p.stem
    y, m = stem.split("-")
    return date(int(y), int(m), 1)


def iter_catalog_files(base: Path):
    # /data/lake/state/catalog_daily/root=ES/2015-01.json
    for root_dir in sorted(base.glob("root=*")):
        for jf in sorted(root_dir.glob("*.json")):
            yield root_dir.name.split("=", 1)[1], jf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog-dir", default="/data/lake/state/catalog_daily", help="catalog_daily directory")
    ap.add_argument("--dataset", default="GLBX.MDP3", help="dataset name stored in JSONs")
    ap.add_argument("--conn", default=None, help="psycopg conn string (or use PG* env vars)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    catalog_dir = Path(args.catalog_dir)

    # Connect: either conn string, or libpq env vars (PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE)
    conn = psycopg.connect(args.conn) if args.conn else psycopg.connect()
    conn.row_factory = dict_row

    upsert_catalog_sql = """
    INSERT INTO md.symbology_catalog_month (root, month, dataset, parent_symbol, catalog_path)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (root, month, dataset)
    DO UPDATE SET
      parent_symbol = EXCLUDED.parent_symbol,
      catalog_path  = EXCLUDED.catalog_path;
    """

    upsert_contract_sql = """
    INSERT INTO md.contracts (
      dataset, root, symbol, instrument_id, valid_from, valid_to, is_spread,
      first_seen_month, last_seen_month
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (dataset, root, symbol, instrument_id, valid_from, valid_to)
    DO UPDATE SET
      is_spread = EXCLUDED.is_spread,
      first_seen_month = LEAST(md.contracts.first_seen_month, EXCLUDED.first_seen_month),
      last_seen_month  = GREATEST(md.contracts.last_seen_month, EXCLUDED.last_seen_month);
    """

    files = list(iter_catalog_files(catalog_dir))
    print(f"Found {len(files)} catalog JSON files under {catalog_dir}")

    inserted_catalogs = 0
    upserted_contract_rows = 0

    with conn:
        with conn.cursor() as cur:
            for root, jf in tqdm(files, desc="Loading catalogs"):
                month = parse_month_from_filename(jf)
                payload = json.loads(jf.read_text())

                parent = payload.get("parent", f"{root}.FUT")
                dataset = payload.get("dataset", args.dataset)
                catalog_path = str(jf)

                if not args.dry_run:
                    cur.execute(upsert_catalog_sql, (root, month, dataset, parent, catalog_path))
                inserted_catalogs += 1

                # Raw response is under raw_response_parent_to_instrument_id
                raw = payload.get("raw_response_parent_to_instrument_id", {})
                result = raw.get("result", {}) if isinstance(raw, dict) else {}
                if not isinstance(result, dict):
                    continue

                # Each key is a child symbol; value is list of validity dicts, usually one.
                for symbol, rows in result.items():
                    if not isinstance(symbol, str) or not isinstance(rows, list) or not rows:
                        continue
                    r0 = rows[0]
                    if not isinstance(r0, dict):
                        continue
                    d0 = r0.get("d0")
                    d1 = r0.get("d1")
                    s = r0.get("s")  # instrument_id string in your JSON
                    if not (d0 and d1 and s):
                        continue

                    try:
                        valid_from = date.fromisoformat(d0)
                        valid_to = date.fromisoformat(d1)
                        instrument_id = int(s)
                    except Exception:
                        continue

                    is_spread = "-" in symbol

                    if not args.dry_run:
                        cur.execute(
                            upsert_contract_sql,
                            (
                                dataset,
                                root,
                                symbol,
                                instrument_id,
                                valid_from,
                                valid_to,
                                is_spread,
                                month,
                                month,
                            ),
                        )
                    upserted_contract_rows += 1

    print(f"Catalog months upserted: {inserted_catalogs}")
    print(f"Contract rows upserted:  {upserted_contract_rows}")

    conn.close()


if __name__ == "__main__":
    main()