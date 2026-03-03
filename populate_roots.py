import argparse
from pathlib import Path
import psycopg

def read_roots(path: Path):
    roots = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        roots.append(s)
    return sorted(set(roots))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots-file", default="roots.txt")
    ap.add_argument("--dataset", default="GLBX.MDP3")
    ap.add_argument("--conn", default=None,
                    help="Optional connection string. If omitted uses local defaults.")
    args = ap.parse_args()

    roots_file = Path(args.roots_file).expanduser().resolve()
    if not roots_file.exists():
        raise SystemExit(f"roots file not found: {roots_file}")

    roots = read_roots(roots_file)
    print(f"Loaded {len(roots)} roots from {roots_file}")

    conn = psycopg.connect(args.conn) if args.conn else psycopg.connect()
    with conn, conn.cursor() as cur:
        for root in roots:
            cur.execute(
                """
                INSERT INTO md.roots (dataset, root)
                VALUES (%s, %s)
                ON CONFLICT (dataset, root)
                DO UPDATE SET active=true;
                """,
                (args.dataset, root),
            )

    print("md.roots populated successfully.")

if __name__ == "__main__":
    main()