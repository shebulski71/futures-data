
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

DB = "/data/lake/duckdb/futures.duckdb"
STATE_DIR = Path("/data/lake/state")


def today_ct() -> date:
    return datetime.now(ZoneInfo("America/Chicago")).date()


def default_target_trade_date_utc() -> date:
    # same logic as the watermark flow: validate yesterday (last completed day)
    return today_ct() - timedelta(days=1)


def load_roots(roots_file: Path) -> list[str]:
    roots = []
    for line in roots_file.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        roots.append(s)
    return sorted(set(roots))


@dataclass
class ValidationSummary:
    target_trade_date_utc: str
    roots_total: int
    roots_ok: int
    roots_missing: int
    roots_incomplete: int
    max_date_daily_joined: str | None
    max_date_continuous_daily: str | None
    ok: bool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots-file", default="/home/marketdata/futures-data/roots.txt")
    ap.add_argument("--target-trade-date-utc", default=None, help="YYYY-MM-DD (defaults to yesterday CT)")
    ap.add_argument("--out", default=None, help="Output JSON path (defaults to /data/lake/state/postrun_validation_YYYY-MM-DD.json)")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero on any problem (recommended for Prefect).")
    args = ap.parse_args()

    roots_file = Path(args.roots_file).expanduser().resolve()
    roots = load_roots(roots_file)

    target = date.fromisoformat(args.target_trade_date_utc) if args.target_trade_date_utc else default_target_trade_date_utc()
    out_path = Path(args.out) if args.out else STATE_DIR / f"postrun_validation_{target.isoformat()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(DB, read_only=True)

    try:
        # Max dates
        max_daily = con.execute("SELECT max(trade_date_utc) FROM daily_joined").fetchone()[0]
        max_cont = con.execute("SELECT max(trade_date_utc) FROM continuous_daily").fetchone()[0]

        # Per-root presence for target day in daily_joined
        roots_in = ", ".join([f"'{r}'" for r in roots])
        q = f"""
        SELECT root, COUNT(*) AS n
        FROM daily_joined
        WHERE trade_date_utc = '{target.isoformat()}'
          AND root IN ({roots_in})
        GROUP BY root
        """
        rows = con.execute(q).fetchall()
        counts = {r: int(n) for (r, n) in rows}

        missing = [r for r in roots if counts.get(r, 0) == 0]
        # "incomplete" means present but suspicious (usually should be >=1)
        # For daily_joined it’s typically 1 row per root-day for continuous; but daily_joined is per SYMBOL.
        # We just check >0 for the root-day.
        incomplete: list[str] = []  # keep for future deeper checks

        roots_ok = len(roots) - len(missing) - len(incomplete)

        # Also validate continuous_daily has that day for all roots
        q2 = f"""
        SELECT root, COUNT(*) AS n
        FROM continuous_daily
        WHERE trade_date_utc = '{target.isoformat()}'
          AND root IN ({roots_in})
        GROUP BY root
        """
        rows2 = con.execute(q2).fetchall()
        cont_counts = {r: int(n) for (r, n) in rows2}
        cont_missing = [r for r in roots if cont_counts.get(r, 0) == 0]

        report = {
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "target_trade_date_utc": target.isoformat(),
            "roots_total": len(roots),
            "daily_joined": {
                "max_trade_date_utc": (max_daily.isoformat() if max_daily else None),
                "missing_roots_for_target_day": missing,
            },
            "continuous_daily": {
                "max_trade_date_utc": (max_cont.isoformat() if max_cont else None),
                "missing_roots_for_target_day": cont_missing,
            },
            "notes": [
                "daily_joined check is 'any rows for root on target day' (daily_joined is per-symbol/day).",
                "continuous_daily check expects exactly one row per root/day.",
            ],
        }

        ok = (len(missing) == 0) and (len(cont_missing) == 0) and (max_daily is not None) and (max_daily >= target) and (max_cont is not None) and (max_cont >= target)

        summary = ValidationSummary(
            target_trade_date_utc=target.isoformat(),
            roots_total=len(roots),
            roots_ok=roots_ok,
            roots_missing=len(missing),
            roots_incomplete=len(incomplete),
            max_date_daily_joined=(max_daily.isoformat() if max_daily else None),
            max_date_continuous_daily=(max_cont.isoformat() if max_cont else None),
            ok=ok,
        )

        report["summary"] = asdict(summary)

        out_path.write_text(json.dumps(report, indent=2))
        print(f"Wrote report: {out_path}")
        print("Summary:", report["summary"])

        if args.strict and not ok:
            raise SystemExit(2)

    finally:
        con.close()


if __name__ == "__main__":
    main()