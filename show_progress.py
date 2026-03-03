from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Iterable, Dict, Any

PROGRESS_PATH_DEFAULT = "/data/lake/state/ingest_daily_progress.json"
SCHEMAS_DAILY = ["ohlcv-1d", "statistics"]


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def month_ranges(start: date, end: date) -> Iterable[str]:
    y, m = start.year, start.month
    while True:
        cur = date(y, m, 1)
        if cur >= end:
            break
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            m = 1
            y += 1


def load_progress(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def schema_done(progress: dict, root: str, month: str, schema: str) -> bool:
    return schema in progress.get("schemas_done", {}).get(root, {}).get(month, [])


def normalized_done(progress: dict, root: str, month: str) -> bool:
    return bool(progress.get("normalized_done", {}).get(root, {}).get(month, False))


# ---------------------------------------------------
# Build JSON report
# ---------------------------------------------------
def build_report(progress: dict, roots: list[str], months: list[str]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "dataset": progress.get("dataset"),
        "roots": {},
        "overall": {},
    }

    total_units = len(roots) * len(months) * len(SCHEMAS_DAILY)
    done_units = 0

    norm_total = len(roots) * len(months)
    norm_done = 0

    for root in roots:
        root_total = len(months) * len(SCHEMAS_DAILY)
        root_done = 0

        pending = []

        for mo in months:
            missing = [s for s in SCHEMAS_DAILY if not schema_done(progress, root, mo, s)]

            # schema completion accounting
            root_done += (len(SCHEMAS_DAILY) - len(missing))
            done_units += (len(SCHEMAS_DAILY) - len(missing))

            if not normalized_done(progress, root, mo):
                pending.append(
                    {
                        "month": mo,
                        "missing_schemas": missing,
                        "normalized": False,
                    }
                )
            else:
                norm_done += 1

        pct = (root_done / root_total * 100.0) if root_total else 0.0

        report["roots"][root] = {
            "schema_units_done": root_done,
            "schema_units_total": root_total,
            "percent_complete": round(pct, 3),
            "pending": pending,
        }

    overall_pct = (done_units / total_units * 100.0) if total_units else 0.0
    norm_pct = (norm_done / norm_total * 100.0) if norm_total else 0.0

    report["overall"] = {
        "schema_units_done": done_units,
        "schema_units_total": total_units,
        "percent_complete": round(overall_pct, 3),
        "normalized_months_done": norm_done,
        "normalized_months_total": norm_total,
        "normalized_percent": round(norm_pct, 3),
    }

    return report


# ---------------------------------------------------
# Pretty table output
# ---------------------------------------------------
def print_table(report: dict):
    print(f"Dataset: {report.get('dataset')}\n")

    for root, data in report["roots"].items():
        pct = data["percent_complete"]
        done = data["schema_units_done"]
        total = data["schema_units_total"]

        if not data["pending"]:
            status = "COMPLETE"
        else:
            nxt = data["pending"][0]
            missing = ",".join(nxt["missing_schemas"]) if nxt["missing_schemas"] else "none"
            status = f"next={nxt['month']} missing={missing}"

        print(f"{root:>6}  {pct:6.2f}%  ({done}/{total})  {status}")

    print("")
    ov = report["overall"]
    print(
        f"Overall schemas done: {ov['schema_units_done']}/{ov['schema_units_total']}  "
        f"({ov['percent_complete']:.2f}%)"
    )
    print(
        f"Overall months normalized: {ov['normalized_months_done']}/"
        f"{ov['normalized_months_total']}  ({ov['normalized_percent']:.2f}%)"
    )


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Show ingest progress (table or JSON).")
    ap.add_argument("--progress", type=str, default=PROGRESS_PATH_DEFAULT)
    ap.add_argument("--roots-file", type=str, default="roots.txt")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--root", type=str, default=None, help="Filter to single root")
    ap.add_argument("--json", action="store_true", help="Print JSON output")
    ap.add_argument("--out", type=str, default=None, help="Write JSON to file")
    args = ap.parse_args()

    progress_path = Path(args.progress)
    roots_path = Path(args.roots_file)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    roots = [
        r.strip()
        for r in roots_path.read_text().splitlines()
        if r.strip() and not r.startswith("#")
    ]

    if args.root:
        roots = [r for r in roots if r == args.root]

    months = list(month_ranges(start, end))
    progress = load_progress(progress_path)

    report = build_report(progress, roots, months)

    if args.json:
        js = json.dumps(report, indent=2)
        print(js)
        if args.out:
            Path(args.out).write_text(js)
    else:
        print_table(report)


if __name__ == "__main__":
    main()