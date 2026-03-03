from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple

PROGRESS_PATH_DEFAULT = "/data/lake/state/ingest_daily_progress.json"
SCHEMAS_DAILY = ["ohlcv-1d", "statistics"]


def month_ranges(start: date, end: date) -> List[str]:
    y, m = start.year, start.month
    out = []
    while True:
        cur = date(y, m, 1)
        if cur >= end:
            break
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out


def load_progress(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def schema_done(progress: dict, root: str, month: str, schema: str) -> bool:
    return schema in progress.get("schemas_done", {}).get(root, {}).get(month, [])


def normalized_done(progress: dict, root: str, month: str) -> bool:
    return bool(progress.get("normalized_done", {}).get(root, {}).get(month, False))


def bar(pct: float, width: int = 28) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round((pct / 100.0) * width))
    return "█" * filled + "░" * (width - filled)


def clear_screen() -> None:
    # ANSI clear + move cursor home
    sys.stdout.write("\033[2J\033[H")


def compute_root_stats(progress: dict, root: str, months: List[str]) -> Tuple[int, int, float, str]:
    total_units = len(months) * len(SCHEMAS_DAILY)
    done_units = 0

    next_pending = None  # (month, missing_schemas, needs_normalize)

    for mo in months:
        missing = [s for s in SCHEMAS_DAILY if not schema_done(progress, root, mo, s)]
        done_units += (len(SCHEMAS_DAILY) - len(missing))

        needs_norm = not normalized_done(progress, root, mo)
        if next_pending is None and (missing or needs_norm):
            next_pending = (mo, missing, needs_norm)

    pct = (done_units / total_units * 100.0) if total_units else 0.0

    if next_pending is None:
        status = "COMPLETE"
    else:
        mo, missing, needs_norm = next_pending
        if missing:
            status = f"next {mo} missing {','.join(missing)}"
        else:
            status = f"next {mo} normalize pending" if needs_norm else f"next {mo}"

    return done_units, total_units, pct, status


def compute_overall(progress: dict, roots: List[str], months: List[str]) -> Dict[str, Any]:
    total_units = len(roots) * len(months) * len(SCHEMAS_DAILY)
    done_units = 0

    norm_total = len(roots) * len(months)
    norm_done = 0

    for r in roots:
        for mo in months:
            for s in SCHEMAS_DAILY:
                if schema_done(progress, r, mo, s):
                    done_units += 1
            if normalized_done(progress, r, mo):
                norm_done += 1

    pct = (done_units / total_units * 100.0) if total_units else 0.0
    norm_pct = (norm_done / norm_total * 100.0) if norm_total else 0.0

    return {
        "schema_done": done_units,
        "schema_total": total_units,
        "schema_pct": pct,
        "norm_done": norm_done,
        "norm_total": norm_total,
        "norm_pct": norm_pct,
    }


def main():
    ap = argparse.ArgumentParser(description="Live CLI progress bars for ingest_daily_all_roots.py")
    ap.add_argument("--progress", default=PROGRESS_PATH_DEFAULT, help="Path to ingest progress JSON")
    ap.add_argument("--roots-file", default="roots.txt", help="roots.txt used for ingest")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD (exclusive-ish)")
    ap.add_argument("--interval", type=float, default=2.0, help="Refresh interval seconds")
    ap.add_argument("--top", type=int, default=20, help="Show top N roots (sorted by least complete)")
    ap.add_argument("--root", default=None, help="Show only one root (e.g., ES)")
    ap.add_argument("--show-complete", action="store_true", help="Include completed roots in the list")
    args = ap.parse_args()

    progress_path = Path(args.progress).expanduser()
    roots_path = Path(args.roots_file).expanduser()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    months = month_ranges(start, end)

    roots = [
        r.strip()
        for r in roots_path.read_text().splitlines()
        if r.strip() and not r.strip().startswith("#")
    ]
    if args.root:
        roots = [r for r in roots if r == args.root]

    if not roots:
        raise SystemExit("No roots found (check --roots-file or --root).")

    try:
        while True:
            progress = load_progress(progress_path)
            dataset = progress.get("dataset", "unknown")

            rows = []
            for r in roots:
                done_u, tot_u, pct, status = compute_root_stats(progress, r, months)
                rows.append((pct, r, done_u, tot_u, status))

            # Sort by least complete first (most urgent)
            rows.sort(key=lambda x: x[0])

            if not args.show_complete:
                rows = [x for x in rows if x[0] < 100.0 or "normalize pending" in x[4]]

            overall = compute_overall(progress, roots, months)

            clear_screen()
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"Ingest progress (daily truth)  {now}")
            print(f"dataset={dataset}  roots={len(roots)}  months={len(months)}  schemas={SCHEMAS_DAILY}")
            print(
                f"overall schemas: {overall['schema_done']}/{overall['schema_total']} "
                f"({overall['schema_pct']:.2f}%)   "
                f"months normalized: {overall['norm_done']}/{overall['norm_total']} "
                f"({overall['norm_pct']:.2f}%)"
            )
            print("")

            show = rows[: args.top] if args.top > 0 else rows
            for pct, r, done_u, tot_u, status in show:
                b = bar(pct, width=28)
                print(f"{r:>6}  {pct:6.2f}%  {b}  ({done_u}/{tot_u})  {status}")

            if len(rows) > len(show):
                print(f"\n... {len(rows) - len(show)} more roots hidden (use --top or --show-complete)")

            sys.stdout.flush()
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()