"""Score table_to_text() on the public tables.json with the homework's own table_score().

    env -u PYTHONPATH .venv/bin/python tools/check_tables.py [--data-folder public_test] [-v]

Task 1.1 is graded with exactly this function, so a table that scores 1.0 here is safe on the
private set (same 12 table kinds).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hw4 import table_score, table_to_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("-v", "--verbose", action="store_true", help="print the text of each table")
    args = parser.parse_args()

    tables = json.load(open(Path(args.data_folder) / "tables.json", encoding="utf-8"))
    rows = []
    for table in tables:
        try:
            text = table_to_text(table["html"])
        except Exception as exc:  # noqa: BLE001 - a crashing table is a 0 for that table
            print(f"FAIL {table['source']}: {exc!r}")
            rows.append((table["caption"][:44], 0.0, 0, 0))
            continue
        rows.append((table["caption"][:44], table_score(text, table), len(table["cells"]),
                     len([line for line in text.splitlines() if line.strip()])))
        if args.verbose:
            print("=" * 100)
            print(table["caption"])
            print("-" * 100)
            print(text)

    print(f"{'caption':46} {'score':>6} {'cells':>6} {'lines':>6}")
    for caption, score, cells, lines in rows:
        flag = "" if score >= 0.999 else ("  <-- check" if score < 0.9 else "  <- partial")
        print(f"{caption:46} {score:6.3f} {cells:6d} {lines:6d}{flag}")
    mean = sum(r[1] for r in rows) / len(rows)
    print(f"\nmean over {len(rows)} tables: {mean:.4f}   -> Task 1.1 = {8 * mean:.2f} / 8")


if __name__ == "__main__":
    main()
