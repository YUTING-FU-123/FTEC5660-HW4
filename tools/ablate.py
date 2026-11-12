"""Task 5: the ablation study. Runs the same pipeline with one component switched at a time.

    env -u PYTHONPATH .venv/bin/python tools/ablate.py [--data-folder public_test]

Every configuration is the previous one plus exactly one change (or, for B0, the Tutorial-7 baseline).
Writes tools/ablation.csv and prints the table that goes into README.md.

The switches are read by hw4.py from the environment:
    HW4_NO_TABLETEXT   table_to_text() falls back to BeautifulSoup get_text()
    HW4_NAIVE_CHUNK    one flat text + RecursiveCharacterTextSplitter, required metadata only
    HW4_NO_PARSING     ignore parse_query() entirely (no company/year/doc_type routing)
    HW4_NO_METADATA    no metadata filtering inside retrieve()
    HW4_NO_COMPARISON  no per-company merge for comparison questions
    HW4_NO_ABSTAIN     never return []
    HW4_LLM_CHECK      add the optional LLM answerability check
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASELINE = {
    "HW4_NO_TABLETEXT": "1",
    "HW4_NAIVE_CHUNK": "1",
    "HW4_NO_PARSING": "1",
    "HW4_NO_METADATA": "1",
    "HW4_NO_COMPARISON": "1",
    "HW4_NO_ABSTAIN": "1",
}

CONFIGS: list[tuple[str, str, dict[str, str]]] = [
    ("B0", "naive RAG: get_text + RecursiveCharacterTextSplitter + BM25 top-5 over the whole corpus, "
           "no query parsing, never []", {}),
    ("V1", "+ table_to_text (keep each value with its row label and column header)", {"HW4_NO_TABLETEXT": "0"}),
    ("V2", "+ section chunks with a contextual header (metadata: section, kind)", {"HW4_NAIVE_CHUNK": "0"}),
    ("V3", "+ parse_query and metadata filtering (company / fiscal year / doc type)",
     {"HW4_NO_PARSING": "0", "HW4_NO_METADATA": "0"}),
    ("V4", "+ per-company retrieval for comparison questions", {"HW4_NO_COMPARISON": "0"}),
    ("V5", "+ abstention gates (company not in corpus, year not covered, item never mentioned)",
     {"HW4_NO_ABSTAIN": "0"}),
    ("V6", "+ LLM answerability check (rejected by hand: costs recall)", {"HW4_LLM_CHECK": "1"}),
]

POINT_RE = re.compile(r"^  Task ([\d.]+)\s+([\d.]+) / (\d+)$", re.M)
METRIC_RES = {
    "recall": re.compile(r"recall@5 on answerable questions: ([\d.]+)%"),
    "accuracy": re.compile(r"Task 4   answer accuracy: ([\d.]+)%"),
    "kept": re.compile(r"Task 3   answerable questions with chunks: ([\d.]+)%"),
    "rejected": re.compile(r"unanswerable questions with \[\]: ([\d.]+)%"),
    "seconds": re.compile(r"^Time: ([\d.]+)s", re.M),
    "cost": re.compile(r"Generator usage: .*'cost_usd': ([\d.]+)"),
    "chunks": re.compile(r"(\d+) chunks to index"),
}


def run(config_env: dict[str, str], out_csv: Path, data_folder: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HW4_")}
    env.update(config_env)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, str(ROOT / "hw4.py"), "--data-folder", data_folder,
                           "--output", str(out_csv)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    text = proc.stdout + proc.stderr
    row: dict[str, str | float] = {}
    for task, value, _ in POINT_RE.findall(text):
        row[f"task{task}"] = float(value)
    for key, pattern in METRIC_RES.items():
        match = pattern.search(text)
        row[key] = float(match.group(1)) if match else ""
    total = re.search(r"Total\s+([\d.]+) /", text)
    row["total"] = float(total.group(1)) if total else 0.0
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("--start", default="B0", help="resume from this configuration")
    args = parser.parse_args()

    jobs = []
    env: dict[str, str] = dict(BASELINE)
    for name, note, changes in CONFIGS:
        env = {**env, **changes}
        jobs.append((name, note, dict(env)))
    start_index = next(i for i, job in enumerate(jobs) if job[0] == args.start)

    rows = []
    for name, note, config_env in jobs[start_index:]:
        print(f"--- {name}: {note}", flush=True)
        result = run(config_env, Path(f"/tmp/ablation_{name}.csv"), args.data_folder)
        result["config"] = name
        result["change"] = note
        rows.append(result)
        print("    " + " ".join(f"{k}={v}" for k, v in result.items() if k not in ("config", "change")),
              flush=True)

    columns = ["config", "change", "task1.1", "task1.2", "task2.1", "task2.2", "task2.3", "task3",
               "task4", "recall", "accuracy", "kept", "rejected", "chunks", "seconds", "cost", "total"]
    out = ROOT / "tools" / "ablation.csv"
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
