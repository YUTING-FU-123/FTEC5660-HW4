"""The abstention trade-off, one line per question: how much of what it asks for the corpus mentions.

    env -u PYTHONPATH .venv/bin/python tools/abstain_probe.py

Sorted by coverage so the gap between answerable and unanswerable questions is visible. The chosen
threshold (hw4.METRIC_COVERAGE_MIN) should sit inside that gap, with margin on both sides.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hw4  # noqa: E402


def main() -> None:
    folder = Path("public_test")
    corpus = hw4.load_corpus(folder / "corpus")
    chunks_by_file = {n: hw4.chunk_document(n, h) for n, h in corpus.items()}
    chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
              if len(c.page_content) <= hw4.MAX_CHUNK_CHARS]
    questions = hw4.load_json(folder / "questions.json")
    truth = hw4.load_json(folder / "ground_truth.json")
    index = hw4.build_index(chunks)

    rows = []
    for qid, question in questions.items():
        gold = truth[qid]
        parsed = hw4.parse_query(question)
        keys = [hw4.company_key(c) for c in parsed["companies"]]
        known = [k for k in keys if k in index["postings"]["company"]]
        asked = hw4._asked_terms(question, parsed)
        if not known:
            rows.append((0.0, qid, "not-in-corpus", question))
            continue
        cand = hw4._candidate_ids(index, known, parsed["fiscal_year"], parsed["doc_type"])
        raw = index["bm25"].get_scores(hw4._tokenize(question))
        ranked = sorted(cand, key=lambda i: -hw4._boost(index, i, raw[i], parsed["fiscal_year"],
                                                        parsed["doc_type"], question, asked))
        pool = set()
        for i in ranked[:12]:
            pool |= index["fingerprints"][i]
        coverage = len(asked & pool) / len(asked) if asked else 1.0
        missing = sorted(asked - pool)
        label = "ANSWERABLE" if gold["evidence"] else "unanswerable"
        rows.append((coverage, qid, label, question, missing))

    rows.sort()
    print(f"threshold now: {hw4.METRIC_COVERAGE_MIN}\n")
    for row in rows:
        coverage, qid, label, question, *rest = row
        missing = rest[0] if rest else []
        print(f"{coverage:5.2f}  {qid}  {label:22} {question[:78]}")
        if missing and label.startswith("unans"):
            print(f"{'':9}missing: {missing}")

    answerable = [r[0] for r in rows if len(r) > 4 and r[2] == "ANSWERABLE"]
    unanswerable = [r[0] for r in rows if len(r) > 4 and r[2] == "unanswerable"]
    if answerable and unanswerable:
        print(f"\nlowest answerable   : {min(answerable):.2f}")
        print(f"highest unanswerable: {max(unanswerable):.2f}")
        print(f"gap                 : {min(answerable) - max(unanswerable):+.2f}")


if __name__ == "__main__":
    main()
