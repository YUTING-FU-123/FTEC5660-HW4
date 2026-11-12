"""Score parse_query() on the public questions with the homework's own parse_score().

    env -u PYTHONPATH .venv/bin/python tools/eval_parse.py [--data-folder public_test] [-v]

The company registry is filled by chunk_document(), exactly as the runner does it (it chunks the
corpus before it parses the questions), so this script walks the corpus first.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hw4 import (chunk_document, company_key, load_corpus, load_json, parse_query,  # noqa: E402
                 parse_score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every question")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    for name, html in corpus.items():
        chunk_document(name, html)

    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json") or {}

    scores, field_misses = [], Counter()
    for qid, question in questions.items():
        gold = ground_truth.get(qid)
        parsed = parse_query(question)
        if not gold:
            print(qid, parsed)
            continue
        score = parse_score(parsed, gold)
        scores.append(score)
        expected = gold["query"]
        problems = []
        got_keys = {company_key(c) for c in (parsed.get("companies") or [])}
        want_keys = {company_key(c) for c in expected["companies"]}
        if got_keys != want_keys:
            problems.append(f"companies {parsed.get('companies')} != {expected['companies']}")
            field_misses["companies"] += 1
        if parsed.get("fiscal_year") != expected["fiscal_year"]:
            problems.append(f"year {parsed.get('fiscal_year')} != {expected['fiscal_year']}")
            field_misses["fiscal_year"] += 1
        if parsed.get("doc_type") != expected["doc_type"]:
            problems.append(f"doc_type {parsed.get('doc_type')} != {expected['doc_type']}")
            field_misses["doc_type"] += 1
        if problems and args.verbose:
            print(f"[{score:.2f}] {qid}: {question}\n      " + "; ".join(problems))

    mean = sum(scores) / len(scores)
    print(f"\nscored {len(scores)} questions, mean parse_score = {mean:.4f}  -> Task 2.2 = {10 * mean:.2f} / 10")
    print("field misses:", dict(field_misses) or "none")


if __name__ == "__main__":
    main()
