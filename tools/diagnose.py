"""Per-question diagnosis: what was parsed, what was retrieved, whether it hit, and why.

    env -u PYTHONPATH .venv/bin/python tools/diagnose.py [--data-folder public_test] [--only misses]

Prints one block per question plus a summary of the failure modes, so a fix can be aimed at a
category instead of guessed at. Uses the homework's own retrieval_hit()/answer_correct().
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hw4  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("--only", choices=["misses", "all", "abstained"], default="misses")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = hw4.load_corpus(folder / "corpus")
    chunks_by_file = {name: hw4.chunk_document(name, html) for name, html in corpus.items()}
    chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
              if len(c.page_content) <= hw4.MAX_CHUNK_CHARS]
    questions = hw4.load_json(folder / "questions.json")
    truth = hw4.load_json(folder / "ground_truth.json") or {}
    index = hw4.build_index(chunks)

    modes = Counter()
    for qid, question in questions.items():
        gold = truth.get(qid) or {}
        parsed = hw4.parse_query(question)
        got = hw4.retrieve(index, question)
        answerable = bool(gold.get("evidence"))
        hit = hw4.retrieval_hit(got, gold) if answerable else None
        abstained = not got
        if not answerable and not abstained:
            modes["unanswerable but returned chunks"] += 1
        if answerable and abstained:
            modes["answerable but abstained"] += 1
        if answerable and not abstained and not hit:
            modes["answerable, chunks, no hit"] += 1
        if answerable and hit:
            modes["answerable + hit"] += 1

        interesting = (args.only == "all"
                       or (args.only == "misses" and (hit is False or (answerable and abstained)))
                       or (args.only == "abstained" and abstained))
        if not interesting:
            continue
        print("=" * 100)
        print(f"{qid} [{gold.get('category')}] {'ANSWERABLE' if answerable else 'NOT ANSWERABLE'}: {question}")
        print(f"   parsed     : {parsed}")
        print(f"   expected   : {gold.get('query')}")
        print(f"   retrieved  : {sorted({c.metadata.get('source') for c in got}) or '[]'}")
        if answerable:
            need = [[f"{file}:{text}" for file, text in group] for group in gold.get("evidence", [])]
            print(f"   evidence   : {need}")
            print(f"   hit        : {hit}")
        else:
            asked = set(hw4._content_words(question)) - set(
                hw4._content_words(" ".join(parsed.get("companies") or [])))
            pool = set()
            for c in got:
                pool.update(hw4._stem(t) for t in hw4._tokenize(c.page_content))
            coverage = len(asked & pool) / len(asked) if asked else 1.0
            print(f"   asked words: {sorted(asked)}")
            print(f"   coverage   : {coverage:.2f} (threshold {hw4.METRIC_COVERAGE_MIN})")

    print("\n--- failure modes ---")
    for mode, count in modes.most_common():
        print(f"  {count:3d}  {mode}")
    answerable = [q for q, g in truth.items() if g.get("evidence")]
    print(f"  answerable questions: {len(answerable)}")


if __name__ == "__main__":
    main()
