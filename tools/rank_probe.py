"""Where did the evidence chunk rank? Diagnoses recall failures chunk by chunk.

    env -u PYTHONPATH .venv/bin/python tools/rank_probe.py q040 q017 ...

For each question: the chunk that holds the evidence, its BM25 rank inside the candidate set, the
number of chunks sharing its table, and whether it survived the 5-chunk cut.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hw4  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("qids", nargs="*")
    parser.add_argument("--data-folder", default="public_test")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = hw4.load_corpus(folder / "corpus")
    chunks_by_file = {n: hw4.chunk_document(n, h) for n, h in corpus.items()}
    chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
              if len(c.page_content) <= hw4.MAX_CHUNK_CHARS]
    questions = hw4.load_json(folder / "questions.json")
    truth = hw4.load_json(folder / "ground_truth.json") or {}
    index = hw4.build_index(chunks)
    scores = index["bm25"]

    for qid in args.qids:
        question, gold = questions[qid], truth[qid]
        parsed = hw4.parse_query(question)
        keys = [hw4.company_key(c) for c in parsed["companies"]]
        known = [k for k in keys if k in index["postings"]["company"]]
        cand = hw4._candidate_ids(index, known, parsed["fiscal_year"], parsed["doc_type"])
        raw = scores.get_scores(hw4._tokenize(question))
        ranked = sorted(cand, key=lambda i: -hw4._boost(index, i, raw[i], parsed["fiscal_year"],
                                                        parsed["doc_type"], question))
        picked_docs = hw4.retrieve(index, question)
        picked = {next((i for i, c in enumerate(chunks)
                        if c.page_content == doc.page_content and c.metadata.get("source") == doc.metadata.get("source")), -1)
                  for doc in picked_docs}
        print("=" * 100)
        print(f"{qid}: {question}")
        print(f"  parsed={parsed}  candidates={len(cand)}")
        for group in gold.get("evidence", []):
            for file, text in group[:3]:
                holders = [i for i, c in enumerate(chunks)
                           if c.metadata.get("source") == file and hw4.contains(c.page_content, text)]
                for i in holders:
                    rank = ranked.index(i) + 1 if i in ranked else None
                    print(f"  evidence {file}:{text} -> chunk#{i} (section {chunks[i].metadata.get('section')!r},"
                          f" kind {chunks[i].metadata.get('kind')}, len {len(chunks[i].page_content)})"
                          f" rank={rank} picked={i in picked} bm25={raw[i]:.2f}")
                if not holders:
                    print(f"  evidence {file}:{text} -> NOT IN ANY CHUNK")
        print(f"  top5: {[ (chunks[i].metadata.get('source'), chunks[i].metadata.get('section')) for i in ranked[:5] ]}")


if __name__ == "__main__":
    main()
