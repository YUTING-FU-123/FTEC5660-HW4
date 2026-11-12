"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================

import html as html_lib
import math
import unicodedata

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# BM25 is imported lazily inside build_index so that table/chunk stages never pay for it.

# Task 3: share of the words the question asks about that must appear somewhere in the candidates.
# Tuned on public_test: the lowest coverage among answerable questions is well above this, while the
# "figure not disclosed" questions fall below it (no chunk mentions e.g. lobbying expenses at all).
METRIC_COVERAGE_MIN = float(os.environ.get("HW4_COVERAGE", "0.30"))

# --------------------------------------------------------------------------- helpers


def _clean_text(value: Any) -> str:
    """Collapse whitespace but keep every character of a printed figure (commas, parentheses)."""
    text = str(value if value is not None else "")
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def _cell_text(cell: Any) -> str:
    return _clean_text(cell.get_text(" ", strip=True))


def _span(cell: Any) -> int:
    """The colspan of a cell as an int (1 when absent or malformed)."""
    raw = cell.get("colspan")
    try:
        return max(1, int(str(raw)))
    except (TypeError, ValueError):
        return 1


def _table_parts(table_html: str) -> tuple[str, list[str], list[list[tuple[str, int, str]]]]:
    """(caption, header cells, body rows as (tag, colspan, text)) for one <table> element."""
    soup = BeautifulSoup(table_html, "lxml")
    table = soup.find("table") or soup
    caption_el = table.find("caption")
    caption = _cell_text(caption_el) if caption_el else ""

    thead = table.find("thead")
    headers: list[str] = []
    if thead:
        header_rows = thead.find_all("tr")
        if header_rows:
            headers = [_cell_text(c) for c in header_rows[-1].find_all(["th", "td"])]

    body_rows: list[list[tuple[str, int, str]]] = []
    thead_trs = set(id(tr) for tr in (thead.find_all("tr") if thead else []))
    for tr in table.find_all("tr"):
        if id(tr) in thead_trs:
            continue
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        body_rows.append([(c.name, _span(c), _cell_text(c)) for c in cells])
    return caption, headers, body_rows


def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    caption, headers, body_rows = _table_parts(table_html)
    lines: list[str] = []
    if caption:
        lines.append(caption)

    for row in body_rows:
        # A single wide <th> is a section label ("Assets", "Operating activities"), not a data row.
        if len(row) == 1 and row[0][0] == "th" and row[0][1] > 1:
            if row[0][2]:
                lines.append(row[0][2])
            continue

        label = row[0][2]
        values = [text for _, colspan, text in row[1:] if colspan == 1]
        if not label or not values:
            continue
        data_headers = headers[-len(values):] if len(headers) >= len(values) else headers + [""] * (len(values) - len(headers))
        pairs = [(hdr or f"column {i + 1}", value) for i, (hdr, value) in enumerate(zip(data_headers, values))]

        single = " | ".join([label] + [f"{hdr}: {value}" for hdr, value in pairs if value])
        if len(single) <= 400:
            lines.append(single)
        else:  # keep every value on its own short line so nothing exceeds the 400-character limit
            lines.extend(f"{label} | {hdr}: {value}" for hdr, value in pairs if value)
    return "\n".join(lines)


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    limit = MAX_CHUNK_CHARS
    soup = BeautifulSoup(html, "lxml")
    body = soup.body or soup
    company, fiscal_year, doc_type = _document_identity(soup, body)
    _register_company(company)

    header = f"[{company} | FY{fiscal_year} | {doc_type} | {{section}}]"
    chunks: list[Document] = []

    def emit(section: str, text: str, kind: str) -> None:
        head = header.format(section=section)
        text = text.strip()
        if not text:
            return
        room = limit - len(head) - 2
        if len(text) <= room:
            pieces = [text]
        elif kind == "table":
            pieces = _pack_lines(text.splitlines(), room)
        else:
            pieces = _split_with_overlap(text, room)
        for i, piece in enumerate(pieces):
            chunks.append(Document(page_content=f"{head}\n{piece}",
                                   metadata={"source": file_name, "company": company,
                                             "fiscal_year": fiscal_year, "doc_type": doc_type,
                                             "section": section, "kind": kind, "part": i}))

    section = "front matter"
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            emit(section, "\n".join(buffer), "text")
            buffer.clear()

    for node in _walk_block_nodes(body):
        if node.name in ("h1", "h2", "h3"):
            flush()
            section = _clean_text(node.get_text(" ", strip=True))
        elif node.name == "table":
            flush()
            emit(section, table_to_text(str(node)), "table")
        else:
            text = _clean_text(node.get_text(" ", strip=True))
            if text:
                buffer.append(text)
    flush()
    return chunks


def _walk_block_nodes(root: Any):
    """Yield the body's block-level elements in reading order, skipping the nav bar and scripts."""
    for child in root.children:
        name = getattr(child, "name", None)
        if name is None:
            continue
        if name in ("nav", "script", "style", "head"):
            continue
        if name in ("h1", "h2", "h3", "table", "p", "li", "dt", "dd", "caption"):
            yield child
            continue
        if name in ("ul", "ol", "dl"):
            yield from _walk_block_nodes(child)
            continue
        if name in ("tr", "td", "th", "thead", "tbody"):
            continue
        yield from _walk_block_nodes(child)


def _document_identity(soup: Any, body: Any) -> tuple[str, int, str]:
    """(company, fiscal_year, doc_type) from the first lines of the document."""
    heading = soup.find("h1")
    company = _clean_text(heading.get_text(" ", strip=True)) if heading else ""
    company = re.sub(r"\s*\((?:[A-Za-z0-9.:]+\s*)+\)\s*$", "", company).strip()  # drop "(NYSE: NGR)"

    # The deck of the document: everything up to the first section heading, plus the <title>.
    lead_parts: list[str] = []
    for node in body.children:
        name = getattr(node, "name", None)
        if name in ("h2", "table"):
            break
        if name in ("h1", "p", "div", "span"):
            lead_parts.append(_clean_text(node.get_text(" ", strip=True)))
    title = _clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    head = " ".join([title] + [p for p in lead_parts if p])

    doc_type = "annual_report"
    if re.search(r"equity research|results review", head, re.I):
        doc_type = "research_report"
    elif re.search(r"earnings (?:conference )?call|conference call", head, re.I):
        doc_type = "earnings_call"

    fiscal_year = 0
    for pattern in (r"\bFY\s*(\d{4})\b",
                    r"annual report\D{0,32}?(\d{4})",
                    r"full year\D{0,32}?(\d{4})",
                    r"\b(20\d{2})\b"):
        match = re.search(pattern, head, re.I)
        if match:
            fiscal_year = int(match.group(1))
            break
    return company, fiscal_year, doc_type


def _pack_lines(lines: list[str], room: int) -> list[str]:
    """Group lines into pieces of at most `room` characters, never splitting a line."""
    pieces, current = [], []
    for line in lines:
        if current and len("\n".join(current + [line])) > room:
            pieces.append("\n".join(current))
            current = []
        current.append(line[:room])
    if current:
        pieces.append("\n".join(current))
    return pieces


def _split_with_overlap(text: str, room: int, overlap: int = 200) -> list[str]:
    """Split long prose on whitespace, keeping a small overlap so a sentence is never cut in half."""
    if room <= 0:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        end = min(start + room, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + int(room * 0.6), end)
            end = cut if cut > start else end
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    found: list[str] = []
    known = sorted(_COMPANY_REGISTRY.values(), key=len, reverse=True)
    q_norm = normalize(question)

    for name in known:  # longest first, so "Northgate Robotics" wins over a bare "Northgate"
        key = company_key(name)
        if key and f" {key} " in q_norm:
            found.append(name)

    # A company the corpus does not hold looks like a near miss of one it does
    # ("Northgate Robotic Systems" next to "Northgate Robotics"). The question still names it,
    # so it belongs in the answer even though nothing can be retrieved for it.
    known_keys = [company_key(name) for name in known]
    first_tokens = {key.split()[0] for key in known_keys if key}
    for phrase in _capitalized_phrases(question):
        key = company_key(phrase)
        if not key or _is_intermediary(phrase):
            continue
        if any(key == company_key(name) for name in found):
            continue
        if any(key in other or other in key for other in known_keys):
            continue
        if key.split()[0] in first_tokens:
            found.append(phrase)

    return {"companies": found,
            "fiscal_year": _question_year(question),
            "doc_type": _question_doc_type(question)}


# --------------------------------------------------------------------------- query parsing helpers

_COMPANY_REGISTRY: dict[str, str] = {}   # company_key -> name as printed in the corpus


def _register_company(name: str) -> None:
    """Remember every company the corpus mentions, so parse_query() needs no hard-coded list."""
    key = company_key(name)
    if key:
        _COMPANY_REGISTRY.setdefault(key, name)


def _capitalized_phrases(question: str) -> list[str]:
    """Runs of capitalised words ("Northgate Robotic Systems"), longest first, possessives removed."""
    phrases = []
    for match in re.finditer(r"\b[A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*){0,3}", question):
        phrase = match.group(0).rstrip(".'-")
        phrases.append(phrase)
    return sorted(set(phrases), key=len, reverse=True)


def _is_intermediary(phrase: str) -> bool:
    """Brokers, research houses and audit firms are named in questions but are not the company studied."""
    return bool(re.search(r"\b(equity research|research|securities|capital|partners|llp|audit|"
                          r"associates|advisors|advisers|bank|broking|investments?)\b", phrase, re.I))


def _question_year(question: str) -> int | None:
    match = re.search(r"\bFY\s*'?(\d{2,4})\b", question, re.I)
    if match:
        value = match.group(1)
        return int(value) if len(value) == 4 else 2000 + int(value)
    match = re.search(r"(?:31 December|December|year ended|year to)\D{0,12}(\d{4})", question, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(20\d{2})\b", question)
    return int(match.group(1)) if match else None


def _question_doc_type(question: str) -> str:
    if re.search(r"earnings (?:conference )?call|conference call|fourth[- ]quarter|"
                 r"\bQ4\b|on (?:its|the) (?:latest )?call|management (?:guided|guide)", question, re.I):
        return "earnings_call"
    if re.search(r"\brating\b|target price|\bforecast|estimate|results review|\bbroker|analyst|"
                 r"price target", question, re.I):
        return "research_report"
    return "annual_report"


def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    from rank_bm25 import BM25Okapi

    tokenized = [_tokenize(chunk.page_content) for chunk in chunks]
    postings: dict[str, dict[str, list[int]]] = {"company": {}, "doc_type": {}, "year": {}}
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        key = company_key(meta.get("company", ""))
        postings["company"].setdefault(key, []).append(i)
        postings["doc_type"].setdefault(str(meta.get("doc_type", "")), []).append(i)
        postings["year"].setdefault(str(meta.get("fiscal_year", "")), []).append(i)

    years = [int(m["fiscal_year"]) for m in (c.metadata or {} for c in chunks)
             if str(m.get("fiscal_year", "")).lstrip("-").isdigit()]
    return {
        "chunks": chunks,
        "bm25": BM25Okapi(tokenized) if tokenized else None,
        "postings": postings,
        # The FY<oldest> report carries one comparative column, and the calls/notes look one year ahead.
        "year_min": (min(years) - 1) if years else None,
        "year_max": (max(years) + 1) if years else None,
        "token_sets": [set(tokens) for tokens in tokenized],
        "fingerprints": [_fingerprint(tokens) for tokens in tokenized],
    }


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.
    """
    if not index or not index.get("chunks"):
        return []
    chunks: list[Document] = index["chunks"]
    parsed = {"companies": [], "fiscal_year": None, "doc_type": "annual_report"}
    if not _switch("HW4_NO_PARSING"):
        parsed = parse_query(question)

    keys = [company_key(name) for name in parsed.get("companies") or []]
    known = [key for key in keys if key in index["postings"]["company"]]
    if _switch("HW4_NO_PARSING"):
        known = list(index["postings"]["company"])  # naive baseline: ignore what the question is about
    if keys and not known:
        return []                                    # every company named is outside the corpus

    year = None if _switch("HW4_NO_PARSING") else parsed.get("fiscal_year")
    if year is not None and index["year_min"] is not None and not (index["year_min"] <= year <= index["year_max"]):
        return []                                    # a year no document can cover (e.g. FY2021)

    doc_type = None if _switch("HW4_NO_PARSING") else parsed.get("doc_type")
    if _switch("HW4_NO_METADATA"):
        candidates = range(len(chunks))
    else:
        candidates = _candidate_ids(index, known, year, doc_type)

    scores = index["bm25"].get_scores(_tokenize(question))
    asked = _asked_terms(question, parsed)
    ranked = sorted(candidates,
                    key=lambda i: -_boost(index, i, scores[i], year, doc_type, question, asked))

    if _switch("HW4_NO_COMPARISON") or len(known) < 2:
        picked = ranked[:MAX_RETRIEVED]
    else:
        picked = _merge_per_company(index, ranked, known)

    if not picked:
        return []
    picked_docs = [chunks[i] for i in picked]

    if not _switch("HW4_NO_ABSTAIN") and not _switch("HW4_NO_PARSING"):
        if not _answers_the_question(index, asked, picked, ranked):
            return []
        if _switch("HW4_LLM_CHECK") and not _llm_says_answerable(index, question, picked):
            return []
    return picked_docs[:MAX_RETRIEVED]


# --------------------------------------------------------------------------- retrieval helpers

STOPWORDS = {
    "a", "an", "the", "of", "for", "in", "on", "at", "to", "and", "or", "did", "do", "does", "was",
    "were", "is", "are", "be", "been", "what", "which", "who", "whom", "whose", "how", "much", "many",
    "its", "it", "their", "they", "this", "that", "these", "those", "with", "from", "by", "as", "than",
    "more", "higher", "lower", "company", "companies", "usd", "million", "millions", "percent", "year",
    "fiscal", "end", "ended", "december", "31", "due", "report", "reported", "value", "amount", "total",
    "there", "any", "all", "about", "would", "could", "should", "had", "has", "have", "also",
}


def _switch(name: str) -> bool:
    """Ablation switch for Task 5 (an environment variable that turns one component off)."""
    return os.environ.get(name, "") not in ("", "0", "false", "False")


def _stem(word: str) -> str:
    """A very small stemmer: enough to make 'headquartered'/'headquarters' and 'announced'/'announce' agree."""
    for suffix in ("ings", "ing", "ies", "es", "ed", "d", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _fingerprint(tokens) -> set[str]:
    """Compare words by their first five characters: robust to the morphology the documents use."""
    prints = set()
    for token in tokens:
        token = str(token)
        if len(token) >= 5 and not token.replace(",", "").replace(".", "").isdigit():
            prints.add(token[:5])
        else:
            prints.add(token)
    return prints


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9,.\-]*", str(text)):
        word = raw.strip(".-").lower()
        if not word or re.fullmatch(r"(?:fy|q)\d{2,4}", word):
            continue                                 # "FY2025" is metadata, not content
        plain = word.replace(",", "")
        if plain != word:
            tokens.append(plain)                     # "1,574.7" should also match "1574.7"
        if word in STOPWORDS or plain in STOPWORDS:
            continue
        tokens.append(word)
        if word != plain:
            tokens.append(plain)
    return tokens or ["_"]


def _content_words(text: str) -> list[str]:
    return [_stem(t) for t in _tokenize(text) if t not in STOPWORDS and not t.replace(",", "").isdigit()
            and not re.fullmatch(r"\d+(\.\d+)?", t.replace(",", ""))]


def _covers(meta: dict) -> set[int]:
    """The fiscal years whose figures a document prints (the comparative column counts)."""
    year = meta.get("fiscal_year")
    if not isinstance(year, int):
        return set()
    if meta.get("doc_type") == "annual_report":
        return {year, year - 1}
    return {year, year + 1}


def _candidate_ids(index: Any, companies: list[str], year: int | None, doc_type: str | None) -> list[int]:
    chunks: list[Document] = index["chunks"]
    if not companies:
        return list(range(len(chunks)))
    ids: set[int] = set()
    for company in companies:
        ids.update(index["postings"]["company"].get(company, []))
    if not ids:
        return []
    typed = [i for i in ids if doc_type and chunks[i].metadata.get("doc_type") == doc_type]
    if len(typed) >= 3:
        ids = set(typed)
    if year is not None:
        cited = [i for i in ids if year in _covers(chunks[i].metadata or {})]
        if len(cited) >= 3:
            ids = set(cited)
    return sorted(ids)


def _boost(index: Any, i: int, score: float, year: int | None, doc_type: str | None,
           question: str, asked: set[str] | None = None) -> float:
    meta = index["chunks"][i].metadata or {}
    value = score
    if year is not None and year in _covers(meta):
        value *= 1.25                                   # the document actually prints that year
    if doc_type and meta.get("doc_type") == doc_type:
        value *= 1.10
    if meta.get("kind") == "table" and re.search(
            r"how much|in USD millions|in percent|what (?:was|were)|which company|more |higher|lower|"
            r"larger|number of|how many", question, re.I):
        value *= 1.15                                   # a figure question is answered by a table line
    if asked:
        # A chunk that actually names the asked-for item ("Inventories", "Total assets", "Employees")
        # beats a chunk that merely talks about the company at length.
        covered = len(asked & index["fingerprints"][i]) / len(asked)
        value *= 1.0 + 0.9 * covered
        section_terms = _fingerprint(_tokenize(str(meta.get("section", ""))))
        if section_terms:
            value *= 1.0 + 0.5 * len(asked & section_terms) / len(asked)
    return value


def _merge_per_company(index: Any, ranked: list[int], companies: list[str]) -> list[int]:
    """A comparison needs a chunk for each company: give every company its best chunks first."""
    chunks: list[Document] = index["chunks"]
    buckets: dict[str, list[int]] = {company: [] for company in companies}
    for i in ranked:
        meta = chunks[i].metadata or {}
        key = company_key(meta.get("company", ""))
        if key not in buckets or len(buckets[key]) >= 3:
            continue
        if len(buckets[key]) >= 2:                      # third slot only from an unused document
            used = {chunks[j].metadata.get("source") for j in buckets[key]}
            if meta.get("source") in used:
                continue
        buckets[key].append(i)
    ordered: list[int] = []
    for round_index in range(3):
        for company in companies:
            if round_index < len(buckets[company]):
                ordered.append(buckets[company][round_index])
    return ordered[:MAX_RETRIEVED]


def _asked_terms(question: str, parsed: dict) -> set[str]:
    """The words the question asks about, minus the company names (they are handled by metadata)."""
    asked = set(_content_words(question))
    for name in parsed.get("companies") or []:
        asked -= set(_content_words(name))
    return _fingerprint(asked)


def _answers_the_question(index: Any, asked: set[str], picked: list[int], ranked: list[int]) -> bool:
    """Task 3: does any candidate chunk actually mention what the question asks for?"""
    if not asked:
        return True
    if not picked:
        return False
    pool: set[str] = set()
    for i in picked:
        pool |= index["fingerprints"][i]
    for i in ranked[:12]:
        pool |= index["fingerprints"][i]
    coverage = len(asked & pool) / len(asked)
    return coverage >= METRIC_COVERAGE_MIN


ANSWERABILITY_PROMPT = """Context:
{context}

Question: {question}

Is the item the question asks for stated somewhere in the context above (the exact figure, name or
fact)? If the context does not contain it at all, the reports never disclose it.
Answer YES if the context contains it, NO if it does not. Answer with one word:"""


def _llm_says_answerable(index: Any, question: str, picked: list[int]) -> bool:
    """An optional second opinion from the reader model (allowed in retrieve, not in the other two)."""
    try:
        llm = make_llm(temperature=0.0, max_tokens=4)
        context = "\n\n".join(index["chunks"][i].page_content[:900] for i in picked[:3])
        verdict = str(llm.invoke(ANSWERABILITY_PROMPT.format(context=context, question=question)).content)
    except Exception:  # noqa: BLE001 - a failed check must never lose a good retrieval
        return True
    return "NO" not in verdict.upper()


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
