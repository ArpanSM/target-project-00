#!/usr/bin/env python3
"""
common_sentence_filter.py
-------------------
• Reads a JSON-Lines product file.
• Counts how often every sentence occurs across the corpus.
• Removes the top-N most frequent sentences.
• Writes a *cleaned* JSON-Lines file plus a text report of the dropped sentences.

Usage
-----
$ python common_sentence_filter.py --input target_products.jl --output common_sentence_filtered.jl --report common_sentence_filtered_report.txt --top_n 10
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

# ---------- pre‑processing helpers -----------------------------------------
# Regex to remove the section headers that clutter the corpus
_HEADER_REGEX = re.compile(r"\b(?:Description|Specifications?|Other\s+Details)\s*:\s*", flags=re.I)
# Very simple sentence splitter: '.', '!', '?' followed by whitespace or end‑of‑string
_SENT_SPLIT_REGEX = re.compile(
    r"""
    (?<=[.!?])   # keep the delimiter
    [\s]+       # whitespace afterwards
    """,
    flags=re.VERBOSE,
)


def _preprocess(text: str) -> str:
    """Strip section headers & collapse new‑lines into spaces."""
    # Replace newlines with single spaces so we don't break sentences early
    text = text.replace("\n", " ")
    # Remove section labels
    text = _HEADER_REGEX.sub("", text)
    # Squash multiple spaces that may have been introduced
    return re.sub(r"\s+", " ", text).strip()


# ---------- sentence splitting ---------------------------------------------

def split_sentences(raw_text: str) -> List[str]:
    """Return a list of trimmed sentences (post‑preprocessing)."""
    text = _preprocess(raw_text)
    sents = _SENT_SPLIT_REGEX.split(text)
    return [s.strip() for s in sents if s.strip()]


# ---------- frequency pass --------------------------------------------------

def count_sentence_freq(lines: List[str]) -> Counter:
    """Return Counter mapping sentence → frequency across all docs."""
    freq = Counter()
    for line in lines:
        obj = json.loads(line)
        for sent in split_sentences(obj["text"]):
            freq[sent] += 1
    return freq


# ---------- cleaning pass ---------------------------------------------------

def remove_top_n_sentences(line: str, top_sentences: set) -> Tuple[str, bool]:
    """Return cleaned JSON‑line and boolean flag whether we removed something."""
    obj = json.loads(line)
    cleaned_sentences = [
        s for s in split_sentences(obj["text"]) if s not in top_sentences
    ]
    changed = len(cleaned_sentences) != len(split_sentences(obj["text"]))
    obj["text"] = " ".join(cleaned_sentences)
    return json.dumps(obj, ensure_ascii=False), changed


# ---------- main ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to original jsonl")
    ap.add_argument("--output", required=True, help="path for cleaned jsonl")
    ap.add_argument("--report", required=True, help="file listing dropped sentences")
    ap.add_argument("--top_n", type=int, default=10, help="how many sentences to drop")
    args = ap.parse_args()

    raw_lines = Path(args.input).read_text(encoding="utf-8").splitlines()

    # 1) count frequencies
    freq = count_sentence_freq(raw_lines)
    most_common: List[Tuple[str, int]] = freq.most_common(args.top_n)
    top_sentences = {s for s, _ in most_common}

    # 2) second pass – clean
    cleaned_lines = []
    removed_count = 0
    for line in raw_lines:
        cleaned, changed = remove_top_n_sentences(line, top_sentences)
        if changed:
            removed_count += 1
        cleaned_lines.append(cleaned)

    # 3) write outputs
    Path(args.output).write_text("\n".join(cleaned_lines), encoding="utf-8")

    with Path(args.report).open("w", encoding="utf-8") as fh:
        fh.write(f"Top {args.top_n} boiler‑plate sentences and counts:\n\n")
        for sent, cnt in most_common:
            fh.write(f"{cnt:3d} × {sent}\n")

    print(
        f"Cleaned {removed_count}/{len(raw_lines)} docs "
        f"(removed {args.top_n} frequent sentences)."
    )


if __name__ == "__main__":
    main()  
