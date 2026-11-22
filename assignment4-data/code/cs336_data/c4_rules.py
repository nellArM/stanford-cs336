"""C4-style line cleaning (Raffel et al., 2020, Section 2.2).

The evaluation set (Paloma C4-100-domains) was produced by C4's heuristics, so applying the
same heuristics to Common Crawl text moves the training distribution toward it: virtually
every line of the validation data ends in terminal punctuation and has at least 5 words.
"""

from __future__ import annotations

import regex as re

TERMINAL_PUNCTUATION = (".", "!", "?", '"', "”", "'", "’")
MIN_WORDS_PER_LINE = 5
MIN_SENTENCES = 3
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)]*(?:\s|$)")
_BOILERPLATE_RE = re.compile(
    r"javascript|cookies?\b.*\b(?:policy|consent|accept|browser|experience)|all rights reserved|terms of (?:use|service)"
    r"|privacy policy|click here|sign up|log ?in\b|subscribe|newsletter|©",
    re.IGNORECASE,
)


def clean_lines(text: str) -> str:
    """Keep only prose-like lines: >= 5 words, ending in terminal punctuation, no boilerplate phrases."""
    kept = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.endswith(TERMINAL_PUNCTUATION):
            continue
        if len(line.split()) < MIN_WORDS_PER_LINE:
            continue
        if _BOILERPLATE_RE.search(line):
            continue
        # Stray lines in another script inside an otherwise English page.
        if sum(ord(ch) > 0x24F for ch in line) > 0.2 * len(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def count_sentences(text: str) -> int:
    return len(_SENTENCE_END_RE.findall(text))


def page_rejection_reason(cleaned: str) -> str | None:
    """Page-level C4 rules applied to already line-cleaned text."""
    lowered = cleaned.lower()
    if "lorem ipsum" in lowered:
        return "lorem_ipsum"
    if "{" in cleaned:
        return "curly_brace"  # code / templating residue
    if count_sentences(cleaned) < MIN_SENTENCES:
        return "too_few_sentences"
    return None
