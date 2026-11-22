"""HTML -> plain text."""

from __future__ import annotations

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """Decode raw HTML bytes (UTF-8 first, detected encoding otherwise) and extract the visible text."""
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        try:
            html = html_bytes.decode(encoding, errors="replace")
        except (LookupError, TypeError):
            html = html_bytes.decode("utf-8", errors="replace")
    try:
        return extract_plain_text(html)
    except Exception:
        return None
