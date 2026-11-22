"""Masking of personally identifiable information: e-mail addresses, US phone numbers, IPv4 addresses."""

from __future__ import annotations

import regex as re

EMAIL_MASK = "|||EMAIL_ADDRESS|||"
PHONE_MASK = "|||PHONE_NUMBER|||"
IP_MASK = "|||IP_ADDRESS|||"

# local-part@domain.tld; the local part may not start or end with a dot.
_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9_%+-]+(?:\.[A-Za-z0-9_%+-]+)*@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?![\w-])")

# North-American numbers: optional +1, 3-digit area code (optionally parenthesised),
# 3-digit exchange and 4-digit line number, separated by nothing, space, dot or dash.
_PHONE_RE = re.compile(
    r"""
    (?<![\w.+-])                      # not in the middle of a longer number/word
    (?:\+?1[\s.-]?)?                  # country code
    (?:\(\d{3}\)|\d{3})               # area code
    [\s.-]{0,2}
    \d{3}
    [\s.-]?
    \d{4}
    (?![\w-]|\.\d)                    # not followed by more digits/letters
    """,
    re.VERBOSE,
)

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IPV4_RE = re.compile(rf"(?<![\w.]){_OCTET}(?:\.{_OCTET}){{3}}(?![\w]|\.\d)")


def mask_emails(text: str) -> tuple[str, int]:
    return _EMAIL_RE.subn(EMAIL_MASK, text)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    return _PHONE_RE.subn(PHONE_MASK, text)


def mask_ips(text: str) -> tuple[str, int]:
    return _IPV4_RE.subn(IP_MASK, text)


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """Apply all three masks.  IPs go before phone numbers so dotted quads are not half-eaten."""
    text, n_email = mask_emails(text)
    text, n_ip = mask_ips(text)
    text, n_phone = mask_phone_numbers(text)
    return text, {"email": n_email, "ip": n_ip, "phone": n_phone}
