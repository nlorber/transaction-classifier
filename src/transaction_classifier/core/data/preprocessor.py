"""Text cleaning and normalisation for French accounting transactions."""

import re
from html import unescape

from bs4 import BeautifulSoup

# Sentinel string used by upstream accounting systems for missing values
NULL_SENTINEL = "NULL"

_TAG_RE = re.compile(r"<[^>]+>")
_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_ACCENTED_KEEP = re.compile(r"[^a-z0-9àâäéèêëïîôùûüçœæ\s\-]")
_ASCII_ONLY = re.compile(r"[^a-z0-9\s\-]")


def strip_html(raw: str) -> str:
    """Remove HTML markup and decode character entities.

    Uses a fast regex pass first; falls back to BeautifulSoup
    only when residual tags remain after the initial pass.
    """
    if not raw or raw == NULL_SENTINEL:
        return ""

    text = unescape(raw)
    text = _BREAK_RE.sub(" ", text)

    # Fast path: drop obvious tags with a regex
    text = _TAG_RE.sub(" ", text)

    # If anything that looks like a tag survived, use the full parser
    if "<" in text:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")

    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_text(text: str, preserve_accents: bool = True) -> str:
    """Lower-case and sanitise French accounting text for vectorisation."""
    if not text:
        return ""

    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text)

    pattern = _ACCENTED_KEEP if preserve_accents else _ASCII_ONLY
    text = pattern.sub(" ", text)

    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_html_for_extraction(text: str) -> str:
    """Decode HTML entities and strip tags for structured field extraction.

    Lighter than ``strip_html`` — no BeautifulSoup fallback — suitable for
    pre-extraction cleaning of banking comment fields.
    """
    if not text or text == NULL_SENTINEL or str(text).lower() == "nan":
        return ""
    from html import unescape

    decoded = unescape(str(text)).replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", decoded).strip()
