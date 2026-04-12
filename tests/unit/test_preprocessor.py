"""Unit tests for text preprocessing."""

from transaction_classifier.core.data.preprocessor import normalize_text, strip_html


def test_clean_html_removes_tags():
    assert strip_html("<b>hello</b>") == "hello"


def test_clean_html_decodes_entities():
    result = strip_html("a&amp;b")
    assert "a" in result and "b" in result


def test_clean_html_replaces_br():
    result = strip_html("line1<br/>line2")
    assert "line1" in result
    assert "line2" in result


def test_clean_html_empty_input():
    assert strip_html("") == ""
    assert strip_html("NULL") == ""


def test_preprocess_text_lowercases():
    assert normalize_text("HELLO WORLD") == "hello world"


def test_preprocess_text_normalizes_whitespace():
    assert normalize_text("hello   world") == "hello world"


def test_preprocess_text_keeps_french_accents():
    result = normalize_text("éléphant café", preserve_accents=True)
    assert "éléphant" in result
    assert "café" in result


def test_preprocess_text_removes_accents():
    result = normalize_text("éléphant", preserve_accents=False)
    assert "é" not in result


def test_preprocess_text_empty():
    assert normalize_text("") == ""
