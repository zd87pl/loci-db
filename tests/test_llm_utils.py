"""Unit tests for research._llm_utils error handling."""

from __future__ import annotations

import pytest

from research._llm_utils import (
    LLMResponseError,
    extract_text,
    parse_json_array,
    parse_json_object,
    require_fields,
)


class _FakeTextBlock:
    def __init__(self, text, type_="text"):
        self.text = text
        self.type = type_


class _FakeMessage:
    def __init__(self, content):
        self.content = content


def test_extract_text_happy_path():
    msg = _FakeMessage([_FakeTextBlock('{"a": 1}')])
    assert extract_text(msg) == '{"a": 1}'


def test_extract_text_empty_content_raises():
    msg = _FakeMessage([])
    with pytest.raises(LLMResponseError, match="no content blocks"):
        extract_text(msg)


def test_extract_text_non_text_block_raises():
    class ImageBlock:
        type = "image"

    with pytest.raises(LLMResponseError, match="not a text block"):
        extract_text(_FakeMessage([ImageBlock()]))


def test_extract_text_empty_string_raises():
    msg = _FakeMessage([_FakeTextBlock("   ")])
    with pytest.raises(LLMResponseError, match="empty"):
        extract_text(msg)


def test_parse_json_object_plain():
    assert parse_json_object('{"x": 1, "y": "z"}') == {"x": 1, "y": "z"}


def test_parse_json_object_with_fence():
    text = 'some preamble\n```json\n{"x": 1}\n```\nepilogue'
    assert parse_json_object(text) == {"x": 1}


def test_parse_json_object_with_preamble_no_fence():
    text = 'Here is the JSON:\n{"x": 1}'
    assert parse_json_object(text) == {"x": 1}


def test_parse_json_object_invalid_raises():
    with pytest.raises(LLMResponseError, match="not valid JSON"):
        parse_json_object("this is not json at all")


def test_parse_json_object_rejects_array():
    with pytest.raises(LLMResponseError, match="Expected JSON object"):
        parse_json_object("[1, 2, 3]")


def test_parse_json_array_plain():
    assert parse_json_array("[1, 2, 3]") == [1, 2, 3]


def test_parse_json_array_rejects_object():
    with pytest.raises(LLMResponseError, match="Expected JSON array"):
        parse_json_array('{"x": 1}')


def test_require_fields_all_present():
    require_fields({"a": 1, "b": 2}, ["a", "b"], context="test")


def test_require_fields_missing_raises():
    with pytest.raises(LLMResponseError, match="missing required fields"):
        require_fields({"a": 1}, ["a", "b"], context="Analyzer")
