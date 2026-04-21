"""Shared LLM response parsing utilities.

Centralises the robust extraction of JSON objects/arrays from Anthropic
messages, along with helpful error messages when the model returns
malformed or empty content.
"""

from __future__ import annotations

import json
import re
from typing import Any


class LLMResponseError(RuntimeError):
    """Raised when an LLM response cannot be parsed as expected JSON."""


_OBJECT_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_ARRAY_FENCE_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
_OBJECT_RAW_RE = re.compile(r"(\{[\s\S]*\})")
_ARRAY_RAW_RE = re.compile(r"(\[[\s\S]*\])")


def _strip_fences(text: str, kind: str) -> str:
    """Return the inner JSON blob if wrapped in ```json ... ``` fences."""
    pattern = _OBJECT_FENCE_RE if kind == "object" else _ARRAY_FENCE_RE
    m = pattern.search(text)
    if m:
        return m.group(1)
    raw = _OBJECT_RAW_RE if kind == "object" else _ARRAY_RAW_RE
    m = raw.search(text)
    return m.group(1) if m else text


def extract_text(message: Any) -> str:
    """Safely extract the first text block from an Anthropic message.

    Raises:
        LLMResponseError: if the message has no content blocks or the first
            block is not a text block with a non-empty ``text`` attribute.
    """
    content = getattr(message, "content", None)
    if not content:
        raise LLMResponseError("LLM response has no content blocks")
    first = content[0]
    text = getattr(first, "text", None)
    if text is None:
        block_type = getattr(first, "type", type(first).__name__)
        raise LLMResponseError(
            f"LLM response first block is not a text block (got type={block_type!r})"
        )
    if not isinstance(text, str) or not text.strip():
        raise LLMResponseError("LLM response text block is empty")
    return text


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating markdown fences."""
    candidate = _strip_fences(text.strip(), "object")
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError as exc:
        preview = text[:200].replace("\n", " ")
        raise LLMResponseError(
            f"LLM output was not valid JSON: {exc.msg} at pos {exc.pos}. "
            f"Preview: {preview!r}"
        ) from exc
    if not isinstance(result, dict):
        raise LLMResponseError(
            f"Expected JSON object, got {type(result).__name__}"
        )
    return result


def parse_json_array(text: str) -> list[Any]:
    """Parse a JSON array from model output, tolerating markdown fences."""
    candidate = _strip_fences(text.strip(), "array")
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError as exc:
        preview = text[:200].replace("\n", " ")
        raise LLMResponseError(
            f"LLM output was not valid JSON: {exc.msg} at pos {exc.pos}. "
            f"Preview: {preview!r}"
        ) from exc
    if not isinstance(result, list):
        raise LLMResponseError(
            f"Expected JSON array, got {type(result).__name__}"
        )
    return result


def require_fields(data: dict[str, Any], fields: list[str], context: str) -> None:
    """Ensure *data* contains all *fields*; raise with a descriptive error otherwise."""
    missing = [f for f in fields if f not in data]
    if missing:
        raise LLMResponseError(
            f"{context}: LLM response missing required fields: {missing}"
        )
