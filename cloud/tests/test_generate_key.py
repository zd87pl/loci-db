"""Unit tests for the generate_key.py CLI helpers (pure functions, no DB)."""

from __future__ import annotations

import generate_key
import pytest


def test_validate_namespace_accepts_lowercase_alnum():
    assert generate_key.validate_namespace("acme123") == "acme123"
    assert generate_key.validate_namespace("abc") == "abc"  # min length
    assert generate_key.validate_namespace("a" * 64) == "a" * 64  # max length


@pytest.mark.parametrize(
    "bad",
    [
        "loci_admin",  # underscore — the cross-tenant prefix collision case
        "foo_loci",
        "Admin",  # uppercase
        "ab",  # too short
        "a" * 65,  # too long
        "with-dash",
        "with.dot",
        "",
    ],
)
def test_validate_namespace_rejects_invalid(bad):
    with pytest.raises(ValueError, match="namespace"):
        generate_key.validate_namespace(bad)


def test_default_namespace_is_valid_and_underscore_free():
    """The CLI default must itself satisfy the isolation rule it enforces."""
    assert "_" not in generate_key.DEFAULT_NAMESPACE
    assert generate_key.validate_namespace(generate_key.DEFAULT_NAMESPACE)


def test_key_generation_shape():
    raw = generate_key.generate_raw_key()
    assert raw.startswith("loci_")
    assert len(raw) == len("loci_") + 64
    digest = generate_key.hash_key(raw)
    assert len(digest) == 64
    assert digest != generate_key.hash_key(raw + "x")
