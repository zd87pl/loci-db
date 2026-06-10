"""Tests for loci.payload_filters translation helpers."""

from __future__ import annotations

from qdrant_client.models import FieldCondition, MatchAny, MatchValue

from loci.payload_filters import extra_filter_to_conditions, extra_filter_to_memory

# ---------------------------------------------------------------------------
# extra_filter_to_conditions
# ---------------------------------------------------------------------------


def test_conditions_none_returns_empty_list() -> None:
    assert extra_filter_to_conditions(None) == []


def test_conditions_empty_dict_returns_empty_list() -> None:
    assert extra_filter_to_conditions({}) == []


def test_conditions_scalar_uses_match_value() -> None:
    conditions = extra_filter_to_conditions({"user_id": "abc"})
    assert len(conditions) == 1
    cond = conditions[0]
    assert isinstance(cond, FieldCondition)
    assert cond.key == "user_id"
    assert isinstance(cond.match, MatchValue)
    assert cond.match.value == "abc"


def test_conditions_int_scalar() -> None:
    conditions = extra_filter_to_conditions({"count": 7})
    assert isinstance(conditions[0].match, MatchValue)
    assert conditions[0].match.value == 7


def test_conditions_bool_scalar() -> None:
    conditions = extra_filter_to_conditions({"active": True})
    assert isinstance(conditions[0].match, MatchValue)
    assert conditions[0].match.value is True


def test_conditions_list_uses_match_any() -> None:
    conditions = extra_filter_to_conditions({"tag": ["a", "b", "c"]})
    assert len(conditions) == 1
    cond = conditions[0]
    assert isinstance(cond, FieldCondition)
    assert cond.key == "tag"
    assert isinstance(cond.match, MatchAny)
    assert cond.match.any == ["a", "b", "c"]


def test_conditions_tuple_uses_match_any() -> None:
    conditions = extra_filter_to_conditions({"tag": ("a", "b")})
    cond = conditions[0]
    assert isinstance(cond.match, MatchAny)
    assert cond.match.any == ["a", "b"]


def test_conditions_set_uses_match_any() -> None:
    conditions = extra_filter_to_conditions({"tag": {"a", "b"}})
    cond = conditions[0]
    assert isinstance(cond.match, MatchAny)
    # Sets are unordered; compare contents irrespective of order.
    assert sorted(cond.match.any) == ["a", "b"]


def test_conditions_frozenset_uses_match_any() -> None:
    conditions = extra_filter_to_conditions({"tag": frozenset({1, 2})})
    cond = conditions[0]
    assert isinstance(cond.match, MatchAny)
    assert sorted(cond.match.any) == [1, 2]


def test_conditions_empty_list_still_match_any() -> None:
    conditions = extra_filter_to_conditions({"tag": []})
    assert len(conditions) == 1
    cond = conditions[0]
    assert isinstance(cond.match, MatchAny)
    assert cond.match.any == []


def test_conditions_multiple_keys_mixed_shapes() -> None:
    extra_filter = {"user_id": "abc", "tag": ["x", "y"], "count": 3}
    conditions = extra_filter_to_conditions(extra_filter)
    assert len(conditions) == 3

    by_key = {c.key: c for c in conditions}
    assert set(by_key) == {"user_id", "tag", "count"}

    assert isinstance(by_key["user_id"].match, MatchValue)
    assert by_key["user_id"].match.value == "abc"

    assert isinstance(by_key["tag"].match, MatchAny)
    assert by_key["tag"].match.any == ["x", "y"]

    assert isinstance(by_key["count"].match, MatchValue)
    assert by_key["count"].match.value == 3


def test_conditions_preserves_insertion_order() -> None:
    extra_filter = {"a": 1, "b": 2, "c": 3}
    conditions = extra_filter_to_conditions(extra_filter)
    assert [c.key for c in conditions] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# extra_filter_to_memory
# ---------------------------------------------------------------------------


def test_memory_none_returns_empty_dict() -> None:
    assert extra_filter_to_memory(None) == {}


def test_memory_empty_dict_returns_empty_dict() -> None:
    assert extra_filter_to_memory({}) == {}


def test_memory_scalar_passthrough() -> None:
    assert extra_filter_to_memory({"user_id": "abc"}) == {"user_id": "abc"}


def test_memory_int_scalar_passthrough() -> None:
    assert extra_filter_to_memory({"count": 7}) == {"count": 7}


def test_memory_bool_scalar_passthrough() -> None:
    assert extra_filter_to_memory({"active": False}) == {"active": False}


def test_memory_list_wrapped_in_any() -> None:
    assert extra_filter_to_memory({"tag": ["a", "b"]}) == {"tag": {"any": ["a", "b"]}}


def test_memory_tuple_wrapped_in_any() -> None:
    assert extra_filter_to_memory({"tag": ("a", "b")}) == {"tag": {"any": ["a", "b"]}}


def test_memory_set_wrapped_in_any() -> None:
    result = extra_filter_to_memory({"tag": {"a", "b"}})
    assert set(result) == {"tag"}
    assert sorted(result["tag"]["any"]) == ["a", "b"]


def test_memory_frozenset_wrapped_in_any() -> None:
    result = extra_filter_to_memory({"tag": frozenset({1, 2})})
    assert sorted(result["tag"]["any"]) == [1, 2]


def test_memory_empty_list_wrapped_in_any() -> None:
    assert extra_filter_to_memory({"tag": []}) == {"tag": {"any": []}}


def test_memory_multiple_keys_mixed_shapes() -> None:
    extra_filter = {"user_id": "abc", "tag": ["x", "y"], "count": 3}
    result = extra_filter_to_memory(extra_filter)
    assert result == {
        "user_id": "abc",
        "tag": {"any": ["x", "y"]},
        "count": 3,
    }


def test_memory_does_not_mutate_input() -> None:
    extra_filter = {"tag": ["x", "y"]}
    extra_filter_to_memory(extra_filter)
    # Input dict and its list value remain unchanged.
    assert extra_filter == {"tag": ["x", "y"]}
