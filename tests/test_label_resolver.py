"""Tests for ``app.label_resolver`` (handheld demo task T6).

All tests use fakes: a ``FakeStaticEmbedder`` whose cosine geometry is chosen by
hand, and the real ``HashLabelEmbedder`` (which must disable the embedding
step).  The last test block runs against the real potion-base-8M model when it
is available and is skipped otherwise.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Sequence

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.label_embedding import (  # noqa: E402
    HashLabelEmbedder,
    LabelEmbedder,
    make_embedder,
    static_model_available,
)
from app.label_resolver import (  # noqa: E402
    DEFAULT_ALIASES,
    DEFAULT_DENYLIST,
    LabelResolver,
    Resolution,
    label_variants,
    singularize,
)

COCO_SUBSET = [
    "person",
    "bicycle",
    "dog",
    "bear",
    "handbag",
    "bottle",
    "wine glass",
    "cup",
    "knife",
    "hot dog",
    "chair",
    "couch",
    "potted plant",
    "dining table",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "book",
    "scissors",
    "teddy bear",
]


def _unit(v: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else list(v)


class FakeStaticEmbedder:
    """Hand-built geometry: table words live in the first 6 dims, every other word
    gets its own private axis, so unknown words are orthogonal to everything."""

    name = "static:fake"
    dim = 6 + 64

    TABLE: dict[str, list[float]] = {
        "cup": [1, 0, 0, 0, 0, 0],
        "tumbler": [0.8, 0.6, 0, 0, 0, 0],  # cos(cup) = 0.8
        "beaker": [0.5, 0.0, 0.866, 0, 0, 0],  # cos(cup) = 0.5
        "dog": [0, 0, 0, 1, 0, 0],
        "puppy": [0, 0, 0, 0.9, 0.436, 0],  # cos(dog) = 0.9
        "keyboard": [0, 0, 0, 0, 0, 1],
        "keys": [0, 0, 0, 0, 0, 1],  # cos(keyboard) = 1.0 -> deny-listed
        "key": [0, 0, 0, 0, 0.1, 0.995],  # cos(keyboard) ~ 0.995 -> deny-listed via singular
        "knife": [0, 0, 0, 0, 1, 0],
        "bottle": [0.4, 0, 0, 0, 0.9165, 0],  # cos(knife) ~ 0.92, cos(cup) 0.4
    }

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._axes: dict[str, int] = {}

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        key = " ".join(text.lower().split())
        vec = [0.0] * self.dim
        if key in self.TABLE:
            vec[:6] = [float(x) for x in self.TABLE[key]]
        else:
            axis = self._axes.setdefault(key, 6 + len(self._axes))
            assert axis < self.dim, "too many distinct unknown words for the fake"
            vec[axis] = 1.0
        return _unit(vec)

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class NoEmbedHash:
    """A hash-named embedder that fails loudly if the resolver tries to embed."""

    name = "hash:128"
    dim = 128

    def embed(self, text: str) -> list[float]:
        raise AssertionError(f"embed() must not be called for hash embedders (got {text!r})")

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError("embed_many() must not be called for hash embedders")


def _resolver(vocab=COCO_SUBSET, embedder: LabelEmbedder | None = None, **kw) -> LabelResolver:
    return LabelResolver(vocab, embedder or HashLabelEmbedder(), **kw)


# ---------------------------------------------------------------------------
# helpers: variants / singularisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("word", "singular"),
    [
        ("cups", "cup"),
        ("keys", "key"),
        ("boxes", "box"),
        ("couches", "couch"),
        ("benches", "bench"),
        ("knives", "knife"),
        ("mice", "mouse"),
        ("berries", "berry"),
        ("hot dogs", "hot dog"),
        ("glasses", None),
        ("scissors", None),
        ("cup", None),
        ("bus", None),
        ("", None),
    ],
)
def test_singularize(word: str, singular: str | None) -> None:
    assert singularize(word) == singular


def test_label_variants_order_and_dedupe() -> None:
    phrases, heads = label_variants("  My Cups? ")
    assert phrases == ["my cups", "cups", "my cup", "cup"]
    assert heads == []
    phrases, heads = label_variants("the coffee cups")
    assert phrases[0] == "the coffee cups"
    assert "coffee cups" in phrases and "coffee cup" in phrases
    assert heads == ["cups", "cup"]
    assert label_variants("") == ([], [])
    assert label_variants("?!") == ([], [])


# ---------------------------------------------------------------------------
# exact / alias / plural / determiners
# ---------------------------------------------------------------------------


def test_exact_match() -> None:
    res = _resolver().resolve("cup")
    assert res == Resolution("cup", "exact", 1.0, [])
    assert res.resolved is True
    assert res.to_dict() == {"label": "cup", "how": "exact", "score": 1.0, "suggestions": []}


@pytest.mark.parametrize(
    "text", ["Cup", " cup?", "my cup", "the cup", "my cups", "the cups!", "a cup"]
)
def test_exact_match_with_determiners_and_plurals(text: str) -> None:
    res = _resolver().resolve(text)
    assert (res.label, res.how) == ("cup", "exact")


def test_multiword_exact_beats_head_noun() -> None:
    r = _resolver()
    assert r.resolve("hot dog").label == "hot dog"
    assert r.resolve("hot dogs").label == "hot dog"
    assert r.resolve("teddy bears").label == "teddy bear"
    assert r.resolve("cell phones").label == "cell phone"


def test_alias_mug_to_cup() -> None:
    res = _resolver().resolve("mug")
    assert (res.label, res.how, res.score) == ("cup", "alias", 1.0)
    assert _resolver().resolve("my mugs").label == "cup"
    assert _resolver().resolve("the telly").label == "tv"
    assert _resolver().resolve("remote control").label == "remote"
    assert _resolver().resolve("water bottle").label == "bottle"
    assert _resolver().resolve("sofa").label == "couch"
    assert _resolver().resolve("purse").label == "handbag"


def test_alias_requires_target_in_vocabulary() -> None:
    r = _resolver(vocab=["bottle", "dog"])
    res = r.resolve("mug")
    assert res.label is None
    assert res.how == "none"


def test_exact_vocabulary_match_wins_over_alias() -> None:
    r = _resolver(vocab=[*COCO_SUBSET, "water bottle"])
    assert r.resolve("water bottle") == Resolution("water bottle", "exact", 1.0, [])


def test_head_noun_fallback() -> None:
    r = _resolver()
    assert r.resolve("coffee cup") == Resolution("cup", "exact", 1.0, [])
    assert r.resolve("my tv remote").label == "remote"
    assert r.resolve("kitchen table") == Resolution("dining table", "alias", 1.0, [])
    assert r.resolve("laptop computer").label == "laptop"


def test_plural_only_nouns_are_not_singularised() -> None:
    # "glasses" (spectacles) must not become "glass" -> alias "wine glass"
    r = _resolver()
    assert "glass" in DEFAULT_ALIASES
    assert r.resolve("glasses").label is None
    assert r.resolve("my glasses").label is None
    assert r.resolve("scissors").label == "scissors"


def test_custom_aliases_and_normalisation() -> None:
    r = _resolver(aliases={"  Beverage Holder ": "CUP"})
    assert r.resolve("beverage holder").label == "cup"
    assert r.resolve("mug").label is None  # default table replaced


def test_empty_and_junk_text() -> None:
    r = _resolver()
    for text in ["", "   ", "?", "my", "the"]:
        res = r.resolve(text)
        assert res.label is None
        assert res.how == "none"
        assert res.suggestions == []


# ---------------------------------------------------------------------------
# embedding step (static fake)
# ---------------------------------------------------------------------------


def test_embedding_resolution_with_fake_static() -> None:
    fake = FakeStaticEmbedder()
    r = _resolver(vocab=[*COCO_SUBSET, "cup"], embedder=fake)
    res = r.resolve("tumbler")
    assert res.label == "cup"
    assert res.how == "embedding"
    assert res.score == pytest.approx(0.8, abs=1e-3)
    assert res.suggestions == []


def test_embedding_threshold_respected() -> None:
    fake = FakeStaticEmbedder()
    r = _resolver(embedder=fake, threshold=0.7)
    res = r.resolve("beaker")  # cos(cup) = 0.5 < 0.7
    assert res.label is None
    assert res.how == "none"
    assert res.score == pytest.approx(0.5, abs=1e-3)
    assert res.suggestions == ["cup"]  # still suggested (>= floor 0.3)

    r2 = _resolver(embedder=FakeStaticEmbedder(), threshold=0.45)
    assert r2.resolve("beaker").label == "cup"


def test_denylist_blocks_keys_to_keyboard() -> None:
    fake = FakeStaticEmbedder()
    r = _resolver(embedder=fake)
    res = r.resolve("keys")  # cos(keyboard) == 1.0 but deny-listed
    assert res.label is None
    assert res.how == "none"
    assert res.score == pytest.approx(1.0)
    assert "keyboard" in res.suggestions  # offered as a question, never as the answer
    # singular variant goes through the same deny-list
    assert r.resolve("key").label is None
    assert r.resolve("my keys").label is None
    assert frozenset({"keys", "keyboard"}) in DEFAULT_DENYLIST


def test_denylist_skips_to_next_candidate_above_threshold() -> None:
    # exact match in the vocabulary short-circuits everything
    r = _resolver(embedder=FakeStaticEmbedder(), denylist=[("bottle", "knife")])
    assert r.resolve("bottle").how == "exact"
    # without "bottle" in the vocabulary: knife 0.92 is denied, cup 0.4 is below threshold
    r = _resolver(
        vocab=["knife", "cup"], embedder=FakeStaticEmbedder(), denylist=[("bottle", "knife")]
    )
    res = r.resolve("bottle")
    assert res.label is None
    assert res.how == "none"
    assert res.suggestions == ["knife", "cup"]
    # the same geometry with a lower threshold resolves to the next non-denied label
    r = _resolver(
        vocab=["knife", "cup"],
        embedder=FakeStaticEmbedder(),
        denylist=[("bottle", "knife")],
        threshold=0.35,
    )
    assert r.resolve("bottle") == Resolution("cup", "embedding", pytest.approx(0.4, abs=1e-3), [])
    # and with no deny-list the top hit wins
    r = _resolver(vocab=["knife", "cup"], embedder=FakeStaticEmbedder(), denylist=[])
    assert r.resolve("bottle").label == "knife"


def test_custom_denylist_accepts_any_iterable_pairs() -> None:
    r = _resolver(embedder=FakeStaticEmbedder(), denylist=[["Puppy", "DOG"]])
    assert r.resolve("puppy").label is None
    r = _resolver(embedder=FakeStaticEmbedder(), denylist=set())
    assert r.resolve("puppy").label == "dog"


def test_suggestions_sorted_and_capped() -> None:
    vocab = ["dog", "cup", "keyboard", "knife"]  # deliberately not in score order
    r = _resolver(vocab=vocab, embedder=FakeStaticEmbedder(), threshold=0.99, max_suggestions=2)
    res = r.resolve("bottle")  # knife 0.92, cup 0.4, others 0
    assert res.label is None
    assert res.score == pytest.approx(0.9165, abs=1e-3)
    assert res.suggestions == ["knife", "cup"]
    r = _resolver(vocab=vocab, embedder=FakeStaticEmbedder(), threshold=0.99, max_suggestions=1)
    assert r.resolve("bottle").suggestions == ["knife"]
    r = _resolver(vocab=vocab, embedder=FakeStaticEmbedder(), threshold=0.99, suggestion_floor=0.5)
    assert r.resolve("bottle").suggestions == ["knife"]
    r = _resolver(vocab=vocab, embedder=FakeStaticEmbedder(), threshold=0.99, max_suggestions=0)
    assert r.resolve("bottle").suggestions == []


def test_lexical_steps_run_before_embedding() -> None:
    fake = FakeStaticEmbedder()
    r = _resolver(embedder=fake)
    assert r.resolve("mug") == Resolution("cup", "alias", 1.0, [])
    assert r.resolve("cups") == Resolution("cup", "exact", 1.0, [])
    # neither query was embedded (only the vocabulary may have been)
    assert "mug" not in fake.calls and "cups" not in fake.calls


# ---------------------------------------------------------------------------
# hash embedder: no embedding step at all
# ---------------------------------------------------------------------------


def test_hash_embedder_never_embeds() -> None:
    r = _resolver(embedder=NoEmbedHash())
    assert r.semantic is False
    assert r.resolve("cup").how == "exact"
    assert r.resolve("mug").how == "alias"
    res = r.resolve("tumbler")
    assert res == Resolution(None, "none", 0.0, [])


def test_real_hash_embedder_gives_no_suggestions() -> None:
    r = _resolver(embedder=HashLabelEmbedder())
    res = r.resolve("keys")
    assert res.label is None
    assert res.suggestions == []


# ---------------------------------------------------------------------------
# vocabulary handling
# ---------------------------------------------------------------------------


def test_vocabulary_union_and_caching() -> None:
    fake = FakeStaticEmbedder()
    r = _resolver(vocab=COCO_SUBSET, embedder=fake)
    assert r.resolve("keys").label is None
    r.resolve("tumbler")  # forces the vocabulary matrix (one embed per label)
    embedded_vocab = sum(1 for c in fake.calls if c in COCO_SUBSET)
    assert embedded_vocab == len(COCO_SUBSET)

    # a browser-mode observation adds "keys" to tracked labels -> union makes it exact
    r.set_vocabulary([*COCO_SUBSET, "keys"])
    assert r.resolve("keys") == Resolution("keys", "exact", 1.0, [])
    assert r.resolve("my keys").label == "keys"
    keys_calls_before = fake.calls.count("keys")
    r.resolve("tumbler")
    # only the new label was embedded; the old ones came from the cache
    assert sum(1 for c in fake.calls if c in COCO_SUBSET) == len(COCO_SUBSET)
    assert fake.calls.count("keys") == keys_calls_before + 1

    # setting the same vocabulary again is a no-op
    before = len(fake.calls)
    r.set_vocabulary([*COCO_SUBSET, "keys"])
    r.resolve("tumbler")
    assert len(fake.calls) == before + 1  # just the query


def test_vocabulary_is_normalised_and_deduplicated() -> None:
    r = _resolver(vocab=[" Cup ", "cup", "CELL  PHONE", "", "dog"])
    assert r.vocabulary == ["cup", "cell phone", "dog"]
    assert r.resolve("cell phone").how == "exact"


def test_empty_vocabulary() -> None:
    r = _resolver(vocab=[], embedder=FakeStaticEmbedder())
    assert r.resolve("cup") == Resolution(None, "none", 0.0, [])
    r.set_vocabulary(["cup"])
    assert r.resolve("cup").how == "exact"


def test_resolution_to_dict_rounds_score() -> None:
    assert Resolution("cup", "embedding", 0.123456789, ["a"]).to_dict() == {
        "label": "cup",
        "how": "embedding",
        "score": 0.1235,
        "suggestions": ["a"],
    }


# ---------------------------------------------------------------------------
# real model (skipped without potion-base-8M)
# ---------------------------------------------------------------------------

semantic = pytest.mark.skipif(
    not static_model_available(), reason="potion-base-8M not available (LOCI_EMBED_MODEL_DIR)"
)

COCO80 = [
    *COCO_SUBSET,
    "car",
    "bird",
    "cat",
    "backpack",
    "umbrella",
    "suitcase",
    "fork",
    "spoon",
    "bowl",
    "bed",
    "toilet",
    "microwave",
    "oven",
    "sink",
    "refrigerator",
    "clock",
    "vase",
    "toothbrush",
]


@semantic
def test_real_model_resolution_table() -> None:
    r = LabelResolver(COCO80, make_embedder("static"))
    assert r.semantic is True
    assert r.resolve("mug") == Resolution("cup", "alias", 1.0, [])
    assert r.resolve("television").label == "tv"
    keys = r.resolve("keys")
    assert keys.label is None and keys.how == "none"
    assert keys.suggestions[:1] == ["keyboard"]  # "did you mean keyboard?"
    assert r.resolve("key").label is None
    assert r.resolve("glasses").label is None
    assert r.resolve("wallet").label is None
    assert r.resolve("coffee cup").label == "cup"
    teddy = r.resolve("teddy")
    assert (teddy.label, teddy.how) == ("teddy bear", "embedding")
    assert teddy.score > 0.45
    none = r.resolve("xqzvv")
    assert none.label is None and none.suggestions == []
