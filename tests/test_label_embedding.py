"""Tests for ``app.label_embedding`` (handheld demo task T6).

The hash embedder must reproduce the legacy ``spatial_memory._label_embedding``
byte for byte; the static embedder (potion-base-8M read with tokenizers +
safetensors) is exercised only when the model directory is present
(``LOCI_EMBED_MODEL_DIR`` or ``deploy/handheld/models/potion-base-8M``) and
checked against ``model2vec`` when that package is importable.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.label_embedding import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    HashLabelEmbedder,
    LabelEmbedder,
    StaticLabelEmbedder,
    cosine,
    hash_label_embedding,
    make_embedder,
    normalize_label,
    static_model_available,
)

EMBED_DIM = 128


def _legacy_label_embedding(label: str) -> list[float]:
    """Verbatim copy of the pre-T6 ``spatial_memory._label_embedding``."""
    normalized = label.strip().lower()
    key = f"label:{normalized}"
    digest = hashlib.sha256(key.encode()).digest()
    seed = struct.unpack("<I", digest[:4])[0]

    vec: list[float] = []
    for i in range(EMBED_DIM):
        if i < 8:
            char_val = sum(ord(c) for c in normalized) / max(len(normalized), 1)
            freq = (i // 2 + 1) * math.pi
            if i % 2 == 0:
                vec.append(math.sin(char_val / 128.0 * freq))
            else:
                vec.append(math.cos(char_val / 128.0 * freq))
        else:
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            val = (seed / 0x7FFFFFFF) * 2.0 - 1.0
            vec.append(val)

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 1e-8:
        vec = [v / norm for v in vec]
    return vec


LABELS = ["cup", "keys", "cell phone", "  Dining  Table ", "", "MUG!", "héllo wörld", "a" * 300]


# ---------------------------------------------------------------------------
# hash embedding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", LABELS)
def test_hash_label_embedding_matches_legacy_bytes(label: str) -> None:
    ours = hash_label_embedding(label)
    legacy = _legacy_label_embedding(label)
    assert ours == legacy  # exact float equality, not approx
    assert len(ours) == 128


def test_hash_label_embedding_is_unit_length() -> None:
    for label in ["cup", "bottle", "x"]:
        vec = hash_label_embedding(label)
        assert math.sqrt(sum(v * v for v in vec)) == pytest.approx(1.0, abs=1e-9)


def test_hash_label_embedding_custom_dim() -> None:
    vec = hash_label_embedding("cup", dim=16)
    assert len(vec) == 16
    # The first 8 (sin/cos) dims are shared with the 128-d vector before normalisation,
    # so their ratios agree.
    full = hash_label_embedding("cup")
    assert vec[0] / vec[1] == pytest.approx(full[0] / full[1])


def test_hash_label_embedding_case_and_whitespace_insensitive() -> None:
    assert hash_label_embedding("Cup") == hash_label_embedding(" cup ")
    assert hash_label_embedding("cup") != hash_label_embedding("cups")


def test_hash_embedder_protocol_and_values() -> None:
    emb = HashLabelEmbedder()
    assert isinstance(emb, LabelEmbedder)
    assert emb.name == "hash:128"
    assert emb.dim == 128
    assert emb.embed("cup") == _legacy_label_embedding("cup")
    assert emb.embed("cup") == hash_label_embedding("cup")
    many = emb.embed_many(["cup", "Cup", "dog"])
    assert many[0] == many[1]
    assert many[0] != many[2]
    assert all(len(v) == 128 for v in many)


def test_hash_embedder_returns_fresh_lists() -> None:
    emb = HashLabelEmbedder()
    first = emb.embed("cup")
    first[0] = 42.0
    assert emb.embed("cup")[0] != 42.0


def test_normalize_label() -> None:
    assert normalize_label("  Cell   Phone\t") == "cell phone"
    assert normalize_label("CUP") == "cup"
    assert normalize_label("") == ""


# ---------------------------------------------------------------------------
# cosine
# ---------------------------------------------------------------------------


def test_cosine_basic() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine([3.0, 4.0], [6.0, 8.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cosine([1.0], [1.0, 2.0])


def test_cosine_of_hash_embeddings_is_self_one_other_small() -> None:
    a = hash_label_embedding("cup")
    b = hash_label_embedding("mug")
    assert cosine(a, a) == pytest.approx(1.0)
    assert abs(cosine(a, b)) < 0.5  # hash space: no semantics, just not identical


# ---------------------------------------------------------------------------
# make_embedder / availability
# ---------------------------------------------------------------------------


def test_make_embedder_default_is_hash_with_empty_env() -> None:
    emb = make_embedder(env={})
    assert isinstance(emb, HashLabelEmbedder)
    assert emb.name == "hash:128"
    assert emb.dim == 128


def test_make_embedder_explicit_hash_ignores_model_dir(tmp_path) -> None:
    emb = make_embedder("hash", model_dir=tmp_path / "nope", env={"LOCI_LABEL_EMBEDDER": "static"})
    assert isinstance(emb, HashLabelEmbedder)


def test_make_embedder_env_kind_hash() -> None:
    emb = make_embedder(env={"LOCI_LABEL_EMBEDDER": "HASH"})
    assert isinstance(emb, HashLabelEmbedder)


def test_make_embedder_static_raises_without_model(tmp_path) -> None:
    with pytest.raises((ImportError, OSError, KeyError)):
        make_embedder("static", model_dir=tmp_path / "missing", env={})
    with pytest.raises((ImportError, OSError, KeyError)):
        make_embedder(env={"LOCI_LABEL_EMBEDDER": "static", "LOCI_EMBED_MODEL_DIR": str(tmp_path)})


def test_make_embedder_auto_falls_back_to_hash(tmp_path) -> None:
    emb = make_embedder("auto", model_dir=tmp_path / "missing", env={})
    assert isinstance(emb, HashLabelEmbedder)
    emb = make_embedder(env={"LOCI_LABEL_EMBEDDER": "auto", "LOCI_EMBED_MODEL_DIR": str(tmp_path)})
    assert isinstance(emb, HashLabelEmbedder)


def test_make_embedder_unknown_kind() -> None:
    with pytest.raises(ValueError):
        make_embedder("bogus", env={})


def test_static_model_available_false_for_empty_dir(tmp_path) -> None:
    assert static_model_available(tmp_path) is False
    assert static_model_available(env={"LOCI_EMBED_MODEL_DIR": str(tmp_path)}) is False


def test_static_model_available_false_when_deps_missing(tmp_path, monkeypatch) -> None:
    for name in ("model.safetensors", "tokenizer.json", "config.json"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "tokenizers", None)  # makes `import tokenizers` fail
    assert static_model_available(tmp_path) is False


def test_default_model_dir_points_into_repo() -> None:
    parts = DEFAULT_MODEL_DIR.parts
    assert parts[-4:] == ("deploy", "handheld", "models", "potion-base-8M")
    assert (DEFAULT_MODEL_DIR.parents[3] / "demo_spatial" / "app").is_dir()


def test_static_embedder_missing_files_raise_oserror(tmp_path) -> None:
    pytest.importorskip("tokenizers")
    pytest.importorskip("safetensors")
    with pytest.raises(OSError):
        StaticLabelEmbedder(tmp_path / "missing")
    (tmp_path / "model.safetensors").write_bytes(b"")
    with pytest.raises(OSError):
        StaticLabelEmbedder(tmp_path)  # tokenizer.json missing


def test_static_embedder_without_embeddings_tensor_raises_keyerror(tmp_path) -> None:
    pytest.importorskip("tokenizers")
    st = pytest.importorskip("safetensors.numpy")
    np = pytest.importorskip("numpy")
    st.save_file(
        {"a": np.zeros((2, 4), dtype=np.float32), "b": np.zeros((2, 4), dtype=np.float32)},
        str(tmp_path / "model.safetensors"),
    )
    (tmp_path / "tokenizer.json").write_text("{}")
    with pytest.raises(KeyError):
        StaticLabelEmbedder(tmp_path)


def test_static_embedder_tiny_synthetic_model(tmp_path) -> None:
    """A three-token WordLevel model exercises tokenise -> drop UNK -> mean -> normalise."""
    tokenizers = pytest.importorskip("tokenizers")
    st = pytest.importorskip("safetensors.numpy")
    np = pytest.importorskip("numpy")
    from tokenizers import models, normalizers, pre_tokenizers

    vocab = {"[UNK]": 0, "cup": 1, "red": 2, "dog": 3}
    tok = tokenizers.Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(tmp_path / "tokenizer.json"))
    weights = np.array(
        [[9.0, 9.0, 9.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    st.save_file({"embeddings": weights}, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text('{"normalize": true}')

    emb = StaticLabelEmbedder(tmp_path)
    assert isinstance(emb, LabelEmbedder)
    assert emb.dim == 3
    assert emb.name == f"static:{tmp_path.name}"
    assert emb.embed("cup") == pytest.approx([1.0, 0.0, 0.0])
    # mean of cup (1,0,0) and red (0,2,0) = (0.5, 1, 0) -> normalised
    assert emb.embed("Red CUP") == pytest.approx(
        [0.5 / math.sqrt(1.25), 1.0 / math.sqrt(1.25), 0.0]
    )
    # unknown tokens are dropped, not averaged in as [UNK]
    assert emb.embed("cup zebra") == pytest.approx([1.0, 0.0, 0.0])
    assert emb.embed("zebra") == [0.0, 0.0, 0.0]
    assert emb.embed_many(["cup", "dog"]) == [emb.embed("cup"), emb.embed("dog")]
    assert static_model_available(tmp_path) is True
    assert isinstance(make_embedder("auto", model_dir=tmp_path, env={}), StaticLabelEmbedder)
    assert isinstance(make_embedder(env={"LOCI_EMBED_MODEL_DIR": str(tmp_path)}), HashLabelEmbedder)
    assert isinstance(
        make_embedder(env={"LOCI_LABEL_EMBEDDER": "auto", "LOCI_EMBED_MODEL_DIR": str(tmp_path)}),
        StaticLabelEmbedder,
    )


def test_make_embedder_auto_uses_static_when_constructor_succeeds(tmp_path, monkeypatch) -> None:
    """``auto`` must not fall back when the model is present and loadable."""
    import app.label_embedding as mod

    fake = MagicMock(spec=["name", "dim", "embed", "embed_many"])
    fake.name = "static:fake"
    fake.dim = 4
    monkeypatch.setattr(mod, "static_model_available", lambda *_a, **_k: True)
    monkeypatch.setattr(mod, "StaticLabelEmbedder", lambda _d: fake)
    assert make_embedder("auto", model_dir=tmp_path, env={}) is fake

    def boom(_d):
        raise OSError("bad weights")

    monkeypatch.setattr(mod, "StaticLabelEmbedder", boom)
    assert isinstance(make_embedder("auto", model_dir=tmp_path, env={}), HashLabelEmbedder)


# ---------------------------------------------------------------------------
# semantic tests: need the real potion-base-8M files
# ---------------------------------------------------------------------------

semantic = pytest.mark.skipif(
    not static_model_available(),
    reason="potion-base-8M not available (set LOCI_EMBED_MODEL_DIR or run fetch_models.py)",
)


@pytest.fixture(scope="module")
def potion() -> StaticLabelEmbedder:
    emb = make_embedder("static")
    assert isinstance(emb, StaticLabelEmbedder)
    return emb


@semantic
def test_static_embedder_shape(potion: StaticLabelEmbedder) -> None:
    assert potion.dim == 256
    assert potion.name == "static:potion-base-8M"
    assert potion.vocab_size > 20_000
    assert isinstance(potion, LabelEmbedder)
    vec = potion.embed("cup")
    assert len(vec) == 256
    assert math.sqrt(sum(v * v for v in vec)) == pytest.approx(1.0, abs=1e-5)
    assert potion.embed("cup") == potion.embed(" CUP ")
    assert potion.embed_many(["cup", "dog"]) == [potion.embed("cup"), potion.embed("dog")]


@semantic
def test_static_embedder_semantics(potion: StaticLabelEmbedder) -> None:
    def sim(a: str, b: str) -> float:
        return cosine(potion.embed(a), potion.embed(b))

    assert sim("cup", "mug") > 0.40 > sim("cup", "dog")
    assert sim("keys", "key chain") > 0.45 > sim("keys", "dog")
    assert sim("phone", "cell phone") > 0.70
    assert sim("bottle", "water bottle") > 0.80
    assert sim("tv", "television") > 0.85
    assert sim("couch", "sofa") > 0.60


@semantic
def test_static_embedder_unknown_only_tokens_give_zero_vector(potion: StaticLabelEmbedder) -> None:
    vec = potion.embed("\U0001f642")  # emoji -> [UNK] only
    assert vec == [0.0] * 256
    assert cosine(vec, potion.embed("cup")) == 0.0


@semantic
def test_make_embedder_prefers_static_in_auto_mode() -> None:
    assert isinstance(make_embedder("auto"), StaticLabelEmbedder)
    assert isinstance(make_embedder(env={"LOCI_LABEL_EMBEDDER": "hash"}), HashLabelEmbedder)


@semantic
def test_static_embedder_parity_with_model2vec(potion: StaticLabelEmbedder) -> None:
    m2v = pytest.importorskip("model2vec")
    np = pytest.importorskip("numpy")
    ref_model = m2v.StaticModel.from_pretrained(str(potion.model_dir))
    labels = [
        "cup",
        "mug",
        "keys",
        "key chain",
        "cell phone",
        "dining table",
        "where are my keys",
        "xqzvv",
        "Mug",
        "MUG!",
        "coffee mug",
        "  cell   phone ",
        "\U0001f642",
    ]
    ref = np.asarray(ref_model.encode(labels), dtype=np.float32)
    ours = np.asarray(potion.embed_many(labels), dtype=np.float32)
    singles = np.asarray([potion.embed(x) for x in labels], dtype=np.float32)
    assert ours.shape == ref.shape == (len(labels), 256)
    assert float(np.abs(ours - ref).max()) <= 1e-6
    assert float(np.abs(singles - ref).max()) <= 1e-6
