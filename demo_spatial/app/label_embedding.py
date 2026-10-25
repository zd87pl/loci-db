"""Label embedders for the spatial memory demo.

Two interchangeable implementations of the :class:`LabelEmbedder` protocol:

* :class:`HashLabelEmbedder` -- the deterministic 128-d construction that the
  demo has always used (``spatial_memory._label_embedding``).  No model files,
  byte-identical to the legacy function, meaningless semantics (only exact
  label matches score high).  This is the default so that tests stay
  deterministic and CI needs no weights.
* :class:`StaticLabelEmbedder` -- a model2vec-format static embedding model
  (``potion-base-8M``: 256-d, 30 MB) read with ``tokenizers`` + ``safetensors``
  + numpy only.  Tokenise without special tokens, drop ``[UNK]``, mean-pool the
  token vectors and L2-normalise; this reproduces ``model2vec.StaticModel.encode``
  bit for bit without importing model2vec.

Selection goes through :func:`make_embedder` (``LOCI_LABEL_EMBEDDER`` =
``hash`` | ``static`` | ``auto``, model directory from ``LOCI_EMBED_MODEL_DIR``).
Heavy imports are deferred to the constructors so that ``import app.label_embedding``
never needs the optional packages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MODEL_DIR",
    "HASH_DIM",
    "HashLabelEmbedder",
    "LabelEmbedder",
    "StaticLabelEmbedder",
    "cosine",
    "hash_label_embedding",
    "make_embedder",
    "normalize_label",
    "static_model_available",
]

HASH_DIM = 128
"""Dimension of the legacy hash embedding (``spatial_memory.EMBED_DIM``)."""

DEFAULT_MODEL_DIR: Path = (
    Path(__file__).resolve().parents[2] / "deploy" / "handheld" / "models" / "potion-base-8M"
)
"""Where ``fetch_models.py`` vendors potion-base-8M inside the repository."""

_MODEL_FILES = ("model.safetensors", "tokenizer.json", "config.json")
_ENV_KIND = "LOCI_LABEL_EMBEDDER"
_ENV_MODEL_DIR = "LOCI_EMBED_MODEL_DIR"
_CACHE_LIMIT = 4096


@runtime_checkable
class LabelEmbedder(Protocol):
    """Anything that turns a label into a fixed-size unit vector."""

    name: str
    dim: int

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]: ...


def normalize_label(text: str) -> str:
    """Strip, lower-case and collapse internal whitespace."""
    return " ".join(text.strip().lower().split())


def hash_label_embedding(label: str, dim: int = HASH_DIM) -> list[float]:
    """Deterministic embedding of a label (exact port of the legacy function).

    First 8 dims are sin/cos of the mean character code, the remainder comes
    from a 31-bit LCG seeded with the sha256 of ``"label:<normalised>"``; the
    result is L2-normalised.  Only ``strip().lower()`` is applied to the label so
    that the vector is byte-identical to ``spatial_memory._label_embedding``.
    """
    normalized = label.strip().lower()
    key = f"label:{normalized}"
    digest = hashlib.sha256(key.encode()).digest()
    seed = struct.unpack("<I", digest[:4])[0]

    char_val = sum(ord(c) for c in normalized) / max(len(normalized), 1)
    vec: list[float] = []
    for i in range(dim):
        if i < 8:
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


class HashLabelEmbedder:
    """The legacy 128-d hash embedding behind the :class:`LabelEmbedder` protocol."""

    name: str = f"hash:{HASH_DIM}"
    dim: int = HASH_DIM

    def __init__(self, dim: int = HASH_DIM) -> None:
        self.dim = int(dim)
        self.name = f"hash:{self.dim}"
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        key = text.strip().lower()
        hit = self._cache.get(key)
        if hit is None:
            if len(self._cache) >= _CACHE_LIMIT:
                self._cache.clear()
            hit = hash_label_embedding(key, self.dim)
            self._cache[key] = hit
        return list(hit)

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def __repr__(self) -> str:
        return f"HashLabelEmbedder(dim={self.dim})"


class StaticLabelEmbedder:
    """model2vec-format static embeddings loaded with tokenizers + safetensors.

    Expects a directory holding ``model.safetensors`` (tensor ``embeddings``,
    shape ``(vocab, dim)``), ``tokenizer.json`` and ``config.json``.  Raises
    ``ImportError`` when the optional packages are missing, ``OSError`` when a
    file is missing or unreadable and ``KeyError`` when the tensor is absent.
    """

    def __init__(self, model_dir: str | os.PathLike[str]) -> None:
        import numpy as np
        from safetensors.numpy import load_file
        from tokenizers import Tokenizer

        d = Path(model_dir)
        if not d.is_dir():
            raise FileNotFoundError(f"embedding model directory not found: {d}")
        for fname in _MODEL_FILES[:2]:
            if not (d / fname).is_file():
                raise FileNotFoundError(f"embedding model file missing: {d / fname}")

        tensors = load_file(str(d / "model.safetensors"))
        if "embeddings" in tensors:
            weights = tensors["embeddings"]
        elif len(tensors) == 1:
            weights = next(iter(tensors.values()))
        else:
            raise KeyError(f"no 'embeddings' tensor in {d} (found {sorted(tensors)})")
        if weights.ndim != 2:
            raise ValueError(f"embedding tensor must be 2-d, got shape {weights.shape}")

        config: dict = {}
        cfg_path = d / "config.json"
        if cfg_path.is_file():
            try:
                config = json.loads(cfg_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:  # unreadable config: keep defaults
                logger.warning("ignoring unreadable %s: %s", cfg_path, exc)

        self._np = np
        self._weights = weights.astype(np.float32, copy=False)
        self._tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._unk_id = self._unk_token_id(self._tok)
        self.normalize = bool(config.get("normalize", True))
        self.config = config
        self.model_dir = d
        self.dim = int(self._weights.shape[1])
        self.vocab_size = int(self._weights.shape[0])
        self.name = f"static:{d.name}"
        self._cache: dict[str, list[float]] = {}

    @staticmethod
    def _unk_token_id(tok) -> int | None:  # noqa: ANN001 - tokenizers.Tokenizer
        model = getattr(tok, "model", None)
        unk = getattr(model, "unk_token", None)
        if unk is not None:
            return tok.token_to_id(unk)
        return tok.token_to_id("[UNK]")

    # -- encoding -----------------------------------------------------------

    def _ids(self, text: str) -> list[int]:
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if self._unk_id is not None:
            ids = [i for i in ids if i != self._unk_id]
        return ids

    def _pool(self, ids: list[int]) -> list[float]:
        """Mean-pool token rows and L2-normalise, in exactly model2vec's order.

        The reductions run on a ``(1, n, dim)`` / ``(1, dim)`` view so that numpy
        picks the same accumulation path as ``StaticModel._encode_batch`` and the
        result is bit-identical to ``model2vec`` (a 1-d ``np.linalg.norm`` would
        go through BLAS and differ by one float32 ulp).
        """
        np = self._np
        if not ids:
            return [0.0] * self.dim
        rows = self._weights[np.asarray([ids], dtype=np.int64)]  # (1, n, dim)
        out = rows.mean(axis=1).astype(np.float32)  # (1, dim)
        if self.normalize:
            norm = np.linalg.norm(out, axis=1, keepdims=True) + 1e-32
            out = (out / norm).astype(np.float32)
        return [float(v) for v in out[0]]

    def embed(self, text: str) -> list[float]:
        key = normalize_label(text)
        hit = self._cache.get(key)
        if hit is None:
            if len(self._cache) >= _CACHE_LIMIT:
                self._cache.clear()
            hit = self._pool(self._ids(key))
            self._cache[key] = hit
        return list(hit)

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [normalize_label(t) for t in texts]
        misses = sorted({k for k in keys if k not in self._cache})
        if misses:
            encodings = self._tok.encode_batch(misses, add_special_tokens=False)
            for key, enc in zip(misses, encodings, strict=True):
                ids = enc.ids
                if self._unk_id is not None:
                    ids = [i for i in ids if i != self._unk_id]
                if len(self._cache) >= _CACHE_LIMIT:
                    self._cache.clear()
                self._cache[key] = self._pool(ids)
        return [list(self._cache[k]) for k in keys]

    def tokens(self, text: str) -> list[str]:
        """Token strings for a label (diagnostics)."""
        return list(self._tok.encode(normalize_label(text), add_special_tokens=False).tokens)

    def __repr__(self) -> str:
        return f"StaticLabelEmbedder({str(self.model_dir)!r}, dim={self.dim})"


# -- selection ----------------------------------------------------------------


def _resolve_model_dir(model_dir: str | os.PathLike[str] | None, env: Mapping[str, str]) -> Path:
    if model_dir:
        return Path(model_dir)
    from_env = env.get(_ENV_MODEL_DIR, "").strip()
    return Path(from_env) if from_env else DEFAULT_MODEL_DIR


def static_model_available(
    model_dir: str | os.PathLike[str] | None = None, *, env: Mapping[str, str] | None = None
) -> bool:
    """True when ``tokenizers``/``safetensors`` import and the three model files exist."""
    d = _resolve_model_dir(model_dir, os.environ if env is None else env)
    try:
        import safetensors.numpy  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return all((d / f).is_file() for f in _MODEL_FILES)


def make_embedder(
    kind: str | None = None,
    model_dir: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> LabelEmbedder:
    """Build the embedder selected by ``kind`` or ``LOCI_LABEL_EMBEDDER``.

    ``hash`` (default) never touches the file system; ``static`` raises when the
    model or its packages are missing; ``auto`` prefers static and falls back to
    hash with a warning.
    """
    environ = os.environ if env is None else env
    choice = (kind or environ.get(_ENV_KIND) or "hash").strip().lower()
    if choice == "hash":
        return HashLabelEmbedder()

    directory = _resolve_model_dir(model_dir, environ)
    if choice == "static":
        return StaticLabelEmbedder(directory)
    if choice == "auto":
        if static_model_available(directory):
            try:
                return StaticLabelEmbedder(directory)
            except (ImportError, OSError, KeyError, ValueError) as exc:
                logger.warning("static embedder unusable (%s); falling back to hash", exc)
        else:
            logger.info("static embedding model not available at %s; using hash", directory)
        return HashLabelEmbedder()
    raise ValueError(f"unknown {_ENV_KIND} value {choice!r} (expected hash|static|auto)")


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 when either vector is (near) zero."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 1e-24 or nb <= 1e-24:
        return 0.0
    return dot / math.sqrt(na * nb)
