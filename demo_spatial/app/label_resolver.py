"""Map a spoken object name onto the detector's closed vocabulary.

The camera only knows the labels its model was trained on (COCO-80 for the
handheld demo) plus whatever labels have already been stored in memory.  A
blind user will say "mug", "my keys" or "the telly"; :class:`LabelResolver`
turns that into one of three outcomes, in this order:

1. ``exact``     -- the normalised noun (also with "my"/"the" stripped and the
   plural removed) is in the vocabulary;
2. ``alias``     -- a curated synonym maps onto a vocabulary label
   (``mug -> cup``, ``telly -> tv``);
3. ``embedding`` -- only with a static (semantic) embedder: the nearest
   vocabulary label by cosine similarity, when it clears ``threshold`` and the
   pair is not on the deny-list (``keys`` must never become ``keyboard``);
4. ``none``      -- nothing matched; ``suggestions`` carries the nearest labels
   so the answer can say "did you mean keyboard?" without pretending.

Resolution lives in the voice/API layer only; ``SpatialMemory.where_is`` keeps
filtering on exact labels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .label_embedding import LabelEmbedder, normalize_label

__all__ = [
    "DEFAULT_ALIASES",
    "DEFAULT_DENYLIST",
    "PLURAL_ONLY",
    "LabelResolver",
    "Resolution",
    "label_variants",
    "singularize",
]


@dataclass
class Resolution:
    """Outcome of :meth:`LabelResolver.resolve`."""

    label: str | None
    how: str  # "exact" | "alias" | "embedding" | "none"
    score: float
    suggestions: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.label is not None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "how": self.how,
            "score": round(float(self.score), 4),
            "suggestions": list(self.suggestions),
        }


DEFAULT_ALIASES: dict[str, str] = {
    "mug": "cup",
    "phone": "cell phone",
    "mobile": "cell phone",
    "television": "tv",
    "telly": "tv",
    "sofa": "couch",
    "settee": "couch",
    "table": "dining table",
    "water bottle": "bottle",
    "purse": "handbag",
    "bag": "handbag",
    "remote control": "remote",
    "plant": "potted plant",
    "computer": "laptop",
    "glass": "wine glass",
    "bike": "bicycle",
}
"""Spoken name -> detector label.  Only applied when the target is in the vocabulary."""

DEFAULT_DENYLIST: set[frozenset[str]] = {
    frozenset({"keys", "keyboard"}),
    frozenset({"dog", "hot dog"}),
    frozenset({"bear", "teddy bear"}),
    # potion-base-8M puts "glasses" (spectacles) 0.54 from "wine glass": never auto-resolve.
    frozenset({"glasses", "wine glass"}),
}
"""Pairs that look similar to an embedder but must never resolve onto each other."""

PLURAL_ONLY: frozenset[str] = frozenset(
    {
        "glasses",
        "sunglasses",
        "scissors",
        "jeans",
        "pants",
        "shorts",
        "trousers",
        "headphones",
        "earphones",
        "earbuds",
        "clothes",
        "tongs",
        "pliers",
        "binoculars",
    }
)
"""Nouns whose trailing "s" is not a plural; never singularised."""

_DETERMINERS = ("my ", "the ", "a ", "an ", "our ", "your ")
_IRREGULAR_PLURALS = {"mice": "mouse", "people": "person", "knives": "knife", "children": "child"}
_PUNCTUATION = "?!.,;:'\"()[]"


def singularize(word: str) -> str | None:
    """Return the singular of a (normalised) noun phrase, or None when it has none."""
    if not word or word in PLURAL_ONLY:
        return None
    head = word.rsplit(" ", 1)
    prefix, last = (head[0] + " ", head[1]) if len(head) == 2 else ("", head[0])
    if last in PLURAL_ONLY:
        return None
    if last in _IRREGULAR_PLURALS:
        return prefix + _IRREGULAR_PLURALS[last]
    if len(last) > 4 and last.endswith("ies"):
        return prefix + last[:-3] + "y"
    if len(last) > 3 and last.endswith("ves"):
        return prefix + last[:-3] + "f"
    if len(last) > 4 and last.endswith(("sses", "shes", "ches", "xes", "zes")):
        return prefix + last[:-2]
    if len(last) > 3 and last.endswith("s") and not last.endswith(("ss", "us", "is")):
        return prefix + last[:-1]  # cups -> cup, keys -> key; bus/iris/glass untouched
    return None


def label_variants(text: str) -> tuple[list[str], list[str]]:
    """Candidate spellings for a spoken noun.

    Returns ``(phrase_variants, head_variants)``: the full phrase (normalised,
    determiners stripped, singularised) first, then the head noun of a
    multi-word phrase ("coffee cup" -> "cup").  Both lists are ordered by
    priority and free of duplicates.
    """
    norm = normalize_label(text).strip(_PUNCTUATION).strip()
    if not norm:
        return [], []
    phrases: list[str] = [norm]
    stripped = norm
    for det in _DETERMINERS:
        if stripped.startswith(det) and len(stripped) > len(det):
            stripped = stripped[len(det) :].strip()
            break
    if stripped != norm:
        phrases.append(stripped)
    for p in list(phrases):
        s = singularize(p)
        if s and s not in phrases:
            phrases.append(s)

    heads: list[str] = []
    words = stripped.split()
    if len(words) > 1:
        head = words[-1]
        heads.append(head)
        s = singularize(head)
        if s and s not in heads:
            heads.append(s)
    heads = [h for h in heads if h not in phrases]
    return phrases, heads


class LabelResolver:
    """Resolve spoken nouns onto a detector vocabulary (see module docstring)."""

    def __init__(
        self,
        vocabulary: Iterable[str],
        embedder: LabelEmbedder,
        *,
        aliases: Mapping[str, str] = DEFAULT_ALIASES,
        denylist: Iterable[Iterable[str]] = DEFAULT_DENYLIST,
        threshold: float = 0.45,
        max_suggestions: int = 5,
        suggestion_floor: float = 0.3,
    ) -> None:
        self._embedder = embedder
        self._aliases = {normalize_label(k): normalize_label(v) for k, v in aliases.items()}
        self._denylist = _expand_denylist(denylist)
        self.threshold = float(threshold)
        self.max_suggestions = int(max_suggestions)
        self.suggestion_floor = float(suggestion_floor)
        self._vocab: list[str] = []
        self._vocab_set: set[str] = set()
        self._emb_cache: dict[str, np.ndarray] = {}
        self._matrix: np.ndarray | None = None
        self.set_vocabulary(vocabulary)

    # -- vocabulary ---------------------------------------------------------

    @property
    def embedder(self) -> LabelEmbedder:
        return self._embedder

    @property
    def semantic(self) -> bool:
        """True when the embedder carries meaning (static model), so step 3 runs."""
        return str(getattr(self._embedder, "name", "")).startswith("static")

    @property
    def vocabulary(self) -> list[str]:
        return list(self._vocab)

    def set_vocabulary(self, labels: Iterable[str]) -> None:
        """Replace the vocabulary; embeddings of labels seen before are reused."""
        seen: set[str] = set()
        vocab: list[str] = []
        for raw in labels:
            lab = normalize_label(str(raw))
            if lab and lab not in seen:
                seen.add(lab)
                vocab.append(lab)
        if vocab == self._vocab:
            return
        self._vocab = vocab
        self._vocab_set = seen
        self._matrix = None  # rebuilt lazily on the first semantic lookup

    def _ensure_matrix(self) -> np.ndarray:
        if self._matrix is None:
            missing = [lab for lab in self._vocab if lab not in self._emb_cache]
            if missing:
                for lab, vec in zip(missing, self._embedder.embed_many(missing), strict=True):
                    self._emb_cache[lab] = _unit(np.asarray(vec, dtype=np.float64))
            if self._vocab:
                self._matrix = np.stack([self._emb_cache[lab] for lab in self._vocab])
            else:
                self._matrix = np.zeros((0, int(self._embedder.dim)), dtype=np.float64)
        return self._matrix

    # -- resolution ---------------------------------------------------------

    def _denied(self, variants: Sequence[str], label: str) -> bool:
        return any(frozenset((v, label)) in self._denylist for v in variants)

    def _lexical(self, candidates: Sequence[str]) -> Resolution | None:
        for cand in candidates:
            if cand in self._vocab_set:
                return Resolution(cand, "exact", 1.0)
        for cand in candidates:
            target = self._aliases.get(cand)
            if target is not None and target in self._vocab_set:
                return Resolution(target, "alias", 1.0)
        return None

    def _scores(self, variants: Sequence[str]) -> np.ndarray:
        """Best cosine per vocabulary label over all phrase variants."""
        matrix = self._ensure_matrix()
        if matrix.shape[0] == 0:
            return np.zeros(0)
        queries = np.stack(
            [_unit(np.asarray(v, dtype=np.float64)) for v in self._embedder.embed_many(variants)]
        )
        return (matrix @ queries.T).max(axis=1)

    def resolve(self, text: str) -> Resolution:
        phrases, heads = label_variants(text)
        if not phrases:
            return Resolution(None, "none", 0.0)

        hit = self._lexical(phrases) or self._lexical(heads)
        if hit is not None:
            return hit

        if not self.semantic or not self._vocab:
            return Resolution(None, "none", 0.0)

        scores = self._scores(phrases)
        order = np.argsort(-scores, kind="stable")
        best_score = float(scores[order[0]]) if len(order) else 0.0
        for idx in order:
            label = self._vocab[int(idx)]
            score = float(scores[idx])
            if score < self.threshold:
                break
            if self._denied(phrases, label):
                continue
            return Resolution(label, "embedding", score)

        suggestions = [
            self._vocab[int(i)]
            for i in order[: self.max_suggestions]
            if float(scores[i]) >= self.suggestion_floor
        ]
        return Resolution(None, "none", best_score, suggestions)


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else vec


def _expand_denylist(denylist: Iterable[Iterable[str]]) -> set[frozenset[str]]:
    """Normalise deny-list pairs and add their singular spellings.

    ``{"keys", "keyboard"}`` must also block ``"key"`` (the plural-stripped
    variant the resolver tries), so every member is paired with its singular.
    """
    out: set[frozenset[str]] = set()
    for pair in denylist:
        members = [normalize_label(x) for x in pair]
        forms: list[set[str]] = []
        for m in members:
            alt = {m}
            s = singularize(m)
            if s:
                alt.add(s)
            forms.append(alt)
        if len(forms) != 2:
            out.add(frozenset(members))
            continue
        for a in forms[0]:
            for b in forms[1]:
                out.add(frozenset((a, b)))
    return out
