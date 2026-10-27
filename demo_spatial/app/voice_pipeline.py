"""Voice pipeline: Whisper STT → intent extraction → spatial query → TTS output.

Supports three TTS engines (configured via TTS_ENGINE env var):
  - "openai"  (default): OpenAI TTS API — best quality, requires API key
  - "piper":  Local Piper TTS — zero cloud dependency, requires model download
  - "edge":   Microsoft Edge TTS — free cloud, no API key needed

STT uses OpenAI Whisper API by default. Falls back to faster-whisper
for fully offline operation if WHISPER_LOCAL=1 is set.

Handheld additions (design section 4.10): ``parse_intent`` captures a room
("what changed in the kitchen"), answers go through ``SpatialMemory.resolve_label``
(alias "mug" -> "cup", honest refusal for unknown nouns) and ``phrasing`` (walk-in /
egocentric wording), ``handle_text_query(synthesize_response=False)`` skips TTS,
``VoicePipeline.audio_mime`` sniffs the audio container and Piper >= 1.3's
``synthesize_wav`` API is supported.  Memories without the handheld API (the
pre-room ``SpatialMemory``) keep today's behaviour.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import logging
import os
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import phrasing
from .phrasing import (
    format_age,
    format_clock,
    image_position_phrase,
)

if TYPE_CHECKING:
    from .detection_source import CameraPose
    from .label_resolver import Resolution
    from .spatial_memory import SpatialMemory
    from .vlm_client import VLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent model
# ---------------------------------------------------------------------------


@dataclass
class QueryIntent:
    """Parsed intent from a voice query."""

    kind: str  # "where_is" | "history" | "list_objects" | "changes" | "where_am_i" | "unknown"
    object_name: str  # the target object, empty string if not applicable
    raw_text: str  # original transcribed text
    room: str = ""  # room named in the query ("in the kitchen"), "" when none

    def to_dict(self) -> dict:
        return {"kind": self.kind, "object_name": self.object_name, "room": self.room}


# The patterns run on normalised text (see ``_normalise``): lower case, no punctuation
# but apostrophes, no wake word or filler in front, no time phrase or "please" at the end.
_WHERE_PATTERNS = [
    # "where did my keys go" before the generic form, which would take "did my keys go"
    r"where\s+(?:did|has|have)\s+(?:my\s+|the\s+|our\s+)?(.+?)\s+(?:go|gone|went|end up|ended up)$",
    r"where(?:'s| is| are| was| were| did (?:i|we|you) (?:last )?(?:put|leave|place|set|drop|see)"
    r"| have (?:i|we) (?:put|left))?\s+(?:my\s+|the\s+)?(.+?)$",
    r"(?:find|locate|show me)\s+(?:my\s+|the\s+)?(.+?)$",
    r"(?:have you seen|did you see|can you see|do you see)\s+"
    r"(?:my\s+|the\s+|a\s+|an\s+|any\s+)?(.+?)$",
    r"(?:looking|searching|hunting)\s+for\s+(?:my\s+|the\s+)?(.+?)$",
    r"(?:^|\s)lost\s+(?:my\s+|the\s+)(.+?)$",
    r"what happened to\s+(?:my\s+|the\s+)?(.+?)$",
    # "is there a bottle (in the kitchen)", "is my laptop in the kitchen"; without a room the
    # second shape is a yes/no question about something else ("is my laptop charged")
    r"^(?:is|are)\s+there\s+(?:a\s+|an\s+|any\s+|my\s+|the\s+)?(.+)$",
    r"^(?:is|are)\s+(?:my\s+|the\s+|a\s+|an\s+|our\s+|your\s+)(.+?\s+(?:in|inside|around)\s+.+)$",
]

_HISTORY_PATTERNS = [
    r"(?:when|last time)\s+(?:was|did|have)\s+(?:i\s+)?"
    r"(?:seen|used|placed|put|left)\s+(?:my\s+|the\s+)?(.+?)$",
    r"when\s+(?:did you|did i|have you)\s+(?:last\s+)?"
    r"(?:see|seen|spot|notice)\s+(?:my\s+|the\s+)?(.+?)$",
    r"the\s+last\s+time\s+(?:you|i|we)\s+(?:saw|seen|spotted|noticed|had|used)\s+(?:my\s+|the\s+)?(.+?)$",
    r"when\s+(?:was\s+)?(?:my\s+|the\s+)?(.+?)\s+(?:last\s+)?(?:seen|spotted|detected)",
    r"how\s+long\s+(?:ago|since)\s+(?:was|did|have)\s+(?:my\s+|the\s+)?(.+?)(?:$|seen|spotted)",
    r"how\s+long\s+since\s+(?:my\s+|the\s+)?(.+?)\s+(?:was|were)\s+(?:last\s+)?(?:seen|spotted)$",
    r"how\s+long\s+since\s+(?:you|i)\s+(?:last\s+)?(?:saw|seen)\s+(?:my\s+|the\s+)?(.+?)$",
    r"history\s+(?:of\s+)?(?:my\s+)?(.+?)$",
]

_LIST_PATTERNS = [
    r"(?:what(?:'s| is| are)?\s+(?:in\s+the\s+room|around|nearby|here|visible))",
    r"(?:list|show|tell me)\s+(?:all\s+)?(?:the\s+)?objects",
    r"^(?:list|name)\s+(?:all|everything|every|what)\b",
    r"what\s+(?:can|could|did|have)\s+you\s+(?:see|seen|spot|spotted|notice|noticed|remember)",
    r"what\s+do\s+(?:i|you)\s+(?:have|see|know|remember)",
    r"(?:tell me|show me)\s+what\s+(?:you\s+)?(?:can\s+)?"
    r"(?:see|is around|'s around|is here|is nearby)",
    r"^describe\b",
    # "what objects do you have in your memory", "what things are in the kitchen"
    r"what\s+(?:objects|things|items|stuff)\b"
    r"(?:.*?\s+in\s+(?:the\s+|my\s+|your\s+)?(?P<room>[\w' ]+?))?$",
    # handheld: "what's in the kitchen" / "what is in my living room"
    r"what(?:'s| is)\s+in\s+(?:the\s+|my\s+|your\s+)?(?P<room>[\w' ]+?)$",
]

_CHANGES_PATTERNS = [
    r"what(?:'s| has| have)\s+(?:changed|moved|been moved)",
    r"(?:any|what)\s+(?:new|recent)\s+(?:changes|updates|movements)",
    r"(?:has\s+)?anything\s+(?:changed|moved|new|different)"
    r"(?:\s+(?:in|around)\s+(?:the\s+|my\s+)?(?P<room>[\w' ]+?))?$",
    # handheld: "what changed in the kitchen", "what's new around my living room".
    # The auxiliary is optional: the spec's "what(?:'s| has| have| is)" alone misses
    # the plain past tense "what changed", which is the demo's own phrase.
    r"what(?:'s|s| has| have| is)?\s+(?:changed|moved|been moved|new|different)"
    r"(?:\s+(?:in|around)\s+(?:the\s+|my\s+)?(?P<room>[\w' ]+?))?$",
]

_WHERE_AM_I_PATTERNS = [
    r"^where am i(?:\s+(?:right\s+)?now)?$",
    r"^(?:which|what) room (?:am i in|is this)$",
]

# Trailing "in the kitchen" / "in my living room" on an object name.
_ROOM_TAIL = re.compile(
    r"^(?P<obj>.+)\s+(?:in|inside|around)\s+(?:the\s+|my\s+|our\s+|your\s+)?(?P<room>[\w' ]+?)\s*$"
)

# Captured "rooms" that mean "here", not a room name.
_NOT_A_ROOM = frozenset(
    {"room", "the room", "here", "view", "frame", "scene", "picture", "front", "front of me"}
)

# A spoken room that is not a known room still counts when it ends in one of these, so
# "where is my mug in the bedroom" gets an honest "not seen in the bedroom" while
# "in your memory" or "in the car" is not taken for a room.
_ROOM_NOUNS = frozenset(
    {
        "attic", "balcony", "basement", "bathroom", "bedroom", "cellar", "closet", "corridor",
        "den", "downstairs", "entrance", "entryway", "foyer", "garage", "garden", "hall",
        "hallway", "kitchen", "kitchenette", "landing", "laundry", "lobby", "loft", "lounge",
        "nursery", "office", "pantry", "patio", "porch", "restroom", "room", "shed", "study",
        "upstairs", "washroom", "workshop", "yard",
    }
)  # fmt: skip

_SPELLINGS = {
    "wheres": "where's",
    "whats": "what's",
    "where'd": "where did",
    "where're": "where are",
}

# Wake word, fillers and "can you" in front of a question; "please" and thanks anywhere.
_LEAD_FILLER = re.compile(
    r"^(?:(?:hey|hi|hello|ok|okay)\s+loci|loci|hey|hi|hello|ok|okay|um+|uh+|erm|er|hmm+|so|well|"
    r"and|now|right|alright|all right|excuse me|sorry|yeah|oh|do you know|i wonder)\s+"
)
_LEAD_POLITE = re.compile(
    r"^(?:can|could|would|will)\s+you\s+(?=(?:tell|find|locate|show|help|check|look|describe|list|remind)\b)"
)
_COURTESY = re.compile(r"\b(?:please|thank you very much|thank you|thanks)\b")

# A time phrase at the end ("since lunch", "yesterday", "right now") is not part of the
# object or room; "how long since ..." keeps its "since".
_TIME_TAIL = re.compile(
    r"\s+(?:(?<!long )since\s+.+|yesterday|today|tonight|this (?:morning|afternoon|evening|week)"
    r"|last night|earlier(?: today)?|recently|lately|just now|right now|now|at the moment"
    r"|(?:a|an|one|two|three|\d+|a few|a couple of)\s+(?:minutes?|hours?|days?)\s+ago|a while ago)$"
)

_OBJ_HEAD = re.compile(r"^(?:my|the|a|an|our|your|any|some|that|this)\s+")
_OBJ_TAIL = re.compile(r"\s+(?:is|are|was|were|go|gone|went|anywhere|again|for me|at the moment)$")

# Words that make leftover text a sentence rather than an object name.
_NOT_IN_OBJECTS = frozenset(
    {
        "what", "where", "when", "who", "why", "how", "which", "whose",
        "is", "are", "was", "were", "am", "do", "does", "did", "don", "doesn", "didn",
        "can", "could", "would", "will", "won", "should", "shall", "have", "has", "had",
        "may", "might", "isn", "aren", "wasn", "let",
        "i", "you", "we", "me", "it", "he", "she", "they", "us", "them",
        "tell", "say", "thank", "thanks",
    }
)  # fmt: skip
_NOT_OBJECTS = frozenset(
    {"yes", "no", "hello", "hi", "hey", "bye", "goodbye", "okay", "ok", "stop", "cancel",
     "nothing", "never mind", "nevermind", "help", "loci", "sorry", "test", "testing"}
)  # fmt: skip


def _clean_room(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", " ").split()).rstrip("?!.,")


def match_room(captured: str | None, known_rooms: Iterable[str] = ()) -> str:
    """Canonicalise a room captured from speech.

    Known room names win case-insensitively with ``"_"``/``" "`` equivalence
    ("living room" -> "living_room"); a known room that prefixes the capture
    ("kitchen since lunch") is accepted too.  Unknown text is returned cleaned
    (lower case, spaces) so ``where_is(room=...)`` can still slug it.  Words that
    mean "here" ("what's in the room") map to "".
    """
    cap = _clean_room(captured or "")
    if not cap or cap in _NOT_A_ROOM:
        return ""
    canon: dict[str, str] = {}
    for name in known_rooms or ():
        if isinstance(name, str) and name.strip():
            canon[_clean_room(name)] = name
    if cap in canon:
        return canon[cap]
    for norm in sorted(canon, key=len, reverse=True):
        if cap.startswith(norm + " "):
            return canon[norm]
    return cap


def _room(captured: str | None, known_rooms: Iterable[str]) -> str:
    """``match_room``, but only for a known room or a phrase that ends in a room noun."""
    known = list(known_rooms or ())
    room = match_room(captured, known)
    if not room or room in known:
        return room
    return room if room.split()[-1] in _ROOM_NOUNS else ""


def _split_room(obj: str, known_rooms: Iterable[str]) -> tuple[str, str]:
    """ "mug in the living room" -> ("mug", "living_room").

    A tail that is not a room ("cup in your memory", "phone in the car") is dropped, so the
    object is still found wherever it was seen last.
    """
    m = _ROOM_TAIL.match(obj)
    if not m:
        return obj, ""
    return m.group("obj").strip(), _room(m.group("room"), known_rooms)


def _clean_object(obj: str) -> str:
    """Strip determiners in front and "is" / "anywhere" / "go" at the end."""
    obj = obj.strip()
    prev = None
    while obj != prev:
        prev = obj
        obj = _OBJ_TAIL.sub("", _OBJ_HEAD.sub("", obj)).strip()
    return obj


def _normalise(text: str) -> str:
    """Lower case, punctuation off, wake word / fillers / courtesy / trailing time phrase off."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^\w' -]+", " ", t)
    t = " ".join(_SPELLINGS.get(w, w) for w in t.split())
    t = " ".join(_COURTESY.sub(" ", t).split())
    prev = None
    while t != prev:
        prev = t
        t = _LEAD_POLITE.sub("", _LEAD_FILLER.sub("", t))
        t = _TIME_TAIL.sub("", t).strip()
    return t


def _search(patterns: Sequence[str], cleaned: str) -> tuple[re.Match[str] | None, str | None]:
    """First match over ``patterns``; a match carrying a ``room`` group is preferred."""
    first: re.Match[str] | None = None
    for pat in patterns:
        m = re.search(pat, cleaned)
        if not m:
            continue
        if first is None:
            first = m
        if "room" in m.groupdict() and m.group("room"):
            return m, m.group("room")
    return first, None


def parse_intent(text: str, known_rooms: Iterable[str] = ()) -> QueryIntent:
    """Extract structured intent (kind, object, room) from a transcribed query."""
    cleaned = _normalise(text)
    rooms = [r for r in (known_rooms or ()) if isinstance(r, str)]
    if not cleaned:  # "Thank you." (Whisper's favourite on silence), "um", "hey loci"
        return QueryIntent(kind="unknown", object_name="", raw_text=text)

    for pat in _WHERE_AM_I_PATTERNS:
        if re.search(pat, cleaned):
            return QueryIntent(kind="where_am_i", object_name="", raw_text=text)

    # Check change/movement queries
    m, room = _search(_CHANGES_PATTERNS, cleaned)
    if m:
        return QueryIntent(kind="changes", object_name="", raw_text=text, room=_room(room, rooms))

    # Check list-all queries
    m, room = _search(_LIST_PATTERNS, cleaned)
    if m:
        return QueryIntent(
            kind="list_objects", object_name="", raw_text=text, room=_room(room, rooms)
        )

    # History, then where-is: the object is what the pattern captured, minus a room tail
    for kind, patterns in (("history", _HISTORY_PATTERNS), ("where_is", _WHERE_PATTERNS)):
        for pat in patterns:
            m = re.search(pat, cleaned)
            if m:
                obj, room_name = _split_room(m.group(1).strip(), rooms)
                obj = _clean_object(obj)
                if obj:
                    return QueryIntent(kind=kind, object_name=obj, raw_text=text, room=room_name)

    # Bare object names ("my keys", "glasses", "mug in the kitchen"): at most three words
    # and nothing that makes it a sentence, so "what time is it" or "tell me a joke" stay
    # unknown instead of becoming a search for an object called that.
    obj, room_name = _split_room(cleaned, rooms)
    obj = _clean_object(obj)
    words = set(re.split(r"[ ']", cleaned))
    if obj and len(obj.split()) <= 3 and not words & _NOT_IN_OBJECTS and obj not in _NOT_OBJECTS:
        return QueryIntent(kind="where_is", object_name=obj, raw_text=text, room=room_name)

    return QueryIntent(kind="unknown", object_name="", raw_text=text)


# Backwards-compatible private aliases (older callers/tests imported these).
_format_age = format_age
_format_clock_time = format_clock
_describe_position = image_position_phrase


def _describe_position_relative(target, all_objects: list) -> str:
    """Describe target position relative to other nearby tracked objects (image frame)."""
    if not all_objects or len(all_objects) < 2:
        return ""
    others = [o for o in all_objects if o.label != target.label]
    if not others:
        return ""
    # Find closest other object by Euclidean distance
    closest = min(others, key=lambda o: ((o.cx - target.cx) ** 2 + (o.cy - target.cy) ** 2) ** 0.5)
    dx = target.cx - closest.cx
    dy = target.cy - closest.cy
    dist = (dx**2 + dy**2) ** 0.5
    if dist > 0.5:
        return ""  # too far apart to be useful
    if abs(dx) > abs(dy):
        direction = "to the right of" if dx > 0 else "to the left of"
    else:
        direction = "in front of" if dy > 0 else "behind"
    return f"It's {direction} the {closest.label}."


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

#: Labels offered in the unknown-noun refusal when the vocabulary is unknown.
EXAMPLE_LABELS: tuple[str, ...] = (
    "cup",
    "bottle",
    "cell phone",
    "remote",
    "book",
    "laptop",
    "backpack",
    "chair",
)
CHANGES_WINDOW_S = 600.0
_LIST_MAX = 6


@dataclass
class AnswerBundle:
    """What ``answer_query`` found: the spoken text plus the structured evidence."""

    text: str
    resolution: Resolution | None = None
    results: list[Any] = field(default_factory=list)
    changes: list[Any] = field(default_factory=list)


def _has_method(obj: Any, name: str) -> bool:
    return callable(getattr(obj, name, None))


def _accepts_kw(fn: Any, name: str) -> bool:
    """True when ``fn`` takes keyword ``name`` (or ``**kwargs``); True when unknowable."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _str_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(v) for v in value if isinstance(v, str)]
    return []


def _obs_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _room_spec(memory: Any, room: str | None) -> Any:
    """``memory.rooms.get(room)`` when the memory has a RoomMap, else None."""
    if not room:
        return None
    rooms = getattr(memory, "rooms", None)
    if rooms is None or not _has_method(rooms, "get"):
        return None
    try:
        spec = rooms.get(room)
    except Exception:  # pragma: no cover - defensive against odd RoomMap fakes
        return None
    # A spec must at least carry a string anchor_name or a landmarks mapping.
    if isinstance(getattr(spec, "anchor_name", None), str) or isinstance(
        getattr(spec, "landmarks", None), dict
    ):
        return spec
    return None


def _room_specs_for(memory: Any, observations: Sequence[Any]) -> dict[str, Any]:
    specs: dict[str, Any] = {}
    for obs in observations:
        room = getattr(obs, "room", None)
        if room and room not in specs:
            spec = _room_spec(memory, room)
            if spec is not None:
                specs[room] = spec
    return specs


def example_labels(memory: Any, limit: int = 5) -> list[str]:
    """Labels to offer in the refusal: seen labels first, then common COCO names."""
    tracked = _str_list(getattr(memory, "tracked_labels", None))
    vocab = None
    for attr in ("vocabulary", "_vocabulary", "_vocab"):  # T4 keeps it as _vocab
        v = _str_list(getattr(memory, attr, None))
        if v:
            vocab = {x.lower() for x in v}
            break
    out: list[str] = []
    for label in [*tracked[:2], *EXAMPLE_LABELS]:
        if vocab is not None and label.lower() not in vocab:
            continue
        if label not in out:
            out.append(label)
        if len(out) >= limit:
            break
    if not out and vocab:
        out = sorted(vocab)[:limit]
    return out or list(EXAMPLE_LABELS[:limit])


def _started_at(memory: Any) -> int | None:
    v = getattr(memory, "started_at_ms", None)
    return int(v) if isinstance(v, int) and not isinstance(v, bool) else None


def _where_is(memory: Any, label: str, limit: int, room: str) -> list[Any]:
    if room and _accepts_kw(memory.where_is, "room"):
        results = memory.where_is(label, limit=limit, room=room)
    else:
        results = memory.where_is(label, limit=limit)
        if room:
            want = phrasing.room_display(room)
            results = [
                r for r in results if phrasing.room_display(getattr(r, "room", None)) == want
            ]
    return _obs_list(results)


def _resolve(memory: Any, name: str) -> Resolution | None:
    if not _has_method(memory, "resolve_label"):
        return None
    try:
        res = memory.resolve_label(name)
    except Exception as e:  # pragma: no cover - resolver bugs must not 500 the API
        logger.warning("resolve_label(%r) failed: %s", name, e)
        return None
    if isinstance(getattr(res, "how", None), str):
        return res
    return None


def _answer_object(
    intent: QueryIntent, memory: Any, current_pose: CameraPose | None
) -> AnswerBundle:
    spoken = intent.object_name
    res = _resolve(memory, spoken)
    if res is not None and getattr(res, "label", None) is None:
        return AnswerBundle(phrasing.unknown_label_answer(spoken, res, example_labels(memory)), res)

    label = str(res.label) if res is not None else spoken
    hedge = res is not None and res.how == "embedding" and float(res.score) < 0.6
    limit = 20 if intent.kind == "history" else 5
    results = _where_is(memory, label, limit, intent.room)
    if not results:
        return AnswerBundle(
            phrasing.not_seen_answer(spoken, intent.room or None, resolved_label=label), res, []
        )

    specs = _room_specs_for(memory, results)
    kwargs: dict[str, Any] = {
        "current_pose": current_pose,
        "resolved_label": label,
        "started_at_ms": _started_at(memory),
        "hedge": hedge,
    }
    if intent.kind == "history" and len(results) > 1:
        text = phrasing.history_answer(spoken, results, room_spec_for=specs, **kwargs)
    else:
        obs = results[0]
        text = phrasing.where_is_answer(
            spoken, obs, room_spec=specs.get(getattr(obs, "room", None) or ""), **kwargs
        )
        if (getattr(obs, "frame", "image") or "image") == "image" and _has_method(
            memory, "current_objects"
        ):
            # Legacy browser mode: relate the object to its nearest tracked neighbour.
            rel = _describe_position_relative(obs, _obs_list(memory.current_objects()))
            if rel:
                text += f" {rel}"
    return AnswerBundle(text, res, results)


def _answer_changes(intent: QueryIntent, memory: Any) -> AnswerBundle:
    room = intent.room or None
    if _has_method(memory, "changes"):
        changes = _obs_list(memory.changes(room, CHANGES_WINDOW_S))
        text = phrasing.changes_answer(changes, room, CHANGES_WINDOW_S)
        return AnswerBundle(text, changes=changes)
    # Legacy memory: "recently spotted" wording from the per-label cache.
    recent = _obs_list(memory.recent_changes(window_seconds=60))
    if not recent:
        return AnswerBundle("No objects have moved in the last minute.")
    parts = []
    for o in recent[:5]:
        pos = image_position_phrase(o.cx, o.cy)
        parts.append(f"{o.label} at {format_clock(o.timestamp_ms)}, {pos}")
    return AnswerBundle("Recently spotted: " + "; ".join(parts) + ".", changes=recent)


def _answer_list(intent: QueryIntent, memory: Any) -> AnswerBundle:
    room = intent.room or None
    if room and _accepts_kw(memory.current_objects, "room"):
        objects = _obs_list(memory.current_objects(room=room))
    else:
        objects = _obs_list(memory.current_objects())
        if room:
            objects = [
                o
                for o in objects
                if phrasing.room_display(getattr(o, "room", None)) == phrasing.room_display(room)
            ]
    if not objects:
        if room:
            return AnswerBundle(
                f"I haven't seen any objects in the {phrasing.room_display(room)} yet."
            )
        return AnswerBundle(
            "I haven't seen any objects yet. Try pointing the camera around the room."
        )
    specs = _room_specs_for(memory, objects)
    now_ms = int(time.time() * 1000)
    parts = []
    for o in objects[:_LIST_MAX]:
        where = phrasing.location_phrase(
            o,
            room_spec=specs.get(getattr(o, "room", None) or ""),
            egocentric=False,
            name_room=not room,
        )
        age = format_age((now_ms - int(getattr(o, "timestamp_ms", now_ms))) / 1000.0)
        parts.append(f"{o.label} ({where}, seen {age})")
    head = f"In the {phrasing.room_display(room)}, I'm tracking: " if room else "I'm tracking: "
    result = head + "; ".join(parts) + "."
    if len(objects) > _LIST_MAX:
        result += f" Plus {len(objects) - _LIST_MAX} more objects."
    return AnswerBundle(result, results=objects)


def answer_query(
    intent: QueryIntent,
    memory: Any,
    vlm_answer: str = "",
    *,
    current_pose: CameraPose | None = None,
) -> AnswerBundle:
    """Answer an intent against ``memory``; returns the text and the evidence used."""
    if vlm_answer:
        return AnswerBundle(vlm_answer)

    if intent.kind == "where_am_i":
        spec = _room_spec(memory, getattr(current_pose, "room", None))
        anchor = getattr(spec, "anchor_name", None)
        return AnswerBundle(
            phrasing.where_am_i_answer(
                current_pose,
                anchor_name=anchor if isinstance(anchor, str) else phrasing.DEFAULT_ANCHOR_NAME,
            )
        )

    if intent.kind == "list_objects":
        return _answer_list(intent, memory)

    if intent.kind == "changes":
        return _answer_changes(intent, memory)

    if intent.kind in ("where_is", "history") and intent.object_name:
        return _answer_object(intent, memory, current_pose)

    return AnswerBundle("I'm not sure what you're asking. Try saying 'where is my [object]?'")


def build_response_text(
    intent: QueryIntent,
    memory: SpatialMemory,
    vlm_answer: str = "",
    *,
    current_pose: CameraPose | None = None,
) -> str:
    """Build a natural language response from intent + LOCI-DB query results."""
    return answer_query(intent, memory, vlm_answer, current_pose=current_pose).text


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

WHISPER_SAMPLE_RATE = 16_000


def wav_bytes_to_float32(audio_bytes: bytes, target_rate: int = WHISPER_SAMPLE_RATE) -> Any:
    """Decode RIFF/WAVE PCM bytes to a mono float32 numpy array at ``target_rate``.

    Returns ``None`` for anything that is not a PCM WAV this function can read
    (browser webm/opus, compressed WAV subtypes); the caller then falls back to the
    container decoder.  Used so faster-whisper never needs PyAV for the headset's
    16 kHz 16-bit WAV: faster-whisper's ``decode_audio`` calls
    ``av.open(..., metadata_errors=...)``, which PyAV >= 15 rejects with TypeError.
    """
    if len(audio_bytes) < 44 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        return None
    import wave

    import numpy as np

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError, ValueError):
        return None
    if channels < 1 or rate <= 0:
        return None
    if width == 2:
        pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        pcm = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        pcm = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        return None
    if channels > 1:
        usable = (pcm.size // channels) * channels
        pcm = pcm[:usable].reshape(-1, channels).mean(axis=1)
    if rate != target_rate and pcm.size > 1:
        n_out = max(1, int(round(pcm.size * target_rate / rate)))
        x_old = np.linspace(0.0, 1.0, num=pcm.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        pcm = np.interp(x_new, x_old, pcm).astype(np.float32)
    return np.ascontiguousarray(pcm, dtype=np.float32)


_PYAV_PATCHED = False


def ensure_pyav_compat() -> bool:
    """Make faster-whisper's ``av.open(..., metadata_errors="ignore")`` call survive PyAV >= 15.

    PyAV 15 removed the ``metadata_errors`` / ``metadata_encoding`` keyword arguments;
    faster-whisper (<= 1.2) still passes them, so every non-WAV transcription would
    fail with TypeError.  Installed once; returns True when ``av`` is importable.
    """
    global _PYAV_PATCHED
    if _PYAV_PATCHED:
        return True
    try:
        import av
    except Exception:
        return False
    original = av.open
    if getattr(original, "_loci_compat", False):
        _PYAV_PATCHED = True
        return True

    def _open(*args: Any, **kwargs: Any) -> Any:
        try:
            return original(*args, **kwargs)
        except TypeError as exc:
            dropped = {
                k: kwargs.pop(k) for k in ("metadata_errors", "metadata_encoding") if k in kwargs
            }
            if not dropped or "metadata" not in str(exc):
                raise
            return original(*args, **kwargs)

    _open._loci_compat = True  # type: ignore[attr-defined]
    av.open = _open
    _PYAV_PATCHED = True
    return True


class WhisperSTT:
    """Speech-to-text via OpenAI Whisper API or local faster-whisper."""

    def __init__(self) -> None:
        self._use_local = os.environ.get("WHISPER_LOCAL", "0") == "1"
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")
        self._model_name = os.environ.get("WHISPER_MODEL", "whisper-1")
        self._local_model = None
        self._openai_client = None

    def _get_openai_client(self):
        if self._openai_client is None:
            import openai

            self._openai_client = openai.AsyncOpenAI(api_key=self._openai_key)
        return self._openai_client

    def _load_local_model(self):
        if self._local_model is not None:
            return
        try:
            from faster_whisper import WhisperModel

            model_size = os.environ.get("WHISPER_LOCAL_MODEL", "base")
            self._local_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info("Loaded local Whisper model: %s", model_size)
        except ImportError:
            logger.warning("faster-whisper not installed — falling back to OpenAI API")
            self._use_local = False

    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        """Transcribe audio bytes to text. Returns empty string on failure."""
        if self._use_local:
            return await self._transcribe_local(audio_bytes, language)
        if self._openai_key:
            return await self._transcribe_openai(audio_bytes, language)
        logger.warning("No STT backend available (set OPENAI_API_KEY or WHISPER_LOCAL=1)")
        return ""

    async def _transcribe_openai(self, audio_bytes: bytes, language: str) -> str:
        try:
            client = self._get_openai_client()  # ImportError without the openai package
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.webm"
            response = await client.audio.transcriptions.create(
                model=self._model_name,
                file=audio_file,
                language=language,
                response_format="text",
            )
            return str(response).strip()
        except Exception as e:
            logger.error("Whisper API transcription failed: %s", e)
            return ""

    async def _transcribe_local(self, audio_bytes: bytes, language: str) -> str:
        self._load_local_model()
        if self._local_model is None:
            return await self._transcribe_openai(audio_bytes, language)
        try:
            # PCM WAV (headset loop, phone page recordings converted server-side) is
            # decoded with the stdlib and handed over as float32 samples, bypassing
            # PyAV entirely; other containers (browser webm/opus) still go through
            # faster-whisper's decoder with the PyAV >= 15 keyword shim in place.
            audio: Any = wav_bytes_to_float32(audio_bytes)
            if audio is None:
                ensure_pyav_compat()
                audio = io.BytesIO(audio_bytes)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: list(
                    self._local_model.transcribe(
                        audio,
                        language=language,
                        beam_size=1,
                    )[0]
                ),
            )
            return " ".join(seg.text.strip() for seg in result)
        except Exception as e:
            logger.error("Local Whisper transcription failed: %s", e)
            return ""

    @property
    def is_available(self) -> bool:
        return bool(self._openai_key) or self._use_local


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


class TTS:
    """Text-to-speech with pluggable backends.

    Engine selection via TTS_ENGINE env var:
      - "openai"  (default): OpenAI TTS API (tts-1 / tts-1-hd)
      - "piper":  Local Piper TTS (model files must be present)
      - "edge":   Microsoft Edge TTS (free, cloud)
    """

    def __init__(self) -> None:
        self._engine = os.environ.get("TTS_ENGINE", "openai").lower()
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")
        # nova, alloy, echo, fable, onyx, shimmer
        self._openai_voice = os.environ.get("TTS_VOICE", "nova")
        self._openai_model = os.environ.get("TTS_MODEL", "tts-1")
        self._piper_model = os.environ.get("PIPER_MODEL_PATH", "")
        self._edge_voice = os.environ.get("EDGE_TTS_VOICE", "en-US-AnaNeural")
        self._openai_client = None
        self._piper_voice = None

    def _get_openai_client(self):
        if self._openai_client is None:
            import openai

            self._openai_client = openai.AsyncOpenAI(api_key=self._openai_key)
        return self._openai_client

    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (MP3 or WAV depending on engine)."""
        if not text.strip():
            return b""

        if self._engine == "piper":
            result = await self._synthesize_piper(text)
            if result:
                return result
            # Fall through to next engine
            logger.warning("Piper TTS failed, falling back to edge-tts")

        if self._engine in ("edge", "piper"):
            result = await self._synthesize_edge(text)
            if result:
                return result
            logger.warning("Edge TTS failed, falling back to OpenAI")

        # Default: OpenAI TTS
        if self._openai_key:
            return await self._synthesize_openai(text)

        logger.error("No TTS backend available")
        return b""

    async def _synthesize_openai(self, text: str) -> bytes:
        try:
            client = self._get_openai_client()  # ImportError without the openai package
            response = await client.audio.speech.create(
                model=self._openai_model,
                voice=self._openai_voice,
                input=text,
                response_format="mp3",
            )
            return response.content
        except Exception as e:
            logger.error("OpenAI TTS failed: %s", e)
            return b""

    async def _synthesize_piper(self, text: str) -> bytes:
        """Synthesize using local Piper TTS model."""
        if not self._piper_model:
            logger.debug("PIPER_MODEL_PATH not set")
            return b""
        try:
            import piper

            if self._piper_voice is None:
                self._piper_voice = piper.PiperVoice.load(self._piper_model)

            loop = asyncio.get_event_loop()
            audio_bytes = await loop.run_in_executor(None, self._run_piper_sync, text)
            return audio_bytes
        except ImportError:
            logger.debug("piper-tts not installed")
            return b""
        except Exception as e:
            logger.error("Piper TTS error: %s", e)
            return b""

    def _run_piper_sync(self, text: str) -> bytes:
        """Run Piper synthesis synchronously (called from executor).

        piper-tts >= 1.3 renamed the file API to ``synthesize_wav(text, wav_file)``;
        older releases expose ``synthesize(text, wav_file)``.  Both write the WAV
        header themselves.
        """
        import wave

        voice = self._piper_voice
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            if hasattr(voice, "synthesize_wav"):
                voice.synthesize_wav(text, wf)
            else:
                voice.synthesize(text, wf)
        return buf.getvalue()

    async def _synthesize_edge(self, text: str) -> bytes:
        """Synthesize using Microsoft Edge TTS (edge-tts library)."""
        try:
            import edge_tts

            communicate = edge_tts.Communicate(text, self._edge_voice)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf.getvalue()
        except ImportError:
            logger.debug("edge-tts not installed")
            return b""
        except Exception as e:
            logger.error("Edge TTS error: %s", e)
            return b""

    @property
    def engine(self) -> str:
        return self._engine


# ---------------------------------------------------------------------------
# High-level VoicePipeline
# ---------------------------------------------------------------------------


class VoicePipeline:
    """End-to-end voice query pipeline: audio → text → answer → audio.

    Usage::

        pipeline = VoicePipeline(memory, vlm_client, pose_provider=localizer.current)
        result = await pipeline.handle_audio_query(audio_bytes)
        # result.answer_text  — text of the answer
        # result.answer_audio — audio bytes for playback
    """

    def __init__(
        self,
        memory: SpatialMemory,
        vlm_client: VLMClient | None = None,
        *,
        pose_provider: Callable[[], CameraPose | None] | None = None,
    ) -> None:
        self._memory = memory
        self._vlm = vlm_client
        self._pose_provider = pose_provider
        self._stt = WhisperSTT()
        self._tts = TTS()

    @dataclass
    class QueryResult:
        transcription: str
        intent: QueryIntent
        answer_text: str
        answer_audio: bytes
        latency_ms: float
        resolution: Resolution | None = None
        results: list[Any] = field(default_factory=list)
        changes: list[Any] = field(default_factory=list)

        @property
        def audio_mime(self) -> str | None:
            return VoicePipeline.audio_mime(self.answer_audio) if self.answer_audio else None

    @staticmethod
    def audio_mime(audio: bytes) -> str:
        """Sniff the container of TTS output: WAV (Piper), MP3 (OpenAI/edge) or unknown."""
        head = bytes(audio[:4]) if audio else b""
        if head[:4] == b"RIFF":
            return "audio/wav"
        if head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0):
            return "audio/mpeg"
        if head[:4] == b"OggS":
            return "audio/ogg"
        return "application/octet-stream"

    # -- helpers -----------------------------------------------------------

    def current_pose(self) -> CameraPose | None:
        """The localizer's current pose (None without a provider or on error)."""
        if self._pose_provider is None:
            return None
        try:
            return self._pose_provider()
        except Exception as e:
            logger.debug("pose_provider failed: %s", e)
            return None

    def known_rooms(self) -> list[str]:
        """Configured room names plus rooms seen, for ``parse_intent``."""
        names: list[str] = []
        rooms = getattr(self._memory, "rooms", None)
        if rooms is not None and _has_method(rooms, "names"):
            with contextlib.suppress(Exception):  # a RoomMap fake without names()
                names.extend(_str_list(rooms.names()))
        names.extend(_str_list(getattr(self._memory, "rooms_seen", None)))
        out: list[str] = []
        for n in names:
            if n not in out:
                out.append(n)
        return out

    def parse(self, text: str) -> QueryIntent:
        return parse_intent(text, self.known_rooms())

    def _answer(self, intent: QueryIntent) -> AnswerBundle:
        return answer_query(intent, self._memory, current_pose=self.current_pose())

    async def _generate_answer(self, intent: QueryIntent) -> str:
        """Query LOCI-DB and build a rule-based answer. Fast path: no LLM call."""
        return self._answer(intent).text

    # -- entry points ------------------------------------------------------

    async def handle_audio_query(
        self,
        audio_bytes: bytes,
        language: str = "en",
        synthesize_response: bool = True,
    ) -> VoicePipeline.QueryResult:
        """Full pipeline: audio → transcribe → understand → query → respond.

        Args:
            audio_bytes: Raw audio (WebM/WAV/MP3 from browser mic).
            language: ISO 639-1 language code for STT.
            synthesize_response: If False, skip TTS (text-only response).

        Returns:
            QueryResult with transcription, intent, answer text, and audio.
        """
        t0 = time.perf_counter()

        # Step 1: Transcribe
        transcript = await self._stt.transcribe(audio_bytes, language)
        if not transcript:
            answer = "Sorry, I couldn't understand the audio. Please try again."
            audio = await self._tts.synthesize(answer) if synthesize_response else b""
            return self.QueryResult(
                transcription="",
                intent=QueryIntent("unknown", "", ""),
                answer_text=answer,
                answer_audio=audio,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Step 2: Parse intent
        intent = self.parse(transcript)

        # Step 3: Query spatial memory + generate answer
        bundle = self._answer(intent)

        # Step 4: Synthesize speech
        audio = await self._tts.synthesize(bundle.text) if synthesize_response else b""

        latency_ms = (time.perf_counter() - t0) * 1000
        return self.QueryResult(
            transcription=transcript,
            intent=intent,
            answer_text=bundle.text,
            answer_audio=audio,
            latency_ms=round(latency_ms, 1),
            resolution=bundle.resolution,
            results=bundle.results,
            changes=bundle.changes,
        )

    async def handle_text_query(
        self, text: str, *, synthesize_response: bool = True
    ) -> VoicePipeline.QueryResult:
        """Text-only pipeline (skip STT; TTS unless ``synthesize_response=False``)."""
        t0 = time.perf_counter()
        intent = self.parse(text)
        bundle = self._answer(intent)
        audio = await self._tts.synthesize(bundle.text) if synthesize_response else b""
        return self.QueryResult(
            transcription=text,
            intent=intent,
            answer_text=bundle.text,
            answer_audio=audio,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            resolution=bundle.resolution,
            results=bundle.results,
            changes=bundle.changes,
        )

    def answer_text(self, text: str) -> str:
        """Synchronous text answer (no TTS); handy for the headset loop's ``answer_fn``."""
        return self._answer(self.parse(text)).text

    @property
    def stt_available(self) -> bool:
        return self._stt.is_available

    @property
    def tts_engine(self) -> str:
        return self._tts.engine

    def status(self) -> dict:
        return {
            "stt_available": self.stt_available,
            "tts_engine": self.tts_engine,
            "vlm_available": self._vlm.is_available if self._vlm else False,
            "pose_provider": self._pose_provider is not None,
        }
