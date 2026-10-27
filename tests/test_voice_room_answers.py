"""Tests for the handheld additions to demo_spatial/app/voice_pipeline.py (section 4.10).

Room capture in ``parse_intent``, resolver-aware ``build_response_text`` against a
``MagicMock`` memory (the post-T4 API) and against a legacy fake (today's API),
``handle_text_query(synthesize_response=False)``, ``audio_mime`` and the Piper >= 1.3
``synthesize_wav`` path.  No STT/TTS backend, camera or LOCI client is used.
"""

from __future__ import annotations

import asyncio
import io
import sys
import time
import types
import wave
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo_spatial"))

from app import phrasing as P  # noqa: E402
from app import voice_pipeline as vp  # noqa: E402
from app.detection_source import CameraPose, camera_pose_T  # noqa: E402
from app.label_resolver import Resolution  # noqa: E402
from app.room_anchor import RoomSpec  # noqa: E402
from app.voice_pipeline import (  # noqa: E402
    TTS,
    QueryIntent,
    VoicePipeline,
    answer_query,
    build_response_text,
    match_room,
    parse_intent,
)

# Answers compute ages from time.time(). Sightings are placed relative to NOW_MS, and the
# clock is pinned to NOW_MS for every test below, so "3 minutes ago" stays "3 minutes ago"
# however long the rest of the suite takes to reach this file (CI needs over a minute).
NOW_MS = int(time.time() * 1000)
TS = NOW_MS - 180_000


@pytest.fixture(autouse=True)
def _frozen_clock(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: NOW_MS / 1000)


ROOMS = ["kitchen", "living_room"]
T_KITCHEN = camera_pose_T((0.0, 1.4, 2.5), 0.0)
KITCHEN = RoomSpec(name="kitchen", anchor_tag=0, landmarks={"table": (1.0, 0.75, 1.2)})
LIVING = RoomSpec(name="living_room", anchor_tag=1, anchor_name="hall tag")
SPECS = {"kitchen": KITCHEN, "living_room": LIVING}


@dataclass
class FakeObs:
    label: str = "cup"
    cx: float = 0.5
    cy: float = 0.5
    confidence: float = 0.9
    timestamp_ms: int = TS
    state_id: str = "s1"
    depth_m: float | None = None
    room: str | None = None
    frame: str = "image"
    pos_m: tuple[float, float, float] | None = None
    xyz_cam_m: tuple[float, float, float] | None = None
    anchored: bool = False
    anchor_confidence: float = 0.0
    restored: bool = False

    @property
    def age_seconds(self) -> float:
        return (NOW_MS - self.timestamp_ms) / 1000.0


@dataclass
class FakeChange:
    kind: str
    label: str
    room: str | None = "kitchen"
    last_seen_ms: int = TS
    first_seen_ms: int | None = None
    pos_m: tuple[float, float, float] | None = None
    pos_before_m: tuple[float, float, float] | None = None
    delta_m: float | None = None


def room_obs(pos=(1.0, 0.8, 1.0), *, room="kitchen", anchored=True, **kw) -> FakeObs:
    return FakeObs(frame="room", room=room, pos_m=pos, anchored=anchored, **kw)


def kitchen_pose(anchored: bool = True) -> CameraPose:
    return CameraPose(
        room="kitchen",
        frame="room",
        T_room_cam=T_KITCHEN,
        anchored=anchored,
        anchor_confidence=1.0 if anchored else 0.0,
        tag_ids=(0,),
        age_s=0.2,
    )


def mock_memory(*, resolution=None, results=(), changes=(), objects=()) -> MagicMock:
    """A MagicMock with the section 4.8 surface configured for one answer."""
    memory = MagicMock(name="SpatialMemory")
    memory.resolve_label.return_value = resolution or Resolution("cup", "alias", 1.0)
    memory.where_is.return_value = list(results)
    memory.changes.return_value = list(changes)
    memory.current_objects.return_value = list(objects)
    memory.rooms.names.return_value = list(ROOMS)
    memory.rooms.get.side_effect = lambda r: SPECS.get(r)
    memory.rooms_seen = ["kitchen"]
    memory.tracked_labels = ["cup", "bottle"]
    memory.started_at_ms = NOW_MS - 3_600_000
    return memory


class LegacyMemory:
    """Today's SpatialMemory API: no resolver, rooms, changes or room keywords."""

    def __init__(self, observations: list[FakeObs]) -> None:
        self._obs = observations
        self.calls: list[tuple] = []

    def where_is(self, label, limit=5, min_confidence=None):
        self.calls.append(("where_is", label, limit))
        return [o for o in self._obs if o.label == label][:limit]

    def current_objects(self):
        return list(self._obs)

    def recent_changes(self, window_seconds=30.0):
        return [o for o in self._obs if o.timestamp_ms >= NOW_MS - window_seconds * 1000]

    @property
    def tracked_labels(self):
        return [o.label for o in self._obs]


# ---------------------------------------------------------------------------
# parse_intent: room capture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind", "obj", "room"),
    [
        ("what changed in the kitchen", "changes", "", "kitchen"),
        ("What changed in the kitchen?", "changes", "", "kitchen"),
        ("what's changed in my living room", "changes", "", "living_room"),
        ("what has moved around the living room?", "changes", "", "living_room"),
        ("what is new in the kitchen since lunch", "changes", "", "kitchen"),
        ("what changed", "changes", "", ""),
        ("what's new", "changes", "", ""),
        ("any recent changes", "changes", "", ""),
        ("where is my mug in the living room", "where_is", "mug", "living_room"),
        ("Where is my mug in the living room?", "where_is", "mug", "living_room"),
        ("have you seen the bottle in the kitchen", "where_is", "bottle", "kitchen"),
        ("where did I put my keys in the kitchen", "where_is", "keys", "kitchen"),
        ("mug in the kitchen", "where_is", "mug", "kitchen"),
        ("where is my mug", "where_is", "mug", ""),
        ("my keys?", "where_is", "keys", ""),
        ("what's in the kitchen", "list_objects", "", "kitchen"),
        ("What is in my living room?", "list_objects", "", "living_room"),
        ("what's in the room", "list_objects", "", ""),
        ("what's around", "list_objects", "", ""),
        ("list all objects", "list_objects", "", ""),
        ("when did you last see my cup in the kitchen", "history", "cup", "kitchen"),
        ("when did I last see my phone", "history", "phone", ""),
        ("where am I", "where_am_i", "", ""),
        ("Which room am I in?", "where_am_i", "", ""),
    ],
)
def test_parse_intent_rooms(text, kind, obj, room):
    intent = parse_intent(text, ROOMS)
    assert (intent.kind, intent.object_name, intent.room) == (kind, obj, room)
    assert intent.raw_text == text
    assert intent.to_dict() == {"kind": kind, "object_name": obj, "room": room}


def test_parse_intent_without_known_rooms_keeps_spoken_room():
    intent = parse_intent("where is my mug in the living room")
    assert (intent.kind, intent.object_name, intent.room) == ("where_is", "mug", "living room")
    assert parse_intent("what changed in the Kitchen").room == "kitchen"
    assert parse_intent("what's in the room").room == ""


def test_parse_intent_unknown_room_text_is_kept_for_an_honest_not_seen():
    intent = parse_intent("where is my mug in the bedroom", ROOMS)
    assert (intent.object_name, intent.room) == ("mug", "bedroom")


def test_parse_intent_legacy_shapes_unchanged():
    assert parse_intent("where is my water bottle?").object_name == "water bottle"
    assert parse_intent("find the remote").kind == "where_is"
    unknown = parse_intent("tell me a long story about absolutely nothing at all please thanks")
    assert unknown.kind == "unknown" and unknown.room == ""
    assert QueryIntent("unknown", "", "").room == ""


@pytest.mark.parametrize(
    ("text", "kind", "obj", "room"),
    [
        # list questions, including the board's own "what can you see"
        ("What objects do you have in your memory?", "list_objects", "", ""),
        ("what can you see", "list_objects", "", ""),
        ("what have you seen", "list_objects", "", ""),
        ("what do you remember", "list_objects", "", ""),
        ("tell me what you see", "list_objects", "", ""),
        ("describe the room", "list_objects", "", ""),
        ("list everything", "list_objects", "", ""),
        ("what things are in the kitchen", "list_objects", "", "kitchen"),
        ("what's in your memory", "list_objects", "", ""),
        # where-is with wake word, fillers, courtesy, embedded questions, time phrases
        ("is my laptop in the kitchen", "where_is", "laptop", "kitchen"),
        ("is there a bottle in the living room", "where_is", "bottle", "living_room"),
        ("can you find the bottle please", "where_is", "bottle", ""),
        ("hey loci where are my glasses", "where_is", "glasses", ""),
        ("Hey Loci, um, where's the remote?", "where_is", "remote", ""),
        ("where did my keys go", "where_is", "keys", ""),
        ("where'd my keys go", "where_is", "keys", ""),
        ("do you know where my wallet is", "where_is", "wallet", ""),
        ("could you tell me where the cup is", "where_is", "cup", ""),
        ("have you seen my keys anywhere", "where_is", "keys", ""),
        ("where are my keys right now", "where_is", "keys", ""),
        ("where did I put my keys yesterday", "where_is", "keys", ""),
        ("I'm looking for my wallet", "where_is", "wallet", ""),
        ("where is my cup in your memory", "where_is", "cup", ""),
        ("where is my phone in the car", "where_is", "phone", ""),
        ("when was the last time you saw my keys", "history", "keys", ""),
        ("what changed since this morning", "changes", "", ""),
        ("has anything moved in the kitchen", "changes", "", "kitchen"),
        # not object searches: Whisper's silence hallucination, chit-chat, yes/no
        ("Thank you.", "unknown", "", ""),
        ("thanks", "unknown", "", ""),
        ("hey loci", "unknown", "", ""),
        ("Hey, Lucy.", "unknown", "", ""),
        ("Hey, Lucy, where did I leave my keys?", "where_is", "keys", ""),
        ("what time is it", "unknown", "", ""),
        ("how are you", "unknown", "", ""),
        ("tell me a joke", "unknown", "", ""),
        ("what's the weather like", "unknown", "", ""),
        ("is my laptop charged", "unknown", "", ""),
        ("hello", "unknown", "", ""),
        # bare object names still work
        ("glasses?", "where_is", "glasses", ""),
        ("the water bottle", "where_is", "water bottle", ""),
    ],
)
def test_parse_intent_everyday_phrasings(text, kind, obj, room):
    intent = parse_intent(text, ROOMS)
    assert (intent.kind, intent.object_name, intent.room) == (kind, obj, room)


def test_parse_intent_rooms_need_a_known_room_or_a_room_word():
    assert parse_intent("what's in the garage").room == "garage"
    assert parse_intent("where is my mug in the guest room").room == "guest room"
    assert parse_intent("what objects do you have in your memory").room == ""
    assert parse_intent("how long since my keys were seen").kind == "history"


def test_match_room_canonicalises():
    assert match_room("Living Room", ROOMS) == "living_room"
    assert match_room("living_room", ROOMS) == "living_room"
    assert match_room("kitchen since lunch", ROOMS) == "kitchen"
    assert match_room("kitchenette", ROOMS) == "kitchenette"
    assert match_room("the room", ROOMS) == ""
    assert match_room("", ROOMS) == ""
    assert match_room(None, ROOMS) == ""
    assert match_room("Guest Room", []) == "guest room"


# ---------------------------------------------------------------------------
# build_response_text with the post-T4 memory API (MagicMock)
# ---------------------------------------------------------------------------


def test_alias_disclosure_room_answer():
    memory = mock_memory(results=[room_obs()])
    intent = parse_intent("where is my mug", ROOMS)
    text = build_response_text(intent, memory)
    memory.resolve_label.assert_called_once_with("mug")
    memory.where_is.assert_called_once_with("cup", limit=5)
    assert text.startswith(
        "Your mug (I track it as a cup) was last seen in the kitchen, near the table."
    )
    assert "before I restarted" not in text


def test_room_filter_is_forwarded_and_walk_in_wording_without_landmark():
    memory = mock_memory(results=[room_obs((-0.8, 0.7, 1.0), room="living_room")])
    intent = parse_intent("where is my mug in the living room", ROOMS)
    text = build_response_text(intent, memory)
    memory.where_is.assert_called_once_with("cup", limit=5, room="living_room")
    assert "in the living room, about 1 metre into the room and 1 metre to your right" in text
    assert "as you walk in from the hall tag, at table height." in text


def test_camera_frame_answer_is_hedged():
    memory = mock_memory(results=[FakeObs(frame="camera", xyz_cam_m=(0.0, 0.4, 1.6))])
    text = build_response_text(parse_intent("where is my cup", ROOMS), memory)
    assert text.startswith(
        "Your cup was last seen about 1.5 metres in front of the camera, below eye line, "
        "roughly — I wasn't sure of my position."
    )


def test_held_pose_room_answer_carries_unanchored_suffix():
    memory = mock_memory(results=[room_obs((0.1, 0.8, 2.0), anchored=False)])
    text = build_response_text(parse_intent("where's the cup", ROOMS), memory)
    assert P.UNANCHORED_SUFFIX in text


def test_unknown_noun_refusal_never_queries_memory():
    res = Resolution(None, "none", 0.2, ["keyboard"])
    memory = mock_memory(resolution=res)
    text = build_response_text(parse_intent("where are my keys", ROOMS), memory)
    assert text == (
        "I can't recognise 'keys' with this camera. Did you mean keyboard? "
        "I can look for cup, bottle, cell phone, remote, book."
    )
    memory.where_is.assert_not_called()


def test_restored_sighting_says_so():
    memory = mock_memory(results=[room_obs(timestamp_ms=NOW_MS - 7_200_000)])
    text = build_response_text(parse_intent("where is my mug", ROOMS), memory)
    assert text.endswith(" That was before I restarted.")


def test_not_seen_in_room():
    memory = mock_memory(results=[])
    text = build_response_text(parse_intent("where is my mug in the kitchen", ROOMS), memory)
    assert text == "I haven't seen your mug in the kitchen yet (I look for it as a cup)."


def test_low_confidence_embedding_match_is_hedged():
    memory = mock_memory(
        resolution=Resolution("bird", "embedding", 0.51), results=[room_obs(label="bird")]
    )
    text = build_response_text(parse_intent("where is my beaker", ROOMS), memory)
    assert text.startswith("Your beaker (I think you mean a bird) was last seen")
    memory = mock_memory(
        resolution=Resolution("couch", "embedding", 0.68), results=[room_obs(label="couch")]
    )
    text = build_response_text(parse_intent("where is my sofa", ROOMS), memory)
    assert text.startswith("Your sofa (I track it as a couch) was last seen")


def test_history_with_rooms():
    results = [
        room_obs(),
        room_obs((-0.8, 0.7, 1.0), room="living_room", timestamp_ms=NOW_MS - 3_600_000),
        FakeObs(cx=0.9, cy=0.9, timestamp_ms=NOW_MS - 7_200_000),
    ]
    memory = mock_memory(results=results)
    intent = parse_intent("when did you last see my mug", ROOMS)
    assert intent.kind == "history"
    text = build_response_text(intent, memory)
    memory.where_is.assert_called_once_with("cup", limit=20)
    assert text.startswith(
        "Your mug (I track it as a cup) was last seen in the kitchen, near the table."
    )
    assert "Previous sightings:" in text
    assert (
        "in the living room, about 1 metre into the room and 1 metre to your right "
        "as you walk in from the hall tag"
    ) in text
    assert "on the right, near the front." in text


def test_egocentric_answer_with_current_pose():
    memory = mock_memory(results=[room_obs((-1.0, 0.8, -0.5))])
    intent = parse_intent("where is my mug", ROOMS)
    text = build_response_text(intent, memory, current_pose=kitchen_pose())
    assert "about 3 metres ahead of you and 1 metre to your left, at table height" in text
    text = build_response_text(intent, memory, current_pose=kitchen_pose(anchored=False))
    assert "as you walk in from the door tag" in text


def test_changes_answer_from_memory():
    changes = [FakeChange("moved", "cup", delta_m=2.4), FakeChange("gone", "laptop")]
    memory = mock_memory(changes=changes)
    intent = parse_intent("what changed in the kitchen", ROOMS)
    text = build_response_text(intent, memory)
    memory.changes.assert_called_once_with("kitchen", 600.0)
    assert text == (
        "In the kitchen in the last 10 minutes: the cup moved about 2.5 metres; "
        "I no longer see the laptop."
    )
    memory = mock_memory(changes=[])
    assert build_response_text(parse_intent("what's new", ROOMS), memory) == (
        "Nothing changed in the last 10 minutes."
    )
    memory.changes.assert_called_once_with(None, 600.0)


def test_list_objects_in_room():
    objects = [room_obs(), room_obs((-2.0, 0.1, 3.0), label="bottle", anchored=False)]
    memory = mock_memory(objects=objects)
    text = build_response_text(parse_intent("what's in the kitchen", ROOMS), memory)
    memory.current_objects.assert_called_once_with(room="kitchen")
    assert text.startswith(
        "In the kitchen, I'm tracking: cup (near the table, seen 3 minutes ago); bottle ("
    )
    assert "to your right as you walk in from the door tag, on the floor, roughly" in text
    memory = mock_memory(objects=[])
    assert build_response_text(parse_intent("what's in the kitchen", ROOMS), memory) == (
        "I haven't seen any objects in the kitchen yet."
    )
    memory = mock_memory(objects=[room_obs()])
    text = build_response_text(parse_intent("what do you see", ROOMS), memory)
    memory.current_objects.assert_called_once_with()
    assert text == "I'm tracking: cup (in the kitchen, near the table, seen 3 minutes ago)."


def test_where_am_i_and_vlm_short_circuit():
    memory = mock_memory()
    intent = parse_intent("where am I", ROOMS)
    assert build_response_text(intent, memory, current_pose=kitchen_pose()).startswith(
        "You're in the kitchen, about 2.5 metres in from the door tag"
    )
    assert build_response_text(intent, memory).startswith("I'm not sure which room you're in")
    assert build_response_text(intent, memory, vlm_answer="It is a cup.") == "It is a cup."
    assert build_response_text(QueryIntent("unknown", "", "blah"), memory).startswith(
        "I'm not sure what"
    )


def test_answer_query_returns_evidence():
    results = [room_obs()]
    memory = mock_memory(results=results)
    bundle = answer_query(parse_intent("where is my mug", ROOMS), memory)
    assert bundle.resolution == Resolution("cup", "alias", 1.0)
    assert bundle.results == results and bundle.changes == []
    changes = [FakeChange("new", "bottle")]
    bundle = answer_query(parse_intent("what changed", ROOMS), mock_memory(changes=changes))
    assert bundle.changes == changes and bundle.resolution is None


def test_example_labels_prefers_vocabulary_and_tracked():
    memory = mock_memory()
    assert vp.example_labels(memory) == ["cup", "bottle", "cell phone", "remote", "book"]
    memory.tracked_labels = ["laptop"]
    memory.vocabulary = ["laptop", "cup", "mouse"]
    assert vp.example_labels(memory) == ["laptop", "cup"]


# ---------------------------------------------------------------------------
# Legacy memory API keeps today's wording
# ---------------------------------------------------------------------------


def test_legacy_memory_answers_unchanged():
    # The legacy where_is path measures the age against the real clock at answer
    # time (not FakeObs.age_seconds), so take "now" here rather than at import:
    # a long full-suite run must not turn "just now" into "46 seconds ago".
    obs = FakeObs(label="keys", cx=0.1, cy=0.1, timestamp_ms=int(time.time() * 1000) - 2000)
    memory = LegacyMemory([obs])
    text = build_response_text(parse_intent("where are my keys"), memory)
    clock = P.format_clock(obs.timestamp_ms)
    assert (
        text
        == f"Your keys was last seen on the left, toward the back. That was just now, at {clock}."
    )
    assert memory.calls == [("where_is", "keys", 5)]

    assert build_response_text(parse_intent("where is my mug"), memory) == (
        "I haven't seen your mug yet. Try scanning the area."
    )
    assert build_response_text(parse_intent("list all objects"), memory) == (
        "I'm tracking: keys (on the left, toward the back, seen just now)."
    )
    assert build_response_text(parse_intent("what changed"), LegacyMemory([])) == (
        "No objects have moved in the last minute."
    )
    recent = build_response_text(parse_intent("what changed"), memory)
    assert recent.startswith("Recently spotted: keys at ")


def test_legacy_memory_room_filter_applied_locally():
    memory = LegacyMemory([FakeObs(label="cup")])
    text = build_response_text(parse_intent("where is my cup in the kitchen"), memory)
    assert text == "I haven't seen your cup in the kitchen yet."


# ---------------------------------------------------------------------------
# VoicePipeline
# ---------------------------------------------------------------------------


class RecordingTTS:
    def __init__(self, audio: bytes = b"RIFF\x00\x00\x00\x00WAVEfmt ") -> None:
        self.calls: list[str] = []
        self.audio = audio
        self.engine = "fake"

    async def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        return self.audio


def make_pipeline(memory, pose_provider=None) -> tuple[VoicePipeline, RecordingTTS]:
    pipeline = VoicePipeline(memory, None, pose_provider=pose_provider)
    tts = RecordingTTS()
    pipeline._tts = tts  # noqa: SLF001 - test seam
    return pipeline, tts


def test_handle_text_query_without_tts():
    memory = mock_memory(results=[room_obs()])
    pipeline, tts = make_pipeline(memory)
    result = asyncio.run(pipeline.handle_text_query("where is my mug", synthesize_response=False))
    assert result.answer_audio == b"" and tts.calls == []
    assert result.audio_mime is None
    assert result.intent.kind == "where_is" and result.intent.room == ""
    assert result.resolution == Resolution("cup", "alias", 1.0)
    assert result.results == [room_obs()] and result.changes == []
    assert result.answer_text.startswith(
        "Your mug (I track it as a cup) was last seen in the kitchen"
    )
    assert result.transcription == "where is my mug" and result.latency_ms >= 0


def test_handle_text_query_default_still_synthesizes():
    pipeline, tts = make_pipeline(mock_memory(results=[room_obs()]))
    result = asyncio.run(pipeline.handle_text_query("where is my mug"))
    assert tts.calls == [result.answer_text]
    assert result.answer_audio.startswith(b"RIFF") and result.audio_mime == "audio/wav"


def test_pipeline_uses_known_rooms_and_pose_provider():
    memory = mock_memory(results=[room_obs((-1.0, 0.8, -0.5))])
    pipeline, _ = make_pipeline(memory, pose_provider=kitchen_pose)
    assert pipeline.known_rooms() == ["kitchen", "living_room"]
    result = asyncio.run(
        pipeline.handle_text_query("where is my mug in the living room", synthesize_response=False)
    )
    assert result.intent.room == "living_room"
    memory.where_is.assert_called_with("cup", limit=5, room="living_room")
    result = asyncio.run(pipeline.handle_text_query("where is my mug", synthesize_response=False))
    assert "ahead of you" in result.answer_text
    assert pipeline.status()["pose_provider"] is True
    assert pipeline.answer_text("where am I").startswith("You're in the kitchen")


def test_pose_provider_errors_are_swallowed():
    def boom():
        raise RuntimeError("localizer down")

    pipeline, _ = make_pipeline(
        mock_memory(results=[room_obs((-1.0, 0.8, -0.5))]), pose_provider=boom
    )
    assert pipeline.current_pose() is None
    result = asyncio.run(pipeline.handle_text_query("where is my mug", synthesize_response=False))
    assert "as you walk in" in result.answer_text


def test_handle_audio_query_populates_evidence():
    memory = mock_memory(changes=[FakeChange("moved", "cup", delta_m=1.0)])
    pipeline, tts = make_pipeline(memory)

    async def fake_transcribe(audio_bytes, language="en"):
        return "what changed in the kitchen"

    pipeline._stt.transcribe = fake_transcribe  # noqa: SLF001 - test seam
    result = asyncio.run(pipeline.handle_audio_query(b"\x00" * 16, synthesize_response=False))
    assert result.transcription == "what changed in the kitchen"
    assert result.intent.kind == "changes" and result.intent.room == "kitchen"
    assert result.changes == memory.changes.return_value
    assert result.answer_text.startswith(
        "In the kitchen in the last 10 minutes: the cup moved about 1 metre"
    )
    assert tts.calls == []


def test_legacy_pipeline_construction_still_works():
    pipeline = VoicePipeline(LegacyMemory([]), None)
    assert pipeline.known_rooms() == []
    assert pipeline.current_pose() is None
    assert pipeline.status()["pose_provider"] is False


# ---------------------------------------------------------------------------
# audio_mime and Piper >= 1.3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("audio", "mime"),
    [
        (b"RIFF\x24\x00\x00\x00WAVE", "audio/wav"),
        (b"ID3\x04\x00\x00", "audio/mpeg"),
        (bytes([0xFF, 0xFB, 0x90, 0x00]), "audio/mpeg"),
        (bytes([0xFF, 0xF3, 0x90, 0x00]), "audio/mpeg"),
        (b"OggS\x00\x02", "audio/ogg"),
        (b"", "application/octet-stream"),
        (b"zz", "application/octet-stream"),
        (b"\x00\x01\x02\x03", "application/octet-stream"),
    ],
)
def test_audio_mime_sniff(audio, mime):
    assert VoicePipeline.audio_mime(audio) == mime


def _write_tone(wf) -> None:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(22050)
    wf.writeframes(b"\x00\x01" * 220)


class NewPiperVoice:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize_wav(self, text, wav_file, **kw) -> None:
        self.calls.append(("wav", text))
        _write_tone(wav_file)

    def synthesize(self, text, wav_file) -> None:  # pragma: no cover - must not be used
        raise AssertionError("old API called although synthesize_wav exists")


class OldPiperVoice:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text, wav_file) -> None:
        self.calls.append(("old", text))
        _write_tone(wav_file)


def _piper_tts(monkeypatch, voice) -> TTS:
    monkeypatch.setenv("TTS_ENGINE", "piper")
    monkeypatch.setenv("PIPER_MODEL_PATH", "/nonexistent/voice.onnx")
    tts = TTS()
    tts._piper_voice = voice  # noqa: SLF001 - skip PiperVoice.load
    return tts


def test_run_piper_sync_prefers_synthesize_wav(monkeypatch):
    voice = NewPiperVoice()
    audio = _piper_tts(monkeypatch, voice)._run_piper_sync("hello")
    assert voice.calls == [("wav", "hello")]
    assert audio.startswith(b"RIFF") and VoicePipeline.audio_mime(audio) == "audio/wav"
    with wave.open(io.BytesIO(audio), "rb") as wf:
        assert wf.getframerate() == 22050 and wf.getnframes() == 220


def test_run_piper_sync_falls_back_to_old_api(monkeypatch):
    voice = OldPiperVoice()
    audio = _piper_tts(monkeypatch, voice)._run_piper_sync("hello")
    assert voice.calls == [("old", "hello")]
    assert audio.startswith(b"RIFF")


def test_synthesize_piper_end_to_end_with_fake_module(monkeypatch):
    fake_piper = types.ModuleType("piper")
    fake_piper.PiperVoice = MagicMock()  # load() is bypassed by presetting the voice
    monkeypatch.setitem(sys.modules, "piper", fake_piper)
    tts = _piper_tts(monkeypatch, NewPiperVoice())
    audio = asyncio.run(tts.synthesize("Your mug is on the table."))
    assert audio.startswith(b"RIFF")
    fake_piper.PiperVoice.load.assert_not_called()
    assert asyncio.run(tts.synthesize("   ")) == b""
