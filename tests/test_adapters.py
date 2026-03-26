"""Tests for world model adapters."""

import numpy as np
import pytest

from loci.adapters.dreamer import DreamerV3Adapter
from loci.adapters.generic import GenericAdapter
from loci.adapters.vjepa2 import VJEPA2Adapter
from loci.schema import WorldState


class TestVJEPA2Adapter:
    def test_tubelet_to_world_state(self):
        adapter = VJEPA2Adapter()
        embedding = np.random.randn(1408).astype(np.float32)
        ws = adapter.tubelet_to_world_state(
            tubelet_embedding=embedding,
            patch_position=(0, 5, 10),
            scene_bounds=(20.0, 20.0, 10.0),
            timestamp_ms=1000,
            scene_id="scene_1",
        )
        assert isinstance(ws, WorldState)
        assert len(ws.vector) == 1408
        assert ws.timestamp_ms == 1000
        assert ws.scene_id == "scene_1"
        assert 0.0 <= ws.x <= 1.0
        assert 0.0 <= ws.y <= 1.0

    def test_tubelet_rejects_2d(self):
        adapter = VJEPA2Adapter()
        with pytest.raises(ValueError, match="1D"):
            adapter.tubelet_to_world_state(
                tubelet_embedding=np.zeros((2, 1408)),
                patch_position=(0, 0, 0),
                scene_bounds=(1.0, 1.0, 1.0),
                timestamp_ms=0,
                scene_id="s",
            )

    def test_batch_clip_to_states(self):
        adapter = VJEPA2Adapter()
        clip = np.random.randn(2, 3, 4, 1408).astype(np.float32)
        states = adapter.batch_clip_to_states(
            clip_embeddings=clip,
            scene_bounds=(10.0, 10.0, 5.0),
            start_timestamp_ms=0,
            scene_id="clip_1",
        )
        assert len(states) == 2 * 3 * 4  # T * H * W
        assert all(isinstance(s, WorldState) for s in states)
        assert states[0].timestamp_ms == 0
        assert states[-1].timestamp_ms == 33  # second temporal patch

    def test_batch_rejects_3d(self):
        adapter = VJEPA2Adapter()
        with pytest.raises(ValueError, match="4D"):
            adapter.batch_clip_to_states(
                clip_embeddings=np.zeros((2, 3, 1408)),
                scene_bounds=(1.0, 1.0, 1.0),
                start_timestamp_ms=0,
                scene_id="s",
            )


class TestDreamerV3Adapter:
    def test_rssm_to_world_state(self):
        adapter = DreamerV3Adapter()
        h_t = np.random.randn(512).astype(np.float32)
        z_t = np.random.randn(1024).astype(np.float32)
        ws = adapter.rssm_to_world_state(
            h_t=h_t,
            z_t=z_t,
            position=(0.5, 0.5, 0.5),
            timestamp_ms=1000,
            scene_id="dream_1",
        )
        assert isinstance(ws, WorldState)
        assert len(ws.vector) == 512 + 1024
        assert ws.x == 0.5
        assert 0.0 <= ws.confidence <= 1.0

    def test_explicit_confidence(self):
        adapter = DreamerV3Adapter()
        ws = adapter.rssm_to_world_state(
            h_t=np.zeros(64),
            z_t=np.zeros(32),
            position=(0.0, 0.0, 0.0),
            timestamp_ms=0,
            scene_id="s",
            confidence=0.42,
        )
        assert ws.confidence == 0.42

    def test_rejects_2d_h_t(self):
        adapter = DreamerV3Adapter()
        with pytest.raises(ValueError, match="1D"):
            adapter.rssm_to_world_state(
                h_t=np.zeros((2, 64)),
                z_t=np.zeros(32),
                position=(0.0, 0.0, 0.0),
                timestamp_ms=0,
                scene_id="s",
            )


class TestGenericAdapter:
    def test_from_numpy(self):
        adapter = GenericAdapter()
        embedding = np.random.randn(512).astype(np.float32)
        ws = adapter.from_numpy(
            embedding=embedding,
            position=(0.1, 0.2, 0.3),
            timestamp_ms=5000,
            scene_id="generic_1",
        )
        assert isinstance(ws, WorldState)
        assert len(ws.vector) == 512
        assert ws.x == 0.1

    def test_dimension_validation(self):
        adapter = GenericAdapter(expected_dim=256)
        with pytest.raises(ValueError, match="256-dim"):
            adapter.from_numpy(
                embedding=np.zeros(512),
                position=(0.0, 0.0, 0.0),
                timestamp_ms=0,
                scene_id="s",
            )

    def test_accepts_correct_dim(self):
        adapter = GenericAdapter(expected_dim=256)
        ws = adapter.from_numpy(
            embedding=np.zeros(256),
            position=(0.0, 0.0, 0.0),
            timestamp_ms=0,
            scene_id="s",
        )
        assert len(ws.vector) == 256

    def test_rejects_2d(self):
        adapter = GenericAdapter()
        with pytest.raises(ValueError, match="1D"):
            adapter.from_numpy(
                embedding=np.zeros((2, 256)),
                position=(0.0, 0.0, 0.0),
                timestamp_ms=0,
                scene_id="s",
            )
