"""Warehouse robot simulation with LOCI episodic memory."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from loci import LocalLociClient, WorldState

from .embeddings import EMBEDDING_DIM, generate_embedding

logger = logging.getLogger(__name__)

# Grid size
GRID_W = 20
GRID_H = 20

# Object types
SHELF = "shelf"
CHARGER = "charger"
DOCK = "dock"
OBSTACLE = "obstacle"
ANOMALY = "anomaly"


@dataclass
class WarehouseObject:
    x: int
    y: int
    obj_type: str


def generate_warehouse() -> list[WarehouseObject]:
    """Create a warehouse layout with shelves, chargers, docks, and obstacles."""
    objects: list[WarehouseObject] = []

    # Shelving rows (aisles at x=2-3, 6-7, 10-11, 14-15)
    for shelf_col in [3, 4, 7, 8, 11, 12, 15, 16]:
        for row in range(2, 18):
            if row in (6, 7, 12, 13):
                continue  # gaps for cross-aisles
            objects.append(WarehouseObject(shelf_col, row, SHELF))

    # Charging stations
    for pos in [(0, 0), (0, 19), (19, 0)]:
        objects.append(WarehouseObject(pos[0], pos[1], CHARGER))

    # Loading docks along bottom wall
    for x in range(5, 15):
        objects.append(WarehouseObject(x, 19, DOCK))

    # A few obstacles
    for pos in [(9, 4), (10, 9), (5, 15)]:
        objects.append(WarehouseObject(pos[0], pos[1], OBSTACLE))

    return objects


def generate_patrol_route() -> list[tuple[int, int]]:
    """Generate a patrol route that snakes through warehouse aisles."""
    route: list[tuple[int, int]] = []

    # Start at bottom-left, snake through aisles
    aisle_xs = [2, 5, 6, 9, 10, 13, 14, 17]

    for i, ax in enumerate(aisle_xs):
        if i % 2 == 0:
            # Go up
            for y in range(18, 1, -1):
                route.append((ax, y))
        else:
            # Go down
            for y in range(2, 19):
                route.append((ax, y))

    # Return path along top
    for x in range(17, 1, -1):
        route.append((x, 1))

    # Close the loop along left side
    for y in range(1, 19):
        route.append((1, y))
    for x in range(1, 3):
        route.append((x, 18))

    return route


@dataclass
class Simulation:
    """Manages the warehouse, robot, and LOCI memory."""

    client: LocalLociClient = field(init=False)
    warehouse: list[WarehouseObject] = field(init=False)
    warehouse_grid: dict[tuple[int, int], str] = field(init=False)
    patrol_route: list[tuple[int, int]] = field(init=False)
    anomalies: list[tuple[int, int]] = field(default_factory=list)

    # Robot state
    robot_x: int = 2
    robot_y: int = 18
    route_idx: int = 0
    tick_count: int = 0
    start_time_ms: int = 0

    # Control
    running: bool = False
    subscribers: set[Any] = field(default_factory=set)

    # Recent embeddings for prediction
    recent_embeddings: list[list[float]] = field(default_factory=list)
    recent_positions: list[tuple[int, int]] = field(default_factory=list)
    recent_state_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.client = LocalLociClient(
            vector_size=EMBEDDING_DIM,
            epoch_size_ms=5000,
            decay_lambda=1e-4,
            distance="cosine",
        )
        self.warehouse = generate_warehouse()
        self.warehouse_grid = {(o.x, o.y): o.obj_type for o in self.warehouse}
        self.patrol_route = generate_patrol_route()

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.__post_init__()
        self.anomalies = []
        self.robot_x = 2
        self.robot_y = 18
        self.route_idx = 0
        self.tick_count = 0
        self.start_time_ms = 0
        self.running = False
        self.recent_embeddings = []
        self.recent_positions = []
        self.recent_state_ids = []

    def get_visible_objects(self, x: int, y: int, radius: int = 3) -> list[dict]:
        """Find all objects within radius of (x, y)."""
        visible = []
        for ox, oy in list(self.warehouse_grid.keys()) + [(ax, ay) for ax, ay in self.anomalies]:
            dx = ox - x
            dy = oy - y
            if dx * dx + dy * dy <= radius * radius:
                obj_type = self.warehouse_grid.get((ox, oy))
                if obj_type is None and (ox, oy) in self.anomalies:
                    obj_type = ANOMALY
                if obj_type:
                    visible.append({"x": ox, "y": oy, "type": obj_type})
        return visible

    def get_visible_object_keys(self, x: int, y: int, radius: int = 3) -> list[str]:
        """Get string keys for visible objects (for embedding generation)."""
        visible = self.get_visible_objects(x, y, radius)
        return [f"{v['type']}@{v['x']},{v['y']}" for v in visible]

    @property
    def elapsed_ms(self) -> int:
        return self.tick_count * 500

    @property
    def memory_count(self) -> int:
        return self.client.store.total_points

    def get_warehouse_layout(self) -> list[dict]:
        """Return warehouse objects as serializable dicts."""
        result = [{"x": o.x, "y": o.y, "type": o.obj_type} for o in self.warehouse]
        for ax, ay in self.anomalies:
            result.append({"x": ax, "y": ay, "type": ANOMALY})
        return result

    async def tick(self) -> dict | None:
        """Advance simulation by one step. Returns tick data for broadcast."""
        if not self.running:
            return None

        if self.start_time_ms == 0:
            self.start_time_ms = int(time.time() * 1000)

        # Move robot along patrol route
        self.robot_x, self.robot_y = self.patrol_route[self.route_idx]
        self.route_idx = (self.route_idx + 1) % len(self.patrol_route)

        # Calculate velocity (direction to next position)
        next_x, next_y = self.patrol_route[self.route_idx]
        vel_x = next_x - self.robot_x
        vel_y = next_y - self.robot_y

        # Observe surroundings
        visible = self.get_visible_objects(self.robot_x, self.robot_y)
        visible_keys = self.get_visible_object_keys(self.robot_x, self.robot_y)

        # Generate embedding
        embedding = generate_embedding(self.robot_x, self.robot_y, visible_keys)

        # Insert into LOCI
        timestamp_ms = self.start_time_ms + self.elapsed_ms
        state = WorldState(
            x=self.robot_x / (GRID_W - 1),
            y=self.robot_y / (GRID_H - 1),
            z=0.5,
            timestamp_ms=timestamp_ms,
            vector=embedding,
            scene_id="warehouse",
            scale_level="patch",
        )
        state_id = self.client.insert(state)

        # Track recent history for prediction
        self.recent_embeddings.append(embedding)
        self.recent_positions.append((self.robot_x, self.robot_y))
        self.recent_state_ids.append(state_id)
        if len(self.recent_embeddings) > 20:
            self.recent_embeddings.pop(0)
            self.recent_positions.pop(0)
            self.recent_state_ids.pop(0)

        self.tick_count += 1

        # Build tick data
        tick_data = {
            "type": "tick",
            "robot": {"x": self.robot_x, "y": self.robot_y, "velocity": [vel_x, vel_y]},
            "tick": self.tick_count,
            "elapsed_ms": self.elapsed_ms,
            "memory_count": self.memory_count,
            "visible_objects": visible,
            "observation_radius": 3,
            "timestamp_ms": timestamp_ms,
        }

        # Broadcast to WebSocket subscribers
        await self.broadcast(tick_data)

        return tick_data

    async def broadcast(self, data: dict) -> None:
        """Send data to all WebSocket subscribers."""
        dead = set()
        for ws in self.subscribers:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self.subscribers -= dead

    async def run_loop(self) -> None:
        """Main simulation loop — runs as asyncio background task."""
        self.running = True
        while self.running:
            await self.tick()
            await asyncio.sleep(0.5)

    def make_predictor(self, steps_ahead: int = 10):
        """Create a simple linear extrapolation predictor."""
        recent = self.recent_embeddings[-3:] if len(self.recent_embeddings) >= 3 else self.recent_embeddings[:]

        def predictor_fn(context: list[float]) -> list[float]:
            if len(recent) < 2:
                return context

            # Average the deltas between recent embeddings
            deltas = []
            for i in range(1, len(recent)):
                delta = [recent[i][j] - recent[i - 1][j] for j in range(len(recent[0]))]
                deltas.append(delta)

            avg_delta = [
                sum(d[j] for d in deltas) / len(deltas) for j in range(len(deltas[0]))
            ]

            # Extrapolate
            predicted = [
                context[j] + avg_delta[j] * steps_ahead for j in range(len(context))
            ]

            # L2 normalize
            import math

            norm = math.sqrt(sum(v * v for v in predicted))
            if norm > 1e-8:
                predicted = [v / norm for v in predicted]

            return predicted

        return predictor_fn
