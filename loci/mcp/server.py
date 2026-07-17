"""MCP server exposing Loci as a spatial memory for AI agents.

Implements a FastMCP server named ``loci-memory`` with tools to remember
observations (embedding + pose + time), recall them by similarity, place,
or time, score how novel a new observation is, and walk trajectories.

Configuration is read from environment variables at first use:

===================  =========================================================
``LOCI_MCP_MODE``    ``local`` (default, in-memory), ``qdrant``, or ``cloud``.
``QDRANT_URL``       Qdrant URL for ``qdrant`` mode (default
                     ``http://localhost:6333``).
``LOCI_CLOUD_URL``   Base URL for ``cloud`` mode (required in that mode).
``LOCI_API_KEY``     API key for ``cloud`` mode (required in that mode);
                     also passed to Qdrant in ``qdrant`` mode if set.
``LOCI_VECTOR_SIZE``    Embedding dimensionality (default 512).
``LOCI_EPOCH_SIZE_MS``  Temporal shard width in ms (default 5000).
``LOCI_DISTANCE``    ``cosine`` (default), ``dot``, or ``euclidean``.
``LOCI_SCENE_ID``    Default scene name for tools (default ``default``).
===================  =========================================================

The tool implementations are plain functions so they can be imported and
tested without the MCP SDK installed; :func:`build_server` registers them
onto a FastMCP instance and :func:`main` runs it on stdio.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loci.client import LociClient
from loci.local_client import LocalLociClient
from loci.retrieval.predict import PredictRetrieveResult
from loci.schema import ScoredWorldState, WorldState
from loci.temporal.sharding import epoch_id

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

_INSTALL_HINT = (
    "The MCP SDK is not installed. Install it with: pip install 'loci-stdb[mcp]' "
    "(or: uv pip install 'mcp>=1.2,<2')"
)

_VALID_MODES = ("local", "qdrant", "cloud")

# Over-fetch factor for recalls that need post-search filtering or
# re-ranking (place-only recency sort, cloud-mode scene post-filter).
_RECALL_OVERFETCH = 5


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServerConfig:
    """Resolved server configuration (see module docstring for the env vars)."""

    mode: str
    vector_size: int
    epoch_size_ms: int
    distance: str
    default_scene_id: str
    qdrant_url: str
    cloud_url: str | None
    api_key: str | None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from None
    if value <= 0:
        raise RuntimeError(f"{name} must be positive, got {value}")
    return value


def load_config() -> ServerConfig:
    """Read server configuration from the environment (validating as we go)."""
    mode = os.environ.get("LOCI_MCP_MODE", "local").strip().lower() or "local"
    if mode not in _VALID_MODES:
        raise RuntimeError(f"LOCI_MCP_MODE must be one of {list(_VALID_MODES)}, got {mode!r}")

    distance = os.environ.get("LOCI_DISTANCE", "cosine").strip().lower() or "cosine"
    if distance not in ("cosine", "dot", "euclidean"):
        raise RuntimeError(
            f"LOCI_DISTANCE must be one of ['cosine', 'dot', 'euclidean'], got {distance!r}"
        )

    cloud_url = os.environ.get("LOCI_CLOUD_URL") or None
    api_key = os.environ.get("LOCI_API_KEY") or None
    if mode == "cloud":
        if not cloud_url:
            raise RuntimeError("LOCI_MCP_MODE=cloud requires LOCI_CLOUD_URL to be set")
        if not api_key:
            raise RuntimeError("LOCI_MCP_MODE=cloud requires LOCI_API_KEY to be set")

    return ServerConfig(
        mode=mode,
        vector_size=_env_int("LOCI_VECTOR_SIZE", 512),
        epoch_size_ms=_env_int("LOCI_EPOCH_SIZE_MS", 5000),
        distance=distance,
        default_scene_id=os.environ.get("LOCI_SCENE_ID", "default") or "default",
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        cloud_url=cloud_url,
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Client lifecycle (lazy singleton with a reset hook for tests)
# ---------------------------------------------------------------------------

_client: LociClient | LocalLociClient | None = None
_config: ServerConfig | None = None


def _make_client(config: ServerConfig) -> LociClient | LocalLociClient:
    common: dict[str, Any] = {
        "vector_size": config.vector_size,
        "epoch_size_ms": config.epoch_size_ms,
        "distance": config.distance,
    }
    if config.mode == "local":
        return LocalLociClient(**common)
    if config.mode == "qdrant":
        return LociClient(qdrant_url=config.qdrant_url, api_key=config.api_key, **common)
    return LociClient(base_url=config.cloud_url, api_key=config.api_key, **common)


def get_config() -> ServerConfig:
    """Return the active configuration, loading it from the environment once."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_client() -> LociClient | LocalLociClient:
    """Return the lazily constructed Loci client for the configured mode."""
    global _client
    if _client is None:
        _client = _make_client(get_config())
    return _client


def reset_client() -> None:
    """Drop the cached client and config so the next call re-reads the env.

    Test hook: call between tests after changing ``LOCI_*`` env vars.
    """
    global _client, _config
    if _client is not None:
        close = getattr(_client, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()
    _client = None
    _config = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)


def _error(message: str) -> dict[str, Any]:
    return {"error": message}


def _friendly(exc: Exception) -> dict[str, Any]:
    """Turn an unexpected exception into a friendly error payload (no traceback)."""
    if isinstance(exc, ValueError):
        return _error(str(exc))
    return _error(f"{type(exc).__name__}: {exc}")


def _check_vector(vector: list[float]) -> str | None:
    """Return an error message if the vector has the wrong dimensionality."""
    expected = get_config().vector_size
    if len(vector) != expected:
        return (
            f"vector has dimension {len(vector)}, but this memory is configured for "
            f"{expected}-dimensional embeddings (LOCI_VECTOR_SIZE={expected})."
        )
    return None


def _check_position(
    x: float | None, y: float | None, z: float | None
) -> tuple[float, float, float] | None | str:
    """Validate an optional (x, y, z) triple.

    Returns the triple when fully given, None when fully omitted, and an
    error message string when partially given or out of range.
    """
    given = [c for c in (x, y, z) if c is not None]
    if not given:
        return None
    if len(given) != 3:
        return "provide all three of x, y, z for a spatial query, or none of them."
    assert x is not None and y is not None and z is not None  # for the type checker
    for name, value in (("x", x), ("y", y), ("z", z)):
        if not 0.0 <= value <= 1.0:
            return f"{name} must be in [0, 1], got {value}."
    return (x, y, z)


def _bounds_around(x: float, y: float, z: float, radius: float) -> dict[str, float]:
    return {
        "x_min": max(0.0, x - radius),
        "x_max": min(1.0, x + radius),
        "y_min": max(0.0, y - radius),
        "y_max": min(1.0, y + radius),
        "z_min": max(0.0, z - radius),
        "z_max": min(1.0, z + radius),
    }


def _cosine(a: list[float], b: list[float]) -> float:
    """Clamped-to-[0, 1] cosine similarity; 0.0 for empty/mismatched vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(p * q for p, q in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(p * p for p in a))
    norm_b = math.sqrt(sum(q * q for q in b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _state_dict(state: WorldState, score: float | None) -> dict[str, Any]:
    return {
        "id": state.id,
        "x": state.x,
        "y": state.y,
        "z": state.z,
        "timestamp_ms": state.timestamp_ms,
        "scene_id": state.scene_id,
        "score": score,
        "metadata": state.metadata,
    }


def _time_window(
    time_start_ms: int | None, time_end_ms: int | None
) -> tuple[int, int] | None | str:
    """Build a (start, end) window; missing start=0, missing end=now."""
    if time_start_ms is None and time_end_ms is None:
        return None
    start = 0 if time_start_ms is None else time_start_ms
    end = _now_ms() if time_end_ms is None else time_end_ms
    if start < 0:
        return f"time_start_ms must be non-negative, got {start}."
    if end < start:
        return f"time window is empty: time_end_ms ({end}) < time_start_ms ({start})."
    return (start, end)


# ---------------------------------------------------------------------------
# Tools (plain functions — registered onto FastMCP in build_server)
# ---------------------------------------------------------------------------


def remember(
    vector: list[float],
    x: float,
    y: float,
    z: float,
    timestamp_ms: int | None = None,
    scene_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store an observation in spatial memory.

    Saves an embedding vector together with where (x, y, z) and when
    (timestamp_ms) it was observed, so it can later be recalled by
    similarity, by place, or by time.

    Args:
        vector: The embedding of the observation (e.g. from a vision or text
            encoder). Must match the memory's configured dimensionality.
        x: Normalised position in [0, 1] (e.g. position in the environment).
        y: Normalised position in [0, 1].
        z: Normalised position in [0, 1].
        timestamp_ms: Unix time in milliseconds. Defaults to the current time.
        scene_id: Scene/episode name grouping related observations. Defaults
            to the server's configured default scene. Observations sharing a
            scene_id form a trajectory that can be walked with `trajectory`.
        metadata: Optional JSON-serialisable notes stored verbatim and
            returned on recall (e.g. {"label": "red door", "source": "cam0"}).

    Returns:
        {"id": <state id>, "epoch": <temporal shard index>} on success, or
        {"error": <message>} on invalid input (never a traceback).
    """
    try:
        dim_error = _check_vector(vector)
        if dim_error is not None:
            return _error(dim_error)
        config = get_config()
        ts = _now_ms() if timestamp_ms is None else timestamp_ms
        state = WorldState(
            x=x,
            y=y,
            z=z,
            timestamp_ms=ts,
            vector=vector,
            scene_id=scene_id if scene_id is not None else config.default_scene_id,
            metadata=metadata or {},
        )
        state_id = get_client().insert(state)
        return {"id": state_id, "epoch": epoch_id(ts, config.epoch_size_ms)}
    except Exception as exc:  # never leak a traceback to the model
        return _friendly(exc)


def recall(
    vector: list[float] | None = None,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    radius: float = 0.1,
    time_start_ms: int | None = None,
    time_end_ms: int | None = None,
    scene_id: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Recall stored observations by similarity, place, time, or any combination.

    Modes (combine freely):
    - Similarity: pass `vector` to rank memories by embedding similarity.
    - Place: pass all of `x`, `y`, `z` (plus optional `radius`, default 0.1)
      to restrict to memories within that box around the position (bounds
      clamped to [0, 1]).
    - Time: pass `time_start_ms` and/or `time_end_ms` (missing start means 0,
      missing end means now — future-dated memories need an explicit end).

    At least one of the three must be given. Without `vector`, matches are
    found by the spatial/time filters and ordered by recency (newest first)
    with `score` null; when more matches exist than an internal over-fetch
    can see, the returned subset may be arbitrary — pass a vector for fully
    ranked recall. With `vector`, results are ordered by similarity and
    `score` is the backend similarity (higher is better).

    `scene_id` filters to one scene. In local/qdrant mode it is applied in
    the database; in cloud mode it is applied client-side after an
    over-fetch, so distant matches beyond the over-fetch window may be missed.

    Returns:
        A list of {id, x, y, z, timestamp_ms, scene_id, score, metadata},
        or {"error": <message>} on invalid input.
    """
    try:
        position = _check_position(x, y, z)
        if isinstance(position, str):
            return _error(position)
        window = _time_window(time_start_ms, time_end_ms)
        if isinstance(window, str):
            return _error(window)
        if vector is None and position is None and window is None:
            return _error(
                "provide at least one of: a query vector, a position (x, y, z), "
                "or a time window (time_start_ms / time_end_ms)."
            )
        if vector is not None:
            dim_error = _check_vector(vector)
            if dim_error is not None:
                return _error(dim_error)
        if limit <= 0:
            return _error(f"limit must be positive, got {limit}.")
        if radius <= 0:
            return _error(f"radius must be positive, got {radius}.")

        config = get_config()
        client = get_client()
        bounds = _bounds_around(*position, radius) if position is not None else None

        # Without a query vector the ANN search still needs one; use an
        # arbitrary constant vector, over-fetch, and re-rank by recency.
        by_similarity = vector is not None
        query_vector = vector if vector is not None else [1.0] * config.vector_size
        fetch_limit = limit if by_similarity else max(limit * _RECALL_OVERFETCH, 50)

        scored: list[ScoredWorldState]
        if config.mode == "cloud":
            # Cloud mode: no scored queries and no payload filters — fetch
            # plain results, score client-side, post-filter by scene.
            fetch_limit = max(fetch_limit, limit * _RECALL_OVERFETCH)
            states = client.query(
                vector=query_vector,
                spatial_bounds=bounds,
                time_window_ms=window,
                limit=fetch_limit,
            )
            if scene_id is not None:
                states = [s for s in states if s.scene_id == scene_id]
            scored = [
                ScoredWorldState(state=s, score=_cosine(query_vector, s.vector), decayed_score=0.0)
                for s in states
            ]
        else:
            scene_filter = {"scene_id": scene_id} if scene_id is not None else None
            scored = client.query_scored(
                vector=query_vector,
                spatial_bounds=bounds,
                time_window_ms=window,
                limit=fetch_limit,
                _extra_payload_filter=scene_filter,
            )

        if by_similarity:
            return [_state_dict(item.state, float(item.score)) for item in scored[:limit]]
        ordered = sorted(scored, key=lambda item: item.state.timestamp_ms, reverse=True)
        return [_state_dict(item.state, None) for item in ordered[:limit]]
    except Exception as exc:  # never leak a traceback to the model
        return _friendly(exc)


def novelty(
    vector: list[float],
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
) -> dict[str, Any]:
    """Score how novel an observation is compared to everything remembered.

    Answers "have I seen something like this before?". Returns a novelty
    score in [0, 1]: ~0.0 means an essentially exact match exists in memory,
    ~1.0 means nothing similar has ever been remembered (an empty memory is
    maximally novel). Use it to decide whether to explore carefully (high
    novelty) or rely on prior experience (low novelty).

    Optionally pass all of `x`, `y`, `z` to restrict the comparison to
    memories near that position (radius 0.3), i.e. "is this novel *here*?".

    Args:
        vector: The embedding of the current observation.
        x: Optional normalised position in [0, 1] (give all three or none).
        y: Optional normalised position in [0, 1].
        z: Optional normalised position in [0, 1].

    Returns:
        {"novelty": <0..1>, "best_cosine": <0..1 similarity of the closest
        memory>, "nearest": [up to 3 of {id, x, y, z, timestamp_ms, scene_id,
        score (cosine to the query), metadata}]}, or {"error": <message>}.
    """
    try:
        dim_error = _check_vector(vector)
        if dim_error is not None:
            return _error(dim_error)
        position = _check_position(x, y, z)
        if isinstance(position, str):
            return _error(position)

        config = get_config()
        client = get_client()
        nearest_states: list[WorldState]
        if config.mode == "cloud":
            # Cloud mode cannot run the scored predict-and-retrieve path;
            # compute the same absolute novelty (1 - best cosine) client-side,
            # honouring the optional position restriction (same 0.3 radius).
            bounds = _bounds_around(*position, 0.3) if position is not None else None
            nearest_states = client.query(vector=vector, spatial_bounds=bounds, limit=3)
            best = max((_cosine(vector, s.vector) for s in nearest_states), default=0.0)
            novelty_score = 1.0 - best
        else:
            # Identity predictor: score the observation itself through the
            # calibrated absolute-novelty path (1 - best clamped cosine).
            result = client.predict_and_retrieve(
                context_vector=vector,
                predictor_fn=lambda v: v,
                limit=3,
                current_position=position,
                return_prediction=True,
            )
            assert isinstance(result, PredictRetrieveResult)  # return_prediction=True
            nearest_states = result.results
            novelty_score = result.prediction_novelty
            best = max((_cosine(vector, s.vector) for s in nearest_states), default=0.0)

        return {
            "novelty": novelty_score,
            "best_cosine": best,
            "nearest": [_state_dict(s, _cosine(vector, s.vector)) for s in nearest_states[:3]],
        }
    except Exception as exc:  # never leak a traceback to the model
        return _friendly(exc)


def trajectory(
    state_id: str,
    steps_back: int = 20,
    steps_forward: int = 20,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Walk the trajectory around a remembered observation.

    Given the id of a remembered state (as returned by `remember` or
    `recall`), returns the time-ordered sequence of states in the same scene:
    up to `steps_back` states before it, the state itself, and up to
    `steps_forward` states after it. Use it to replay "what happened around
    the time I saw this".

    Returns:
        A chronologically ordered list of {id, x, y, z, timestamp_ms}, or
        {"error": <message>} if the id is unknown.
    """
    try:
        if steps_back < 0 or steps_forward < 0:
            return _error("steps_back and steps_forward must be non-negative.")
        states = get_client().get_trajectory(
            state_id, steps_back=steps_back, steps_forward=steps_forward
        )
        if not states:
            return _error(f"no remembered state found with id {state_id!r}.")
        return [
            {
                "id": s.id,
                "x": s.x,
                "y": s.y,
                "z": s.z,
                "timestamp_ms": s.timestamp_ms,
            }
            for s in states
        ]
    except Exception as exc:  # never leak a traceback to the model
        return _friendly(exc)


def memory_stats() -> dict[str, Any]:
    """Describe this spatial memory: backend mode, dimensionality, and size.

    Returns:
        {"mode": "local"|"qdrant"|"cloud", "vector_size": int,
        "distance": str, "epoch_size_ms": int, "default_scene_id": str,
        "total_states": int or "unknown", "oldest_timestamp_ms": int or null,
        "newest_timestamp_ms": int or null}.
        `total_states` is exact in local mode and "unknown" for backends
        where counting is not cheap; the timestamp bounds span raw and
        consolidated memories and are null when the memory is empty or the
        backend cannot report them cheaply. Note: local mode is in-memory
        and per-process — memories vanish when the server exits.
    """
    try:
        config = get_config()
        client = get_client()

        total_states: int | str = "unknown"
        oldest_ms: int | None = None
        newest_ms: int | None = None
        store = getattr(client, "_store", None)
        if store is not None:
            with contextlib.suppress(Exception):
                total_states = int(store.total_points)  # property on MemoryStore
            with contextlib.suppress(Exception):
                # Bounded layout: one raw data collection plus one summary
                # collection; min/max timestamp across both is the span.
                for attr in ("_data_collection", "_summary_collection"):
                    collection = getattr(client, attr, None)
                    if collection is None:
                        continue
                    value_range = store.payload_value_range(collection, "timestamp_ms")
                    if value_range is None:
                        continue
                    lo, hi = int(value_range[0]), int(value_range[1])
                    oldest_ms = lo if oldest_ms is None else min(oldest_ms, lo)
                    newest_ms = hi if newest_ms is None else max(newest_ms, hi)

        return {
            "mode": config.mode,
            "vector_size": config.vector_size,
            "distance": config.distance,
            "epoch_size_ms": config.epoch_size_ms,
            "default_scene_id": config.default_scene_id,
            "total_states": total_states,
            "oldest_timestamp_ms": oldest_ms,
            "newest_timestamp_ms": newest_ms,
        }
    except Exception as exc:  # never leak a traceback to the model
        return _friendly(exc)


# NOTE: no `forget` tool is shipped. Neither LociClient nor LocalLociClient
# exposes a public targeted-deletion API (retention purging is an internal,
# policy-driven, whole-epoch mechanism configured at construction time), and
# the contract for this server is to omit the tool rather than ship a stub.


# ---------------------------------------------------------------------------
# Server assembly
# ---------------------------------------------------------------------------

_TOOLS = (remember, recall, novelty, trajectory, memory_stats)


def build_server() -> FastMCP:
    """Create the FastMCP server and register all Loci memory tools."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc

    server = FastMCP(
        "loci-memory",
        instructions=(
            "Spatial memory backed by Loci, a 4D spatiotemporal vector database. "
            "Use `remember` to store embeddings with a position (x, y, z in [0, 1]) "
            "and time, `recall` to retrieve them by similarity/place/time, `novelty` "
            "to check how unfamiliar an observation is, and `trajectory` to replay a "
            "scene around a remembered state."
        ),
    )
    for fn in _TOOLS:
        server.tool()(fn)
    return server


def main() -> None:
    """Entry point for the ``loci-mcp`` console script (stdio transport)."""
    parser = argparse.ArgumentParser(
        prog="loci-mcp",
        description=(
            "Run the Loci MCP server (spatial memory for agents) on stdio. "
            "Configure via LOCI_MCP_MODE, QDRANT_URL, LOCI_CLOUD_URL, LOCI_API_KEY, "
            "LOCI_VECTOR_SIZE, LOCI_EPOCH_SIZE_MS, LOCI_DISTANCE, LOCI_SCENE_ID. "
            "See docs/MCP_SERVER.md."
        ),
    )
    parser.parse_args()

    try:
        load_config()  # fail fast on bad configuration, before stdio starts
    except RuntimeError as exc:
        print(f"loci-mcp: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
