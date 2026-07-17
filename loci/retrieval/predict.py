"""Predict-then-retrieve pipeline.

The core novelty: use a world model to predict future state,
then retrieve historical states matching that prediction.

This is the "hippocampus" primitive — grounding model predictions
against empirical memory.

Analogous to HyDE (Hypothetical Document Embeddings, ACL 2023)
but for spatiotemporal world models.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loci.schema import ScoredWorldState

if TYPE_CHECKING:
    from loci.schema import WorldState


@dataclass
class PredictRetrieveResult:
    """Result of a predict-then-retrieve operation.

    Attributes:
        results: WorldStates ranked by combined score.
        prediction_novelty: ``1.0 - best_cosine`` where ``best_cosine`` is the
            highest cosine similarity (clamped to [0, 1]) between the predicted
            vector and any retrieved state's stored vector. 0.0 = an essentially
            exact historical match, 1.0 = no analog (orthogonal/opposed matches
            or an empty result set). Absolute and metric-independent.
        predicted_vector: The predicted future embedding (if return_prediction=True).
        retrieval_latency_ms: Time spent on the retrieval query.
        predictor_call_ms: Time spent calling the predictor function.
        novelty_samples: Number of historical observations used for calibration.
    """

    results: list[WorldState] = field(default_factory=list)
    prediction_novelty: float = 1.0
    predicted_vector: list[float] | None = None
    retrieval_latency_ms: float = 0.0
    predictor_call_ms: float = 0.0
    novelty_samples: int = 0


def rerank_prediction_candidates(
    candidates: list[ScoredWorldState],
    *,
    now_ms: int,
    future_horizon_ms: int,
    alpha: float,
    limit: int,
    use_temporal_proximity: bool = False,
    predicted_vector: list[float] | None = None,
    time_window_ms: tuple[int, int] | None = None,
) -> tuple[list[WorldState], float]:
    """Re-rank scored retrieval candidates and return the best match score.

    Ranking blends normalised backend scores with temporal proximity. When
    *use_temporal_proximity* is set, proximity is measured against the midpoint
    of *time_window_ms* (scaled by its half-width) when an explicit window is
    given, otherwise against ``now_ms + future_horizon_ms / 2``.

    The returned best score is the highest clamped cosine similarity between
    *predicted_vector* and any candidate's stored vector — an absolute match
    quality suited to novelty scoring, independent of the backend's score
    scale and of the ranking above. When *predicted_vector* is omitted, the
    best combined ranking score is returned instead (legacy behaviour).
    """
    if not candidates:
        return [], 0.0

    vector_scores = _normalize_prediction_scores([candidate.score for candidate in candidates])
    if time_window_ms is not None:
        start_ms, end_ms = time_window_ms
        center_ms = (start_ms + end_ms) / 2
        half_width_ms = (end_ms - start_ms) / 2
    else:
        center_ms = now_ms + future_horizon_ms / 2
        half_width_ms = future_horizon_ms / 2

    combined: list[tuple[float, float, WorldState]] = []
    for candidate, vector_sim in zip(candidates, vector_scores, strict=True):
        if use_temporal_proximity and half_width_ms > 0:
            t_dist = abs(candidate.state.timestamp_ms - center_ms)
            temporal_prox = max(0.0, 1.0 - t_dist / half_width_ms)
            score = alpha * vector_sim + (1.0 - alpha) * temporal_prox
        else:
            score = vector_sim

        combined.append((score, candidate.decayed_score, candidate.state))

    combined.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if predicted_vector is not None:
        best_score = max(
            _clamped_cosine(predicted_vector, candidate.state.vector) for candidate in candidates
        )
    else:
        best_score = combined[0][0]
    results = [state for _, _, state in combined[:limit]]
    return results, best_score


def _normalize_prediction_scores(scores: list[float]) -> list[float]:
    """Map retrieval scores to [0, 1] without relying on rank alone."""
    if not scores:
        return []

    baseline = [_sigmoid(score) for score in scores]
    lo = min(scores)
    hi = max(scores)
    if hi - lo <= 1e-9:
        return baseline

    spread = hi - lo
    return [
        max(0.0, min(1.0, 0.5 * baseline[idx] + 0.5 * ((score - lo) / spread)))
        for idx, score in enumerate(scores)
    ]


def _sigmoid(score: float) -> float:
    bounded = max(-60.0, min(60.0, score))
    return 1.0 / (1.0 + math.exp(-bounded))


def _cosine(a: Any, b: Any) -> float:
    """Cosine similarity; 0.0 for missing, empty, mismatched or zero-norm vectors."""
    if a is None or b is None or len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def _clamped_cosine(a: Any, b: Any) -> float:
    return max(0.0, min(1.0, _cosine(a, b)))


def _validate_predicted_vector(predicted: Any, expected_len: int) -> None:
    """Reject malformed predictor output before it can poison novelty scoring."""
    if isinstance(predicted, (str, bytes)) or not hasattr(predicted, "__len__"):
        raise ValueError(
            f"predictor_fn returned {type(predicted).__name__}, expected a sequence of floats"
        )
    if len(predicted) == 0:
        raise ValueError("predictor_fn returned an empty vector")
    if len(predicted) != expected_len:
        raise ValueError(
            f"predictor_fn returned a vector of length {len(predicted)}, "
            f"expected {expected_len} to match context_vector"
        )
    for idx, value in enumerate(predicted):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"predictor_fn returned non-numeric value {value!r} at index {idx}"
            ) from None
        if not math.isfinite(numeric):
            raise ValueError(
                f"predictor_fn returned non-finite value {value!r} (NaN/inf) at index {idx}"
            )


class PredictThenRetrieve:
    """The core novelty: use a world model to predict future state,
    then retrieve historical states matching that prediction.

    This turns LOCI into a novelty detector for physical agents:
    - novelty ~ 0.0 → "I've seen this before" → use retrieved experience
    - novelty ~ 1.0 → "This is new territory" → alert, proceed carefully

    Novelty is absolute: ``1.0 - best cosine similarity`` between the
    prediction and any retrieved state's stored vector, so it is comparable
    across backends and distance metrics.

    When a :class:`~loci.retrieval.novelty.NoveltyCalibrator` is supplied,
    novelty scores are calibrated against a running historical distribution
    rather than using the raw cosine heuristic.
    """

    def __init__(self, client: Any, calibrator: Any = None) -> None:
        self._client = client
        self._calibrator = calibrator

    def retrieve(
        self,
        context_vector: list[float],
        predictor_fn: Callable[[list[float]], list[float]],
        future_horizon_ms: int,
        current_position: tuple[float, float, float] | None = None,
        current_timestamp_ms: int | None = None,
        spatial_search_radius: float = 0.3,
        limit: int = 10,
        alpha: float = 0.7,
        return_prediction: bool = False,
        search_time_window_ms: tuple[int, int] | None = None,
    ) -> PredictRetrieveResult:
        """Run the predict-then-retrieve pipeline.

        Pipeline:
        1. Call predictor_fn(context_vector) → predicted_vector (timed,
           validated: finite floats, same dimension as context_vector)
        2. Query store with predicted_vector, filtered by spatial bounds
           and by search_time_window_ms only when explicitly provided
        3. Score results: alpha * vector_sim + (1-alpha) * temporal_proximity
           when a time window is explicit (proximity centred on the window
           midpoint); otherwise rank by vector similarity
        4. Compute prediction_novelty as 1 - best cosine similarity between
           the prediction and any retrieved state's stored vector

        Args:
            context_vector: Current-state embedding vector.
            predictor_fn: Maps embedding → predicted future embedding.
                Called exactly once. Must return a finite float vector with
                the same dimension as ``context_vector`` (ValueError otherwise).
            future_horizon_ms: How far into the future to search (ms).
            current_position: Optional (x, y, z) for spatial filtering.
            current_timestamp_ms: Current time in ms (defaults to now).
            spatial_search_radius: Radius around current_position to search.
            limit: Maximum number of results.
            alpha: Weight for vector similarity vs temporal proximity.
                0.7 = 70% vector similarity, 30% temporal proximity.
            return_prediction: Whether to include predicted_vector in result.
            search_time_window_ms: Optional explicit timestamp range to search.
                By default, searches all stored history for analogs rather than
                assuming future-dated states already exist in the database.

        Returns:
            PredictRetrieveResult with ranked results and novelty score.
        """
        now_ms = (
            current_timestamp_ms if current_timestamp_ms is not None else int(time.time() * 1000)
        )

        # Step 1: Call predictor
        t0 = time.perf_counter()
        predicted_vector = predictor_fn(context_vector)
        predictor_call_ms = (time.perf_counter() - t0) * 1000
        _validate_predicted_vector(predicted_vector, len(context_vector))

        # Step 2: Build query parameters.  The default is historical analog
        # search across stored memories; callers can opt into a concrete
        # absolute-time search window when their data is future-scheduled.
        time_window = search_time_window_ms

        spatial_bounds = None
        if current_position is not None:
            x, y, z = current_position
            spatial_bounds = {
                "x_min": max(0.0, x - spatial_search_radius),
                "x_max": min(1.0, x + spatial_search_radius),
                "y_min": max(0.0, y - spatial_search_radius),
                "y_max": min(1.0, y + spatial_search_radius),
                "z_min": max(0.0, z - spatial_search_radius),
                "z_max": min(1.0, z + spatial_search_radius),
            }

        # Step 3: Retrieve — a single query, timed end to end.  Prefer the
        # scored API; fall back to plain query for duck-typed clients.
        t1 = time.perf_counter()
        raw_candidates: list[ScoredWorldState] | None = None
        query_scored = getattr(self._client, "query_scored", None)
        if callable(query_scored):
            scored_response = query_scored(
                vector=predicted_vector,
                spatial_bounds=spatial_bounds,
                time_window_ms=time_window,
                limit=limit * 2,  # over-fetch for re-ranking
            )
            if isinstance(scored_response, list):
                raw_candidates = scored_response
        if raw_candidates is None:
            raw_results = (
                self._client.query(
                    vector=predicted_vector,
                    spatial_bounds=spatial_bounds,
                    time_window_ms=time_window,
                    limit=limit * 2,
                )
                or []
            )
            # No backend scores here: score by actual cosine similarity to
            # the prediction, never by rank position.
            raw_candidates = []
            for ws in raw_results:
                sim = _clamped_cosine(predicted_vector, ws.vector)
                raw_candidates.append(ScoredWorldState(state=ws, score=sim, decayed_score=sim))
        retrieval_latency_ms = (time.perf_counter() - t1) * 1000

        # Step 4: Combined ranking + absolute novelty
        results, best_score = rerank_prediction_candidates(
            raw_candidates,
            now_ms=now_ms,
            future_horizon_ms=future_horizon_ms,
            alpha=alpha,
            limit=limit,
            use_temporal_proximity=search_time_window_ms is not None,
            predicted_vector=predicted_vector,
            time_window_ms=search_time_window_ms,
        )

        # Calibrate novelty if a calibrator is attached.  Score first so the
        # current sample is judged against history that excludes it.
        if self._calibrator is not None:
            prediction_novelty = self._calibrator.calibrated_novelty(best_score)
            novelty_samples = len(self._calibrator)
            self._calibrator.observe(best_score)
        else:
            prediction_novelty = max(0.0, min(1.0, 1.0 - best_score))
            novelty_samples = 0

        return PredictRetrieveResult(
            results=results,
            prediction_novelty=prediction_novelty,
            predicted_vector=predicted_vector if return_prediction else None,
            retrieval_latency_ms=retrieval_latency_ms,
            predictor_call_ms=predictor_call_ms,
            novelty_samples=novelty_samples,
        )


# ---------------------------------------------------------------------------
# Backward-compatible module-level function
# ---------------------------------------------------------------------------


def predict_and_retrieve(
    client: Any,
    context_vector: list[float],
    predictor_fn: Callable[[list[float]], list[float]],
    future_horizon_ms: int = 1000,
    limit: int = 5,
    search_time_window_ms: tuple[int, int] | None = None,
) -> list[WorldState]:
    """Run the predict-then-retrieve primitive (legacy API).

    For the full-featured API with novelty scoring, use
    :class:`PredictThenRetrieve` directly.
    """
    predicted_vector = predictor_fn(context_vector)
    results: list[WorldState] = client.query(
        vector=predicted_vector,
        time_window_ms=search_time_window_ms,
        limit=limit,
    )
    return results
