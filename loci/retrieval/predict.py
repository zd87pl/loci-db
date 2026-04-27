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

if TYPE_CHECKING:
    from loci.schema import ScoredWorldState, WorldState


@dataclass
class PredictRetrieveResult:
    """Result of a predict-then-retrieve operation.

    Attributes:
        results: WorldStates ranked by combined score.
        prediction_novelty: 0.0 = well-known situation (strong historical match),
            1.0 = novel situation (no historical analog found).
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
) -> tuple[list[WorldState], float]:
    """Re-rank scored retrieval candidates for predict-and-retrieve."""
    if not candidates:
        return [], 1.0

    vector_scores = _normalize_prediction_scores([candidate.score for candidate in candidates])
    mid_ms = now_ms + future_horizon_ms // 2
    combined: list[tuple[float, float, WorldState]] = []
    for candidate, vector_sim in zip(candidates, vector_scores, strict=False):
        if future_horizon_ms > 0:
            t_dist = abs(candidate.state.timestamp_ms - mid_ms)
            temporal_prox = max(0.0, 1.0 - t_dist / (future_horizon_ms / 2))
        else:
            temporal_prox = 1.0

        score = alpha * vector_sim + (1.0 - alpha) * temporal_prox
        combined.append((score, candidate.decayed_score, candidate.state))

    combined.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score = combined[0][0]
    results = [state for _, _, state in combined[:limit]]
    prediction_novelty = max(0.0, min(1.0, 1.0 - best_score))
    return results, prediction_novelty


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


class PredictThenRetrieve:
    """The core novelty: use a world model to predict future state,
    then retrieve historical states matching that prediction.

    This turns LOCI into a novelty detector for physical agents:
    - novelty ~ 0.0 → "I've seen this before" → use retrieved experience
    - novelty ~ 1.0 → "This is new territory" → alert, proceed carefully

    When a :class:`~loci.retrieval.novelty.NoveltyCalibrator` is supplied,
    novelty scores are calibrated against a running historical distribution
    rather than using a raw heuristic.
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
    ) -> PredictRetrieveResult:
        """Run the predict-then-retrieve pipeline.

        Pipeline:
        1. Call predictor_fn(context_vector) → predicted_vector (timed)
        2. Query store with predicted_vector, filtered by time window
           and spatial bounds (if current_position provided)
        3. Score results: alpha * vector_sim + (1-alpha) * temporal_proximity
        4. Compute prediction_novelty from best match score

        Args:
            context_vector: Current-state embedding vector.
            predictor_fn: Maps embedding → predicted future embedding.
                Called exactly once.
            future_horizon_ms: How far into the future to search (ms).
            current_position: Optional (x, y, z) for spatial filtering.
            current_timestamp_ms: Current time in ms (defaults to now).
            spatial_search_radius: Radius around current_position to search.
            limit: Maximum number of results.
            alpha: Weight for vector similarity vs temporal proximity.
                0.7 = 70% vector similarity, 30% temporal proximity.
            return_prediction: Whether to include predicted_vector in result.

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

        # Step 2: Build query parameters
        time_window = (now_ms, now_ms + future_horizon_ms)

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

        # Step 3: Retrieve
        t1 = time.perf_counter()
        query_scored = getattr(self._client, "query_scored", None)
        raw_candidates: list[ScoredWorldState] = []
        if callable(query_scored):
            scored_response = query_scored(
                vector=predicted_vector,
                spatial_bounds=spatial_bounds,
                time_window_ms=time_window,
                limit=limit * 2,  # over-fetch for re-ranking
            )
            if isinstance(scored_response, list):
                raw_candidates = scored_response
        retrieval_latency_ms = (time.perf_counter() - t1) * 1000

        # Step 4: Combined scoring
        best_score = 0.0
        if raw_candidates:
            results, best_score = rerank_prediction_candidates(
                raw_candidates,
                now_ms=now_ms,
                future_horizon_ms=future_horizon_ms,
                alpha=alpha,
                limit=limit,
            )
        else:
            raw_results = self._client.query(
                vector=predicted_vector,
                spatial_bounds=spatial_bounds,
                time_window_ms=time_window,
                limit=limit * 2,
            )
            if raw_results:
                mid_ms = now_ms + future_horizon_ms // 2
                scored = []
                for i, ws in enumerate(raw_results):
                    vector_sim = max(0.0, 1.0 - i / max(len(raw_results), 1))
                    if future_horizon_ms > 0:
                        t_dist = abs(ws.timestamp_ms - mid_ms)
                        temporal_prox = max(0.0, 1.0 - t_dist / (future_horizon_ms / 2))
                    else:
                        temporal_prox = 1.0

                    combined = alpha * vector_sim + (1.0 - alpha) * temporal_prox
                    scored.append((combined, ws))

                scored.sort(key=lambda x: x[0], reverse=True)
                results = [ws for _, ws in scored[:limit]]
                best_score = scored[0][0] if scored else 0.0
            else:
                results = []
                best_score = 0.0

        # Calibrate novelty if a calibrator is attached
        if self._calibrator is not None:
            self._calibrator.observe(best_score)
            prediction_novelty = self._calibrator.calibrated_novelty(best_score)
            novelty_samples = len(self._calibrator)
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
) -> list[WorldState]:
    """Run the predict-then-retrieve primitive (legacy API).

    For the full-featured API with novelty scoring, use
    :class:`PredictThenRetrieve` directly.
    """
    predicted_vector = predictor_fn(context_vector)
    now_ms = int(time.time() * 1000)
    results: list[WorldState] = client.query(
        vector=predicted_vector,
        time_window_ms=(now_ms, now_ms + future_horizon_ms),
        limit=limit,
    )
    return results
