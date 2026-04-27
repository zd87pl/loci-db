"""Novelty score calibration for predict-then-retrieve.

The raw heuristic ``1.0 - best_score`` is sensitive to the predictor function
and the embedding space. This module provides a *calibrated* novelty score
based on a running historical distribution of match quality.
"""

from __future__ import annotations

import statistics
from collections import deque


class NoveltyCalibrator:
    """Calibrate novelty scores against a running historical distribution.

    As the agent operates, the calibrator collects best-match scores from
    each ``predict_and_retrieve`` call. Novelty is reported as a z-score
    relative to that history, making it robust across different predictor
    functions and embedding spaces.

    Example:
        >>> calibrator = NoveltyCalibrator(window_size=100)
        >>> ptr = PredictThenRetrieve(client, calibrator=calibrator)
        >>> for _ in range(20):
        ...     result = ptr.retrieve(...)
        ...     print(result.prediction_novelty)  # 0.0 = familiar, 1.0 = novel
    """

    def __init__(self, window_size: int = 200, min_samples: int = 10) -> None:
        self._window: deque[float] = deque(maxlen=window_size)
        self._min_samples = min(min_samples, window_size)

    def observe(self, best_score: float) -> None:
        """Record a new best-match score (0.0 → poor match, 1.0 → perfect match)."""
        self._window.append(float(best_score))

    def calibrated_novelty(self, best_score: float) -> float:
        """Return a calibrated novelty score in [0.0, 1.0].

        Before ``min_samples`` observations are collected, falls back to the
        raw heuristic ``1.0 - best_score``.
        """
        if len(self._window) < self._min_samples:
            return max(0.0, min(1.0, 1.0 - best_score))

        mu = statistics.mean(self._window)
        try:
            sigma = statistics.stdev(self._window)
        except statistics.StatisticsError:
            sigma = 0.0

        if sigma == 0.0:
            return max(0.0, min(1.0, 1.0 - best_score))

        z = (best_score - mu) / sigma
        # Negative z = worse than average match → high novelty
        novelty = max(0.0, min(1.0, -z * 0.5 + 0.5))
        return novelty

    def __len__(self) -> int:
        return len(self._window)

    @property
    def ready(self) -> bool:
        """True when enough samples have been collected for calibration."""
        return len(self._window) >= self._min_samples

    def stats(self) -> dict:
        """Return current distribution statistics."""
        if not self._window:
            return {"count": 0, "mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(self._window),
            "mean": statistics.mean(self._window),
            "stdev": statistics.stdev(self._window) if len(self._window) > 1 else 0.0,
            "min": min(self._window),
            "max": max(self._window),
        }
