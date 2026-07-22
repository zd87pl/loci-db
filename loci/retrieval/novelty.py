"""Novelty score calibration for predict-then-retrieve.

The raw heuristic ``1.0 - best_score`` is sensitive to the predictor function
and the embedding space. This module provides two calibrators:

- :class:`NoveltyCalibrator` — a smooth z-score squash against a running
  historical distribution of match quality (heuristic, continuous).
- :class:`ConformalNoveltyCalibrator` — a conformal alarm with a
  distribution-free guaranteed false-alarm rate (RFC-0001 R2).
"""

from __future__ import annotations

import math
import statistics
from collections import deque

# Floor for the window standard deviation. Near-constant score windows would
# otherwise turn sub-percent fluctuations into full-scale novelty swings.
_STDEV_FLOOR = 0.05


class NoveltyCalibrator:
    """Calibrate novelty scores against a running historical distribution.

    As the agent operates, the calibrator collects best-match scores from
    each ``predict_and_retrieve`` call. Novelty is reported as a smooth
    logistic squash of the z-score relative to that history, making it robust
    across different predictor functions and embedding spaces.

    Mapping: ``novelty = 1 / (1 + exp(z))`` with
    ``z = (best_score - mean) / max(stdev, 0.05)``. An average match scores
    0.5; +/-2 sigma maps to roughly 0.12 / 0.88; the curve approaches 0 and 1
    asymptotically with no hard saturation.

    Score a sample with :meth:`calibrated_novelty` *before* recording it with
    :meth:`observe`, so it is judged against history that excludes it.

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

        Uses the logistic mapping described in the class docstring: 0.5 for an
        average match, low novelty for better-than-average matches, high for
        worse. Call before :meth:`observe` for the same sample.

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
        sigma = max(sigma, _STDEV_FLOOR)

        z = (best_score - mu) / sigma
        # Better-than-average match (z > 0) → low novelty; worse → high.
        bounded = max(-60.0, min(60.0, z))
        return 1.0 / (1.0 + math.exp(bounded))

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


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class ConformalNoveltyCalibrator:
    """Conformal novelty alarm with a guaranteed false-alarm rate.

    Turns the novelty heuristic into a *rate*: :meth:`is_novel` fires on at
    most ~``alpha`` of in-distribution observations, distribution-free, with
    no tuning of a raw threshold.

    Method (inductive/online conformal prediction; Vovk, Gammerman & Shafer,
    *Algorithmic Learning in a Random World*, Springer 2005; Papadopoulos et
    al. 2002 split/inductive conformal): each best-match similarity ``s`` is
    mapped to a nonconformity score ``a = 1 - clamp(s, 0, 1)`` (for LOCI, a
    LOW best-match similarity means MORE nonconforming). A sliding window of
    the most recent nonconformity scores serves as the calibration set. For a
    new observation with nonconformity ``a*`` against a window of size ``n``,
    the conformal p-value is::

        p = (1 + #{a_i in window : a_i >= a*}) / (n + 1)

    and the alarm fires when ``p <= alpha``.

    Guarantee: if the current observation is exchangeable with the window
    contents (e.g. an i.i.d. in-distribution stream), the p-value is
    super-uniform, so ``P(false alarm) = P(p <= alpha) <= alpha`` — a
    finite-sample, distribution-free bound valid at any window size.

    Honest caveats: the guarantee rests on exchangeability. A drifting score
    distribution breaks it — the sliding window *adapts* to slow drift (old
    regimes are evicted, so the alarm recovers) but at the cost of exact
    finite-sample validity during the transition. Alarms are also correlated
    across time (the window is shared between nearby observations); the
    ``alpha`` bound is on the marginal false-alarm rate, not on independence
    of alarms.

    Duck-typed interface: :meth:`observe`, :meth:`calibrated_novelty` and
    ``__len__`` match what ``PredictThenRetrieve`` expects, so an instance
    slots into ``predict_and_retrieve(calibrator=...)`` unchanged. Score with
    :meth:`is_novel` / :meth:`p_value` / :meth:`calibrated_novelty` *before*
    recording with :meth:`observe` (the pipeline already does this), which is
    the correct online-conformal discipline: a sample must be judged against
    history that excludes it.

    Not thread-safe (same policy as :class:`NoveltyCalibrator`): confine an
    instance to one pipeline/thread.

    Example:
        >>> calibrator = ConformalNoveltyCalibrator(alpha=0.05)
        >>> result = client.predict_and_retrieve(..., calibrator=calibrator)
        >>> if calibrator.warmed_up and result.prediction_novelty >= 1 - 0.05:
        ...     print("novel: fires on <= 5% of in-distribution observations")
    """

    def __init__(self, alpha: float = 0.05, window: int = 512, min_samples: int = 30) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if min_samples < 2:
            raise ValueError(f"min_samples must be >= 2, got {min_samples}")
        if window < min_samples:
            raise ValueError(f"window ({window}) must be >= min_samples ({min_samples})")
        self._alpha = float(alpha)
        self._min_samples = int(min_samples)
        # Stored values are nonconformity scores (1 - similarity), not raw scores.
        self._window: deque[float] = deque(maxlen=int(window))

    @property
    def alpha(self) -> float:
        """Configured marginal false-alarm rate bound."""
        return self._alpha

    @property
    def warmed_up(self) -> bool:
        """True once the window holds at least ``min_samples`` observations."""
        return len(self._window) >= self._min_samples

    def observe(self, best_score: float) -> None:
        """Record a best-match score (0.0 → poor match, 1.0 → perfect match).

        Appends its nonconformity to the sliding calibration window, evicting
        the oldest entry once the window is full.
        """
        self._window.append(1.0 - _clamp_unit(best_score))

    def p_value(self, best_score: float) -> float:
        """Conformal p-value of *best_score* against the calibration window.

        Small p ⇒ the current match is worse (more nonconforming) than almost
        all recent history ⇒ novel. Under exchangeability the p-value is
        super-uniform: ``P(p <= alpha) <= alpha`` for any alpha, any window
        size. Ties count against novelty (the ``>=`` comparison), keeping the
        p-value conservative. With an empty window the p-value is 1.0.
        """
        nonconformity = 1.0 - _clamp_unit(best_score)
        exceed = sum(1 for stored in self._window if stored >= nonconformity)
        return (1 + exceed) / (len(self._window) + 1)

    def is_novel(self, best_score: float) -> bool:
        """The conformal alarm: True when ``p_value(best_score) <= alpha``.

        For in-distribution (exchangeable) data this fires with probability
        at most ``alpha`` — a guaranteed false-alarm rate. Valid at any
        window occupancy; with fewer than ``ceil(1/alpha) - 1`` stored
        samples the p-value cannot reach ``alpha``, so the alarm simply
        cannot fire (conservative during warm-up).
        """
        return self.p_value(best_score) <= self._alpha

    def calibrated_novelty(self, best_score: float) -> float:
        """Continuous novelty in [0, 1] for the pipeline interface.

        Returns ``1 - p_value(best_score)``: monotone (worse matches ⇒ higher
        novelty), and thresholding at ``1 - alpha`` reproduces
        :meth:`is_novel` exactly. Before ``min_samples`` observations are
        collected, falls back to the raw absolute novelty
        ``1 - clamp(best_score)`` — check :attr:`warmed_up` before treating
        the value as conformal.
        """
        if not self.warmed_up:
            return 1.0 - _clamp_unit(best_score)
        return 1.0 - self.p_value(best_score)

    def __len__(self) -> int:
        return len(self._window)

    def stats(self) -> dict:
        """Return current calibration-window statistics (nonconformity scale)."""
        occupancy = len(self._window)
        base = {
            "count": occupancy,
            "alpha": self._alpha,
            "warmed_up": self.warmed_up,
        }
        if not self._window:
            return {**base, "mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            **base,
            "mean": statistics.mean(self._window),
            "min": min(self._window),
            "max": max(self._window),
        }
