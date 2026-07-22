"""The five RFC-0001 R3 tasks, each returning a plain metrics dict.

Every task takes a fresh (not yet set up) :class:`SystemUnderTest`, builds
its own SYNTHETIC dataset and its own brute-force oracle where needed, and
tears the system down before returning.  All randomness flows from the
task's ``seed`` through ``numpy.random.default_rng`` — no wall-clock values
ever enter a metric (data timestamps come from the datasets' fixed logical
clock; latency measurements use ``time.perf_counter`` because latency is a
wall-clock quantity by definition).

Tasks:

- :func:`future_analog_recall` — recall@k against the brute-force oracle for
  noisy "predicted future state" queries, with and without a time window.
- :func:`novelty_auc` — streaming ROC-AUC of ``predict_novelty`` over
  OOD-labelled segments (score-before-insert discipline).  Systems without
  a novelty API report ``auc: null``.
- :func:`trajectory_fidelity` — fraction of a scene's true ordered
  trajectory recovered around an anchor.
- :func:`recall_vs_age` — the flight-recorder curve: recall of progressively
  older time windows after a long consolidated stream, plus the resident
  point compression ratio.
- :func:`sustained_load` — p50/p95 insert and query latency under an
  interleaved insert/query stream (no sleeps).
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from .datasets import (
    OodPatrolDataset,
    SmoothPatrolDataset,
    TrajectoryPoint,
    WarehouseDataset,
)
from .systems import DEFAULT_EPOCH_SIZE_MS, BruteForceSystem, QueryHit, SystemUnderTest

TASK_NAMES = [
    "future_analog_recall",
    "novelty_auc",
    "trajectory_fidelity",
    "recall_vs_age",
    "sustained_load",
]


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """ROC-AUC via the rank-sum (Mann-Whitney) formulation, ties averaged.

    Pure numpy — no sklearn dependency.  Requires at least one positive and
    one negative label.
    """
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc requires both positive and negative labels")
    # Average ranks (1-based) with ties sharing their group's mean rank.
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    average_rank_per_value = (cumulative - counts + 1 + cumulative) / 2.0
    ranks = average_rank_per_value[inverse]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fresh_oracle(vector_dim: int, points: Sequence[TrajectoryPoint]) -> BruteForceSystem:
    oracle = BruteForceSystem()
    oracle.setup(vector_dim)
    oracle.insert_many(points)
    return oracle


def _recall(system_hits: list[QueryHit], oracle_hits: list[QueryHit], k: int) -> float:
    truth = {h.timestamp_ms for h in oracle_hits[:k]}
    if not truth:
        return 1.0
    got = {h.timestamp_ms for h in system_hits[:k]}
    return len(truth & got) / len(truth)


# ---------------------------------------------------------------------------
# (a) Future-state analog recall@k
# ---------------------------------------------------------------------------


def future_analog_recall(
    system: SystemUnderTest,
    *,
    n_points: int = 1500,
    vector_dim: int = 32,
    k: int = 10,
    n_queries: int = 60,
    horizon_steps: int = 12,
    query_noise: float = 0.05,
    window_steps: int = 40,
    seed: int = 7,
) -> dict:
    """Recall@k vs the brute-force oracle for noisy future-state queries.

    The query vector is the TRUE embedding ``horizon_steps`` ahead plus
    Gaussian noise — a stand-in for an imperfect world-model prediction.
    System and oracle receive the identical query, so recall isolates the
    retrieval layer.  Reported twice: unfiltered, and restricted to a time
    window centred on the future state.
    """
    ds = SmoothPatrolDataset(n_points=n_points, vector_dim=vector_dim, seed=seed)
    pts = ds.points()
    system.setup(vector_dim)
    system.insert_many(pts)
    oracle = _fresh_oracle(vector_dim, pts)

    rng = np.random.default_rng(seed + 1000)
    max_context = n_points - horizon_steps - 1
    context_indices = np.linspace(0, max_context, num=n_queries, dtype=int)

    half_window_ms = window_steps * ds.step_ms // 2
    recalls_full: list[float] = []
    recalls_windowed: list[float] = []
    for i in context_indices:
        future = pts[i + horizon_steps]
        query = np.asarray(future.embedding) + query_noise * rng.standard_normal(vector_dim)
        query_list = [float(v) for v in query]

        recalls_full.append(
            _recall(system.query(query_list, limit=k), oracle.query(query_list, limit=k), k)
        )

        window = (future.timestamp_ms - half_window_ms, future.timestamp_ms + half_window_ms)
        recalls_windowed.append(
            _recall(
                system.query(query_list, time_window_ms=window, limit=k),
                oracle.query(query_list, time_window_ms=window, limit=k),
                k,
            )
        )

    system.teardown()
    return {
        "task": "future_analog_recall",
        "dataset": ds.name,
        "synthetic": True,
        "n_points": n_points,
        "k": k,
        "n_queries": n_queries,
        "horizon_steps": horizon_steps,
        "recall_at_k": _round(float(np.mean(recalls_full))),
        "recall_at_k_windowed": _round(float(np.mean(recalls_windowed))),
        "oracle": "brute_force",
    }


# ---------------------------------------------------------------------------
# (b) Novelty AUC over OOD segments
# ---------------------------------------------------------------------------


def novelty_auc(
    system: SystemUnderTest,
    *,
    n_points: int = 800,
    vector_dim: int = 32,
    n_ood_segments: int = 3,
    ood_segment_len: int = 40,
    onset_len: int = 5,
    warmup: int = 64,
    seed: int = 13,
) -> dict:
    """Streaming novelty ROC-AUC over OOD-labelled synthetic segments.

    Discipline: each point is scored with ``predict_novelty`` BEFORE it is
    inserted, so a point is always judged against history that excludes it.
    The first ``warmup`` points are inserted but not scored.  Systems whose
    ``predict_novelty`` returns None are reported as unsupported
    (``auc: null``) — no proxy score is invented for them.

    Two AUCs are reported because online novelty has an honest wrinkle:
    once the first points of an excursion are inserted, the REST of the
    segment matches its own predecessors and legitimately scores familiar.

    - ``auc``: all OOD-labelled points are positives (the RFC metric).
    - ``auc_onset``: only the first ``onset_len`` points of each segment are
      positives; later in-segment points are excluded entirely — this is the
      "did the alarm fire when the excursion started" number.
    """
    ds = OodPatrolDataset(
        n_points=n_points,
        vector_dim=vector_dim,
        seed=seed,
        n_ood_segments=n_ood_segments,
        ood_segment_len=ood_segment_len,
    )
    system.setup(vector_dim)

    labels: list[bool] = []
    onset: list[bool] = []  # parallel to labels: True = one of a segment's first points
    scores: list[float] = []
    supported = True
    prev_ood = False
    ood_run = 0
    for i, point in enumerate(ds):
        ood_run = ood_run + 1 if (point.is_ood and prev_ood) else (1 if point.is_ood else 0)
        prev_ood = point.is_ood
        if supported and i >= warmup:
            score = system.predict_novelty(point.embedding)
            if score is None:
                supported = False
            else:
                labels.append(point.is_ood)
                onset.append(point.is_ood and ood_run <= onset_len)
                scores.append(float(score))
        system.insert(point)
    system.teardown()

    if not supported or not labels:
        return {
            "task": "novelty_auc",
            "dataset": ds.name,
            "synthetic": True,
            "supported": False,
            "auc": None,
            "auc_onset": None,
            "n_scored": 0,
            "n_ood": 0,
            "median_familiar": None,
            "median_ood": None,
        }

    y = np.asarray(labels, dtype=bool)
    y_onset = np.asarray(onset, dtype=bool)
    s = np.asarray(scores)
    # Onset AUC: drop non-onset OOD points so they count neither way.
    keep = ~y | y_onset
    return {
        "task": "novelty_auc",
        "dataset": ds.name,
        "synthetic": True,
        "supported": True,
        "auc": _round(roc_auc(labels, scores)),
        "auc_onset": _round(roc_auc(y_onset[keep].tolist(), s[keep].tolist())),
        "n_scored": len(labels),
        "n_ood": int(y.sum()),
        "median_familiar": _round(float(np.median(s[~y]))),
        "median_ood": _round(float(np.median(s[y]))),
    }


# ---------------------------------------------------------------------------
# (c) Trajectory reconstruction fidelity
# ---------------------------------------------------------------------------


def trajectory_fidelity(
    system: SystemUnderTest,
    *,
    n_scenes: int = 4,
    n_visits: int = 3,
    points_per_visit: int = 40,
    vector_dim: int = 32,
    steps: int = 15,
    seed: int = 11,
) -> dict:
    """Fraction of a scene's true ordered trajectory recovered around anchors.

    For each scene, the anchor is the midpoint of the scene's timeline; the
    true trajectory is the ordered window of ``steps`` scene points on each
    side.  Systems use whatever API they have (causal scene scan for LOCI,
    payload scroll for the naive baseline, array slice for brute force) —
    the ``method`` field reports which.
    """
    ds = WarehouseDataset(
        n_scenes=n_scenes,
        n_visits=n_visits,
        points_per_visit=points_per_visit,
        vector_dim=vector_dim,
        seed=seed,
    )
    pts = ds.points()
    system.setup(vector_dim)
    system.insert_many(pts)

    by_scene: dict[str, list[TrajectoryPoint]] = {}
    for p in pts:
        by_scene.setdefault(p.scene_id, []).append(p)

    coverages: list[float] = []
    order_scores: list[float] = []
    supported = True
    for scene_pts in by_scene.values():
        scene_pts.sort(key=lambda p: p.timestamp_ms)
        anchor_idx = len(scene_pts) // 2
        anchor = scene_pts[anchor_idx]
        true_window = scene_pts[max(0, anchor_idx - steps) : anchor_idx + steps + 1]
        true_ts = [p.timestamp_ms for p in true_window]

        got = system.get_trajectory(anchor.timestamp_ms, steps, steps)
        if got is None:
            supported = False
            break
        got_ts = [h.timestamp_ms for h in got if not h.is_summary]
        coverages.append(len(set(true_ts) & set(got_ts)) / len(true_ts))
        if len(got_ts) < 2:
            order_scores.append(1.0 if got_ts else 0.0)
        else:
            in_order = sum(1 for a, b in zip(got_ts, got_ts[1:], strict=False) if a < b)
            order_scores.append(in_order / (len(got_ts) - 1))
    system.teardown()

    if not supported:
        return {
            "task": "trajectory_fidelity",
            "dataset": ds.name,
            "synthetic": True,
            "supported": False,
            "coverage": None,
            "order_fidelity": None,
            "n_scenes": n_scenes,
            "steps": steps,
            "method": getattr(system, "trajectory_method", "unknown"),
        }
    return {
        "task": "trajectory_fidelity",
        "dataset": ds.name,
        "synthetic": True,
        "supported": True,
        "coverage": _round(float(np.mean(coverages))),
        "order_fidelity": _round(float(np.mean(order_scores))),
        "n_scenes": n_scenes,
        "steps": steps,
        "method": getattr(system, "trajectory_method", "unknown"),
    }


# ---------------------------------------------------------------------------
# (d) Recall vs age under consolidation — the flight-recorder curve
# ---------------------------------------------------------------------------


def recall_vs_age(
    system: SystemUnderTest,
    *,
    n_points: int = 3000,
    vector_dim: int = 32,
    k: int = 5,
    n_age_buckets: int = 6,
    queries_per_bucket: int = 8,
    epoch_size_ms: int = DEFAULT_EPOCH_SIZE_MS,
    query_noise: float = 0.02,
    seed: int = 7,
) -> dict:
    """Recall of progressively older windows after a long stream.

    The stream is inserted in epoch-sized chunks (so consolidation runs at
    its natural cadence, driven by the stream's own future-dated clock).
    Afterwards the timeline is split into ``n_age_buckets`` windows, oldest
    to newest, and each is probed with noisy embeddings of points that lived
    in it.  Two recall definitions are reported per bucket:

    - ``recall_strict_by_age``: only exact raw points count — shows what
      consolidation deletes.
    - ``recall_covered_by_age``: a consolidated summary whose source range
      covers a ground-truth point also counts — shows what stays findable
      as gist.

    ``compression_ratio`` = inserted points / resident points at the end
    (1.0 means the system kept everything).
    """
    ds = SmoothPatrolDataset(n_points=n_points, vector_dim=vector_dim, seed=seed)
    pts = ds.points()
    system.setup(vector_dim)

    # Insert chunked by logical epoch, in stream order.
    chunk: list[TrajectoryPoint] = []
    current_epoch = pts[0].timestamp_ms // epoch_size_ms
    for p in pts:
        ep = p.timestamp_ms // epoch_size_ms
        if ep != current_epoch:
            system.insert_many(chunk)
            chunk = []
            current_epoch = ep
        chunk.append(p)
    if chunk:
        system.insert_many(chunk)

    oracle = _fresh_oracle(vector_dim, pts)
    rng = np.random.default_rng(seed + 2000)

    t0 = pts[0].timestamp_ms
    t1 = pts[-1].timestamp_ms
    edges = np.linspace(t0, t1 + 1, n_age_buckets + 1)

    strict_by_age: list[float] = []
    covered_by_age: list[float] = []
    for b in range(n_age_buckets):
        lo, hi = int(edges[b]), int(edges[b + 1]) - 1
        bucket_pts = [p for p in pts if lo <= p.timestamp_ms <= hi]
        probe_indices = np.linspace(0, len(bucket_pts) - 1, num=queries_per_bucket, dtype=int)
        strict: list[float] = []
        covered: list[float] = []
        for pi in probe_indices:
            probe = bucket_pts[int(pi)]
            query = np.asarray(probe.embedding) + query_noise * rng.standard_normal(vector_dim)
            query_list = [float(v) for v in query]
            truth = oracle.query(query_list, time_window_ms=(lo, hi), limit=k)
            hits = system.query(query_list, time_window_ms=(lo, hi), limit=k)
            strict.append(_recall(hits, truth, k))
            truth_ts = [(h.timestamp_ms, h.scene_id) for h in truth[:k]]
            covered.append(
                sum(1 for ts, scene in truth_ts if any(h.covers(ts, scene) for h in hits))
                / max(1, len(truth_ts))
            )
        strict_by_age.append(_round(float(np.mean(strict))))
        covered_by_age.append(_round(float(np.mean(covered))))

    resident = system.resident_points()
    system.teardown()
    compression = None if not resident else _round(n_points / resident, 2)
    return {
        "task": "recall_vs_age",
        "dataset": ds.name,
        "synthetic": True,
        "n_points": n_points,
        "k": k,
        "n_age_buckets": n_age_buckets,
        "queries_per_bucket": queries_per_bucket,
        "bucket_span_ms": int(edges[1] - edges[0]),
        "recall_strict_by_age": strict_by_age,  # oldest -> newest
        "recall_covered_by_age": covered_by_age,  # oldest -> newest
        "inserted_points": n_points,
        "resident_points": resident,
        "compression_ratio": compression,
        "consolidation_active": resident is not None and resident < n_points,
        "oracle": "brute_force",
    }


# ---------------------------------------------------------------------------
# (e) Sustained-load latency
# ---------------------------------------------------------------------------


def sustained_load(
    system: SystemUnderTest,
    *,
    n_points: int = 600,
    vector_dim: int = 32,
    query_every: int = 5,
    k: int = 10,
    window_ms: int = 10_000,
    query_noise: float = 0.05,
    seed: int = 7,
) -> dict:
    """p50/p95 latency for interleaved single inserts and windowed queries.

    Data timestamps come from the dataset's logical clock; there are no
    sleeps — the stream is replayed as fast as the store accepts it.  Every
    ``query_every`` inserts, one recent-window query runs (noisy embedding of
    a random already-inserted point, window = trailing ``window_ms``).
    Latency IS wall-clock by definition; every other metric in this suite is
    deterministic, latency is machine-dependent — compare across systems on
    the same host, not across hosts.
    """
    ds = SmoothPatrolDataset(n_points=n_points, vector_dim=vector_dim, seed=seed)
    pts = ds.points()
    rng = np.random.default_rng(seed + 3000)
    system.setup(vector_dim)

    insert_ms: list[float] = []
    query_ms: list[float] = []
    for i, point in enumerate(pts):
        t_start = time.perf_counter()
        system.insert(point)
        insert_ms.append((time.perf_counter() - t_start) * 1000)

        if i > 0 and i % query_every == 0:
            probe = pts[int(rng.integers(0, i))]
            query = np.asarray(probe.embedding) + query_noise * rng.standard_normal(vector_dim)
            query_list = [float(v) for v in query]
            window = (point.timestamp_ms - window_ms, point.timestamp_ms)
            t_start = time.perf_counter()
            system.query(query_list, time_window_ms=window, limit=k)
            query_ms.append((time.perf_counter() - t_start) * 1000)
    system.teardown()

    return {
        "task": "sustained_load",
        "dataset": ds.name,
        "synthetic": True,
        "n_inserts": len(insert_ms),
        "n_queries": len(query_ms),
        "insert_p50_ms": _round(float(np.percentile(insert_ms, 50)), 3),
        "insert_p95_ms": _round(float(np.percentile(insert_ms, 95)), 3),
        "query_p50_ms": _round(float(np.percentile(query_ms, 50)), 3),
        "query_p95_ms": _round(float(np.percentile(query_ms, 95)), 3),
        "latency_note": (
            "wall-clock, machine-dependent; all other wm_bench metrics are deterministic"
        ),
    }
