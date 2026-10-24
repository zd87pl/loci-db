# LOCI — Implemented Capabilities and Research Hypotheses

LOCI combines spatiotemporal indexing, temporal collection routing, and
retrieval using a predicted embedding. This document describes that composition
and the evidence still needed. It does not establish that LOCI is the first
system to provide these ideas, that related work lacks them, or that the
combination is faster or more useful on every workload.

## Implementation Status

| Capability | Implemented behavior | Evidence still needed |
|---|---|---|
| 4D Hilbert bucketing | Encodes normalized coordinates and epoch-local time at multiple resolutions | Benefit over tuned Qdrant range filters at matched recall |
| Overlap and geometric filtering | Expands candidate buckets and applies exact bounds before top-k, with a post-check | Recall across boundaries, selectivities, distributions, and scales |
| Temporal collection routing | Stores epochs separately and searches relevant collections | Long-retention operational cost and bounded fan-out |
| Predict-then-retrieve | Calls a supplied predictor, searches analogs, and reports a novelty score | Held-out improvement with a real predictor and real observations |
| Model-output adapters | Converts supplied V-JEPA 2, DreamerV3, and generic arrays into WorldState records | Validated end-to-end model integration and coordinate semantics |
| Persistence | Qdrant stores records across process sessions | Backup/restore, recovery, concurrency, and retention validation |
| Hot/warm/cold migration | Planned | Lifecycle policy, implementation, and recovery tests |

## 1. Spatial Indexing

A Hilbert index maps a quantized 4D coordinate to an integer. LOCI stores
multiple resolutions and combines an integer membership pre-filter with exact
coordinate/time/confidence ranges in the backend and a final geometric check.
The default query resolution is coarse; adaptive selection can choose finer
stored resolutions.

The hypothesis is that this routing/filtering combination reduces work for
some spatial and temporal query distributions. Integer membership is not
inherently cheaper than several indexed ranges: bucket enumeration, payload
index behavior, selectivity, ANN settings, and shard fan-out all affect cost.
No universal complexity reduction or 5–20x speedup is established here.

Overlap addresses candidate coverage near quantized boundaries. Exact ranges
exclude ineligible records before top-k; they do not guarantee exhaustive ANN
retrieval. Measure recall@k against exact ground truth at the same ranking
objective, including temporal decay when it is enabled.

## 2. Temporal Routing and Persistence

Epochs map to separate Qdrant collections, and explicit time windows limit
the collections searched. Async clients can search those collections in
parallel. Qdrant persistence allows stored records to survive client sessions.

Retention policies can delete old epochs; automatic hot/warm/cold migration
is not implemented. These mechanisms also introduce
collection-count, metadata-discovery, fan-out, and concurrent-write costs.
Cross-agent use requires compatible coordinate frames and embeddings, tenant
boundaries, and tested write semantics; sharding alone does not supply them.

## 3. Predict-Then-Retrieve

The API calls a user-supplied predictor once, queries stored observations using
its output, re-ranks candidates, and computes a retrieval-based novelty score.
By default there is no timestamp restriction. Supply an explicit historical
cutoff when evaluating predictions to prevent future observations from leaking
into retrieval. This is an API composition, not a database transaction.

The research hypothesis is that predicted embeddings retrieve more useful
historical analogs than current embeddings for a defined downstream task.
The synthetic patrol harness is a regression demonstration, not evidence of
real-predictor accuracy, generalization, or customer value. Test the hypothesis
on held-out episodes with current-state retrieval, no-retrieval, and appropriate
simple-predictor baselines before making a performance or novelty claim.

## 4. Retrieval-Based Novelty

`prediction_novelty` is bounded to [0, 1] and reflects the strength of retrieved
matches under the implemented scoring/calibration scheme. It is not a
probability of an unseen event or an independently validated risk estimate.
Missing data, incompatible embeddings, filters, or retrieval failures can
change the apparent familiarity of a situation.

Evaluate familiar/novel separation, false alerts, calibration behavior, and
distribution shift on labeled held-out data. Any proposed behavior change for
a physical agent is a separate downstream validation task.

## 5. Adapters and Integration

Adapters map arrays and supplied positions to the WorldState schema. They do
not load model weights, provide inference, or prove production integration.
The V-JEPA 2 helper uses tubelet grid coordinates, including temporal grid
position as `z`; these are not calibrated physical XYZ coordinates. A physical
retrieval workflow must supply its own verified frame/transform convention.

Embedding version, preprocessing, dimension, distance metric, source artifact,
and timestamp semantics need an explicit integration contract. Integration
effort must be measured in the pilot rather than asserted as a minutes-long
setup for every model.

## Prior-Art and Claim Gates

Before publishing a research priority or comparison claim, build a cited review
of spatially filtered ANN, temporal ANN, retrieval-augmented world models,
novelty estimation, and prediction-conditioned retrieval. SpatCode, TANNS,
WorldMem, RA-DT, and HyDE are starting points to verify, not an exhaustive
comparison or evidence that alternatives lack a capability.

Publish reproducible, task-matched comparisons and describe where LOCI wins,
ties, or loses. Separate a useful engineering composition from a new research
contribution. The [implementation readiness plan](IMPLEMENTATION_READINESS.md)
defines the next customer, reliability, and evaluation gates.
